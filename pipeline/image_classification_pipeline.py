#!/usr/bin/env python3
#datashare/pipeline-workspace/pipelines/Image Classification/image_classification_pipeline.py
"""
Image Classification Pipeline
==============================
Location : datashare/pipeline-workspace/pipelines/Image Classification/
Script   : image_classification_pipeline.py

Writes a structured pipeline_log.parquet to the logs/ subfolder so the
Job Execution Monitor can track every run, detect failures, and send alerts.

Usage
-----
  python image_classification_pipeline.py                         # full run
  python image_classification_pipeline.py --steps descriptions    # one step
  python image_classification_pipeline.py --start-index 500       # resume
"""

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

IMAGE_DIR   = Path(os.environ.get('IMAGE_DIR',
    './images/'))
OUTPUT_DIR  = Path(os.environ.get('OUTPUT_DIR',
    'output'))
OLLAMA_PATH = Path(os.environ.get('OLLAMA_PATH', '/usr/local/bin/ollama'))
MODEL_NAME  = os.environ.get('MODEL_NAME', 'llava-llama3')

START_INDEX   = int(os.environ.get('START_INDEX', 0))
SAVE_EVERY    = int(os.environ.get('SAVE_EVERY', 10))
BATCH_SIZE    = int(os.environ.get('BATCH_SIZE', 5))
MAX_DIM       = int(os.environ.get('MAX_DIM', 4000))
TIMEOUT_SEC   = int(os.environ.get('TIMEOUT_SEC', 180))
SIM_THRESHOLD = float(os.environ.get('SIM_THRESHOLD', 0.45))
STEPS         = os.environ.get('STEPS', 'descriptions,method1,method2,report,metadata')
# LIMIT: if > 0, cap image processing to this many images (test mode)
LIMIT         = int(os.environ.get('LIMIT', 0))

# ── Monitor integration ───────────────────────────────────────────────────────
WORKFLOW_NAME = 'image_classification'
LOG_DIR = Path(os.environ.get('LOG_DIR',
    'logs'))

# ── Derived output paths (directories are created inside main() after args) ───
DESCRIPTIONS_CSV = OUTPUT_DIR / 'image_descriptions.csv'
SIMILARITY_CSV   = OUTPUT_DIR / 'method1_semantic_results.csv'
LLAVA_CLASS_CSV  = OUTPUT_DIR / 'method2_llava_results.csv'
METADATA_CSV     = OUTPUT_DIR / 'image_metadata.csv'
APP_LOG_FILE     = LOG_DIR / 'pipeline_app.log'

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING  (human-readable text log + structured parquet log for the monitor)
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('image_pipeline')

# The shared PipelineLogger lives in utils/pipeline_logger.py
sys.path.insert(0, str(Path(__file__).parent))
from utils.pipeline_logger import PipelineLogger


# ──────────────────────────────────────────────────────────────────────────────
# TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────
TAXONOMY = {
    'Historical racialized performance': [
        'blackface','yellowface','minstrel','minstrelsy','caricature','stereotype',
        'brownface','redface','ethnic parody','racial parody','racial impersonation',
        'racial mockery','costume','exoticism','cultural appropriation','racial satire',
        'racist theatre','vaudeville','racial comedy','racial burlesque','mammy','sambo',
        'coon caricature','racial impersonator','racial drag',
    ],
    'Human remains': [
        'skeleton','skull','bones','mummy','mummified','corpse','cadaver','grave',
        'burial','coffin','remains','archaeology','anthropology','tomb','ossuary',
        'funerary','embalmed','mortuary','body','exhumed','cranium','fossil',
        'skeletal','ancestral remains','excavation',
    ],
    'Native American imagery': [
        'chief','headdress','war bonnet','powwow','ceremonial dress','native regalia',
        'indigenous','tribal costume','totem','tribe','chief illiniwek','native dance',
        'ceremonial feathers','sacred objects','appropriation','tribal identity',
        'indian mascot','indigenous symbols','tribal regalia','spiritual costume',
        'tribal headdress','tribal powwow','native caricature','native impersonation',
        'cultural stereotype',
    ],
    'Nudity/sexual content': [
        'nude','naked','explicit','pornographic','erotic','topless','genitalia',
        'sexual','intercourse','sex','lewd','seductive','suggestive','provocative',
        'stripper','striptease','playboy','centerfold','adult content','orgy',
        'minors','child pornography','pedophilia','molestation','sexual exploitation',
        'rape','assault','incest','porn','obscene',
    ],
    'Violence/graphic content': [
        'blood','weapon','knife','gun','shooting','stabbing','murder','killing',
        'injury','corpse','beheading','decapitation','gore','explosion','grenade',
        'rifle','pistol','carcass','slaughter','assault','massacre','terrorist',
        'hostage','execution','battle','war','suicide','bomb','violence',
    ],
    'Hate symbols': [
        'swastika','kkk','klan','confederate flag','nazism','neo-nazi','hitler',
        'fascism','white power','88','heil','skinhead','blood drop cross','14 words',
        'iron cross','totenkopf','stormfront','racist symbol','white supremacy',
        'third reich','gas chamber','holocaust denial','aryan','fascist salute',
        'racist graffiti',
    ],
    'Medical/health records': [
        'medical record','patient chart','hospital','x-ray','mri','ct scan',
        'lab result','diagnosis','treatment','medication','prescription','doctor',
        'nurse','healthcare','surgery','clinic','hospital wristband','health form',
        'insurance card','vaccine','pathology','medical note','confidential',
        'HIPAA','health report',
    ],
    'Student/PII records': [
        'student id','social security number','ssn','address','phone number','email',
        'transcript','grades','disciplinary record','enrollment form','financial aid',
        'birthdate','passport','driver license','uin','registration','student loan',
        'FERPA','identity card','contact info','emergency contact','application',
        'admission letter','personal info','resume','cv',
    ],
    'Other sensitive categories': [
        'terrorist','isis','al-qaeda','bomb','explosive','extremist','jihad',
        'hostage','narcotics','cocaine','heroin','meth','drug use','drug paraphernalia',
        'overdose','suicide','self-harm','cutting','razor','noose','hanging','poison',
        'anorexia','bulimia','eating disorder','addiction','alcohol abuse','gambling',
    ],
}

CATEGORIES_LIST = '\n'.join(f'- {k}' for k in TAXONOMY)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def discover_images(image_dir: Path, start_index: int = 0) -> pd.DataFrame:
    images = []
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg']:
        images.extend(image_dir.glob(ext))
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f'No images found in {image_dir}')
    log.info(f'Found {len(images)} images in {image_dir}')
    rows = [{'row_number': i + 1, 'file_path': str(p.resolve()), 'image_name': p.name}
            for i, p in enumerate(images)]
    df = pd.DataFrame(rows)
    return df[df['row_number'] > start_index].reset_index(drop=True)


def already_processed(csv_path: Path, key_col: str = 'file_path') -> set:
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path)
            if key_col in existing.columns:
                paths = set(existing[key_col].astype(str))
                log.info(f'Resume: {len(paths)} already in {csv_path.name}')
                return paths
        except Exception as e:
            log.warning(f'Could not read {csv_path.name}: {e}')
    return set()


def append_to_csv(rows: list, csv_path: Path):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(csv_path, mode='a', index=False,
                               header=not csv_path.exists())
    log.info(f'Saved {len(rows)} rows → {csv_path.name}')


def run_ollama(image_path: str, prompt: str) -> tuple:
    cmd = [str(OLLAMA_PATH), 'run', MODEL_NAME,
           f'<image>{image_path}</image>\n\n{prompt}']
    try:
        result = subprocess.run(cmd, input='', capture_output=True,
                                text=True, timeout=TIMEOUT_SEC)
        return (result.stdout.strip() or result.stderr.strip()), result.returncode
    except subprocess.TimeoutExpired:
        log.warning(f'Ollama timeout: {image_path}')
        return 'TIMEOUT', 1
    except Exception as e:
        log.error(f'Ollama error ({image_path}): {e}')
        return f'ERROR: {e}', 1


def tiff_to_jpeg_temp(tiff_path: str) -> str:
    p = Path(tiff_path)
    if p.suffix.lower() in ('.jpg', '.jpeg'):
        return tiff_path
    tmp_dir = OUTPUT_DIR / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    out = tmp_dir / (p.stem + '.jpg')
    if out.exists():
        return str(out)
    try:
        with Image.open(p) as img:
            try:
                img.seek(0)
            except Exception:
                pass
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            w, h = img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(out, 'JPEG', quality=85)
    except Exception as e:
        log.error(f'TIFF→JPEG failed for {p.name}: {e}')
        return tiff_path
    return str(out)


# ──────────────────────────────────────────────────────────────────────────────
# STEP FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

NATURAL_PROMPT = (
    "Only describe what you actually see in the image. "
    "Do not guess names, locations, or add context. "
    "If unsure, say 'unknown'."
)
STRICT_PROMPT = (
    "Summarize the image in one or two sentences. "
    "Output must be strict and factual, using only observable details. "
    "Do not explain, infer, or add context."
)


def step_generate_descriptions():
    df = discover_images(IMAGE_DIR, START_INDEX)
    done = already_processed(DESCRIPTIONS_CSV)
    df = df[~df['file_path'].isin(done)].reset_index(drop=True)
    if LIMIT > 0:
        df = df.head(LIMIT)
        log.info(f'[TEST MODE] Capped to {LIMIT} images')
    total = len(df)
    log.info(f'Descriptions to generate: {total}')
    if total == 0:
        return

    pending = []
    for i, row in df.iterrows():
        jpg = tiff_to_jpeg_temp(row['file_path'])
        log.info(f'[{i+1}/{total}] {row["image_name"]}')
        t0 = time.perf_counter()
        nat, _ = run_ollama(jpg, NATURAL_PROMPT)
        nat_t = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        strict, _ = run_ollama(jpg, STRICT_PROMPT)
        strict_t = round(time.perf_counter() - t0, 2)
        pending.append({
            'row_number': row['row_number'], 'file_path': row['file_path'],
            'image_name': row['image_name'],
            'image_description_natural': nat, 'image_description_strict': strict,
            'processing_time_natural': nat_t, 'processing_time_strict': strict_t,
        })
        if len(pending) >= SAVE_EVERY:
            append_to_csv(pending, DESCRIPTIONS_CSV)
            pending.clear()
    append_to_csv(pending, DESCRIPTIONS_CSV)
    log.info(f'Descriptions complete: {total} images')


def step_method1_semantic():
    if not DESCRIPTIONS_CSV.exists():
        raise FileNotFoundError(
            f'Descriptions CSV not found: {DESCRIPTIONS_CSV} — run descriptions step first.')
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    except ImportError as e:
        raise ImportError('pip install sentence-transformers') from e

    desc_df = pd.read_csv(DESCRIPTIONS_CSV)
    done = already_processed(SIMILARITY_CSV)
    desc_df = desc_df[~desc_df['file_path'].astype(str).isin(done)].reset_index(drop=True)
    total = len(desc_df)
    log.info(f'Method 1: classifying {total} images')
    if total == 0:
        return

    model = SentenceTransformer('all-mpnet-base-v2')
    tax_embs = {label: model.encode(kws, normalize_embeddings=True)
                for label, kws in TAXONOMY.items()}

    pending = []
    for i, row in desc_df.iterrows():
        combined = ' '.join(filter(None, [
            str(row.get('image_description_natural') or ''),
            str(row.get('image_description_strict')  or ''),
        ])).strip()
        matched_labels, matched_kws, details = [], [], {}
        if combined and combined not in ('TIMEOUT', 'unknown'):
            desc_emb = model.encode([combined], normalize_embeddings=True)
            for label, kw_embs in tax_embs.items():
                sims = cos_sim(desc_emb, kw_embs)[0]
                hits = [(TAXONOMY[label][j], float(sims[j]))
                        for j, s in enumerate(sims) if s >= SIM_THRESHOLD]
                if hits:
                    matched_labels.append(label)
                    matched_kws.extend(kw for kw, _ in hits)
                    details[label] = hits
        pending.append({
            'row_number': row.get('row_number'), 'file_path': row['file_path'],
            'image_name': row.get('image_name'),
            'sensitive': bool(matched_labels),
            'matched_labels': '; '.join(matched_labels),
            'matched_keywords': '; '.join(set(matched_kws)),
            'details_json': json.dumps(details),
        })
        if (i + 1) % BATCH_SIZE == 0:
            log.info(f'  Classified {i+1}/{total}')
        if len(pending) >= SAVE_EVERY:
            append_to_csv(pending, SIMILARITY_CSV)
            pending.clear()
    append_to_csv(pending, SIMILARITY_CSV)
    log.info(f'Method 1 complete: {total} images')


MODERATION_PROMPT = (
    "You are an image moderation classifier.\n"
    "Determine if the image contains any hate symbols or offensive/insensitive content.\n"
    f"If yes, assign it to ONE of these categories:\n{CATEGORIES_LIST}\n\n"
    "If not offensive, set category and rationale to null.\n\n"
    'Output format (strict JSON only, no preamble):\n'
    '{"offensive": true/false, "category": "category_name_or_null", "rationale": "brief_reason_or_null"}'
)


def _parse_llava_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {'offensive': None, 'category': 'PARSE_ERROR', 'rationale': raw[:300]}


def step_method2_llava():
    df = discover_images(IMAGE_DIR, START_INDEX)

    # Load already processed file paths
    done = already_processed(LLAVA_CLASS_CSV)

    # ✅ FIRST: filter out already processed
    df = df[~df['file_path'].isin(done)].reset_index(drop=True)

    # ✅ THEN: apply test limit ONLY to remaining images
    if LIMIT > 0:
        df = df.head(LIMIT)
        log.info(f'[TEST MODE] Capped to {LIMIT} images AFTER resume filtering')

    total = len(df)
    log.info(f'Method 2: classifying {total} images')

    if total == 0:
        log.info('No new images to process for Method 2 — skipping')
        return

    pending = []

    for i, row in df.iterrows():
        jpg = tiff_to_jpeg_temp(row['file_path'])

        log.info(f'[{i+1}/{total}] {row["image_name"]}')

        t0 = time.perf_counter()
        raw, rc = run_ollama(jpg, MODERATION_PROMPT)
        elapsed = round(time.perf_counter() - t0, 2)

        parsed = _parse_llava_json(raw)

        pending.append({
            'row_number': row['row_number'],
            'file_path': row['file_path'],
            'image_name': row['image_name'],
            'offensive': parsed.get('offensive'),
            'category': parsed.get('category'),
            'rationale': parsed.get('rationale'),
            'raw_output': raw[:500],
            'processing_time': elapsed,
            'return_code': rc,
        })

        if len(pending) >= SAVE_EVERY:
            append_to_csv(pending, LLAVA_CLASS_CSV)
            pending.clear()

    append_to_csv(pending, LLAVA_CLASS_CSV)

    log.info(f'Method 2 complete: {total} images')

def step_generate_report():
    report_csv  = OUTPUT_DIR / 'combined_results.csv'
    flagged_csv = OUTPUT_DIR / 'flagged_images.csv'
    summary_txt = OUTPUT_DIR / 'run_summary.txt'

    dfs = {}
    for key, path in [('desc', DESCRIPTIONS_CSV), ('m1', SIMILARITY_CSV), ('m2', LLAVA_CLASS_CSV)]:
        if path.exists():
            dfs[key] = pd.read_csv(path)
        else:
            log.warning(f'{path.name} missing — skipping in report')
    if not dfs:
        raise FileNotFoundError('No output CSVs found. Run earlier steps first.')

    base = dfs.get('desc', dfs.get('m1', dfs.get('m2')))
    combined = base[['row_number', 'file_path', 'image_name']].copy()
    if 'm1' in dfs:
        m1 = dfs['m1'][['file_path', 'sensitive', 'matched_labels', 'matched_keywords']]
        m1 = m1.rename(columns={'sensitive': 'm1_sensitive', 'matched_labels': 'm1_labels',
                                  'matched_keywords': 'm1_keywords'})
        combined = combined.merge(m1, on='file_path', how='left')
    if 'm2' in dfs:
        m2 = dfs['m2'][['file_path', 'offensive', 'category', 'rationale']]
        m2 = m2.rename(columns={'offensive': 'm2_offensive', 'category': 'm2_category',
                                  'rationale': 'm2_rationale'})
        combined = combined.merge(m2, on='file_path', how='left')

    combined.to_csv(report_csv, index=False)
    flag_mask = pd.Series([False] * len(combined), index=combined.index)
    if 'm1_sensitive' in combined.columns:
        flag_mask |= (combined['m1_sensitive'] == True)
    if 'm2_offensive' in combined.columns:
        flag_mask |= (combined['m2_offensive'] == True)
    flagged = combined[flag_mask]
    flagged.to_csv(flagged_csv, index=False)

    total = len(combined)
    lines = ['='*55, 'Image Classification Pipeline — Run Summary', '='*55,
             f'Total images processed  : {total}',
             f'Flagged by Method 1     : {int(combined["m1_sensitive"].sum()) if "m1_sensitive" in combined else "N/A"}',
             f'Flagged by Method 2     : {int((combined["m2_offensive"]==True).sum()) if "m2_offensive" in combined else "N/A"}',
             f'Flagged by either       : {len(flagged)}', '']
    if 'm1_labels' in combined.columns:
        lines.append('Method 1 — label breakdown:')
        for lbl, cnt in combined['m1_labels'].dropna().str.split('; ').explode().value_counts().items():
            if lbl: lines.append(f'  {lbl}: {cnt}')
        lines.append('')
    if 'm2_category' in combined.columns:
        lines.append('Method 2 — category breakdown:')
        for cat, cnt in combined['m2_category'].value_counts().items():
            lines.append(f'  {cat}: {cnt}')
    summary_txt.write_text('\n'.join(lines))
    log.info(f'Report complete → {len(flagged)} flagged images')
    print('\n'.join(lines))


def step_metadata_extraction():
    try:
        from iptcinfo3 import IPTCInfo
        iptc_ok = True
    except ImportError:
        log.warning('iptcinfo3 not installed — IPTC fields will be empty')
        iptc_ok = False

    def _extract(p: Path) -> dict:
        rec = {
            'file_path': str(p.resolve()),
            'image_name': p.name,
            'format': p.suffix.upper().lstrip('.'),
            'file_size_bytes': p.stat().st_size,
            'original_filename': p.name,
            'original_width': None,
            'original_height': None,
            'color_mode': None,
            'error': None,
            'metadata_error': None,
            'dc_title': None,
            'dc_description': None,
            'dc_subject': None,
            'dc_creator': None,
            'dc_date': None,
            'dc_rights': None,
        }

        try:
            with Image.open(p) as img:
                rec['original_width'] = img.width
                rec['original_height'] = img.height
                rec['color_mode'] = img.mode
        except Exception as e:
            rec['error'] = str(e)

        if iptc_ok and p.suffix.lower() in ('.tif', '.tiff', '.jpg', '.jpeg'):
            try:
                info = IPTCInfo(str(p), force=True)

                def _iptc(tag):
                    v = info.data.get(tag)
                    if isinstance(v, list):
                        return '; '.join(
                            x.decode() if isinstance(x, bytes) else str(x) for x in v
                        )
                    return v.decode() if isinstance(v, bytes) else (str(v) if v else None)

                rec['dc_title'] = _iptc('object name')
                rec['dc_description'] = _iptc('caption/abstract')
                rec['dc_subject'] = _iptc('keywords')
                rec['dc_creator'] = _iptc('by-line')
                rec['dc_date'] = _iptc('date created')
                rec['dc_rights'] = _iptc('copyright notice')

            except Exception as e:
                rec['metadata_error'] = str(e)

        return rec

    df = discover_images(IMAGE_DIR, START_INDEX)

    # Load already processed
    done = already_processed(METADATA_CSV)

    # ✅ FIRST: filter out processed images
    df = df[~df['file_path'].isin(done)].reset_index(drop=True)

    # ✅ THEN: apply LIMIT to remaining only
    if LIMIT > 0:
        df = df.head(LIMIT)
        log.info(f'[TEST MODE] Capped to {LIMIT} images AFTER resume filtering')

    total = len(df)

    if total == 0:
        log.info('No new images to process for metadata — skipping')
        return

    pending = []

    for i, row in df.iterrows():
        pending.append(_extract(Path(row['file_path'])))

        if (i + 1) % 100 == 0:
            log.info(f'  Extracted {i+1}/{total}')

        if len(pending) >= SAVE_EVERY:
            append_to_csv(pending, METADATA_CSV)
            pending.clear()

    append_to_csv(pending, METADATA_CSV)

    log.info(f'Metadata extraction complete: {total} images')


# ──────────────────────────────────────────────────────────────────────────────
# STEP REGISTRY  (name → function, script_order)
# ──────────────────────────────────────────────────────────────────────────────
STEP_MAP = {
    'descriptions': (step_generate_descriptions, 1),
    'method1':      (step_method1_semantic,       2),
    'method2':      (step_method2_llava,           3),
    'report':       (step_generate_report,         4),
    'metadata':     (step_metadata_extraction,     5),
}


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir',   type=Path,  default=IMAGE_DIR)
    p.add_argument('--output-dir',  type=Path,  default=OUTPUT_DIR)
    p.add_argument('--log-dir',     type=Path,  default=LOG_DIR)
    p.add_argument('--ollama-path', type=Path,  default=OLLAMA_PATH)
    p.add_argument('--model',       type=str,   default=MODEL_NAME)
    p.add_argument('--start-index', type=int,   default=START_INDEX)
    p.add_argument('--steps',       type=str,   default=STEPS)
    p.add_argument('--threshold',   type=float, default=SIM_THRESHOLD)
    p.add_argument('--timeout',     type=int,   default=TIMEOUT_SEC)
    p.add_argument('--save-every',  type=int,   default=SAVE_EVERY)
    p.add_argument('--limit',       type=int,   default=LIMIT,
                   help='Cap processing to N images (test mode). 0 = no limit.')
    return p.parse_args()


def main():
    global IMAGE_DIR, OUTPUT_DIR, LOG_DIR, OLLAMA_PATH, MODEL_NAME
    global START_INDEX, SIM_THRESHOLD, TIMEOUT_SEC, SAVE_EVERY, STEPS, LIMIT

    if not any('ipykernel' in a for a in sys.argv):
        args = parse_args()
        IMAGE_DIR     = args.image_dir
        OUTPUT_DIR    = args.output_dir
        LOG_DIR       = args.log_dir
        OLLAMA_PATH   = args.ollama_path
        MODEL_NAME    = args.model
        START_INDEX   = args.start_index
        SIM_THRESHOLD = args.threshold
        TIMEOUT_SEC   = args.timeout
        SAVE_EVERY    = args.save_every
        STEPS         = args.steps
        LIMIT         = args.limit
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Rebuild derived paths now that OUTPUT_DIR / LOG_DIR are finalised
        global DESCRIPTIONS_CSV, SIMILARITY_CSV, LLAVA_CLASS_CSV, METADATA_CSV, APP_LOG_FILE
        DESCRIPTIONS_CSV = OUTPUT_DIR / 'image_descriptions.csv'
        SIMILARITY_CSV   = OUTPUT_DIR / 'method1_semantic_results.csv'
        LLAVA_CLASS_CSV  = OUTPUT_DIR / 'method2_llava_results.csv'
        METADATA_CSV     = OUTPUT_DIR / 'image_metadata.csv'
        APP_LOG_FILE     = LOG_DIR    / 'pipeline_app.log'

        # Add file handler now that LOG_DIR exists
        log.addHandler(logging.FileHandler(APP_LOG_FILE))

    steps_to_run = [s.strip() for s in STEPS.split(',') if s.strip()]
    invalid = [s for s in steps_to_run if s not in STEP_MAP]
    if invalid:
        log.error(f'Unknown steps: {invalid}. Valid: {list(STEP_MAP)}')
        sys.exit(1)

    mode_tag = f'[TEST MODE — limit {LIMIT}]' if LIMIT > 0 else '[PRODUCTION]'
    log.info(f'Pipeline starting {mode_tag} — workflow: {WORKFLOW_NAME}')
    log.info(f'Steps: {steps_to_run}  |  Image dir: {IMAGE_DIR}  |  Log dir: {LOG_DIR}')

    version_tag = f"test-limit-{LIMIT}" if LIMIT > 0 else "production"
    pipeline_logger = PipelineLogger(
        workflow_name=WORKFLOW_NAME,
        log_dir=LOG_DIR,
        pipeline_version=version_tag,
    )

    exit_code = 0
    with pipeline_logger.pipeline_run():
        for step_name in steps_to_run:
            fn, order = STEP_MAP[step_name]
            log.info(f'--- Step: {step_name} (order={order}) ---')
            try:
                with pipeline_logger.stage(step_name, order=order):
                    fn()
                    pipeline_logger.set_stage_metadata(
                        output_file=str({
                            'descriptions': DESCRIPTIONS_CSV,
                            'method1':      SIMILARITY_CSV,
                            'method2':      LLAVA_CLASS_CSV,
                            'report':       OUTPUT_DIR / 'combined_results.csv',
                            'metadata':     METADATA_CSV,
                        }.get(step_name, OUTPUT_DIR)),
                        output_file_written=True,
                    )
            except Exception as e:
                log.error(f'Step "{step_name}" failed: {e}')
                exit_code = 1

    log.info(f'Structured log → {pipeline_logger.log_path}')
    log.info(f'Pipeline finished {mode_tag} — exit code {exit_code}')
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
