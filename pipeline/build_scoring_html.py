#!/usr/bin/env python3
"""
build_scoring_html.py
======================
Build an HTML viewer for the variant-comparison audit.

Reads side_by_side.csv and the actual image files, produces a single HTML
file showing each image alongside variant A and C's outputs and inputs for
manual scoring. Designed so the human reviewer never has to alt-tab between
the image, the text outputs, and a spreadsheet.

Output: <run_dir>/analysis/scoring_view.html

The HTML page:
  - Shows all 50 images stacked vertically
  - Each image rendered inline (base64 embedded — single self-contained file)
  - Side-by-side A_natural, A_strict, parsed C JSON fields
  - Y/N radio buttons for the 4 score columns, dropdown for verdict, text
    box for notes
  - Local-only JavaScript that builds an export CSV when you click 'Export'
  - Nothing is saved server-side; export drops the filled CSV in your
    browser's download folder

Usage:
    python3 build_scoring_html.py
    python3 build_scoring_html.py --run-dir output/benchmark_prompts/run_20260512_182032
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BENCHMARK_DIR = Path(
    "./pipeline-workspace/pipelines/"
    "Image Classification/output/benchmark_prompts"
)
DEFAULT_IMAGE_DIR = Path(
    "./image_data/"
    "All_Scanned_Images/Scanned_Images_Combined"
)
MAX_IMAGE_KB = 400   # downscale very large JPEGs for embed; preserves browser perf


def find_latest_run(benchmark_dir: Path) -> Optional[Path]:
    if not benchmark_dir.exists():
        return None
    runs = sorted(
        [p for p in benchmark_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return {}


def image_to_b64_thumb(img_path: Path) -> str:
    """
    Convert image to base64 data URI. Downscales if larger than MAX_IMAGE_KB.
    Keeps the file self-contained — no separate image hosting needed.
    """
    if not img_path.exists():
        return ""
    try:
        # Use PIL if image is large; else read raw
        size_kb = img_path.stat().st_size / 1024
        if size_kb > MAX_IMAGE_KB:
            from PIL import Image
            from io import BytesIO
            img = Image.open(img_path)
            img = img.convert("RGB")
            # Downscale to roughly fit MAX_IMAGE_KB
            max_dim = 1200
            w, h = img.size
            scale = max(w, h) / max_dim
            if scale > 1:
                img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80, optimize=True)
            data = buf.getvalue()
        else:
            data = img_path.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    except Exception as e:
        print(f"  [warn] {img_path.name}: {e}", file=sys.stderr)
        return ""


def build_html(rows: List[Dict[str, str]], image_dir: Path) -> str:
    """Construct the full HTML document as one string."""
    cards: List[str] = []
    for i, row in enumerate(rows, start=1):
        image_name = row["image_name"]
        a_natural = row.get("A_natural", "")
        a_strict = row.get("A_strict", "")
        a_mod = extract_json(row.get("A_moderation", ""))
        b_title = row.get("B_title", "")
        c_obj = extract_json(row.get("C_combined", ""))

        c_title = c_obj.get("title", "")
        c_description = c_obj.get("description", "")
        c_offensive = c_obj.get("offensive", "")
        c_category = c_obj.get("offense_category", "") or ""
        c_reason = c_obj.get("offense_reason", "") or ""
        c_confidence = c_obj.get("confidence", "")

        a_offensive = a_mod.get("offensive", "")
        a_category = a_mod.get("category", "") or "—"
        a_reason = a_mod.get("rationale", "") or "—"

        # Render image
        img_path = image_dir / image_name
        img_data = image_to_b64_thumb(img_path)
        if not img_data:
            img_html = '<div class="img-missing">image not found</div>'
        else:
            img_html = f'<img src="{img_data}" alt="{image_name}" loading="lazy">'

        # Warmup-tagged rows are mostly empty — flag them so reviewer knows to skip
        is_empty = (not a_natural.strip() and not c_description.strip())
        warmup_tag = '<span class="warmup-tag">⚠ WARMUP (low data)</span>' if is_empty else ""

        # Build the card
        card = f"""
<div class="card" id="card-{i}" data-image="{html.escape(image_name)}">
  <div class="card-header">
    <div class="card-title">#{i} — {html.escape(image_name)} {warmup_tag}</div>
  </div>
  <div class="card-body">
    <div class="image-col">
      {img_html}
    </div>
    <div class="text-col">
      <div class="output-block">
        <div class="output-label">A_natural (current pipeline)</div>
        <div class="output-text">{html.escape(a_natural) or "<em>(empty)</em>"}</div>
      </div>
      <div class="output-block">
        <div class="output-label">A_strict</div>
        <div class="output-text">{html.escape(a_strict) or "<em>(empty)</em>"}</div>
      </div>
      <div class="output-block">
        <div class="output-label">A_moderation</div>
        <div class="output-text">
          offensive: <b>{html.escape(str(a_offensive))}</b><br>
          category: {html.escape(str(a_category))}<br>
          rationale: {html.escape(str(a_reason))}
        </div>
      </div>
      <div class="divider"></div>
      <div class="output-block c-block">
        <div class="output-label">C_title (combined prompt)</div>
        <div class="output-text">{html.escape(str(c_title)) or "<em>(empty)</em>"}</div>
      </div>
      <div class="output-block c-block">
        <div class="output-label">C_description</div>
        <div class="output-text">{html.escape(str(c_description)) or "<em>(empty)</em>"}</div>
      </div>
      <div class="output-block c-block">
        <div class="output-label">C_moderation</div>
        <div class="output-text">
          offensive: <b>{html.escape(str(c_offensive))}</b><br>
          category: {html.escape(str(c_category))}<br>
          reason: {html.escape(str(c_reason))}<br>
          confidence: {html.escape(str(c_confidence))}
        </div>
      </div>
      <div class="output-block">
        <div class="output-label">B_title</div>
        <div class="output-text">{html.escape(b_title) or "<em>(empty)</em>"}</div>
      </div>
    </div>
    <div class="score-col">
      <div class="score-heading">Score</div>

      <div class="score-section">A — current pipeline</div>
      <div class="score-row">
        <label>A hallucinated details?</label>
        <span class="yn">
          <label><input type="radio" name="a-hall-{i}" value="Y">Y</label>
          <label><input type="radio" name="a-hall-{i}" value="N">N</label>
        </span>
      </div>
      <div class="score-row">
        <label>A missed details?</label>
        <span class="yn">
          <label><input type="radio" name="a-miss-{i}" value="Y">Y</label>
          <label><input type="radio" name="a-miss-{i}" value="N">N</label>
        </span>
      </div>

      <div class="score-section">C — combined prompt</div>
      <div class="score-row">
        <label>C hallucinated details?</label>
        <span class="yn">
          <label><input type="radio" name="c-hall-{i}" value="Y">Y</label>
          <label><input type="radio" name="c-hall-{i}" value="N">N</label>
        </span>
      </div>
      <div class="score-row">
        <label>C missed details?</label>
        <span class="yn">
          <label><input type="radio" name="c-miss-{i}" value="Y">Y</label>
          <label><input type="radio" name="c-miss-{i}" value="N">N</label>
        </span>
      </div>

      <div class="score-section">Comparison</div>
      <div class="score-row">
        <label>Moderation agrees?</label>
        <span class="yn">
          <label><input type="radio" name="modagree-{i}" value="Y">Y</label>
          <label><input type="radio" name="modagree-{i}" value="N">N</label>
        </span>
      </div>
      <div class="score-row">
        <label>More accurate to image?</label>
        <select name="winner-{i}">
          <option value=""></option>
          <option value="A">A is more accurate</option>
          <option value="C">C is more accurate</option>
          <option value="tied">tied</option>
          <option value="both-wrong">both wrong</option>
        </select>
      </div>
      <div class="score-row">
        <label>Verdict (overall pipeline quality)</label>
        <select name="verdict-{i}">
          <option value=""></option>
          <option value="good">good (either output usable)</option>
          <option value="drift">drift (different but reasonable)</option>
          <option value="wrong">wrong (material errors in both)</option>
        </select>
      </div>
      <div class="score-row">
        <label>Notes</label>
        <textarea name="notes-{i}" rows="2" placeholder="optional..."></textarea>
      </div>
    </div>
  </div>
</div>
"""
        cards.append(card)

    cards_html = "\n".join(cards)

    # Whole-page styles + export script
    html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Variant comparison — scoring audit</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 1.5rem; background: #f5f5f7; color: #1d1d1f;
  }
  h1 { margin: 0 0 0.5rem; }
  .intro {
    background: white; padding: 1rem 1.5rem; border-radius: 6px;
    margin-bottom: 1rem; max-width: 800px;
  }
  .intro p { margin: 0.4rem 0; }
  .intro code { background: #f0f0f3; padding: 0 0.2rem; border-radius: 3px; }
  .export-bar {
    position: sticky; top: 0; background: white; padding: 0.7rem 1rem;
    border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 1rem; display: flex; gap: 0.7rem; align-items: center;
    z-index: 100;
  }
  .export-bar button {
    padding: 0.5rem 1rem; font-size: 0.95rem; border-radius: 4px;
    border: 1px solid #1d6dd8; background: #2e7ee5; color: white;
    cursor: pointer;
  }
  .export-bar button:hover { background: #1d6dd8; }
  .export-bar .status { color: #666; font-size: 0.9rem; }

  .card {
    background: white; border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 1rem; padding: 1rem;
  }
  .card-header { margin-bottom: 0.5rem; }
  .card-title {
    font-weight: 600; font-size: 1.05rem;
  }
  .warmup-tag {
    color: #b88500; background: #fff4d4; padding: 2px 6px;
    border-radius: 3px; font-size: 0.8rem; margin-left: 0.5rem;
  }
  .card-body {
    display: grid; grid-template-columns: 340px 1fr 280px; gap: 1rem;
  }
  .image-col img {
    max-width: 320px; max-height: 320px; width: auto; height: auto;
    border: 1px solid #ddd; border-radius: 4px;
  }
  .image-col .img-missing {
    width: 320px; height: 240px; display: flex; align-items: center;
    justify-content: center; border: 1px dashed #999;
    color: #999; font-style: italic;
  }
  .text-col { font-size: 0.9rem; }
  .output-block { margin-bottom: 0.6rem; }
  .output-label {
    font-weight: 600; color: #555; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.03em; margin-bottom: 0.15rem;
  }
  .c-block .output-label { color: #8b3f8f; }
  .output-text {
    background: #f9f9fb; padding: 0.4rem 0.6rem; border-radius: 3px;
    border-left: 3px solid #ccc;
  }
  .c-block .output-text { border-left-color: #d4a8d6; }
  .divider {
    border-top: 1px solid #e0e0e3; margin: 0.6rem 0;
  }
  .score-col {
    background: #f9f9fb; padding: 0.7rem; border-radius: 4px;
    border: 1px solid #e0e0e3;
  }
  .score-heading {
    font-weight: 600; margin-bottom: 0.5rem; font-size: 0.95rem;
  }
  .score-section {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
    color: #2e7ee5; font-weight: 700;
    margin: 0.6rem 0 0.3rem; padding-bottom: 2px;
    border-bottom: 1px solid #d4dbe8;
  }
  .score-section:first-of-type { margin-top: 0; }
  .score-row {
    margin-bottom: 0.4rem; display: flex; flex-direction: column;
    font-size: 0.82rem;
  }
  .score-row > label {
    font-weight: 500; color: #444; margin-bottom: 0.2rem;
  }
  .yn label {
    margin-right: 0.6rem; font-size: 0.9rem; font-weight: 400;
  }
  .yn input { margin-right: 0.2rem; }
  .score-row select, .score-row textarea {
    font-family: inherit; font-size: 0.85rem; padding: 0.25rem;
    border: 1px solid #ccc; border-radius: 3px;
  }

  @media (max-width: 1100px) {
    .card-body { grid-template-columns: 1fr; }
    .image-col img { max-width: 100%; max-height: 400px; }
  }
</style>
</head>
<body>

<h1>Variant comparison — scoring audit</h1>
<div class="intro">
  <p>For each image, compare what <strong>A</strong> said (current pipeline,
     three prompts) against what <strong>C</strong> said (combined single
     prompt). Fill in the Y/N + verdict for each card.</p>
  <p>Definitions:</p>
  <ul>
    <li><b>Primary subject match</b>: do A and C describe the same main thing?</li>
    <li><b>C added inference</b>: did C invent details A didn't claim?
        (e.g., A says "stadium," C says "stadium where workers are building")</li>
    <li><b>C missed details</b>: did C omit concrete details A captured?</li>
    <li><b>Moderation agrees</b>: do A_offensive and C_offensive match?</li>
    <li><b>Verdict</b>: <code>good</code> (C equivalent or better),
        <code>drift</code> (different but reasonable),
        <code>wrong</code> (C describes something materially incorrect).</li>
  </ul>
  <p>Cards tagged <span class="warmup-tag">⚠ WARMUP</span> have incomplete data
     because their inference calls were excluded from the measured stats as
     warmup. Skip those — there's nothing to score.</p>
  <p>When done, click <b>Export CSV</b> at the top. The file downloads to
     your browser's Downloads folder.</p>
</div>

<div class="export-bar">
  <button onclick="exportCSV()">Export CSV</button>
  <button onclick="exportProgress()">Quick stats</button>
  <span class="status" id="status"></span>
</div>

""" + cards_html + """

<script>
function getRowValues(i) {
  const card = document.getElementById('card-' + i);
  const imageName = card.dataset.image;
  function radioVal(name) {
    const el = card.querySelector('input[name="' + name + '-' + i + '"]:checked');
    return el ? el.value : '';
  }
  function selVal(name) {
    const el = card.querySelector('[name="' + name + '-' + i + '"]');
    return el ? el.value : '';
  }
  function txtVal(name) {
    const el = card.querySelector('[name="' + name + '-' + i + '"]');
    return el ? el.value : '';
  }
  return {
    image_name: imageName,
    score_a_hallucinated: radioVal('a-hall'),
    score_a_missed_details: radioVal('a-miss'),
    score_c_hallucinated: radioVal('c-hall'),
    score_c_missed_details: radioVal('c-miss'),
    score_moderation_agrees: radioVal('modagree'),
    more_accurate: selVal('winner'),
    verdict: selVal('verdict'),
    notes: txtVal('notes'),
  };
}

function getAllRows() {
  const cards = document.querySelectorAll('.card');
  const rows = [];
  for (let i = 1; i <= cards.length; i++) {
    rows.push(getRowValues(i));
  }
  return rows;
}

function exportCSV() {
  const rows = getAllRows();
  const cols = ['image_name',
                'score_a_hallucinated', 'score_a_missed_details',
                'score_c_hallucinated', 'score_c_missed_details',
                'score_moderation_agrees',
                'more_accurate', 'verdict', 'notes'];
  const csv = [cols.join(',')];
  for (const r of rows) {
    csv.push(cols.map(c => '"' + (r[c] || '').replace(/"/g, '""') + '"').join(','));
  }
  const blob = new Blob([csv.join('\\n')], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'scoring_filled.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
  document.getElementById('status').textContent = 'Exported scoring_filled.csv';
}

function exportProgress() {
  const rows = getAllRows();
  let scored = 0, good = 0, drift = 0, wrong = 0;
  let a_better = 0, c_better = 0, tied = 0, both_wrong = 0;
  let a_hall = 0, c_hall = 0;
  for (const r of rows) {
    if (r.verdict) {
      scored++;
      if (r.verdict === 'good') good++;
      else if (r.verdict === 'drift') drift++;
      else if (r.verdict === 'wrong') wrong++;
    }
    if (r.more_accurate === 'A') a_better++;
    else if (r.more_accurate === 'C') c_better++;
    else if (r.more_accurate === 'tied') tied++;
    else if (r.more_accurate === 'both-wrong') both_wrong++;
    if (r.score_a_hallucinated === 'Y') a_hall++;
    if (r.score_c_hallucinated === 'Y') c_hall++;
  }
  document.getElementById('status').textContent =
    'Scored ' + scored + '/' + rows.length +
    ' | good=' + good + ' drift=' + drift + ' wrong=' + wrong +
    ' | A_better=' + a_better + ' C_better=' + c_better + ' tied=' + tied +
    ' | A_hallucinated=' + a_hall + ' C_hallucinated=' + c_hall;
}
</script>

</body>
</html>
"""
    return html_doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
def build_rows_from_jsonl(run_dir: Path) -> List[Dict[str, str]]:
    """
    Build the per-image row structure directly from outputs.jsonl.

    Unlike side_by_side.csv (which analyze_prompt_outputs.py produces with
    warmup calls FILTERED OUT), this includes every call — warmup or not.
    For a content audit you want all 50 images visible, not just the
    non-warmup subset. Latency stats still exclude warmup; content review
    should not.

    Returns a list of dicts with the same column shape build_html expects:
      image_name, A_natural, A_strict, A_moderation, B_title, B_natural,
      B_strict, B_moderation, C_combined
    """
    jsonl_path = run_dir / "outputs.jsonl"
    if not jsonl_path.exists():
        return []

    # repeat 0 only (consistency analysis confirmed repeats identical at temp=0)
    by_image: Dict[str, Dict[str, str]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("repeat", 0) != 0:
                continue
            if not r.get("success", False):
                continue
            img_name = Path(r["image_path"]).name
            col = f"{r['variant']}_{r['prompt_name']}"
            by_image.setdefault(img_name, {"image_name": img_name})[col] = r.get("output", "")

    # Stable column order, fill missing with ""
    column_order = [
        "image_name",
        "A_natural", "A_strict", "A_moderation",
        "B_title", "B_natural", "B_strict", "B_moderation",
        "C_combined",
    ]
    rows = []
    for img_name in sorted(by_image.keys()):
        row = {col: by_image[img_name].get(col, "") for col in column_order}
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--from-sbs", action="store_true",
                   help="Read side_by_side.csv instead of outputs.jsonl "
                        "(excludes warmup rows). Default reads outputs.jsonl "
                        "so all images, including warmup, are shown.")
    args = p.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.benchmark_dir)
        if run_dir is None:
            sys.exit(f"No runs found in {args.benchmark_dir}")
    run_dir = run_dir.resolve()

    analysis_dir = run_dir / "analysis"
    if not analysis_dir.exists():
        analysis_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out or (analysis_dir / "scoring_view.html")

    if args.from_sbs:
        sbs_path = analysis_dir / "side_by_side.csv"
        if not sbs_path.exists():
            sys.exit(f"Missing {sbs_path}\nRun analyze_prompt_outputs.py first, "
                      f"or drop --from-sbs to read outputs.jsonl directly.")
        print(f"Reading: {sbs_path} (warmup rows excluded)")
        with open(sbs_path) as f:
            rows = list(csv.DictReader(f))
    else:
        jsonl_path = run_dir / "outputs.jsonl"
        if not jsonl_path.exists():
            sys.exit(f"Missing {jsonl_path}")
        print(f"Reading: {jsonl_path} (all rows, including warmup)")
        rows = build_rows_from_jsonl(run_dir)

    print(f"Image rows: {len(rows)}")
    print(f"Image dir : {args.image_dir}")
    print(f"Output    : {out_path}")

    html_str = build_html(rows, args.image_dir)

    print(f"Writing {out_path} ({len(html_str):,} chars)...")
    out_path.write_text(html_str, encoding="utf-8")
    print(f"\n  Done. Open this file in a browser:")
    print(f"    {out_path}")
    print(f"\n  Or via Jupyter: navigate to the analysis/ folder and double-click")
    print(f"  scoring_view.html (Jupyter will render it).")


if __name__ == "__main__":
    main()
