#!/usr/bin/env python3
"""
benchmark_prompts.py
====================
Controlled experiment: does a combined single-prompt produce equivalent
per-field output quality to multiple separate prompts?

Why this exists
---------------
The current pipeline calls the model 3 times per image (natural description,
strict description, moderation classification). A combined single-prompt
approach could be 3-4x cheaper if quality holds — but smaller multimodal
models often degrade when asked for many structured fields at once.

This script runs three variants on the same image sample and measures:
  - Per-call latency, output tokens, decode throughput
  - Per-field parse rate (was each field present and well-formed?)
  - Raw model outputs (for human spot-check of content quality)

Variants
--------
  A — Current 3-prompt pipeline (natural, strict, moderation)
  B — 4-prompt pipeline (adds 'title' as a separate call)
  C — Combined single-prompt: title + description + offensive + reason + category + confidence in ONE call

All variants use HTTP /api/generate with keep_alive=24h, so the model stays
resident across all calls. (Subprocess vs HTTP shown to be equivalent in
prior VRAM-trace investigation; HTTP gives us richer timing fields.)

Output
------
  benchmark_prompts/run_<timestamp>/
    inference_timings.parquet    one row per inference call
    resource_samples.parquet     GPU/CPU/memory traces
    stage_timings.parquet        one row per variant
    field_parses.parquet         one row per (image, variant, field) → parsed_ok + value
    outputs.jsonl                all raw model outputs
    prompt_comparison_report.md  human-readable summary
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGE_DIR = Path(
    './images/'
)
DEFAULT_OUTPUT_DIR = Path(
    'output/benchmark_prompts'
)
DEFAULT_MODEL = 'llava-llama3'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 11434
DEFAULT_NUM_IMAGES = 50
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT = 180
DEFAULT_WARMUP = 2
DEFAULT_MAX_DIM = 4000
DEFAULT_JPEG_QUALITY = 85
DEFAULT_NATURAL_NUM_PREDICT = 120   # ~4 sentences cap (user request)

# The 9 allowed sensitivity categories — matches image_classification_pipeline.py taxonomy
ALLOWED_CATEGORIES = {
    "Historical racialized performance",
    "Human remains",
    "Native American imagery",
    "Nudity/sexual content",
    "Violence/graphic content",
    "Hate symbols",
    "Medical/health records",
    "Student/PII records",
    "Other sensitive categories",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


# ──────────────────────────────────────────────────────────────────────────────
# Prompts — Variants A and B reuse the production prompts verbatim
# ──────────────────────────────────────────────────────────────────────────────

NATURAL_PROMPT = (
    "Only describe what you actually see in the image. "
    "Describe in up to four sentences. "
    "Do not guess names, locations, or add context. "
    "If unsure, say 'unknown'."
)
STRICT_PROMPT = (
    "Summarize the image in one or two sentences. "
    "Output must be strict and factual, using only observable details. "
    "Do not explain, infer, or add context."
)
TITLE_PROMPT = (
    "Give a brief title for this image, 4 to 8 words. "
    "Use only observable details. Output only the title, no quotes, no explanation."
)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt v2 (May 13, 2026) — stricter framing on what counts as "offensive"
# ──────────────────────────────────────────────────────────────────────────────
# Two changes from v1:
#   1. Replace the broad "offensive/insensitive content" framing with an
#      explicit rule: only flag if content matches one of the listed
#      categories. Difficult subject matter (war, politics, controversial
#      history) is NOT flag-worthy unless it specifically contains content
#      matching a category (e.g., violence/graphic imagery, hate symbols).
#      Surfaced by image 0000049 (Vietnam protest) being flagged in v1.
#
#   2. Explicitly name "Other sensitive categories" as the catch-all for
#      novel cases that fit the spirit of the taxonomy but don't match a
#      named category. Surfaced by image 0000049 — model invented
#      "War imagery" instead of using the existing catch-all.

CATEGORIES_LIST = "\n".join(f"- {c}" for c in ALLOWED_CATEGORIES)

MODERATION_PROMPT = (
    "You are an image moderation classifier for an archival library workflow.\n\n"
    "Decide whether to flag the image. Flag ONLY if the image contains content "
    "matching one of these categories:\n"
    f"{CATEGORIES_LIST}\n\n"
    "DO NOT flag images merely because they show difficult subjects "
    "(war, protest, politics, controversial history, illness, poverty). "
    "Documentation of difficult topics is not the same as offensive content. "
    "Flag only if the image itself contains the listed category content "
    "(e.g., visible hate symbols, graphic violence in progress, nudity).\n\n"
    "If the image fits the spirit of the taxonomy but doesn't match a named "
    "category, use 'Other sensitive categories'. Do not invent new category "
    "names.\n\n"
    "Output strict JSON only, no preamble:\n"
    '{"offensive": true/false, "category": "exact_category_name_or_null", '
    '"rationale": "brief_reason_or_null"}'
)

COMBINED_PROMPT = (
    # ── C-v4 (May 13, smoke-tested) ──
    # Stronger anti-flag framing with explicit examples; inline categories
    # list removed to test whether the long prompt was inflating decode time.
    # Smoke test on 9 images showed: fixes the Vietnam mis-flag (the v2
    # failure case); regresses ONE image's title to >100 chars; mean wall
    # time ~7.2s/image. This run is the full 50×3 validation.
    "You are analyzing an archival photograph or document scan.\n\n"
    "CRITICAL RULE — READ FIRST:\n"
    "The 'offensive' flag means the IMAGE ITSELF contains content from one of "
    "the allowed sensitivity categories. It does NOT mean the image's SUBJECT "
    "is controversial, sad, political, or about a difficult historical period.\n\n"
    "EXAMPLES OF WHAT IS NOT OFFENSIVE (set offensive=false):\n"
    "- A photo of war protesters holding 'Stop the War' signs.\n"
    "- A photo of soldiers in uniform.\n"
    "- Civil rights marches, political rallies, demonstrations.\n"
    "- Documentation of any difficult historical event.\n\n"
    "EXAMPLES OF WHAT IS OFFENSIVE (set offensive=true):\n"
    "- A swastika or other hate symbol visibly displayed.\n"
    "- Graphic violence happening in the image (blood, weapon mid-use).\n"
    "- Visible nudity.\n"
    "- Visible patient records or medical chart with identifiable info.\n\n"
    "OTHER RULES:\n"
    "- title: MAX 8 WORDS. Noun phrase, NOT a sentence.\n"
    "- description: up to 4 sentences. Only observable details.\n"
    "- offense_category: if offensive=true, exactly ONE category from the list "
    "of 9 sensitivity categories used by this workflow (historical racialized "
    "performance, human remains, Native American imagery, nudity/sexual "
    "content, violence/graphic content, hate symbols, medical/health records, "
    "student/PII records, or 'Other sensitive categories' as a catch-all). "
    "Use the exact category name. NEVER invent new categories.\n"
    "- offense_reason: if offensive=true, one sentence. Null otherwise.\n"
    "- confidence: low / medium / high.\n\n"
    "Output a single JSON object:\n"
    "{\n"
    '  "title": "<noun phrase, MAX 8 words>",\n'
    '  "description": "<up to 4 sentences>",\n'
    '  "offensive": <true or false>,\n'
    '  "offense_category": "<category name or null>",\n'
    '  "offense_reason": "<one sentence or null>",\n'
    '  "confidence": "<low | medium | high>"\n'
    "}\n\n"
    "Output strict JSON only. No preamble. No markdown fences."
)

# Per-variant prompt lists
VARIANT_A_PROMPTS = [
    ("natural", NATURAL_PROMPT, "text"),
    ("strict", STRICT_PROMPT, "text"),
    ("moderation", MODERATION_PROMPT, "json"),
]
VARIANT_B_PROMPTS = [
    ("title", TITLE_PROMPT, "text"),
    ("natural", NATURAL_PROMPT, "text"),
    ("strict", STRICT_PROMPT, "text"),
    ("moderation", MODERATION_PROMPT, "json"),
]
VARIANT_C_PROMPTS = [
    ("combined", COMBINED_PROMPT, "json"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Console output helpers
# ──────────────────────────────────────────────────────────────────────────────

# Module-level flag toggled by --quiet. When True, per-call info() output is
# suppressed; banner() and ok()/fail() still print so progress is trackable.
QUIET = False


def banner(s: str):
    print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}", flush=True)


def info(s: str = ""):
    if not QUIET:
        print(f"  {s}", flush=True)


def info_always(s: str = ""):
    """Print regardless of --quiet — for milestones, not per-call detail."""
    print(f"  {s}", flush=True)


def ok(s: str):
    print(f"  ✓ {s}", flush=True)


def fail(s: str):
    print(f"  ✗ {s}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────────────────────

def pick_images(image_dir: Path, n: int) -> List[Path]:
    images: List[Path] = []
    for ext in ('*.tif', '*.tiff', '*.jpg', '*.jpeg'):
        images.extend(image_dir.glob(ext))
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")
    return images[:n]


def prepare_jpeg(src: Path, cache_dir: Path,
                 max_dim: int = DEFAULT_MAX_DIM,
                 quality: int = DEFAULT_JPEG_QUALITY) -> Path:
    if src.suffix.lower() in ('.jpg', '.jpeg'):
        return src
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / (src.stem + '.jpg')
    if dst.exists():
        return dst
    with Image.open(src) as img:
        if getattr(img, 'n_frames', 1) > 1:
            img.seek(0)
        img = img.convert('RGB')
        w, h = img.size
        scale = max(w / max_dim, h / max_dim, 1.0)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
        img.save(dst, 'JPEG', quality=quality, optimize=True)
    return dst


def jpeg_to_base64(jpeg_path: Path) -> str:
    with open(jpeg_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


# ──────────────────────────────────────────────────────────────────────────────
# HTTP call
# ──────────────────────────────────────────────────────────────────────────────

def call_http(host: str, port: int, model: str, jpeg_path: Path,
              prompt: str, num_predict: Optional[int],
              timeout: int) -> Tuple[str, Dict[str, Any], bool, Optional[str]]:
    url = f"http://{host}:{port}/api/generate"
    image_b64 = jpeg_to_base64(jpeg_path)
    options: Dict[str, Any] = {"temperature": 0}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "keep_alive": "24h",
        "options": options,
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8', errors='replace')
        parsed = json.loads(body)
        return parsed.get("response", "").strip(), parsed, True, None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode('utf-8', errors='replace')
        except Exception:
            err_body = ""
        return "", {}, False, f"HTTP {e.code}: {e.reason} | {err_body[:200]}"
    except Exception as e:
        return "", {}, False, f"{type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Parsing — JSON and field validators
# ──────────────────────────────────────────────────────────────────────────────

def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try strict JSON first; fall back to first {...} substring."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences if present
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find the first balanced JSON object
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


@dataclass
class FieldCheck:
    name: str
    present: bool
    valid: bool                # passed schema check
    value: Any = None
    reason: Optional[str] = None   # why it failed, if invalid


def check_title(v: Any) -> FieldCheck:
    if v is None:
        return FieldCheck("title", False, False, None, "missing")
    if not isinstance(v, str):
        return FieldCheck("title", True, False, v, f"not a string ({type(v).__name__})")
    s = v.strip()
    if not (1 <= len(s) <= 100):
        return FieldCheck("title", True, False, v, f"length {len(s)} not in [1,100]")
    return FieldCheck("title", True, True, s)


def check_description(v: Any) -> FieldCheck:
    if v is None:
        return FieldCheck("description", False, False, None, "missing")
    if not isinstance(v, str):
        return FieldCheck("description", True, False, v, f"not a string ({type(v).__name__})")
    s = v.strip()
    if len(s) < 10:
        return FieldCheck("description", True, False, v, f"length {len(s)} < 10 (likely degenerate)")
    return FieldCheck("description", True, True, s)


def check_offensive(v: Any) -> FieldCheck:
    if v is None:
        return FieldCheck("offensive", False, False, None, "missing")
    if isinstance(v, bool):
        return FieldCheck("offensive", True, True, v)
    # Accept string forms "true"/"false" since some models emit them
    if isinstance(v, str):
        if v.lower() in ("true", "false"):
            return FieldCheck("offensive", True, True, v.lower() == "true",
                              "coerced from string")
    return FieldCheck("offensive", True, False, v, f"not bool ({type(v).__name__})")


def check_offense_category(v: Any, offensive: Optional[bool]) -> FieldCheck:
    # If we know offensive=false, category should be null/missing
    if offensive is False:
        if v in (None, "null", "", "None"):
            return FieldCheck("offense_category", True, True, None)
        # Some valid output: model said "not offensive" but still labeled. Lenient — allow.
        return FieldCheck("offense_category", True, True, v, "extra label on non-offensive")
    # offensive=true or unknown — must be in allowed list
    if v is None:
        return FieldCheck("offense_category", False, False, None,
                          "missing while offensive=true")
    if not isinstance(v, str):
        return FieldCheck("offense_category", True, False, v,
                          f"not a string ({type(v).__name__})")
    # Exact match first
    if v in ALLOWED_CATEGORIES:
        return FieldCheck("offense_category", True, True, v)
    # Lenient match: case-insensitive equality, then substring containment
    # in either direction. Catches "Hate Symbol" vs "Hate symbols" and
    # "Native American" vs "Native American imagery". Will NOT match
    # fabricated labels like "War imagery" against any allowed category.
    v_lower = v.strip().lower()
    for allowed in ALLOWED_CATEGORIES:
        allowed_lower = allowed.lower()
        if v_lower == allowed_lower:
            return FieldCheck("offense_category", True, True, allowed,
                              "case-normalized")
        # Strict substring match: one must contain the other entirely
        if v_lower in allowed_lower or allowed_lower in v_lower:
            return FieldCheck("offense_category", True, True, allowed,
                              f"fuzzy-matched from {v!r}")
    return FieldCheck("offense_category", True, False, v,
                      "not in allowed categories")


def check_offense_reason(v: Any, offensive: Optional[bool]) -> FieldCheck:
    if offensive is False:
        if v in (None, "null", "", "None"):
            return FieldCheck("offense_reason", True, True, None)
        return FieldCheck("offense_reason", True, True, v,
                          "extra reason on non-offensive")
    if v is None:
        return FieldCheck("offense_reason", False, False, None,
                          "missing while offensive=true")
    if not isinstance(v, str) or len(v.strip()) < 5:
        return FieldCheck("offense_reason", True, False, v,
                          "absent or too short")
    return FieldCheck("offense_reason", True, True, v.strip())


def check_confidence(v: Any) -> FieldCheck:
    if v is None:
        return FieldCheck("confidence", False, False, None, "missing")
    if not isinstance(v, str):
        return FieldCheck("confidence", True, False, v, f"not a string ({type(v).__name__})")
    s = v.strip().lower()
    if s not in ALLOWED_CONFIDENCE:
        return FieldCheck("confidence", True, False, v, f"not in {ALLOWED_CONFIDENCE}")
    return FieldCheck("confidence", True, True, s)


# ──────────────────────────────────────────────────────────────────────────────
# Per-variant runners
# ──────────────────────────────────────────────────────────────────────────────

def run_variant_a_or_b(perf, prepared: List[Tuple[Path, Path]],
                       prompts: List[Tuple[str, str, str]],
                       host: str, port: int, model: str, timeout: int,
                       variant_name: str, warmup_calls: int,
                       repeats: int, out_dir: Path) -> List[Dict[str, Any]]:
    """
    Run a multi-prompt variant with N repeats per (image, prompt) cell.
    Returns list of per-(image, repeat) aggregated records.
    """
    total_calls = len(prompts) * len(prepared) * repeats
    banner(f"VARIANT {variant_name} — {len(prompts)} prompts × {len(prepared)} images "
           f"× {repeats} repeats = {total_calls} calls")
    per_image: List[Dict[str, Any]] = []
    call_idx = 0
    with perf.stage(f"variant_{variant_name}") as ctx:
        for rep in range(repeats):
            for img_idx, (orig, jpeg) in enumerate(prepared, start=1):
                image_record: Dict[str, Any] = {
                    "image": str(orig),
                    "image_name": orig.name,
                    "variant": variant_name,
                    "repeat": rep,
                    "raw_outputs": {},
                    "wall_total": 0.0,
                    "output_tokens_total": 0,
                }
                for prompt_name, prompt_text, _kind in prompts:
                    call_idx += 1
                    is_warmup = call_idx <= warmup_calls
                    # Apply token cap to the natural prompt only (user request)
                    num_predict = DEFAULT_NATURAL_NUM_PREDICT if prompt_name == "natural" else None

                    t0 = time.perf_counter()
                    call_start = datetime.now(timezone.utc)
                    output, response_dict, success, err = call_http(
                        host, port, model, jpeg, prompt_text, num_predict, timeout
                    )
                    wall = time.perf_counter() - t0
                    call_end = datetime.now(timezone.utc)

                    perf.inference.record_http(
                        image_path=str(orig),
                        prompt_name=f"{variant_name}_{prompt_name}_r{rep}",
                        model=model,
                        wall_time_sec=wall,
                        success=success,
                        ollama_response=response_dict if success else None,
                        error=err,
                        call_start_ts=call_start,
                        call_end_ts=call_end,
                        is_warmup=is_warmup,
                    )
                    image_record["raw_outputs"][prompt_name] = output
                    if not is_warmup and success:
                        image_record["wall_total"] += wall
                        image_record["output_tokens_total"] += (
                            response_dict.get("eval_count", 0) or 0
                        )

                    tpot_ns = response_dict.get("eval_duration", 0) if response_dict else 0
                    tpot_n = response_dict.get("eval_count", 0) if response_dict else 0
                    tpot_str = f" tpot={(tpot_ns / tpot_n) / 1e6:.0f}ms" if tpot_n else ""
                    marker = " [WARMUP]" if is_warmup else ""
                    info(f"  [{variant_name}/r{rep}][{img_idx:02d}/{len(prepared)}] "
                         f"{orig.name} {prompt_name:11s} {wall:5.2f}s{tpot_str} "
                         f"{'OK' if success else 'FAIL'}{marker}")
                    _append_output_jsonl(out_dir, variant_name, str(orig),
                                         prompt_name, output, success, err,
                                         is_warmup, repeat=rep)

                per_image.append(image_record)
                perf.inference._flush()
            # Per-repeat milestone (visible even with --quiet)
            info_always(f"[{variant_name}] repeat {rep + 1}/{repeats} complete "
                        f"({call_idx} calls so far)")
        ctx.set_metadata(images=len(prepared), calls=call_idx,
                         warmup_calls=warmup_calls, prompts_per_image=len(prompts),
                         repeats=repeats)
    return per_image


def run_variant_c(perf, prepared: List[Tuple[Path, Path]],
                  host: str, port: int, model: str, timeout: int,
                  warmup_calls: int, repeats: int,
                  out_dir: Path) -> List[Dict[str, Any]]:
    """Run the combined single-prompt variant with N repeats."""
    total_calls = len(prepared) * repeats
    banner(f"VARIANT C — combined single prompt × {len(prepared)} images "
           f"× {repeats} repeats = {total_calls} calls")
    per_image: List[Dict[str, Any]] = []
    call_idx = 0
    with perf.stage("variant_C") as ctx:
        for rep in range(repeats):
            for img_idx, (orig, jpeg) in enumerate(prepared, start=1):
                call_idx += 1
                is_warmup = call_idx <= warmup_calls

                t0 = time.perf_counter()
                call_start = datetime.now(timezone.utc)
                output, response_dict, success, err = call_http(
                    host, port, model, jpeg, COMBINED_PROMPT,
                    # v2: cap output. 300 tokens is enough for a complete
                    # 6-field JSON with reasonable lengths, not enough for
                    # a runaway description leaking into the title field.
                    num_predict=300, timeout=timeout,
                )
                wall = time.perf_counter() - t0
                call_end = datetime.now(timezone.utc)

                perf.inference.record_http(
                    image_path=str(orig),
                    prompt_name=f"C_combined_r{rep}",
                    model=model,
                    wall_time_sec=wall,
                    success=success,
                    ollama_response=response_dict if success else None,
                    error=err,
                    call_start_ts=call_start,
                    call_end_ts=call_end,
                    is_warmup=is_warmup,
                )

                image_record = {
                    "image": str(orig),
                    "image_name": orig.name,
                    "variant": "C",
                    "repeat": rep,
                    "raw_outputs": {"combined": output},
                    "wall_total": 0.0 if is_warmup else wall,
                    "output_tokens_total": 0,
                }
                if response_dict and not is_warmup:
                    image_record["output_tokens_total"] = response_dict.get("eval_count", 0) or 0

                tpot_ns = response_dict.get("eval_duration", 0) if response_dict else 0
                tpot_n = response_dict.get("eval_count", 0) if response_dict else 0
                tpot_str = f" tpot={(tpot_ns / tpot_n) / 1e6:.0f}ms" if tpot_n else ""
                marker = " [WARMUP]" if is_warmup else ""
                info(f"  [C/r{rep}][{img_idx:02d}/{len(prepared)}] {orig.name} "
                     f"combined    {wall:5.2f}s{tpot_str} "
                     f"{'OK' if success else 'FAIL'}{marker}")
                _append_output_jsonl(out_dir, "C", str(orig),
                                     "combined", output, success, err,
                                     is_warmup, repeat=rep)

                per_image.append(image_record)
                perf.inference._flush()
            info_always(f"[C] repeat {rep + 1}/{repeats} complete "
                        f"({call_idx} calls so far)")
        ctx.set_metadata(images=len(prepared), calls=call_idx,
                         warmup_calls=warmup_calls, prompts_per_image=1,
                         repeats=repeats)
    return per_image


def _append_output_jsonl(out_dir: Path, variant: str, image_path: str,
                         prompt_name: str, output: str, success: bool,
                         err: Optional[str], is_warmup: bool,
                         repeat: int = 0):
    path = out_dir / "outputs.jsonl"
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "image_path": image_path,
        "prompt_name": prompt_name,
        "repeat": repeat,
        "success": success,
        "error": err,
        "is_warmup": is_warmup,
        "output": output,
    }
    try:
        with open(path, 'a') as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[warn] could not append output: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Field extraction per variant
# ──────────────────────────────────────────────────────────────────────────────

def extract_fields_a(raw_outputs: Dict[str, str]) -> Dict[str, FieldCheck]:
    """
    Variant A: 3 separate prompts. 'natural' and 'strict' are free text;
    'moderation' is JSON with offensive/category/rationale.
    """
    out: Dict[str, FieldCheck] = {}
    out["description"] = check_description(raw_outputs.get("natural"))
    # No title in variant A
    mod_raw = raw_outputs.get("moderation", "")
    mod_obj = extract_json(mod_raw) or {}
    out["offensive"] = check_offensive(mod_obj.get("offensive"))
    off_val = out["offensive"].value if out["offensive"].valid else None
    out["offense_category"] = check_offense_category(
        mod_obj.get("category"), off_val
    )
    out["offense_reason"] = check_offense_reason(
        mod_obj.get("rationale"), off_val
    )
    # No confidence in variant A
    return out


def extract_fields_b(raw_outputs: Dict[str, str]) -> Dict[str, FieldCheck]:
    """Variant B: A + a separate title prompt."""
    out = extract_fields_a(raw_outputs)
    out["title"] = check_title(raw_outputs.get("title"))
    return out


def extract_fields_c(raw_outputs: Dict[str, str]) -> Dict[str, FieldCheck]:
    """Variant C: single combined JSON output with all 6 fields."""
    raw = raw_outputs.get("combined", "")
    obj = extract_json(raw) or {}
    out = {
        "title":            check_title(obj.get("title")),
        "description":      check_description(obj.get("description")),
        "offensive":        check_offensive(obj.get("offensive")),
    }
    off_val = out["offensive"].value if out["offensive"].valid else None
    out["offense_category"] = check_offense_category(obj.get("offense_category"), off_val)
    out["offense_reason"] = check_offense_reason(obj.get("offense_reason"), off_val)
    out["confidence"] = check_confidence(obj.get("confidence"))
    return out


VARIANT_EXTRACTORS = {
    "A": extract_fields_a,
    "B": extract_fields_b,
    "C": extract_fields_c,
}


# ──────────────────────────────────────────────────────────────────────────────
# Report builder
# ──────────────────────────────────────────────────────────────────────────────

def build_field_parses(per_image_by_variant: Dict[str, List[Dict[str, Any]]]
                       ) -> List[Dict[str, Any]]:
    """Build flat rows: one per (image, variant, repeat, field) for the parquet."""
    rows = []
    for variant, records in per_image_by_variant.items():
        extractor = VARIANT_EXTRACTORS[variant]
        for rec in records:
            checks = extractor(rec["raw_outputs"])
            for field_name, fc in checks.items():
                rows.append({
                    "variant": variant,
                    "image_name": rec["image_name"],
                    "image_path": rec["image"],
                    "repeat": rec.get("repeat", 0),
                    "field": field_name,
                    "present": fc.present,
                    "valid": fc.valid,
                    "value": str(fc.value)[:500] if fc.value is not None else None,
                    "reason": fc.reason,
                })
    return rows


def summarize_variant(records: List[Dict[str, Any]],
                      field_rows: List[Dict[str, Any]],
                      variant: str) -> Dict[str, Any]:
    """Per-variant summary: latency, throughput, parse rates per field."""
    measured = [r for r in records if r["wall_total"] > 0]   # excludes warmups
    summary: Dict[str, Any] = {
        "variant": variant,
        "images_measured": len(measured),
        "total_wall_sec": sum(r["wall_total"] for r in measured),
        "total_output_tokens": sum(r["output_tokens_total"] for r in measured),
    }
    if measured:
        walls = sorted(r["wall_total"] for r in measured)
        n = len(walls)
        summary["per_image_wall_median"] = walls[n // 2]
        summary["per_image_wall_mean"] = sum(walls) / n
        summary["per_image_wall_min"] = walls[0]
        summary["per_image_wall_max"] = walls[-1]
        if n > 1:
            mean = summary["per_image_wall_mean"]
            summary["per_image_wall_std"] = (
                sum((w - mean) ** 2 for w in walls) / (n - 1)
            ) ** 0.5
    if summary["total_wall_sec"] > 0:
        summary["images_per_sec"] = len(measured) / summary["total_wall_sec"]
        summary["tokens_per_sec"] = (
            summary["total_output_tokens"] / summary["total_wall_sec"]
        )

    # Per-field parse rates — across ALL measured (image, repeat) cells
    measured_keys = {(r["image_name"], r.get("repeat", 0)) for r in measured}
    variant_rows = [
        r for r in field_rows
        if r["variant"] == variant
        and (r["image_name"], r.get("repeat", 0)) in measured_keys
    ]
    by_field: Dict[str, Dict[str, int]] = {}
    for r in variant_rows:
        f = r["field"]
        if f not in by_field:
            by_field[f] = {"present": 0, "valid": 0, "total": 0}
        by_field[f]["total"] += 1
        if r["present"]:
            by_field[f]["present"] += 1
        if r["valid"]:
            by_field[f]["valid"] += 1
    summary["fields"] = {
        f: {
            "present_pct": (d["present"] / d["total"] * 100) if d["total"] else None,
            "valid_pct":   (d["valid"]   / d["total"] * 100) if d["total"] else None,
            "n": d["total"],
        }
        for f, d in by_field.items()
    }
    return summary


def analyze_repeat_consistency(per_image_by_variant: Dict[str, List[Dict[str, Any]]]
                                ) -> Dict[str, Dict[str, Any]]:
    """
    For temp=0 runs, repeats SHOULD produce identical outputs. This function
    flags any cells where they didn't — that's evidence of nondeterminism in
    the inference engine.

    Returns: variant -> {
        "total_cells": int,                # (image, prompt) pairs
        "identical_cells": int,            # cells where all repeats matched
        "divergent_cells": int,
        "divergent_examples": list of (image_name, prompt_name, [outputs]),
    }
    """
    result: Dict[str, Dict[str, Any]] = {}
    for variant, records in per_image_by_variant.items():
        # Group by image_name → list of (repeat, raw_outputs)
        by_image: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
        for rec in records:
            by_image.setdefault(rec["image_name"], []).append(
                (rec.get("repeat", 0), rec["raw_outputs"])
            )

        total_cells = 0
        identical = 0
        divergent = 0
        examples: List[Tuple[str, str, List[str]]] = []

        for image_name, reps in by_image.items():
            if len(reps) < 2:
                continue
            # For each prompt key seen in the first repeat, compare across all repeats
            first_prompts = reps[0][1].keys()
            for prompt_name in first_prompts:
                outputs = [r_dict.get(prompt_name, "") for _, r_dict in reps]
                total_cells += 1
                if len(set(outputs)) == 1:
                    identical += 1
                else:
                    divergent += 1
                    if len(examples) < 5:
                        examples.append((image_name, prompt_name, outputs))
        result[variant] = {
            "total_cells": total_cells,
            "identical_cells": identical,
            "divergent_cells": divergent,
            "identical_pct": (identical / total_cells * 100) if total_cells else None,
            "divergent_examples": examples,
        }
    return result


def write_report(out_dir: Path, run_id: str, summaries: List[Dict[str, Any]],
                 per_image_by_variant: Dict[str, List[Dict[str, Any]]],
                 num_images: int, warmup_calls: int, repeats: int,
                 consistency: Optional[Dict[str, Dict[str, Any]]] = None):
    lines = [
        f"# Prompt Shape Experiment — {run_id}",
        "",
        f"- Date: {datetime.now(timezone.utc).isoformat()}",
        f"- Host: {socket.gethostname()}",
        f"- Images: {num_images}  |  Repeats per cell: {repeats}  "
        f"|  Warmup excluded: {warmup_calls} calls/variant",
        "",
        "## Variant overview",
        "",
        "| Variant | Calls per image | What it asks for |",
        "|---|---|---|",
        "| A | 3 | natural desc + strict desc + moderation (current pipeline) |",
        "| B | 4 | title + natural desc + strict desc + moderation |",
        "| C | 1 | combined JSON: title + description + offensive + category + reason + confidence |",
        "",
        "## Latency & throughput (per-image, aggregated across all repeats)",
        "",
        "| Variant | Cells measured | Wall median (s/img) | Wall mean (s/img) | Wall std (s) | Wall min/max | Total wall (s) | Tokens/sec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        v = s["variant"]
        std = s.get('per_image_wall_std')
        std_str = f"{std:.2f}" if std is not None else "—"
        wall_min = s.get('per_image_wall_min')
        wall_max = s.get('per_image_wall_max')
        minmax = (f"{wall_min:.2f}/{wall_max:.2f}"
                  if wall_min is not None and wall_max is not None else "—")
        lines.append(
            f"| {v} | {s.get('images_measured', 0)} | "
            f"{s.get('per_image_wall_median', float('nan')):.2f} | "
            f"{s.get('per_image_wall_mean', float('nan')):.2f} | "
            f"{std_str} | "
            f"{minmax} | "
            f"{s.get('total_wall_sec', float('nan')):.1f} | "
            f"{s.get('tokens_per_sec', float('nan')):.1f} |"
        )

    # Speedup of C vs A
    a_summary = next((s for s in summaries if s["variant"] == "A"), None)
    c_summary = next((s for s in summaries if s["variant"] == "C"), None)
    if a_summary and c_summary and a_summary.get("per_image_wall_mean", 0) > 0:
        speedup = a_summary["per_image_wall_mean"] / c_summary["per_image_wall_mean"]
        lines.append("")
        lines.append(
            f"**Combined-prompt speedup vs current 3-prompt: {speedup:.2f}x** "
            f"(based on per-image wall mean)"
        )

    lines.append("")
    lines.append("## Per-field parse rate")
    lines.append("")
    lines.append("`valid` means: field present AND passed the schema check "
                 "(correct type, length, allowed value set, etc).")
    lines.append("")

    # Gather all field names across variants for column union
    all_fields: List[str] = []
    seen_fields = set()
    for s in summaries:
        for fname in s.get("fields", {}).keys():
            if fname not in seen_fields:
                seen_fields.add(fname)
                all_fields.append(fname)
    if all_fields:
        header = "| Variant | " + " | ".join(all_fields) + " |"
        sep = "|---|" + "|".join("---" for _ in all_fields) + "|"
        lines.append(header)
        lines.append(sep)
        for s in summaries:
            row = [s["variant"]]
            for fname in all_fields:
                fd = s.get("fields", {}).get(fname)
                if fd is None:
                    row.append("n/a")
                else:
                    row.append(f"{fd['valid_pct']:.0f}%")
            lines.append("| " + " | ".join(row) + " |")

    # ── Repeat-consistency analysis ──────────────────────────────────────────
    if consistency and repeats > 1:
        lines.append("")
        lines.append(f"## Repeat consistency (temperature=0, {repeats} repeats)")
        lines.append("")
        lines.append("At temperature=0, the model SHOULD produce identical outputs "
                     "across repeats of the same (image, prompt) cell. Any divergence "
                     "indicates nondeterminism in the inference engine — useful to "
                     "know for production stability.")
        lines.append("")
        lines.append("| Variant | Cells | Identical across repeats | Divergent | Identical % |")
        lines.append("|---|---|---|---|---|")
        for variant in sorted(consistency.keys()):
            c = consistency[variant]
            pct = c.get("identical_pct")
            pct_str = f"{pct:.0f}%" if pct is not None else "—"
            lines.append(
                f"| {variant} | {c['total_cells']} | {c['identical_cells']} "
                f"| {c['divergent_cells']} | {pct_str} |"
            )

        # Show a couple of divergent examples per variant
        for variant in sorted(consistency.keys()):
            examples = consistency[variant].get("divergent_examples", [])
            if not examples:
                continue
            lines.append("")
            lines.append(f"**Divergent examples for variant {variant}** "
                         f"(up to 5 cells where repeats disagreed):")
            lines.append("")
            for image_name, prompt_name, outputs in examples:
                lines.append(f"- `{image_name}` / `{prompt_name}`:")
                for r_idx, out in enumerate(outputs):
                    trunc = out[:150].replace("\n", " ") + ("…" if len(out) > 150 else "")
                    lines.append(f"  - r{r_idx}: {trunc}")

    lines.append("")
    lines.append("## Per-image content spot-check (repeat 0 only)")
    lines.append("")
    lines.append("Side-by-side outputs for each image, so you can eyeball whether "
                 "the combined prompt's description is comparably useful to the "
                 "separate prompts. Truncated to first 200 chars per field. Only "
                 "the first repeat is shown here; see `outputs.jsonl` for the rest.")
    lines.append("")

    # Map by image_name → variant → fields. Use REPEAT 0 only.
    by_image: Dict[str, Dict[str, Dict[str, str]]] = {}
    for variant, records in per_image_by_variant.items():
        extractor = VARIANT_EXTRACTORS[variant]
        for rec in records:
            if rec.get("repeat", 0) != 0:
                continue
            checks = extractor(rec["raw_outputs"])
            by_image.setdefault(rec["image_name"], {})[variant] = {
                f: (str(c.value) if c.value is not None else "—")
                for f, c in checks.items()
            }

    for image_name in sorted(by_image.keys()):
        lines.append(f"### {image_name}")
        lines.append("")
        for variant in ("A", "B", "C"):
            v_fields = by_image[image_name].get(variant)
            if v_fields is None:
                continue
            lines.append(f"**Variant {variant}:**")
            for fname, val in v_fields.items():
                trunc = val[:200] + ("…" if len(val) > 200 else "")
                lines.append(f"- `{fname}`: {trunc}")
            lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `inference_timings.parquet` — one row per inference call (HTTP)")
    lines.append("- `resource_samples.parquet` — GPU/CPU/memory traces")
    lines.append("- `stage_timings.parquet` — one row per variant")
    lines.append("- `field_parses.parquet` — one row per (image, variant, field)")
    lines.append("- `outputs.jsonl` — raw model outputs for every call")

    report_path = out_dir / "prompt_comparison_report.md"
    report_path.write_text("\n".join(lines))

    print("\n" + "\n".join(lines) + "\n")
    ok(f"Report written: {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ──────────────────────────────────────────────────────────────────────────────

def preflight(host: str, port: int, model: str) -> bool:
    banner("PRE-FLIGHT")
    try:
        with socket.create_connection((host, port), timeout=3):
            ok(f"Ollama HTTP server reachable on {host}:{port}")
    except Exception as e:
        fail(f"Cannot reach Ollama: {e}")
        return False
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/tags", timeout=10
        ) as r:
            data = json.loads(r.read().decode())
        names = [m.get("name", "") for m in data.get("models", [])]
        if any(n == model or n.startswith(f"{model}:") for n in names):
            ok(f"Model present: {model}")
            return True
        fail(f"Model '{model}' not in {names}")
        return False
    except Exception as e:
        fail(f"/api/tags failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir. Defaults to benchmark_prompts/run_<timestamp>/")
    p.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                   help=f"Repeats per (image, prompt) cell. Default: {DEFAULT_REPEATS}")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help=f"Warmup calls per variant. Default: {DEFAULT_WARMUP}")
    p.add_argument("--variants", type=str, default="A,B,C",
                   help="Comma-separated list of variants to run. Default: A,B,C")
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress per-call info lines (keeps banners + milestones). "
                        "Recommended for long unattended runs.")
    p.add_argument("--utils-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Wire up perf_monitor
    utils_dir = args.utils_dir or (Path(__file__).resolve().parent / "utils")
    if not (utils_dir / "perf_monitor.py").exists():
        alt = Path(__file__).resolve().parent.parent / "utils"
        if (alt / "perf_monitor.py").exists():
            utils_dir = alt
        else:
            fail(f"Could not find perf_monitor.py in {utils_dir} or {alt}")
            sys.exit(2)
    sys.path.insert(0, str(utils_dir.parent))
    from utils.perf_monitor import PerfMonitor

    # Output dir
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_DIR / f"run_{ts}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    variants_to_run = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    for v in variants_to_run:
        if v not in ("A", "B", "C"):
            fail(f"Unknown variant '{v}'. Allowed: A, B, C.")
            sys.exit(2)

    # Wire up the module-level QUIET flag
    global QUIET
    QUIET = args.quiet

    # Estimated cost so the operator knows what they're signing up for
    est_calls = 0
    if "A" in variants_to_run:
        est_calls += 3 * args.num_images * args.repeats
    if "B" in variants_to_run:
        est_calls += 4 * args.num_images * args.repeats
    if "C" in variants_to_run:
        est_calls += 1 * args.num_images * args.repeats
    # 4s avg/call for A/B, ~10s/call for C — rough
    est_seconds = est_calls * 4.0  # conservative
    est_minutes = est_seconds / 60

    print(f"Prompt-shape benchmark — {datetime.now().isoformat()}")
    print(f"  run_id     : {run_id}")
    print(f"  output_dir : {out_dir}")
    print(f"  model      : {args.model}")
    print(f"  images     : {args.num_images} from {args.image_dir.name}")
    print(f"  repeats    : {args.repeats} per (image, prompt) cell")
    print(f"  variants   : {','.join(variants_to_run)}")
    print(f"  warmup     : {args.warmup} calls/variant (excluded from measured stats)")
    print(f"  natural num_predict cap: {DEFAULT_NATURAL_NUM_PREDICT} tokens")
    print(f"  quiet      : {'yes' if QUIET else 'no (full per-call log)'}")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  est calls  : {est_calls}  (~{est_minutes:.0f} min @ 4 sec/call avg)")

    if not preflight(args.host, args.port, args.model):
        sys.exit(2)

    # Prepare images (cached as JPEG in output dir)
    banner("Preparing image sample")
    images = pick_images(args.image_dir, args.num_images)
    cache = out_dir / "jpeg_cache"
    prepared: List[Tuple[Path, Path]] = [
        (p, prepare_jpeg(p, cache)) for p in images
    ]
    info_always(f"Prepared {len(prepared)} images: "
                f"{prepared[0][0].name} ... {prepared[-1][0].name}")

    # Start instrumentation
    perf = PerfMonitor(
        out_dir=out_dir, run_id=run_id,
        workflow_name="benchmark_prompts",
        sample_interval_sec=args.sample_interval,
    )
    perf.out_dir = out_dir

    per_image_by_variant: Dict[str, List[Dict[str, Any]]] = {}

    try:
        if "A" in variants_to_run:
            per_image_by_variant["A"] = run_variant_a_or_b(
                perf, prepared, VARIANT_A_PROMPTS,
                args.host, args.port, args.model, args.timeout,
                "A", args.warmup, args.repeats, out_dir,
            )
        if "B" in variants_to_run:
            per_image_by_variant["B"] = run_variant_a_or_b(
                perf, prepared, VARIANT_B_PROMPTS,
                args.host, args.port, args.model, args.timeout,
                "B", args.warmup, args.repeats, out_dir,
            )
        if "C" in variants_to_run:
            per_image_by_variant["C"] = run_variant_c(
                perf, prepared,
                args.host, args.port, args.model, args.timeout,
                args.warmup, args.repeats, out_dir,
            )
    finally:
        perf.close()

    # Build field-parse rows and write parquet
    field_rows = build_field_parses(per_image_by_variant)
    if field_rows:
        try:
            import pandas as pd
            pd.DataFrame(field_rows).to_parquet(
                out_dir / "field_parses.parquet", index=False
            )
            ok(f"Wrote {len(field_rows)} field-parse rows")
        except Exception as e:
            fail(f"Could not write field_parses.parquet: {e}")

    # Repeat-consistency analysis (cheap, runs in memory)
    consistency = (
        analyze_repeat_consistency(per_image_by_variant)
        if args.repeats > 1 else None
    )

    # Summaries + report
    summaries = [
        summarize_variant(per_image_by_variant[v], field_rows, v)
        for v in variants_to_run
    ]
    write_report(out_dir, run_id, summaries,
                 per_image_by_variant, args.num_images, args.warmup,
                 args.repeats, consistency)

    banner("PROMPT-SHAPE BENCHMARK COMPLETE")
    info_always(f"All artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
