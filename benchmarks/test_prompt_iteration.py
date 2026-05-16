#!/usr/bin/env python3
"""
test_prompt_iteration.py
=========================
Cheap smoke test for prompt iteration. NOT a benchmark — no warmup, no repeats,
no perf instrumentation. The job is: take a prompt + a few images, fire one
HTTP call per image, parse the JSON, print results compactly enough that you
can eyeball whether the iteration moved the needle.

Use this BEFORE committing to a full 50×3 validation run.

Workflow
--------
  1. Edit the PROMPT_VARIANTS dict below — add your candidate prompt as a
     new entry, give it a label like 'C-v3' or 'C-v4'.
  2. Run:  python3 test_prompt_iteration.py --variant C-v3
  3. Read the output. Did the failing image (0000049) flip to offensive=false?
     Did any previously-passing image regress to offensive=true?
  4. If promising, run the full validation via benchmark_prompts.py.

Cost
----
~3 sec/image × N images. Test set of ~13 images = under a minute.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_IMAGE_DIR = Path(
    './image_data/'
    'All_Scanned_Images/Scanned_Images_Combined'
)
DEFAULT_MODEL = 'llava-llama3'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 180

# 9 sensitivity categories — matches production
ALLOWED_CATEGORIES = [
    "Historical racialized performance",
    "Human remains",
    "Native American imagery",
    "Nudity/sexual content",
    "Violence/graphic content",
    "Hate symbols",
    "Medical/health records",
    "Student/PII records",
    "Other sensitive categories",
]
CATEGORIES_LIST = "\n".join(f"- {c}" for c in ALLOWED_CATEGORIES)


# ──────────────────────────────────────────────────────────────────────────────
# Test image set — chosen to give coverage of: known failures, likely policy-
# difficult content, and easy-case controls. Edit as needed.
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_TEST_IMAGES = [
    # Known failures from v2 — should be the focus of the iteration
    "0000049.jpg",   # Vietnam War protest — variant C said offensive=true with "War"
    "0000034.jpg",   # Long-title failure in v2 (length 123)
    "0000008.jpg",   # Long-title failure in v1 — see if regression
    # Easy-case controls — should NOT flip behavior
    "0000001.jpg",   # Stadium under construction
    "0000002.jpg",   # Football championship
    "0000005.jpg",   # (sampled)
    "0000010.jpg",   # (sampled)
    "0000022.jpg",   # Long-title failure in v1
    "0000035.jpg",   # 210-char title in v1 — controlling for fix on this
]


# ──────────────────────────────────────────────────────────────────────────────
# Prompt variants — add your candidate prompts here as you iterate
# ──────────────────────────────────────────────────────────────────────────────

# C-v2 — the current production candidate from the v2 run on May 13
# Included as a baseline to compare against.
COMBINED_PROMPT_V2 = (
    "You are analyzing an archival photograph or document scan.\n\n"
    "RULES:\n"
    "- title: MAX 8 WORDS. Must be a noun phrase, NOT a sentence. "
    "Good: 'Empty stadium under construction'. "
    "Bad: 'A black and white photograph of a stadium...'.\n"
    "- description: up to 4 sentences. Only observable details. "
    "Use 'unknown' if unsure.\n"
    "- offensive: true ONLY if the image contains content matching one of the "
    "listed categories. Do NOT mark true merely because the image shows "
    "difficult subjects (war, protest, politics, controversial history). "
    "Documentation is not offensive content.\n"
    "- offense_category: if offensive=true, ONE of the listed categories. "
    "If the image fits the spirit of the taxonomy but no named category "
    "applies, use 'Other sensitive categories'. Do not invent new categories.\n"
    "- offense_reason: if offensive=true, one sentence on why. Null otherwise.\n"
    "- confidence: low / medium / high — how confident you are about the "
    "offensive classification.\n\n"
    "ALLOWED CATEGORIES:\n"
    f"{CATEGORIES_LIST}\n\n"
    "Output a single JSON object with these exact fields:\n"
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


# C-v3 — STRONGER anti-flag rule, same inline categories.
# Hypothesis: making the anti-flag rule more emphatic + giving an explicit
# Vietnam-protest-style example will push the model to set offensive=false
# on documentation of difficult subjects.
COMBINED_PROMPT_V3 = (
    "You are analyzing an archival photograph or document scan.\n\n"
    "CRITICAL RULE — READ FIRST:\n"
    "The 'offensive' flag means the IMAGE ITSELF contains content from the "
    "allowed categories below. It does NOT mean the image's SUBJECT is "
    "controversial, sad, political, or about a difficult historical period.\n\n"
    "EXAMPLES OF WHAT IS NOT OFFENSIVE (set offensive=false):\n"
    "- A photo of war protesters holding 'Stop the War' signs.\n"
    "- A photo of soldiers in uniform.\n"
    "- A photo of a hospital ward (no patient info visible).\n"
    "- Documentation of any difficult historical event.\n"
    "- Civil rights marches, political rallies, demonstrations.\n\n"
    "EXAMPLES OF WHAT IS OFFENSIVE (set offensive=true):\n"
    "- A swastika or other hate symbol visibly displayed.\n"
    "- Graphic violence happening in the image (blood, weapon mid-use).\n"
    "- Visible nudity.\n"
    "- Visible patient records or medical chart with identifiable info.\n\n"
    "OTHER RULES:\n"
    "- title: MAX 8 WORDS. Noun phrase, NOT a sentence. "
    "Good: 'Empty stadium under construction'. "
    "Bad: 'A photograph of a stadium...'.\n"
    "- description: up to 4 sentences. Only observable details.\n"
    "- offense_category: if offensive=true, exactly ONE of the listed categories. "
    "If fits the spirit but no named category, use 'Other sensitive categories'. "
    "NEVER invent new categories.\n"
    "- offense_reason: if offensive=true, one sentence. Null otherwise.\n"
    "- confidence: low / medium / high.\n\n"
    "ALLOWED CATEGORIES:\n"
    f"{CATEGORIES_LIST}\n\n"
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


# C-v4 — V3's anti-flag rule + remove inline categories list.
# Hypothesis: the long inline categories list is bloating the prompt and
# inflating decode time (via the longer-prefilled-context tax). The model
# already knows the categories from the field description, so removing the
# explicit list may recover latency without losing accuracy.
COMBINED_PROMPT_V4 = (
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


# C-v5 — V4's body + stronger title rule placed RIGHT BEFORE the JSON template.
# Hypothesis: instruction proximity to output matters. The v4 title rule is in
# OTHER RULES list near the top; by the time the model emits the JSON it's
# 1200+ chars later in the prompt. Repeating the title rule immediately before
# the JSON template may improve adherence. Negative example uses the actual
# image 0000022 output from the v4 smoke test as a concrete don't-do-this.
COMBINED_PROMPT_V5 = (
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
    "CRITICAL TITLE RULE — APPLIES TO THE 'title' FIELD BELOW:\n"
    "The title field MUST be 8 words or fewer. Count your words.\n"
    "GOOD title examples (8 words or fewer, noun phrase):\n"
    "  'Empty stadium under construction'\n"
    "  'Three men looking at a pig'\n"
    "  'Woman in striped shirt and glasses'\n"
    "  'Farmers with cattle in a pen'\n"
    "BAD title examples (sentence form, too long — DO NOT DO THIS):\n"
    "  'A large white dome shaped building with a circular opening'\n"
    "  'A black and white photograph of a stadium under construction'\n"
    "  'A group of people walking down a sidewalk in a park.'\n"
    "If you cannot describe the image in 8 words, write a noun phrase "
    "capturing only the main subject — do NOT write a full descriptive "
    "sentence.\n\n"
    "Output a single JSON object:\n"
    "{\n"
    '  "title": "<MAX 8 WORDS — see CRITICAL TITLE RULE above>",\n'
    '  "description": "<up to 4 sentences>",\n'
    '  "offensive": <true or false>,\n'
    '  "offense_category": "<category name or null>",\n'
    '  "offense_reason": "<one sentence or null>",\n'
    '  "confidence": "<low | medium | high>"\n'
    "}\n\n"
    "Output strict JSON only. No preamble. No markdown fences."
)


PROMPT_VARIANTS = {
    "C-v2": COMBINED_PROMPT_V2,    # baseline (what's in benchmark_prompts.py now)
    "C-v3": COMBINED_PROMPT_V3,    # stronger anti-flag rule
    "C-v4": COMBINED_PROMPT_V4,    # stronger anti-flag + no inline categories
    "C-v5": COMBINED_PROMPT_V5,    # v4 + title rule repeated before JSON + examples
}


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def jpeg_to_b64(p: Path) -> str:
    with open(p, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')


def call_http(host: str, port: int, model: str, jpeg_path: Path,
              prompt: str, timeout: int = DEFAULT_TIMEOUT,
              num_predict: int = 300) -> Tuple[str, float, bool, Optional[str]]:
    url = f"http://{host}:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [jpeg_to_b64(jpeg_path)],
        "stream": False,
        "keep_alive": "24h",
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode('utf-8', errors='replace'))
        wall = time.perf_counter() - t0
        return body.get("response", "").strip(), wall, True, None
    except Exception as e:
        wall = time.perf_counter() - t0
        return "", wall, False, f"{type(e).__name__}: {e}"


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


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list(PROMPT_VARIANTS.keys()),
                   required=True, help="Which prompt variant to test")
    p.add_argument("--images", nargs="+", default=None,
                   help="Specific image filenames to test (otherwise uses "
                        "DEFAULT_TEST_IMAGES)")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--num-predict", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    prompt = PROMPT_VARIANTS[args.variant]
    images = args.images if args.images else DEFAULT_TEST_IMAGES

    print(f"Smoke test: {args.variant}")
    print(f"  Model: {args.model}")
    print(f"  Image count: {len(images)}")
    print(f"  num_predict cap: {args.num_predict}")
    print(f"  Prompt length: {len(prompt)} chars")
    print()
    print("=" * 90)

    results: List[Dict[str, Any]] = []
    for img_name in images:
        img_path = args.image_dir / img_name
        if not img_path.exists():
            print(f"  {img_name:18s} MISSING")
            continue

        output, wall, success, err = call_http(
            args.host, args.port, args.model, img_path, prompt,
            num_predict=args.num_predict,
        )
        if not success:
            print(f"  {img_name:18s} FAIL ({wall:.2f}s): {err}")
            continue

        obj = extract_json(output)
        title = str(obj.get("title", ""))[:60]
        offensive = obj.get("offensive", "?")
        category = obj.get("offense_category", "")
        reason = str(obj.get("offense_reason", "") or "")[:60]

        flag = "OFFENSIVE" if offensive is True else "ok       "
        title_len = len(str(obj.get("title", "")))
        title_flag = " " if title_len <= 100 else "!"
        print(f"  {img_name:18s} {wall:5.2f}s  {flag}  "
              f"cat={str(category)[:30]:30s}  "
              f"title({title_len:3d}){title_flag} {title!r}")

        results.append({
            "image": img_name,
            "wall": wall,
            "offensive": offensive,
            "category": category,
            "title_length": title_len,
            "raw_output": output,
        })

    # Summary stats
    if results:
        offensives = [r for r in results if r["offensive"] is True]
        long_titles = [r for r in results if r["title_length"] > 100]
        bad_cats = [r for r in results
                    if r["offensive"] is True
                    and r["category"] not in ALLOWED_CATEGORIES]

        print()
        print("=" * 90)
        print(f"Summary for {args.variant}:")
        print(f"  Images tested:        {len(results)}")
        print(f"  Mean wall time:       {sum(r['wall'] for r in results) / len(results):.2f}s")
        print(f"  Flagged offensive:    {len(offensives)}  "
              f"({[r['image'] for r in offensives]})")
        print(f"  Long titles (>100):   {len(long_titles)}  "
              f"({[r['image'] for r in long_titles]})")
        print(f"  Bad categories:       {len(bad_cats)}  "
              f"({[(r['image'], r['category']) for r in bad_cats]})")
        print()
        print("Decision frame:")
        print("  - For 0000049 (Vietnam protest): want offensive=False")
        print("  - For 0000001, 0000002, 0000005, 0000010 (controls): "
              "want offensive=False")
        print("  - For 0000008, 0000022, 0000034, 0000035 (long-title cases): "
              "want title_length<=100")
        print()
        print("If results look good, run the full validation via "
              "benchmark_prompts.py with the chosen prompt.")


if __name__ == "__main__":
    main()
