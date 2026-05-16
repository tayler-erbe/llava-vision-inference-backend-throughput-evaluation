#!/usr/bin/env python3
"""
analyze_prompt_outputs.py
==========================
Structured content-quality audit of a benchmark_prompts.py run.

What it produces (in <run-dir>/analysis/)
-----------------------------------------
  parse_failures.csv     Every (image, variant, field, repeat) cell where the
                         field failed schema validation, with the actual model
                         output text that caused the failure.

  side_by_side.csv       One row per image. Columns for each variant's outputs
                         (repeat 0 only — consistency analysis confirmed
                         repeats are identical at temp=0).

  scoring_template.csv   Same one-row-per-image layout, with blank columns for
                         manual scoring:
                           score_primary_subject_match (Y/N)
                           score_c_added_inference     (Y/N)
                           score_c_missed_details      (Y/N)
                           score_moderation_agrees     (Y/N)
                           verdict (good/drift/wrong)
                           notes

  audit_summary.txt      Aggregated findings + decision frame.

Usage
-----
    python analyze_prompt_outputs.py
    python analyze_prompt_outputs.py --run-dir output/benchmark_prompts/run_20260512_182032
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


DEFAULT_BENCHMARK_DIR = Path(
    "./pipeline-workspace/pipelines/"
    "Image Classification/output/benchmark_prompts"
)


def banner(s: str):
    print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}")


def info(s: str = ""):
    print(f"  {s}")


def find_latest_run(benchmark_dir: Path) -> Optional[Path]:
    if not benchmark_dir.exists():
        return None
    runs = sorted(
        [p for p in benchmark_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    fp_path = run_dir / "field_parses.parquet"
    out_path = run_dir / "outputs.jsonl"
    inf_path = run_dir / "inference_timings.parquet"

    if not fp_path.exists():
        sys.exit(f"Missing {fp_path}")
    if not out_path.exists():
        sys.exit(f"Missing {out_path}")

    info(f"Loading field_parses.parquet ({fp_path.stat().st_size:,} bytes)")
    fp = pd.read_parquet(fp_path)

    info(f"Loading outputs.jsonl ({out_path.stat().st_size:,} bytes)")
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out_df = pd.DataFrame(rows)

    inf_df = None
    if inf_path.exists():
        info(f"Loading inference_timings.parquet ({inf_path.stat().st_size:,} bytes)")
        inf_df = pd.read_parquet(inf_path)

    return fp, out_df, inf_df


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 1 — parse failures
# ──────────────────────────────────────────────────────────────────────────────

def build_parse_failures(field_parses: pd.DataFrame,
                         outputs: pd.DataFrame) -> pd.DataFrame:
    bad = field_parses[~field_parses["valid"]].copy()
    if len(bad) == 0:
        return pd.DataFrame()

    # Map field → producing prompt for variant A/B (variant C uses "combined")
    field_to_prompt = {
        "description":      "natural",
        "offensive":        "moderation",
        "offense_category": "moderation",
        "offense_reason":   "moderation",
        "title":            "title",
    }

    lookup: Dict[tuple, str] = {}
    for _, r in outputs.iterrows():
        img_name = Path(r["image_path"]).name
        key = (r["variant"], img_name, r["prompt_name"], int(r["repeat"]))
        lookup[key] = r["output"]

    enriched = []
    for _, row in bad.iterrows():
        prompt_name = "combined" if row["variant"] == "C" else field_to_prompt.get(
            row["field"], row["field"]
        )
        key = (row["variant"], row["image_name"], prompt_name, int(row["repeat"]))
        raw = lookup.get(key, "")
        enriched.append({
            "variant":    row["variant"],
            "image_name": row["image_name"],
            "repeat":     row["repeat"],
            "field":      row["field"],
            "present":    row["present"],
            "valid":      row["valid"],
            "value":      row["value"],
            "reason":     row["reason"],
            "raw_output": raw,
        })
    return pd.DataFrame(enriched)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 2 — side-by-side
# ──────────────────────────────────────────────────────────────────────────────

def build_side_by_side(outputs: pd.DataFrame) -> pd.DataFrame:
    df = outputs[
        (outputs["repeat"] == 0) &
        (~outputs.get("is_warmup", False)) &
        (outputs["success"])
    ].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["image_name"] = df["image_path"].apply(lambda p: Path(p).name)

    rows: Dict[str, Dict[str, str]] = defaultdict(dict)
    for _, r in df.iterrows():
        col = f"{r['variant']}_{r['prompt_name']}"
        rows[r["image_name"]][col] = r["output"]

    column_order = [
        "image_name",
        "A_natural", "A_strict", "A_moderation",
        "B_title", "B_natural", "B_strict", "B_moderation",
        "C_combined",
    ]
    out_rows = []
    for name in sorted(rows.keys()):
        row: Dict[str, Any] = {"image_name": name}
        for col in column_order[1:]:
            row[col] = rows[name].get(col, "")
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=column_order)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 3 — scoring template (parses C's JSON into individual columns)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict[str, Any]:
    import re
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


def extract_c_fields(text: str) -> Dict[str, str]:
    obj = _extract_json(text)
    return {
        "C_title":       str(obj.get("title", "")),
        "C_description": str(obj.get("description", "")),
        "C_offensive":   str(obj.get("offensive", "")),
        "C_category":    str(obj.get("offense_category", "")),
        "C_reason":      str(obj.get("offense_reason", "")),
        "C_confidence":  str(obj.get("confidence", "")),
    }


def extract_moderation_fields(text: str, prefix: str) -> Dict[str, str]:
    obj = _extract_json(text)
    return {
        f"{prefix}_offensive": str(obj.get("offensive", "")),
        f"{prefix}_category":  str(obj.get("category", "")),
        f"{prefix}_reason":    str(obj.get("rationale", "")),
    }


def build_scoring_template(sbs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sbs.iterrows():
        row: Dict[str, Any] = {"image_name": r["image_name"]}
        row["A_natural"] = r.get("A_natural", "")
        row["A_strict"] = r.get("A_strict", "")
        row.update(extract_moderation_fields(r.get("A_moderation", ""), "A"))
        row["B_title"] = r.get("B_title", "")
        row.update(extract_c_fields(r.get("C_combined", "")))
        row["score_primary_subject_match"] = ""
        row["score_c_added_inference"] = ""
        row["score_c_missed_details"] = ""
        row["score_moderation_agrees"] = ""
        row["verdict"] = ""
        row["notes"] = ""
        rows.append(row)
    cols = [
        "image_name",
        "A_natural", "A_strict",
        "A_offensive", "A_category", "A_reason",
        "B_title",
        "C_title", "C_description",
        "C_offensive", "C_category", "C_reason", "C_confidence",
        "score_primary_subject_match",
        "score_c_added_inference",
        "score_c_missed_details",
        "score_moderation_agrees",
        "verdict",
        "notes",
    ]
    return pd.DataFrame(rows, columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 4 — chart generation (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────
#
# All chart functions degrade gracefully — if matplotlib isn't importable,
# they log a warning and skip. CSV artifacts always still get written.

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for headless servers
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        info(f"matplotlib not available ({e}); skipping charts")
        return None


def make_latency_chart(out_path: Path, inference_df: pd.DataFrame):
    """
    Two-panel figure: boxplot of per-call wall time + histogram by variant.
    Shows distribution shape, not just median/mean.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    non_warmup = inference_df[~inference_df["is_warmup"]].copy()
    non_warmup["variant"] = non_warmup["prompt_name"].str[0]
    variants = sorted(non_warmup["variant"].unique())
    data = [non_warmup[non_warmup["variant"] == v]["wall_time_sec"].values
            for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: boxplot
    bp = axes[0].boxplot(data, labels=variants, showmeans=True,
                          patch_artist=True)
    # Subtle color per variant
    palette = ["#a8d8ea", "#aa96da", "#fcbad3"]
    for patch, color in zip(bp["boxes"], palette[:len(variants)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Wall time per call (sec)")
    axes[0].set_xlabel("Variant")
    axes[0].set_title("Per-call wall time — boxplot (excludes warmup)")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right panel: histogram per variant
    for v, vals, color in zip(variants, data, palette):
        axes[1].hist(vals, bins=25, alpha=0.5, label=f"Variant {v} (n={len(vals)})",
                     color=color, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Wall time per call (sec)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-call wall time — histogram")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Inference latency distribution by variant",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def make_gpu_timeline_chart(out_path: Path,
                             resource_df: pd.DataFrame,
                             stage_df: pd.DataFrame):
    """
    Three-panel timeline: GPU util %, VRAM MB, power W over time, with
    vertical shaded bands marking which variant was running.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return False
    if len(resource_df) == 0:
        info("No resource_samples rows; skipping GPU timeline")
        return False

    res = resource_df.copy()
    res["ts"] = pd.to_datetime(res["ts"], utc=True)
    res = res.sort_values("ts")
    t0 = res["ts"].iloc[0]
    res["elapsed_min"] = (res["ts"] - t0).dt.total_seconds() / 60

    # Variant time windows from stage_timings
    bands = []
    if len(stage_df) > 0:
        stage_df = stage_df.copy()
        stage_df["ts"] = pd.to_datetime(stage_df["ts"], utc=True)
        stage_df["end_ts"] = pd.to_datetime(stage_df["end_ts"], utc=True)
        for _, r in stage_df.iterrows():
            stage_name = str(r.get("stage", ""))
            if not stage_name.startswith("variant_"):
                continue
            label = stage_name.replace("variant_", "")
            start_min = (r["ts"] - t0).total_seconds() / 60
            end_min = (r["end_ts"] - t0).total_seconds() / 60
            bands.append((label, start_min, end_min))

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Variant band colors
    band_colors = {"A": "#a8d8ea", "B": "#aa96da", "C": "#fcbad3",
                   "baseline_idle": "#dcdcdc"}

    panels = [
        ("gpu_util_pct", "GPU utilization (%)", "GPU compute usage"),
        ("vram_used_mb", "VRAM used (MB)", "Model + KV cache resident memory"),
        ("power_w", "GPU power (W)", "Power draw — L4 cap is 72 W"),
    ]
    for ax, (col, ylabel, title) in zip(axes, panels):
        if col not in res.columns or res[col].notna().sum() == 0:
            ax.set_title(f"{title} (no data)")
            continue
        # Plot the time series
        ax.plot(res["elapsed_min"], res[col], linewidth=1.0, color="#333")
        # Shaded variant bands
        for label, sm, em in bands:
            color = band_colors.get(label, "#cccccc")
            ax.axvspan(sm, em, alpha=0.2, color=color)
            # Label the band at its midpoint, near the top of the axis
            ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100
            ax.text((sm + em) / 2, ymax * 0.95, label,
                    ha="center", va="top", fontsize=9, fontweight="bold",
                    color="#555")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Add a horizontal line at 72W on the power panel (L4 cap)
    if "power_w" in res.columns and res["power_w"].notna().sum() > 0:
        axes[2].axhline(y=72, linestyle="--", color="red", alpha=0.5,
                        label="L4 power cap (72W)")
        axes[2].legend(loc="lower right", fontsize=9)

    axes[2].set_xlabel("Elapsed time (minutes)")
    fig.suptitle("GPU resource timeline — sampled at 2 Hz across the benchmark run",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def make_parse_rate_chart(out_path: Path, field_parses: pd.DataFrame):
    """Grouped bar chart: parse validity rate by variant × field."""
    plt = _try_import_matplotlib()
    if plt is None:
        return False
    if len(field_parses) == 0:
        return False

    # Compute valid_pct per (variant, field)
    grp = (
        field_parses.groupby(["variant", "field"])
        .agg(n=("valid", "size"), valid=("valid", "sum"))
        .reset_index()
    )
    grp["valid_pct"] = grp["valid"] / grp["n"] * 100

    variants = sorted(grp["variant"].unique())
    fields = sorted(grp["field"].unique())

    import numpy as np
    bar_width = 0.8 / len(variants)
    x = np.arange(len(fields))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    palette = ["#a8d8ea", "#aa96da", "#fcbad3"]

    for i, variant in enumerate(variants):
        sub = grp[grp["variant"] == variant].set_index("field")
        heights = [
            sub.loc[f, "valid_pct"] if f in sub.index else None
            for f in fields
        ]
        positions = x + (i - len(variants) / 2 + 0.5) * bar_width
        # Replace None with NaN bars at 0 height so they show as gaps
        cleaned = [h if h is not None else 0 for h in heights]
        bars = ax.bar(positions, cleaned, bar_width,
                       label=f"Variant {variant}",
                       color=palette[i % len(palette)],
                       edgecolor="black", linewidth=0.5)
        # Annotate each bar with the percent value
        for bar, h, orig in zip(bars, cleaned, heights):
            if orig is None:
                ax.text(bar.get_x() + bar.get_width() / 2, 2, "n/a",
                        ha="center", va="bottom", fontsize=8, color="#888")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{h:.0f}%", ha="center", va="bottom",
                        fontsize=8, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=30, ha="right")
    ax.set_ylabel("Valid parse rate (%)")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, linestyle=":", color="#444", alpha=0.5)
    ax.set_title("Per-field parse validity rate by variant\n"
                  "(across all repeats, excluding warmup)",
                  fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 5 — labeling template + label guide (for downstream ROC analysis)
# ──────────────────────────────────────────────────────────────────────────────

CATEGORIES_FOR_LABELING = [
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


def build_labeling_template(side_by_side: pd.DataFrame) -> pd.DataFrame:
    """
    One row per image. Three blocks of columns:

      (1) Reference — A_natural and A_strict descriptions so you know what
          the image is without re-opening files.
      (2) Per-category labels — one column per sensitivity category, takes
          values: 0 (clearly not present), 1 (clearly present), ? (uncertain).
          Multi-label by design — an image can be both Native American imagery
          AND Historical racialized performance.
      (3) Overall judgment — `any_sensitive` (Y/N), `confidence` (low/med/high),
          `notes` (free text).

    The per-category columns are what feeds into ROC analysis later: for each
    category, Method 1's cosine similarity score becomes the predictor, and
    these labels become the ground truth.
    """
    rows = []
    for _, r in side_by_side.iterrows():
        row: Dict[str, Any] = {"image_name": r["image_name"]}
        # Reference text — paste-ready in the spreadsheet
        row["A_natural"] = r.get("A_natural", "")
        row["A_strict"] = r.get("A_strict", "")
        # Per-category label columns — blank for manual entry
        for cat in CATEGORIES_FOR_LABELING:
            # Use a short safe column name
            short = (cat.lower()
                     .replace("/", "_")
                     .replace(" ", "_")
                     .replace(",", ""))
            row[f"label_{short}"] = ""
        # Aggregate fields
        row["any_sensitive"] = ""        # Y/N — does any category apply?
        row["label_confidence"] = ""     # low / med / high — how sure are you?
        row["notes"] = ""                # free text
        rows.append(row)
    # Build the column order
    cols = ["image_name", "A_natural", "A_strict"]
    for cat in CATEGORIES_FOR_LABELING:
        short = (cat.lower().replace("/", "_").replace(" ", "_").replace(",", ""))
        cols.append(f"label_{short}")
    cols += ["any_sensitive", "label_confidence", "notes"]
    return pd.DataFrame(rows, columns=cols)


def write_label_guide(out_path: Path):
    """Write the label-guide markdown that defines the labeling rules."""
    guide = """# Label Guide — Sensitivity Ground Truth

## Purpose

You're producing ground truth labels for 50 archival images so we can run
ROC/PR analysis on Method 1 (cosine similarity against the keyword taxonomy)
and on variant C (combined-prompt LLM classification). Without these labels,
neither analysis can be meaningful.

## What to fill in

Open `labeling_template.csv` in a spreadsheet. For each of the 50 rows:

1. **Read the A_natural and A_strict columns** to see what's in the image.
   If those don't give you enough information, open the actual image file
   at `./images/<image_name>`.

2. **For each `label_*` column**, enter:
   - `1` — the category clearly applies to this image
   - `0` — the category clearly does not apply
   - `?` — uncertain; flag for review

3. **Multi-label is expected.** An image can be both Native American imagery
   AND Historical racialized performance if both apply.

4. **Fill in `any_sensitive`** — Y if any label column is 1; N if all are 0;
   `?` if any are uncertain.

5. **Fill in `label_confidence`** — how sure are you about your call:
   - `low` — really not sure; could go either way
   - `med` — reasonably confident but the image has ambiguity
   - `high` — clear-cut, no real doubt

6. **Use `notes` freely** — capture anything that explains a hard call,
   especially anything you want to ask your supervisor about.

## Category rules — when each label applies

### Historical racialized performance
- Apply when: image shows people in racial caricature costume (blackface,
  yellowface, redface), minstrel-show performers, stereotype-driven theatrical
  imagery from the racial-segregation era, racialized vaudeville/burlesque.
- Don't apply for: photographs of people of any race that don't involve
  caricature or performance. A respectful portrait of a Black performer is
  not in this category. A non-Black performer in blackface IS.

### Human remains
- Apply when: skeletons, skulls, mummies, corpses, cadavers, archaeological
  excavations showing bones, anthropological displays of remains.
- Don't apply for: photographs of living people, statues/sculptures of
  human figures, anatomical drawings unless they show actual remains.

### Native American imagery
- Apply when: image shows Native American people, ceremonial regalia,
  powwow scenes, sacred objects, OR mascot-style caricatures (e.g., Chief
  Illiniwek). This is intentionally broad — the workflow treats anything
  in this category as warranting archivist review before public display.
- Don't apply for: imagery that doesn't depict Native American people or
  cultural elements.
- **Edge case:** a respectful documentary photograph of a powwow IS in this
  category, even though it's not "offensive." The category exists to flag
  for review, not to make a judgment about offensiveness.

### Nudity/sexual content
- Apply when: nudity, partial nudity in sexualized context, sexual acts,
  pornography, sexually suggestive imagery.
- **Critical:** never label `1` if minors are involved. Flag with `?` in
  the label column AND in `notes` immediately; escalate to your supervisor
  outside this workflow.
- Don't apply for: classical art with nude figures unless explicitly
  sexualized. Medical illustrations.

### Violence/graphic content
- Apply when: visible blood, weapons being used to harm, injury, war
  carnage, executions, fights in progress.
- Don't apply for: photographs of soldiers, weapons in static display
  (museum, gun shop), wounded people receiving care (medical context).
- **Ambiguous:** war photographs. A photo of a battlefield is not the
  same as a photo mid-battle. Use `med` or `low` confidence and write a
  note.

### Hate symbols
- Apply when: visible swastika, KKK regalia, Confederate flag in
  ideological context, white supremacist symbology (88, 14 words,
  totenkopf, blood drop cross), neo-Nazi imagery.
- Don't apply for: historical documentation of WWII when the symbol
  appears in context of opposition (e.g., Allied propaganda showing a
  swastika to denounce it). Use `med` confidence and a note explaining.

### Medical/health records
- Apply when: visible medical chart, patient ID, prescription, lab result,
  hospital wristband, vaccination record, X-ray or scan with identifiable
  patient info.
- Don't apply for: generic medical imagery (a stethoscope, an empty
  hospital room) with no patient-specific content.

### Student/PII records
- Apply when: visible student ID, SSN, transcript, financial aid form,
  enrollment paperwork, anything with a name + a sensitive datapoint
  (grade, address, etc.).
- Don't apply for: a yearbook photo with a name (that's published data).
  Generic classroom photos.

### Other sensitive categories
- Apply when: extremism imagery (ISIS, etc.), drug paraphernalia in use,
  self-harm depictions, eating-disorder imagery.
- Catchall — use sparingly. If you're using it, put a description in
  notes so we know what specifically triggered it.

## Practical tips

- **Don't change your mind backward** without redoing earlier rows. If you
  decide partway through that the bar for "Violence/graphic content" should
  be higher, go back and re-score earlier images you've already labeled.
- **Use `?` liberally for genuinely uncertain cases.** ROC analysis lets us
  exclude uncertain rows; it can't fix bad confident labels.
- **For an archival workflow, over-flag rather than under-flag.** When in
  doubt between 0 and 1, lean toward 1 with a `med` confidence — that's
  what an archivist would do.
- **Time budget:** ~5-7 minutes per image if you go carefully. ~5 hours
  for 50 images. Split across sessions if you need to.

## What we'll do with this

Once you've filled in all 50 rows, run `roc_analysis.py` (which I'll write
after labels exist). It will:

  - Run Method 1's semantic similarity over all 50 images
  - For each category, build an ROC curve comparing Method 1's cosine score
    against your `label_<category>` column
  - Compute AUC per category
  - Suggest threshold values per category that maximize F1 or recall
  - Also run a precision-recall analysis (more honest on imbalanced data)
  - Cross-check variant C's `offense_category` predictions against your
    `any_sensitive` + `label_<category>` columns

Output: a markdown report with per-category curves, AUC scores, and
recommended threshold changes.
"""
    out_path.write_text(guide)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis 6 — audit summary (was analysis 4 in v1)
# ──────────────────────────────────────────────────────────────────────────────

def write_audit_summary(out_path: Path,
                        run_dir: Path,
                        field_parses: pd.DataFrame,
                        parse_failures: pd.DataFrame,
                        inference_df: Optional[pd.DataFrame]):
    lines = []
    lines.append("=" * 70)
    lines.append("  Prompt-shape audit summary")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Run directory: {run_dir}")
    lines.append("")

    # Field validity rates
    lines.append("Field validity rates (across all measured repeats):")
    lines.append("")
    valid_by = (
        field_parses.groupby(["variant", "field"])
        .agg(n=("valid", "size"), valid=("valid", "sum"))
        .reset_index()
    )
    valid_by["valid_pct"] = (valid_by["valid"] / valid_by["n"] * 100).round(1)
    for variant in sorted(valid_by["variant"].unique()):
        lines.append(f"  Variant {variant}:")
        sub = valid_by[valid_by["variant"] == variant]
        for _, r in sub.iterrows():
            mark = "OK" if r["valid_pct"] == 100 else "!!"
            lines.append(f"    [{mark}] {r['field']:20s} {r['valid_pct']:5.1f}%  "
                         f"({int(r['valid'])}/{int(r['n'])})")
        lines.append("")

    # Parse failures
    if len(parse_failures) > 0:
        lines.append("-" * 70)
        lines.append("Parse failure breakdown:")
        lines.append("")
        by_reason = (
            parse_failures.groupby(["variant", "field", "reason"])
            .size().reset_index(name="count").sort_values("count", ascending=False)
        )
        for _, r in by_reason.iterrows():
            lines.append(f"  Variant {r['variant']} / {r['field']:20s} / "
                         f"{str(r['reason']):40s} : {r['count']}")
        lines.append("")
        lines.append("Unique failing images per variant:")
        for variant in sorted(parse_failures["variant"].unique()):
            sub = parse_failures[parse_failures["variant"] == variant]
            n_unique = sub["image_name"].nunique()
            n_total = len(sub)
            n_reps = sub["repeat"].nunique()
            lines.append(f"  Variant {variant}: {n_unique} unique images, "
                         f"{n_total} total field failures across {n_reps} repeats")
        lines.append("")

    # Latency
    if inference_df is not None:
        lines.append("-" * 70)
        lines.append("Latency framing (mean across all repeats, warmup excluded):")
        lines.append("")
        non_warmup = inference_df[~inference_df["is_warmup"]].copy()
        non_warmup["variant"] = non_warmup["prompt_name"].str[0]
        prompts_per_img = {"A": 3, "B": 4, "C": 1}
        for variant in ["A", "B", "C"]:
            sub = non_warmup[non_warmup["variant"] == variant]
            if len(sub) == 0:
                continue
            mean_call = float(sub["wall_time_sec"].mean())
            n_calls = len(sub)
            ppi = prompts_per_img.get(variant, 1)
            per_image = mean_call * ppi
            lines.append(f"  Variant {variant}: {mean_call:.2f}s/call × "
                         f"{ppi} prompts = {per_image:.2f}s/image  ({n_calls} calls)")
        lines.append("")

    lines.append("-" * 70)
    lines.append("Decision frame:")
    lines.append("")
    lines.append("  This summary tells you SCHEMA validity. It does NOT tell")
    lines.append("  you content validity. To answer the content question, open")
    lines.append("  scoring_template.csv in a spreadsheet and fill in the score_*")
    lines.append("  columns for each image (~5 min for 50 images).")
    lines.append("")
    lines.append("  Three plausible outcomes:")
    lines.append("")
    lines.append("    1. C content matches A on >90% of images")
    lines.append("       → Adopt C. 22% speedup, negligible quality cost.")
    lines.append("")
    lines.append("    2. C drifts on 20-50% of images")
    lines.append("       → Design v2 as hybrid: combined prompt for moderation,")
    lines.append("         separate call for description. ~1.5x speedup, full quality.")
    lines.append("")
    lines.append("    3. C drifts on >50% of images")
    lines.append("       → Reject C. Stay with A or B. Speedup must come from")
    lines.append("         quantization or concurrency instead.")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.benchmark_dir)
        if run_dir is None:
            sys.exit(f"No runs found in {args.benchmark_dir}")
    run_dir = run_dir.resolve()
    out_dir = args.out_dir or (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"Analyzing run: {run_dir.name}")
    info(f"Run dir: {run_dir}")
    info(f"Output:  {out_dir}")

    field_parses, outputs, inference = load_run(run_dir)
    info(f"field_parses rows: {len(field_parses):,}")
    info(f"outputs.jsonl rows: {len(outputs):,}")
    if inference is not None:
        info(f"inference_timings rows: {len(inference):,}")

    # 1. Parse failures
    banner("ANALYSIS 1 - Parse failures")
    parse_failures = build_parse_failures(field_parses, outputs)
    if len(parse_failures) == 0:
        info("No parse failures found. Everything validated cleanly.")
    else:
        pf_path = out_dir / "parse_failures.csv"
        parse_failures.to_csv(pf_path, index=False)
        info(f"Wrote {len(parse_failures)} failure rows -> {pf_path.name}")
        info("")
        info("Failure-mode breakdown:")
        counter = Counter(zip(parse_failures["variant"],
                              parse_failures["field"],
                              parse_failures["reason"]))
        for (variant, field, reason), n in counter.most_common(10):
            info(f"  Variant {variant} / {field} / {reason}: {n}")

    # 2. Side-by-side
    banner("ANALYSIS 2 - Side-by-side comparison")
    sbs = build_side_by_side(outputs)
    if len(sbs) == 0:
        info("No side-by-side data (no repeat-0 outputs?).")
    else:
        sbs_path = out_dir / "side_by_side.csv"
        sbs.to_csv(sbs_path, index=False)
        info(f"Wrote {len(sbs)} image rows -> {sbs_path.name}")

    # 3. Scoring template
    banner("ANALYSIS 3 - Scoring template")
    if len(sbs) > 0:
        st = build_scoring_template(sbs)
        st_path = out_dir / "scoring_template.csv"
        st.to_csv(st_path, index=False, quoting=csv.QUOTE_ALL)
        info(f"Wrote {len(st)} scorable rows -> {st_path.name}")
        info("")
        info("Open this CSV in a spreadsheet and fill in the score_* columns.")
        info("~5 sec per image x 50 images = ~5 minutes total.")

    # 4. Charts
    banner("ANALYSIS 4 - Charts")
    charts_made = []
    if inference is not None and len(inference) > 0:
        chart_path = out_dir / "chart_latency_distribution.png"
        if make_latency_chart(chart_path, inference):
            charts_made.append(chart_path.name)
            info(f"Wrote latency distribution chart -> {chart_path.name}")

    # GPU timeline needs both resource_samples and stage_timings
    res_path = run_dir / "resource_samples.parquet"
    stage_path = run_dir / "stage_timings.parquet"
    if res_path.exists() and stage_path.exists():
        res_df = pd.read_parquet(res_path)
        stage_df = pd.read_parquet(stage_path)
        chart_path = out_dir / "chart_gpu_timeline.png"
        if make_gpu_timeline_chart(chart_path, res_df, stage_df):
            charts_made.append(chart_path.name)
            info(f"Wrote GPU timeline chart -> {chart_path.name}")
    else:
        info("Skipping GPU timeline — missing resource_samples or stage_timings")

    if len(field_parses) > 0:
        chart_path = out_dir / "chart_parse_rate_by_field.png"
        if make_parse_rate_chart(chart_path, field_parses):
            charts_made.append(chart_path.name)
            info(f"Wrote parse-rate chart -> {chart_path.name}")

    if not charts_made:
        info("No charts produced.")

    # 5. Labeling template + label guide (for downstream ROC analysis)
    banner("ANALYSIS 5 - Labeling template + label guide")
    if len(sbs) > 0:
        lt = build_labeling_template(sbs)
        lt_path = out_dir / "labeling_template.csv"
        lt.to_csv(lt_path, index=False, quoting=csv.QUOTE_ALL)
        info(f"Wrote {len(lt)} unlabeled image rows -> {lt_path.name}")
        guide_path = out_dir / "label_guide.md"
        write_label_guide(guide_path)
        info(f"Wrote label guide -> {guide_path.name}")
        info("")
        info("Read label_guide.md FIRST, then label all 50 rows in labeling_template.csv.")
        info("This is the ground-truth set for the ROC/PR analysis on Method 1.")

    # 6. Summary
    banner("ANALYSIS 6 - Audit summary")
    summary_path = out_dir / "audit_summary.txt"
    write_audit_summary(summary_path, run_dir,
                        field_parses, parse_failures, inference)
    info("")
    info(f"Summary written: {summary_path}")

    banner("DONE")
    info(f"All artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
