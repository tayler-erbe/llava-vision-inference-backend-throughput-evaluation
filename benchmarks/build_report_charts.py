#!/usr/bin/env python3
"""
build_report_charts.py — generate the 4 headline visualizations for the
Performance Engineering Report.

Reads parquet files from the three concurrency-experiment runs and writes
four PNG files into output/report_charts/:

  1. 01_throughput_scaling.png   — img/min vs concurrency, 3 configs
  2. 02_gpu_utilization.png      — GPU util % vs concurrency, 3 configs
  3. 03_latency_inflation.png    — median request latency vs concurrency
  4. 04_corpus_time.png          — hours-to-complete for 12,125 images

Schema reality (discovered from the parquet files on this host):
- stage_timings: {stage, concurrency, batch_wall_sec, images, calls}
  -> source of truth; tells us how many calls happened in each stage
- inference_timings: {ts, wall_time_sec, is_warmup, success, stage=None}
  -> stage column is always None; ts is response-receipt time and bunches
     at end-of-batch, so a timestamp-window join doesn't work reliably.
     We bucket sequentially: rows are ordered by call dispatch order,
     stage_timings tells us how many calls each stage made.
- resource_samples: {stage, gpu_util_pct, vram_used_mb, power_w}
  -> per-stage median, joined via stage string parsing

Usage
-----
  python3 build_report_charts.py
  python3 build_report_charts.py --verbose
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE = Path("")
OUT_DIR = BASE / "output" / "report_charts"

RUNS = [
    ("Ollama / llava-llama3 (Q4_0)",
     BASE / "output/benchmark_concurrency/run_20260515_115942",
     "#C0504D", "o"),
    ("Ollama / llava-1.5-7b (Q4_0)",
     BASE / "output/benchmark_concurrency/run_20260516_154541",
     "#E8A33D", "s"),
    ("vLLM / llava-1.5-7b (FP16)",
     BASE / "output/benchmark_concurrency_vllm/run_20260515_151427",
     "#2E75B6", "D"),
]

CORPUS_SIZE = 12_125
CONCURRENCY_LEVELS = [1, 2, 4, 8]

# ─────────────────────────────────────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────────────────────────────────────

def _stage_to_c(stage_val):
    """Extract integer concurrency from stage strings like 'concurrency_8'."""
    if not isinstance(stage_val, str):
        return None
    for tok in stage_val.replace("-", "_").split("_"):
        if tok.isdigit():
            return int(tok)
    return None


def _bucket_inference_by_position(inf_df: pd.DataFrame,
                                  stage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort inference rows by ts and slice them into stages by position.
    stage_df has 'n_images' column giving N images per stage.
    inf_df contains only NON-WARMUP rows already, so the first N rows are
    stage 1's measured calls, next N are stage 2's, etc.
    """
    stage_df_sorted = stage_df.sort_values("concurrency").reset_index(drop=True)
    inf_sorted = inf_df.sort_values("ts").reset_index(drop=True)

    concurrencies = []
    cursor = 0
    for _, srow in stage_df_sorted.iterrows():
        n = int(srow["n_images"])
        c = int(srow["concurrency"])
        for _ in range(n):
            if cursor < len(inf_sorted):
                concurrencies.append(c)
                cursor += 1

    # If there are leftover inference rows, drop them (shouldn't happen but be safe)
    inf_sorted = inf_sorted.iloc[:len(concurrencies)].copy()
    inf_sorted["concurrency"] = concurrencies
    return inf_sorted


def load_run(run_dir: Path) -> pd.DataFrame:
    """
    Returns columns: concurrency, batch_wall_sec, n_images, throughput_img_min,
                     median_latency_s, gpu_util_pct, vram_used_mb, power_w
    """
    # 1. Stage timings — throughput
    stage = pd.read_parquet(run_dir / "stage_timings.parquet")
    if "success" in stage.columns:
        stage = stage[stage["success"]]
    stage = stage[stage["stage"].astype(str).str.startswith("concurrency_")]

    stage_agg = stage.groupby("concurrency", as_index=False).agg(
        batch_wall_sec=("batch_wall_sec", "first"),
        n_calls=("calls", "first"),
        n_images=("images", "first"),
    )
    stage_agg["throughput_img_min"] = (
        stage_agg["n_images"] / stage_agg["batch_wall_sec"] * 60.0
    )

    # 2. Inference timings — bucket by position, then median latency per bucket
    inf = pd.read_parquet(run_dir / "inference_timings.parquet")
    if "is_warmup" in inf.columns:
        inf = inf[~inf["is_warmup"]]
    if "success" in inf.columns:
        inf = inf[inf["success"]]

    if "stage" in inf.columns and inf["stage"].notna().any():
        # If we have explicit stage labels, use them
        inf = inf.dropna(subset=["stage"])
        inf["concurrency"] = inf["stage"].apply(_stage_to_c)
    else:
        # Bucket sequentially using stage_timings.n_images counts
        inf = _bucket_inference_by_position(inf, stage_agg)

    inf = inf.dropna(subset=["concurrency"])
    inf["concurrency"] = inf["concurrency"].astype(int)

    inf_agg = (inf
        .groupby("concurrency")["wall_time_sec"]
        .median()
        .reset_index()
        .rename(columns={"wall_time_sec": "median_latency_s"}))

    # 3. Resource samples — per-stage median GPU util
    rs = pd.read_parquet(run_dir / "resource_samples.parquet")
    rs = rs.dropna(subset=["stage"])
    rs_agg = (rs
        .groupby("stage")
        .agg(gpu_util_pct=("gpu_util_pct", "median"),
             vram_used_mb=("vram_used_mb", "median"),
             power_w=("power_w", "median"))
        .reset_index())
    rs_agg["concurrency"] = rs_agg["stage"].apply(_stage_to_c)
    rs_agg = rs_agg.dropna(subset=["concurrency"])
    rs_agg["concurrency"] = rs_agg["concurrency"].astype(int)

    # 4. Join everything on concurrency
    out = stage_agg.merge(
        inf_agg[["concurrency", "median_latency_s"]],
        on="concurrency", how="left")
    out = out.merge(
        rs_agg[["concurrency", "gpu_util_pct", "vram_used_mb", "power_w"]],
        on="concurrency", how="left")

    return out.sort_values("concurrency").reset_index(drop=True)


def load_all_runs():
    out = []
    for label, run_dir, color, marker in RUNS:
        if not run_dir.exists():
            print(f"  WARNING: {run_dir} not found, skipping", file=sys.stderr)
            continue
        try:
            df = load_run(run_dir)
            out.append((label, color, marker, df))
            last = df.iloc[-1]
            lat_str = f"{last['median_latency_s']:.1f}s" if pd.notna(last['median_latency_s']) else "n/a"
            print(f"  Loaded {label}: {len(df)} levels, "
                  f"C=8 throughput={last['throughput_img_min']:.1f} img/min, "
                  f"GPU util={last['gpu_util_pct']:.0f}%, "
                  f"median latency={lat_str}")
        except Exception as e:
            print(f"  ERROR loading {label}: {e}", file=sys.stderr)
            raise
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Plot styling
# ─────────────────────────────────────────────────────────────────────────────

def style_axes(ax, *, title, xlabel, ylabel, xticks=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def save_fig(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Wrote {out_path.name} ({out_path.stat().st_size:,} bytes)")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def chart_throughput(runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, marker, df in runs:
        ax.plot(df["concurrency"], df["throughput_img_min"],
                marker=marker, markersize=9, linewidth=2.4,
                color=color, label=label)
        last = df.iloc[-1]
        ax.annotate(f"{last['throughput_img_min']:.1f}",
                    xy=(last["concurrency"], last["throughput_img_min"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=color)

    style_axes(ax,
        title="Throughput scaling: Ollama serializes, vLLM batches",
        xlabel="Concurrency (parallel in-flight requests)",
        ylabel="Throughput (images / minute)",
        xticks=CONCURRENCY_LEVELS)
    ax.set_xlim(0.5, 9.5)
    y_max = max(df["throughput_img_min"].max() for _, _, _, df in runs) * 1.15
    ax.set_ylim(0, max(60, y_max))
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    save_fig(fig, "01_throughput_scaling.png")


def chart_gpu_utilization(runs):
    fig, ax = plt.subplots(figsize=(8, 5))

    concurrencies = CONCURRENCY_LEVELS
    n_runs = len(runs)
    bar_width = 0.78 / n_runs
    x = list(range(len(concurrencies)))

    for i, (label, color, marker, df) in enumerate(runs):
        utils = []
        for c in concurrencies:
            sub = df[df["concurrency"] == c]
            utils.append(sub["gpu_util_pct"].iloc[0] if not sub.empty else 0)
        positions = [xi + (i - (n_runs - 1) / 2) * bar_width for xi in x]
        bars = ax.bar(positions, utils, bar_width,
                      color=color, label=label, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, utils):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                        f"{val:.0f}%", ha="center", va="bottom",
                        fontsize=9, color=color, fontweight="bold")

    style_axes(ax,
        title="GPU utilization: vLLM saturates the L4, Ollama leaves it idle",
        xlabel="Concurrency",
        ylabel="Median GPU utilization (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.axhline(y=100, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=0)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    save_fig(fig, "02_gpu_utilization.png")


def chart_latency(runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, marker, df in runs:
        ax.plot(df["concurrency"], df["median_latency_s"],
                marker=marker, markersize=9, linewidth=2.4,
                color=color, label=label)
        last = df.iloc[-1]
        if pd.notna(last['median_latency_s']):
            ax.annotate(f"{last['median_latency_s']:.1f}s",
                        xy=(last["concurrency"], last["median_latency_s"]),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=11, fontweight="bold", color=color)

    style_axes(ax,
        title="Per-request latency: linear queueing on Ollama, near-flat on vLLM",
        xlabel="Concurrency",
        ylabel="Median request latency (seconds)",
        xticks=CONCURRENCY_LEVELS)
    ax.set_xlim(0.5, 9.5)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    save_fig(fig, "03_latency_inflation.png")


def chart_corpus_time(runs):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    bars_data = []
    for label, color, marker, df in runs:
        best = df["throughput_img_min"].max()
        hours = CORPUS_SIZE / best / 60.0
        bars_data.append((label, color, hours, best))

    labels = [b[0] for b in bars_data]
    hours = [b[2] for b in bars_data]
    colors = [b[1] for b in bars_data]
    throughputs = [b[3] for b in bars_data]

    bars = ax.barh(labels, hours, color=colors, edgecolor="white", linewidth=0.5)

    for bar, hrs, tput in zip(bars, hours, throughputs):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{hrs:.1f} hrs   ({tput:.1f} img/min)",
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_title(f"Time to process the full {CORPUS_SIZE:,}-image corpus",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Wall-clock hours (continuous inference, best concurrency for backend)",
                  fontsize=11)
    ax.set_xlim(0, max(hours) * 1.30)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.invert_yaxis()

    save_fig(fig, "04_corpus_time.png")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true",
                        help="Also write a CSV summary of the per-config data")
    args = parser.parse_args()

    print("Loading benchmark runs...")
    runs = load_all_runs()
    if len(runs) < 2:
        print("ERROR: need at least 2 runs for comparison charts", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\nPer-run per-concurrency summary:")
        for label, _, _, df in runs:
            print(f"\n=== {label} ===")
            print(df[["concurrency", "n_images", "batch_wall_sec",
                      "throughput_img_min", "median_latency_s",
                      "gpu_util_pct", "vram_used_mb", "power_w"]]
                  .to_string(index=False))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(
            [df.assign(config=label) for label, _, _, df in runs],
            ignore_index=True)
        csv_path = OUT_DIR / "combined_data.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}")

    print("\nGenerating charts...")
    chart_throughput(runs)
    chart_gpu_utilization(runs)
    chart_latency(runs)
    chart_corpus_time(runs)

    print(f"\nDone. All charts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
