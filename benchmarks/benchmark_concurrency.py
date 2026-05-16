#!/usr/bin/env python3
"""
benchmark_concurrency.py
=========================
Controlled experiment: does Ollama HTTP parallelize, and at what concurrency
level does throughput stop scaling on this single L4?

Why this exists
---------------
The prompt-shape work measured single-stream latency. Single-stream is one
inference at a time — and at ~5s/image for C-v4, the headline throughput is
12 images/minute. That's the floor.

The ceiling comes from concurrency. The L4 has 23 GB VRAM and the model
takes ~6 GB — there's headroom for multiple in-flight requests. The
question is whether Ollama actually serves them in parallel or serializes
them internally. Until we test, we don't know.

What this measures
------------------
At each concurrency level N in {1, 2, 4, 8}:
  - Aggregate throughput (images/sec, completed by the batch)
  - Per-request wall time distribution (does latency balloon with concurrency?)
  - GPU utilization (does it go up — meaning more parallelism — or stay flat?)
  - VRAM ceiling (does each concurrent request claim more memory?)
  - Power draw (does compute density increase?)

What it does NOT measure
------------------------
Output quality. Concurrency shouldn't change model outputs at temperature=0,
and parse rates from the locked C-v4 prompt are already characterized. This
is a pure throughput experiment.

Key methodology notes
---------------------
- All variants run the locked C-v4 COMBINED_PROMPT (imported from
  benchmark_prompts.py — single source of truth).
- Each concurrency level processes the SAME N_IMAGES images. The aggregate
  throughput is the only fair comparison: total_images / wall_clock_to_finish.
- ThreadPoolExecutor is used because Python threads release the GIL during
  HTTP socket I/O — fine for this workload. The work happens in Ollama's
  process, not ours.
- Ollama 0.3.x serializes requests at the daemon level unless
  OLLAMA_NUM_PARALLEL is set on the SERVER process. If concurrency shows
  no speedup, the script prints a diagnostic about this.

Output
------
  benchmark_concurrency/run_<timestamp>/
    inference_timings.parquet    one row per inference call
    resource_samples.parquet     GPU/CPU/memory traces (2 Hz)
    stage_timings.parquet        one row per concurrency level
    outputs.jsonl                all raw model outputs
    concurrency_report.md        per-level summary + scaling analysis
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Reuse the validated machinery from benchmark_prompts.py
# ──────────────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

try:
    from benchmark_prompts import (
        COMBINED_PROMPT,
        pick_images,
        prepare_jpeg,
        call_http,
    )
except ImportError as e:
    sys.exit(f"Could not import from benchmark_prompts.py: {e}\n"
             f"benchmark_concurrency.py must live in the same directory as "
             f"benchmark_prompts.py.")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGE_DIR = Path(
    './image_data/'
    'All_Scanned_Images/Scanned_Images_Combined'
)
DEFAULT_OUTPUT_DIR = Path(
    './pipeline-workspace/pipelines/'
    'Image Classification/output/benchmark_concurrency'
)
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 11434
DEFAULT_MODEL = 'llava-llama3'
DEFAULT_NUM_IMAGES = 30          # smaller than prompt experiments — each level
                                  # processes this set once; total work scales
                                  # with len(levels) * N
DEFAULT_LEVELS = [1, 2, 4, 8]
DEFAULT_WARMUP = 2
DEFAULT_TIMEOUT = 180
DEFAULT_NUM_PREDICT = 300


# ──────────────────────────────────────────────────────────────────────────────
# Console helpers
# ──────────────────────────────────────────────────────────────────────────────

QUIET = False


def banner(s: str):
    print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}", flush=True)


def info(s: str = ""):
    if not QUIET:
        print(f"  {s}", flush=True)


def info_always(s: str = ""):
    print(f"  {s}", flush=True)


def ok(s: str):
    print(f"  ✓ {s}", flush=True)


def fail(s: str):
    print(f"  ✗ {s}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-request worker
# ──────────────────────────────────────────────────────────────────────────────

def fire_one_request(host: str, port: int, model: str, jpeg_path: Path,
                     orig_path: Path, timeout: int, slot_idx: int,
                     concurrency: int, is_warmup: bool
                     ) -> Dict[str, Any]:
    """
    Single request. Returns a record dict (does not depend on perf_monitor;
    we batch-record afterward to avoid threading hazards inside perf_monitor.
    """
    t0 = time.perf_counter()
    call_start = datetime.now(timezone.utc)
    output, response_dict, success, err = call_http(
        host, port, model, jpeg_path, COMBINED_PROMPT,
        num_predict=DEFAULT_NUM_PREDICT, timeout=timeout,
    )
    wall = time.perf_counter() - t0
    call_end = datetime.now(timezone.utc)

    eval_count = response_dict.get("eval_count", 0) if response_dict else 0
    prompt_eval_dur_ns = response_dict.get("prompt_eval_duration", 0) if response_dict else 0
    eval_dur_ns = response_dict.get("eval_duration", 0) if response_dict else 0
    load_dur_ns = response_dict.get("load_duration", 0) if response_dict else 0

    return {
        "slot_idx": slot_idx,
        "concurrency": concurrency,
        "image": str(orig_path),
        "image_name": orig_path.name,
        "wall_time_sec": wall,
        "success": success,
        "error": err,
        "call_start_ts": call_start,
        "call_end_ts": call_end,
        "output": output,
        "is_warmup": is_warmup,
        "output_tokens": eval_count,
        "prompt_eval_sec": prompt_eval_dur_ns / 1e9 if prompt_eval_dur_ns else 0.0,
        "eval_sec": eval_dur_ns / 1e9 if eval_dur_ns else 0.0,
        "load_sec": load_dur_ns / 1e9 if load_dur_ns else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-concurrency-level run
# ──────────────────────────────────────────────────────────────────────────────

def run_at_concurrency(perf, prepared: List[Tuple[Path, Path]],
                       concurrency: int, host: str, port: int, model: str,
                       timeout: int, out_dir: Path) -> Dict[str, Any]:
    """
    Run all images at the given concurrency level. Returns a summary dict.
    All timing is wall-clock from first dispatch to last completion.
    """
    banner(f"CONCURRENCY {concurrency} — {len(prepared)} images")
    info_always(f"Dispatching {len(prepared)} requests with "
                f"ThreadPoolExecutor(max_workers={concurrency})")

    records: List[Dict[str, Any]] = []
    warmup_n = min(DEFAULT_WARMUP, len(prepared))

    with perf.stage(f"concurrency_{concurrency}") as ctx:
        # WARMUP PHASE — sequential, doesn't count
        if warmup_n > 0:
            info(f"Warmup: {warmup_n} sequential calls")
            for i in range(warmup_n):
                orig, jpeg = prepared[i]
                rec = fire_one_request(host, port, model, jpeg, orig,
                                       timeout, i, concurrency,
                                       is_warmup=True)
                records.append(rec)
                _append_output_jsonl(out_dir, concurrency, rec)

        # MEASURED PHASE — fire all N requests concurrently, time the whole batch
        info(f"Measured: {len(prepared)} concurrent calls "
             f"(pool size {concurrency})")
        batch_start = time.perf_counter()
        batch_start_ts = datetime.now(timezone.utc)
        completed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for i, (orig, jpeg) in enumerate(prepared):
                fut = pool.submit(fire_one_request, host, port, model, jpeg,
                                  orig, timeout, i, concurrency,
                                  is_warmup=False)
                futures.append(fut)
            for fut in as_completed(futures):
                rec = fut.result()
                records.append(rec)
                _append_output_jsonl(out_dir, concurrency, rec)
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    info(f"  ... {completed}/{len(futures)} done")
        batch_wall = time.perf_counter() - batch_start
        batch_end_ts = datetime.now(timezone.utc)

        ctx.set_metadata(model=model, images=len(prepared),
                         concurrency=concurrency,
                         batch_wall_sec=batch_wall,
                         calls=len(prepared) + warmup_n)

    # Record into perf_monitor's inference store (after the batch, single-threaded)
    for rec in records:
        # Build a synthetic ollama_response dict for record_http
        ollama_resp = {
            "eval_count": rec["output_tokens"],
            "prompt_eval_duration": int(rec["prompt_eval_sec"] * 1e9),
            "eval_duration": int(rec["eval_sec"] * 1e9),
            "load_duration": int(rec["load_sec"] * 1e9),
        }
        perf.inference.record_http(
            image_path=rec["image"],
            prompt_name=f"conc{rec['concurrency']}_slot{rec['slot_idx']}",
            model=model,
            wall_time_sec=rec["wall_time_sec"],
            success=rec["success"],
            ollama_response=ollama_resp if rec["success"] else None,
            error=rec["error"],
            call_start_ts=rec["call_start_ts"],
            call_end_ts=rec["call_end_ts"],
            is_warmup=rec["is_warmup"],
        )
    perf.inference._flush()

    # Summary stats for this concurrency level
    measured = [r for r in records if not r["is_warmup"] and r["success"]]
    failed = sum(1 for r in records if not r["is_warmup"] and not r["success"])
    if not measured:
        return {
            "concurrency": concurrency,
            "images": len(prepared),
            "batch_wall_sec": batch_wall,
            "throughput_img_per_min": 0.0,
            "median_request_sec": float("nan"),
            "mean_request_sec": float("nan"),
            "min_request_sec": float("nan"),
            "max_request_sec": float("nan"),
            "failed": failed,
            "batch_start_ts": batch_start_ts,
            "batch_end_ts": batch_end_ts,
        }

    walls = sorted(r["wall_time_sec"] for r in measured)
    n = len(walls)
    median = walls[n // 2]
    mean = sum(walls) / n
    p95 = walls[min(n - 1, int(0.95 * n))]

    summary = {
        "concurrency": concurrency,
        "images": len(measured),
        "batch_wall_sec": batch_wall,
        "throughput_img_per_min": (len(measured) / batch_wall) * 60,
        "median_request_sec": median,
        "mean_request_sec": mean,
        "min_request_sec": walls[0],
        "max_request_sec": walls[-1],
        "p95_request_sec": p95,
        "failed": failed,
        "batch_start_ts": batch_start_ts,
        "batch_end_ts": batch_end_ts,
    }
    info_always(f"[C={concurrency}] batch wall {batch_wall:.1f}s, "
                f"throughput {summary['throughput_img_per_min']:.1f} img/min, "
                f"median request {median:.1f}s, max {walls[-1]:.1f}s")
    return summary


def _append_output_jsonl(out_dir: Path, concurrency: int, rec: Dict[str, Any]):
    path = out_dir / "outputs.jsonl"
    record = {
        "ts": rec["call_end_ts"].isoformat(),
        "concurrency": concurrency,
        "slot_idx": rec["slot_idx"],
        "image_path": rec["image"],
        "success": rec["success"],
        "error": rec["error"],
        "is_warmup": rec["is_warmup"],
        "wall_time_sec": rec["wall_time_sec"],
        "output_tokens": rec["output_tokens"],
        "output": rec["output"],
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[warn] could not append output: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Resource samples — get VRAM/GPU util medians per concurrency window
# ──────────────────────────────────────────────────────────────────────────────

def gpu_stats_per_level(resource_df, summaries: List[Dict[str, Any]]
                        ) -> Dict[int, Dict[str, float]]:
    """For each concurrency level, find GPU/VRAM medians within its window."""
    out: Dict[int, Dict[str, float]] = {}
    if resource_df is None or len(resource_df) == 0:
        return out
    import pandas as pd
    res = resource_df.copy()
    res["ts"] = pd.to_datetime(res["ts"], utc=True)
    for s in summaries:
        c = s["concurrency"]
        bs = pd.to_datetime(s["batch_start_ts"])
        be = pd.to_datetime(s["batch_end_ts"])
        window = res[(res["ts"] >= bs) & (res["ts"] <= be)]
        if len(window) == 0:
            continue
        rec = {}
        for col, key in [("vram_used_mb", "vram_mb"),
                         ("power_w", "power_w"),
                         ("gpu_util_pct", "gpu_util"),
                         ("cpu_pct_total", "cpu_pct")]:
            if col in window.columns:
                vals = window[col].dropna()
                if len(vals) > 0:
                    rec[f"{key}_med"] = float(vals.median())
                    rec[f"{key}_max"] = float(vals.max())
        out[c] = rec
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def write_report(out_dir: Path, run_id: str,
                 summaries: List[Dict[str, Any]],
                 gpu_stats: Dict[int, Dict[str, float]],
                 num_images: int, model: str):
    lines = [
        f"# Concurrency Experiment — {run_id}",
        "",
        f"- Date: {datetime.now(timezone.utc).isoformat()}",
        f"- Host: {socket.gethostname()}",
        f"- Model: {model}",
        f"- Images per level: {num_images}  (same set processed at each "
        f"concurrency level)",
        f"- Prompt: locked C-v4 combined prompt",
        f"- Levels tested: {[s['concurrency'] for s in summaries]}",
        "",
        "## Throughput scaling",
        "",
        "| Concurrency | Batch wall (s) | Images/min | Median req (s) | "
        "P95 req (s) | Max req (s) | Failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['concurrency']} | "
            f"{s['batch_wall_sec']:.1f} | "
            f"{s['throughput_img_per_min']:.1f} | "
            f"{s['median_request_sec']:.2f} | "
            f"{s.get('p95_request_sec', float('nan')):.2f} | "
            f"{s['max_request_sec']:.2f} | "
            f"{s['failed']} |"
        )

    # Scaling table
    base = next((s for s in summaries if s["concurrency"] == 1), None)
    if base:
        lines.append("")
        lines.append("## Scaling vs C=1 baseline")
        lines.append("")
        lines.append("| Concurrency | Throughput speedup | Latency inflation | "
                     "Parallel efficiency |")
        lines.append("|---|---|---|---|")
        base_tput = base["throughput_img_per_min"]
        base_lat = base["median_request_sec"]
        for s in summaries:
            tput_speedup = s["throughput_img_per_min"] / base_tput if base_tput else 0
            lat_inflation = s["median_request_sec"] / base_lat if base_lat else 0
            efficiency = tput_speedup / s["concurrency"] * 100 if s["concurrency"] else 0
            lines.append(
                f"| {s['concurrency']} | "
                f"{tput_speedup:.2f}x | "
                f"{lat_inflation:.2f}x | "
                f"{efficiency:.0f}% |"
            )
        lines.append("")
        lines.append("- **Throughput speedup**: aggregate throughput at this "
                     "concurrency divided by C=1 baseline. Linear scaling = "
                     "concurrency value (e.g. C=4 → 4.0x).")
        lines.append("- **Latency inflation**: per-request median latency at "
                     "this concurrency divided by C=1 baseline. 1.0x = no "
                     "latency cost; higher = each request slowed by queuing.")
        lines.append("- **Parallel efficiency**: speedup as % of perfect "
                     "linear scaling. 100% = perfectly parallel; lower = "
                     "diminishing returns.")

    # GPU/VRAM
    if gpu_stats:
        lines.append("")
        lines.append("## GPU resource usage per concurrency level")
        lines.append("")
        lines.append("| Concurrency | Med VRAM (MB) | Max VRAM (MB) | "
                     "Med GPU util (%) | Max GPU util (%) | Med power (W) |")
        lines.append("|---|---|---|---|---|---|")
        for s in summaries:
            c = s["concurrency"]
            g = gpu_stats.get(c, {})
            lines.append(
                f"| {c} | "
                f"{g.get('vram_mb_med', float('nan')):.0f} | "
                f"{g.get('vram_mb_max', float('nan')):.0f} | "
                f"{g.get('gpu_util_med', float('nan')):.0f} | "
                f"{g.get('gpu_util_max', float('nan')):.0f} | "
                f"{g.get('power_w_med', float('nan')):.1f} |"
            )

    lines.append("")
    lines.append("## Reading this")
    lines.append("")
    lines.append("- **If throughput scales linearly with concurrency** (e.g. "
                 "C=4 → 4x): Ollama is parallelizing well; the GPU has "
                 "headroom and concurrency is your speedup lever.")
    lines.append("- **If throughput stays flat** as concurrency rises: "
                 "Ollama is serializing internally. Likely the daemon was "
                 "started with OLLAMA_NUM_PARALLEL=1 (the default in older "
                 "Ollama). Set OLLAMA_NUM_PARALLEL=4 (or higher) on the "
                 "*Ollama daemon* (`systemctl edit ollama`) and restart it.")
    lines.append("- **If throughput scales but latency balloons**: the GPU "
                 "is saturated. Throughput stops scaling once GPU util "
                 "approaches 100%. Past that, you get queuing, not parallelism.")
    lines.append("- **VRAM ceiling**: each concurrent request claims its "
                 "own KV cache. If VRAM grows linearly with concurrency, "
                 "you'll OOM at high concurrency. Plan accordingly.")

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `inference_timings.parquet` — one row per inference call")
    lines.append("- `resource_samples.parquet` — GPU/CPU/memory at 2 Hz")
    lines.append("- `stage_timings.parquet` — one row per concurrency level")
    lines.append("- `outputs.jsonl` — every raw model output")

    report_path = out_dir / "concurrency_report.md"
    report_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines) + "\n")
    ok(f"Report written: {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    p.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
                   help="Concurrency levels to test, e.g. --levels 1 2 4 8")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--utils-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global QUIET
    QUIET = args.quiet

    # perf_monitor
    utils_dir = args.utils_dir or (_THIS_DIR / "utils")
    if not (utils_dir / "perf_monitor.py").exists():
        alt = _THIS_DIR.parent / "utils"
        if (alt / "perf_monitor.py").exists():
            utils_dir = alt
        else:
            fail(f"Could not find perf_monitor.py in {utils_dir} or {alt}")
            sys.exit(2)
    sys.path.insert(0, str(utils_dir.parent))
    from utils.perf_monitor import PerfMonitor

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DEFAULT_OUTPUT_DIR / f"run_{ts}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    print(f"Concurrency benchmark — {datetime.now().isoformat()}")
    print(f"  run_id     : {run_id}")
    print(f"  output_dir : {out_dir}")
    print(f"  model      : {args.model}")
    print(f"  images     : {args.num_images} from {args.image_dir.name}")
    print(f"  levels     : {args.levels}")
    print(f"  prompt     : locked C-v4 combined prompt")
    print(f"  num_predict: {DEFAULT_NUM_PREDICT}")
    print(f"  quiet      : {'yes' if QUIET else 'no'}")

    # Pre-flight
    banner("PRE-FLIGHT")
    try:
        with socket.create_connection((args.host, args.port), timeout=3):
            ok(f"Ollama HTTP server reachable on {args.host}:{args.port}")
    except Exception as e:
        fail(f"Cannot reach Ollama: {e}")
        sys.exit(2)

    est_calls = sum(args.levels) * 0 + len(args.levels) * args.num_images
    info_always(f"Plan: {len(args.levels)} concurrency levels × "
                f"{args.num_images} images = {est_calls} measured calls "
                f"(plus warmup)")

    banner("Preparing image sample")
    images = pick_images(args.image_dir, args.num_images)
    cache = out_dir / "jpeg_cache"
    prepared: List[Tuple[Path, Path]] = [
        (p, prepare_jpeg(p, cache)) for p in images
    ]
    info_always(f"Prepared {len(prepared)} images: "
                f"{prepared[0][0].name} ... {prepared[-1][0].name}")

    # Instrumentation
    perf = PerfMonitor(
        out_dir=out_dir, run_id=run_id,
        workflow_name="benchmark_concurrency",
        sample_interval_sec=args.sample_interval,
    )
    perf.out_dir = out_dir

    summaries: List[Dict[str, Any]] = []
    try:
        for level in args.levels:
            s = run_at_concurrency(
                perf, prepared, level,
                args.host, args.port, args.model,
                args.timeout, out_dir,
            )
            summaries.append(s)
    finally:
        perf.close()

    # Reload parquets for the report
    resource_df = None
    try:
        import pandas as pd
        res_path = out_dir / "resource_samples.parquet"
        if res_path.exists():
            resource_df = pd.read_parquet(res_path)
    except Exception as e:
        fail(f"Could not reload resource_samples.parquet: {e}")

    gpu_stats = gpu_stats_per_level(resource_df, summaries)
    write_report(out_dir, run_id, summaries, gpu_stats,
                 args.num_images, args.model)

    banner("CONCURRENCY BENCHMARK COMPLETE")
    info_always(f"All artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
