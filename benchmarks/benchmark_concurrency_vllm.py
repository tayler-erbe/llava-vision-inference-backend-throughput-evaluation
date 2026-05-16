#!/usr/bin/env python3
"""
benchmark_concurrency_vllm.py
==============================
Concurrent-request benchmark for vLLM, mirroring benchmark_concurrency.py
so the two are directly comparable.

Why this exists
---------------
benchmark_concurrency.py (Ollama) showed throughput stays flat at ~9 img/min
regardless of concurrency. That is the signature of internal serialization —
Ollama 0.3.14 processes one request at a time even when N are dispatched.

vLLM is the production-grade alternative. Its claim to fame is continuous
batching + paged attention: multiple in-flight requests share a single
forward pass when their KV cache states allow. On a single L4 with ~5x
KV cache capacity (per vLLM's own memory profiling at load time), we
should see real throughput scaling with concurrency until either the GPU
saturates or the batching budget fills.

What this measures
------------------
Same as the Ollama version, at each concurrency level N in {1, 2, 4, 8}:
  - Aggregate throughput (images/min)
  - Per-request wall time distribution
  - GPU utilization, VRAM, power (via perf_monitor at 2 Hz)

Identical 30-image sample, identical C-v4 prompt, identical num_predict cap.
The only difference is the serving backend.

Key methodology differences from Ollama version
-----------------------------------------------
- Talks to vLLM's OpenAI-compatible /v1/chat/completions endpoint
- Images sent as data: URIs (base64) inside an `image_url` content block,
  rather than Ollama's `images: [b64]` field
- The model name in the request is the local path (vLLM uses the model
  path as its served_model_name when started from disk)

What it does NOT measure
------------------------
Output quality. vLLM-llava-1.5-7b-hf is a different base model than
Ollama-llava-llama3, so output content will differ. The comparison here
is purely throughput. Output content audit is a separate question.

Prerequisites
-------------
A vLLM server must be running and reachable. To launch it:

    source <conda>/bin/activate vllm
    vllm serve ./models/llava-1.5-7b-hf/ \\
        --dtype float16 \\
        --gpu-memory-utilization 0.85 \\
        --max-model-len 2048 \\
        --enforce-eager \\
        --host 127.0.0.1 \\
        --port 8000

Wait for "Application startup complete" before running this benchmark.

Output
------
  benchmark_concurrency_vllm/run_<timestamp>/
    inference_timings.parquet
    resource_samples.parquet
    stage_timings.parquet
    outputs.jsonl
    concurrency_report.md
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


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
    )
except ImportError as e:
    sys.exit(f"Could not import from benchmark_prompts.py: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_IMAGE_DIR = Path(
    './image_data/'
    'All_Scanned_Images/Scanned_Images_Combined'
)
DEFAULT_OUTPUT_DIR = Path(
    './pipeline-workspace/pipelines/'
    'Image Classification/output/benchmark_concurrency_vllm'
)
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8000
# vLLM uses the model path as the model name when served from disk
DEFAULT_MODEL = './models/llava-1.5-7b-hf/'
DEFAULT_NUM_IMAGES = 30
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
# vLLM HTTP call (OpenAI-compatible /v1/chat/completions)
# ──────────────────────────────────────────────────────────────────────────────

def jpeg_to_data_uri(jpeg_path: Path) -> str:
    with open(jpeg_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_vllm(host: str, port: int, model: str, jpeg_path: Path,
              prompt: str, num_predict: int, timeout: int
              ) -> Tuple[str, Dict[str, Any], bool, Optional[str]]:
    """
    Send one chat-completions request to vLLM. Returns
    (output_text, response_json, success, error_message).
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    data_uri = jpeg_to_data_uri(jpeg_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": num_predict,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return "", {}, False, f"HTTP {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        return text, body, True, None
    except requests.exceptions.Timeout:
        return "", {}, False, f"timeout after {timeout}s"
    except Exception as e:
        return "", {}, False, f"{type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Per-request worker
# ──────────────────────────────────────────────────────────────────────────────

def fire_one_request(host: str, port: int, model: str, jpeg_path: Path,
                     orig_path: Path, timeout: int, slot_idx: int,
                     concurrency: int, is_warmup: bool
                     ) -> Dict[str, Any]:
    t0 = time.perf_counter()
    call_start = datetime.now(timezone.utc)
    output, response_body, success, err = call_vllm(
        host, port, model, jpeg_path, COMBINED_PROMPT,
        num_predict=DEFAULT_NUM_PREDICT, timeout=timeout,
    )
    wall = time.perf_counter() - t0
    call_end = datetime.now(timezone.utc)

    # vLLM returns OpenAI-style usage in response_body['usage']
    completion_tokens = 0
    prompt_tokens = 0
    if success and isinstance(response_body, dict):
        usage = response_body.get("usage") or {}
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

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
        "output_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-concurrency-level run
# ──────────────────────────────────────────────────────────────────────────────

def run_at_concurrency(perf, prepared: List[Tuple[Path, Path]],
                       concurrency: int, host: str, port: int, model: str,
                       timeout: int, out_dir: Path) -> Dict[str, Any]:
    banner(f"CONCURRENCY {concurrency} — {len(prepared)} images")
    info_always(f"Dispatching {len(prepared)} requests with "
                f"ThreadPoolExecutor(max_workers={concurrency})")

    records: List[Dict[str, Any]] = []
    warmup_n = min(DEFAULT_WARMUP, len(prepared))

    with perf.stage(f"concurrency_{concurrency}") as ctx:
        # WARMUP — sequential
        if warmup_n > 0:
            info(f"Warmup: {warmup_n} sequential calls")
            for i in range(warmup_n):
                orig, jpeg = prepared[i]
                rec = fire_one_request(host, port, model, jpeg, orig,
                                       timeout, i, concurrency,
                                       is_warmup=True)
                records.append(rec)
                _append_output_jsonl(out_dir, concurrency, rec)

        # MEASURED — full concurrency
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

    # Record to perf_monitor (single-threaded, after the batch)
    for rec in records:
        # Build a synthetic 'ollama_response' so we can reuse record_http;
        # we only use the eval_count field; other timing fields not provided by vLLM.
        synthetic = {
            "eval_count": rec["output_tokens"],
            "prompt_eval_duration": 0,
            "eval_duration": 0,
            "load_duration": 0,
        }
        perf.inference.record_http(
            image_path=rec["image"],
            prompt_name=f"vllm_conc{rec['concurrency']}_slot{rec['slot_idx']}",
            model=model,
            wall_time_sec=rec["wall_time_sec"],
            success=rec["success"],
            ollama_response=synthetic if rec["success"] else None,
            error=rec["error"],
            call_start_ts=rec["call_start_ts"],
            call_end_ts=rec["call_end_ts"],
            is_warmup=rec["is_warmup"],
        )
    perf.inference._flush()

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
            "p95_request_sec": float("nan"),
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
        "prompt_tokens": rec.get("prompt_tokens", 0),
        "output": rec["output"],
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[warn] could not append output: {e}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Resource samples per level
# ──────────────────────────────────────────────────────────────────────────────

def gpu_stats_per_level(resource_df, summaries: List[Dict[str, Any]]
                        ) -> Dict[int, Dict[str, float]]:
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
                 num_images: int, model: str,
                 ollama_baseline: Optional[Dict[int, float]] = None):
    lines = [
        f"# vLLM Concurrency Experiment — {run_id}",
        "",
        f"- Date: {datetime.now(timezone.utc).isoformat()}",
        f"- Host: {socket.gethostname()}",
        f"- Backend: vLLM 0.6.6, OpenAI-compatible /v1/chat/completions",
        f"- Model: {model}",
        f"- Images per level: {num_images}  (same set per level)",
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

    # Direct comparison vs Ollama (if baseline numbers provided)
    if ollama_baseline:
        lines.append("")
        lines.append("## vLLM vs Ollama — head to head")
        lines.append("")
        lines.append("| Concurrency | Ollama (img/min) | vLLM (img/min) | "
                     "vLLM speedup over Ollama |")
        lines.append("|---|---|---|---|")
        for s in summaries:
            c = s["concurrency"]
            ollama_t = ollama_baseline.get(c, float("nan"))
            vllm_t = s["throughput_img_per_min"]
            speedup = (vllm_t / ollama_t) if ollama_t else float("nan")
            lines.append(
                f"| {c} | "
                f"{ollama_t:.1f} | "
                f"{vllm_t:.1f} | "
                f"{speedup:.2f}x |"
            )

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
    lines.append("- **If throughput scales with concurrency** and Ollama "
                 "stayed flat: vLLM's continuous batching is doing real work. "
                 "The L4 has the headroom; Ollama just couldn't access it.")
    lines.append("- **If latency holds roughly constant** as concurrency rises: "
                 "vLLM is processing requests truly in parallel within a "
                 "single forward pass. Per-request work doesn't grow much "
                 "with batch size at this scale.")
    lines.append("- **If throughput plateaus at some C**: that's where the GPU "
                 "or KV cache budget saturates. Above that, more concurrency "
                 "means queueing, not parallelism. Stop scaling there.")

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
# Pre-flight: is vLLM serving?
# ──────────────────────────────────────────────────────────────────────────────

def check_vllm_ready(host: str, port: int, timeout: float = 3.0) -> Tuple[bool, str]:
    try:
        # OpenAI-style /v1/models endpoint should respond if vLLM is up
        url = f"http://{host}:{port}/v1/models"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            body = resp.json()
            ids = [m.get("id", "") for m in body.get("data", [])]
            return True, f"reachable; models: {', '.join(ids) if ids else '(none reported)'}"
        return False, f"HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "connection refused — is vllm serve running?"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    p.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--ollama-baseline", type=Path, default=None,
                   help="Path to Ollama concurrency report.md to include head-to-head")
    p.add_argument("--utils-dir", type=Path, default=None)
    return p.parse_args()


def parse_ollama_baseline(path: Path) -> Dict[int, float]:
    """Extract per-concurrency throughput from an Ollama concurrency_report.md."""
    if not path.exists():
        return {}
    out: Dict[int, float] = {}
    try:
        text = path.read_text()
        in_throughput = False
        for line in text.splitlines():
            if line.startswith("## Throughput scaling"):
                in_throughput = True
                continue
            if in_throughput and line.startswith("##"):
                break
            if in_throughput and line.startswith("|") and "img" not in line.lower() and "---" not in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    try:
                        c = int(cells[0])
                        t = float(cells[2])
                        out[c] = t
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        print(f"[warn] could not parse ollama baseline: {e}")
    return out


def main():
    args = parse_args()
    global QUIET
    QUIET = args.quiet

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

    print(f"vLLM concurrency benchmark — {datetime.now().isoformat()}")
    print(f"  run_id     : {run_id}")
    print(f"  output_dir : {out_dir}")
    print(f"  endpoint   : http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  model      : {args.model}")
    print(f"  images     : {args.num_images} from {args.image_dir.name}")
    print(f"  levels     : {args.levels}")

    banner("PRE-FLIGHT")
    ready, msg = check_vllm_ready(args.host, args.port)
    if not ready:
        fail(f"vLLM server not reachable: {msg}")
        fail("Start vllm serve first. See module docstring for command.")
        sys.exit(2)
    ok(f"vLLM server reachable on {args.host}:{args.port}")
    info_always(f"  {msg}")

    ollama_baseline = {}
    if args.ollama_baseline:
        ollama_baseline = parse_ollama_baseline(args.ollama_baseline)
        if ollama_baseline:
            ok(f"Loaded Ollama baseline: {ollama_baseline}")
        else:
            info_always(f"  (could not parse Ollama baseline from "
                        f"{args.ollama_baseline})")

    banner("Preparing image sample")
    images = pick_images(args.image_dir, args.num_images)
    cache = out_dir / "jpeg_cache"
    prepared: List[Tuple[Path, Path]] = [
        (p, prepare_jpeg(p, cache)) for p in images
    ]
    info_always(f"Prepared {len(prepared)} images")

    perf = PerfMonitor(
        out_dir=out_dir, run_id=run_id,
        workflow_name="benchmark_concurrency_vllm",
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
                 args.num_images, args.model,
                 ollama_baseline=ollama_baseline)

    banner("VLLM CONCURRENCY BENCHMARK COMPLETE")
    info_always(f"All artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
