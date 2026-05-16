#!/usr/bin/env python3
# utils/perf_monitor.py
#
# Performance instrumentation for the image classification pipeline.
# Drop next to pipeline_logger.py in utils/.
#
# Purpose
# -------
# Capture the metrics that turn "the pipeline is slow" into "here is exactly
# where the time and memory and power went, sampled at 1-second resolution,
# for every image, for every stage." Those numbers feed three things:
#
#   1. Operational visibility — find regressions, confirm fixes
#   2. The benchmark harness (benchmark.py) — controlled subprocess-vs-HTTP comparison
#   3. The Red Hat interview story — concrete TTFT/TPOT/throughput numbers
#      measured on real archival workload, not generic LLM marketing benchmarks
#
# Three components
# ----------------
#   ResourceSampler — background thread, snapshots GPU + CPU + memory every
#                     N seconds into resource_samples.parquet. Wraps with a
#                     context manager.
#
#   InferenceTimer  — records one row per inference call into
#                     inference_timings.parquet. Captures wall-clock time
#                     plus, when available, Ollama's internal timing fields
#                     (load_duration, prompt_eval_duration, eval_duration,
#                     prompt_eval_count, eval_count). Works for both the
#                     subprocess and HTTP paths.
#
#   StageTimer      — context manager that times an entire pipeline stage
#                     and writes a single summary row into stage_timings.parquet.
#                     Composes with ResourceSampler so each stage carries
#                     its own resource trace.
#
# Dependencies
# ------------
#   Required : psutil  (already in llama_env)
#   Optional : pynvml  (install for cleanest GPU metrics; falls back to
#              shelling out to nvidia-smi if missing — same data, slower)
#
#   We do NOT import pandas at module top. We import lazily inside flushes
#   so this file can be imported in lightweight contexts without paying
#   pandas's ~100ms startup cost.
#
# Schema philosophy
# -----------------
# Same convention as pipeline_logger.py — parquet with explicit columns,
# defensive writes that never crash the pipeline. Failure to log must not
# fail the pipeline run.

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# GPU adapter — pynvml if available, nvidia-smi subprocess fallback
# ──────────────────────────────────────────────────────────────────────────────

class _GPUAdapter:
    """
    Uniform interface to GPU stats regardless of whether pynvml is installed.

    With pynvml: ~0.5ms per sample, no subprocess overhead
    Without:     ~50ms per sample shelling out to nvidia-smi
    """

    def __init__(self):
        self.backend = "none"
        self._pynvml = None
        self._handles: List[Any] = []
        self._nvidia_smi_path: Optional[str] = None
        self._init()

    def _init(self) -> None:
        # Try pynvml first
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(device_count)]
            self._pynvml = pynvml
            self.backend = "pynvml"
            return
        except ImportError:
            pass
        except Exception:
            # pynvml present but init failed — fall through to subprocess
            pass

        # Fall back to nvidia-smi subprocess
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            self._nvidia_smi_path = nvidia_smi
            self.backend = "nvidia-smi"

    def available(self) -> bool:
        return self.backend != "none"

    def sample(self) -> List[Dict[str, Any]]:
        """Return one dict per GPU. Empty list if no GPU available."""
        if self.backend == "pynvml":
            return self._sample_pynvml()
        if self.backend == "nvidia-smi":
            return self._sample_nvidia_smi()
        return []

    def _sample_pynvml(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        p = self._pynvml
        for idx, handle in enumerate(self._handles):
            try:
                util = p.nvmlDeviceGetUtilizationRates(handle)
                mem = p.nvmlDeviceGetMemoryInfo(handle)
                try:
                    power_mw = p.nvmlDeviceGetPowerUsage(handle)
                except Exception:
                    power_mw = None
                try:
                    temp_c = p.nvmlDeviceGetTemperature(
                        handle, p.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp_c = None
                rows.append({
                    "gpu_index": idx,
                    "gpu_util_pct": int(util.gpu),
                    "mem_util_pct": int(util.memory),
                    "vram_used_mb": int(mem.used / (1024 * 1024)),
                    "vram_total_mb": int(mem.total / (1024 * 1024)),
                    "power_w": round(power_mw / 1000, 2) if power_mw else None,
                    "temp_c": temp_c,
                })
            except Exception:
                # Single-GPU failure shouldn't break the sample
                continue
        return rows

    def _sample_nvidia_smi(self) -> List[Dict[str, Any]]:
        # CSV output is much cheaper to parse than the default text format
        query = (
            "index,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw,temperature.gpu"
        )
        try:
            result = subprocess.run(
                [self._nvidia_smi_path,
                 f"--query-gpu={query}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return []
        if result.returncode != 0:
            return []

        rows: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "gpu_index": int(parts[0]),
                    "gpu_util_pct": int(float(parts[1])),
                    "mem_util_pct": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "vram_total_mb": int(float(parts[4])),
                    "power_w": float(parts[5]) if parts[5] not in ("", "[N/A]") else None,
                    "temp_c": int(float(parts[6])) if parts[6] not in ("", "[N/A]") else None,
                })
            except (ValueError, IndexError):
                continue
        return rows

    def shutdown(self) -> None:
        if self.backend == "pynvml" and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# ResourceSampler — background thread, samples system + GPU every N seconds
# ──────────────────────────────────────────────────────────────────────────────

class ResourceSampler:
    """
    Background sampler. Writes one parquet row per sample interval per GPU.

    Usage:
        sampler = ResourceSampler(out_dir=Path("logs"), interval_sec=2.0)
        with sampler.session(run_id="abc-123", stage="descriptions"):
            do_work()
        # samples flushed to logs/resource_samples.parquet on exit
    """

    def __init__(
        self,
        out_dir: Path,
        interval_sec: float = 2.0,
        run_id: Optional[str] = None,
        workflow_name: str = "image_classification",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.out_dir / "resource_samples.parquet"
        self.interval_sec = max(0.5, float(interval_sec))
        self.workflow_name = workflow_name
        self.run_id = run_id

        self._gpu = _GPUAdapter()
        self._psutil_ok = False
        try:
            import psutil  # noqa: F401
            self._psutil_ok = True
        except ImportError:
            pass

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._current_stage: Optional[str] = None
        self._hostname = socket.gethostname()

    # ── Public API ───────────────────────────────────────────────────────────

    @contextmanager
    def session(self, stage: Optional[str] = None,
                run_id: Optional[str] = None):
        """Start sampling on enter, stop and flush on exit."""
        if run_id is not None:
            self.run_id = run_id
        self._current_stage = stage
        self.start()
        try:
            yield self
        finally:
            self.stop()
            self.flush()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ResourceSampler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def set_stage(self, stage: Optional[str]) -> None:
        """Switch the stage label without restarting sampling."""
        self._current_stage = stage

    def flush(self) -> None:
        """Append buffered samples to parquet. Safe to call repeatedly."""
        with self._buffer_lock:
            if not self._buffer:
                return
            rows = self._buffer
            self._buffer = []

        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            if self.parquet_path.exists():
                try:
                    existing = pd.read_parquet(self.parquet_path)
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception:
                    # Corrupted prior file — preserve new data
                    pass
            df.to_parquet(self.parquet_path, index=False)
        except Exception as e:
            # Logging must never crash the pipeline
            print(f"[perf_monitor] ResourceSampler flush failed: {e}",
                  file=sys.stderr)

    # ── Internal sampling loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        # Initialize psutil's CPU counter so the first reading isn't bogus
        if self._psutil_ok:
            try:
                import psutil
                psutil.cpu_percent(interval=None, percpu=False)
            except Exception:
                pass

        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            sample_ts = datetime.now(timezone.utc)

            # System sample
            sys_row = self._sample_system()

            # GPU sample(s) — one row per GPU
            gpu_rows = self._gpu.sample()

            if not gpu_rows:
                # No GPU; still emit a system-only row
                gpu_rows = [{}]

            for gpu_row in gpu_rows:
                row: Dict[str, Any] = {
                    "ts": sample_ts,
                    "workflow_name": self.workflow_name,
                    "run_id": self.run_id,
                    "stage": self._current_stage,
                    "hostname": self._hostname,
                    "gpu_backend": self._gpu.backend,
                }
                row.update(sys_row)
                row.update(gpu_row)
                with self._buffer_lock:
                    self._buffer.append(row)

            # Periodic flush so a long stage doesn't hold all samples in memory
            if len(self._buffer) >= 200:
                self.flush()

            # Tick on a fixed cadence; if we fell behind, catch up without piling
            next_tick += self.interval_sec
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)
            else:
                next_tick = time.monotonic()

    def _sample_system(self) -> Dict[str, Any]:
        if not self._psutil_ok:
            return {}
        try:
            import psutil
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            la1, la5, la15 = psutil.getloadavg()
            return {
                "cpu_pct_total": psutil.cpu_percent(interval=None, percpu=False),
                "cpu_count": psutil.cpu_count(logical=True),
                "mem_used_mb": int(vm.used / (1024 * 1024)),
                "mem_total_mb": int(vm.total / (1024 * 1024)),
                "mem_pct": float(vm.percent),
                "swap_used_mb": int(sw.used / (1024 * 1024)),
                "swap_pct": float(sw.percent),
                "loadavg_1m": la1,
                "loadavg_5m": la5,
                "loadavg_15m": la15,
            }
        except Exception:
            return {}


# ──────────────────────────────────────────────────────────────────────────────
# InferenceTimer — one row per inference call
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRecord:
    """
    One inference call's worth of timing data.

    Common to subprocess + HTTP paths:
        wall_time_sec, image_path, prompt_name, success

    Available only from Ollama HTTP /api/generate (nanoseconds → seconds):
        load_sec               (model load into VRAM — should be ~0 after first call)
        prompt_eval_sec        (prefill time)
        eval_sec               (decode time)
        prompt_tokens          (input tokens)
        output_tokens          (generated tokens)

    Derived (computed on write if data available):
        tpot_ms                (eval_sec * 1000 / output_tokens)
        prefill_throughput_tok_s
        decode_throughput_tok_s
        load_overhead_pct      (load_sec / wall_time_sec * 100)
    """
    ts: datetime
    workflow_name: str
    run_id: Optional[str]
    stage: Optional[str]
    image_path: Optional[str]
    prompt_name: str
    inference_mode: str           # "subprocess" or "http"
    model: str
    wall_time_sec: float
    success: bool
    error: Optional[str] = None

    # Inference call window — used to join against resource_samples for
    # "GPU util during inference" vs "GPU util between inference" analysis
    call_start_ts: Optional[datetime] = None
    call_end_ts: Optional[datetime] = None

    # Warmup separation — set True for calls that should be excluded from
    # production-quality summary stats (cold-load amortization, first-call effects)
    is_warmup: bool = False

    # HTTP-only fields
    load_sec: Optional[float] = None
    prompt_eval_sec: Optional[float] = None
    eval_sec: Optional[float] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Derived (filled in by InferenceTimer.log_record)
    tpot_ms: Optional[float] = None
    prefill_throughput_tok_s: Optional[float] = None
    decode_throughput_tok_s: Optional[float] = None
    load_overhead_pct: Optional[float] = None


class InferenceTimer:
    """
    Writer for inference_timings.parquet.

    Two entry points:

      timer.record_subprocess(...) — for the current `ollama run ...` path
      timer.record_http(...)       — for the HTTP /api/generate path

    Both produce uniform rows, with HTTP rows carrying richer fields.

    Records are buffered in memory and flushed in groups of 50 or on close().
    """

    FLUSH_THRESHOLD = 50

    def __init__(
        self,
        out_dir: Path,
        run_id: Optional[str] = None,
        workflow_name: str = "image_classification",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.out_dir / "inference_timings.parquet"
        self.run_id = run_id
        self.workflow_name = workflow_name
        self._buffer: List[Dict[str, Any]] = []
        # Optional callable that returns the current stage name. Set by
        # PerfMonitor so individual record_* calls don't have to pass stage.
        self._default_stage_provider = None

    def _resolve_stage(self, stage: Optional[str]) -> Optional[str]:
        if stage is not None:
            return stage
        if self._default_stage_provider is not None:
            try:
                return self._default_stage_provider()
            except Exception:
                return None
        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def record_subprocess(
        self,
        image_path: Optional[str],
        prompt_name: str,
        model: str,
        wall_time_sec: float,
        success: bool,
        error: Optional[str] = None,
        stage: Optional[str] = None,
        call_start_ts: Optional[datetime] = None,
        call_end_ts: Optional[datetime] = None,
        is_warmup: bool = False,
    ) -> None:
        rec = InferenceRecord(
            ts=datetime.now(timezone.utc),
            workflow_name=self.workflow_name,
            run_id=self.run_id,
            stage=self._resolve_stage(stage),
            image_path=image_path,
            prompt_name=prompt_name,
            inference_mode="subprocess",
            model=model,
            wall_time_sec=wall_time_sec,
            success=success,
            error=error,
            call_start_ts=call_start_ts,
            call_end_ts=call_end_ts,
            is_warmup=is_warmup,
        )
        self._log_record(rec)

    def record_http(
        self,
        image_path: Optional[str],
        prompt_name: str,
        model: str,
        wall_time_sec: float,
        success: bool,
        ollama_response: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        stage: Optional[str] = None,
        call_start_ts: Optional[datetime] = None,
        call_end_ts: Optional[datetime] = None,
        is_warmup: bool = False,
    ) -> None:
        """
        Records a call to Ollama's HTTP /api/generate endpoint.
        Extracts the timing nanos fields if present.
        """
        rec = InferenceRecord(
            ts=datetime.now(timezone.utc),
            workflow_name=self.workflow_name,
            run_id=self.run_id,
            stage=self._resolve_stage(stage),
            image_path=image_path,
            prompt_name=prompt_name,
            inference_mode="http",
            model=model,
            wall_time_sec=wall_time_sec,
            success=success,
            error=error,
            call_start_ts=call_start_ts,
            call_end_ts=call_end_ts,
            is_warmup=is_warmup,
        )
        if ollama_response:
            # Ollama returns durations in nanoseconds
            for src, dst in [
                ("load_duration", "load_sec"),
                ("prompt_eval_duration", "prompt_eval_sec"),
                ("eval_duration", "eval_sec"),
            ]:
                ns = ollama_response.get(src)
                if isinstance(ns, (int, float)):
                    setattr(rec, dst, ns / 1e9)
            rec.prompt_tokens = ollama_response.get("prompt_eval_count")
            rec.output_tokens = ollama_response.get("eval_count")
        self._log_record(rec)

    def close(self) -> None:
        self._flush()

    # ── Internals ────────────────────────────────────────────────────────────

    def _log_record(self, rec: InferenceRecord) -> None:
        # Derived fields
        if rec.output_tokens and rec.eval_sec and rec.output_tokens > 0:
            rec.tpot_ms = (rec.eval_sec * 1000) / rec.output_tokens
            rec.decode_throughput_tok_s = rec.output_tokens / rec.eval_sec
        if rec.prompt_tokens and rec.prompt_eval_sec and rec.prompt_eval_sec > 0:
            rec.prefill_throughput_tok_s = rec.prompt_tokens / rec.prompt_eval_sec
        if rec.load_sec is not None and rec.wall_time_sec > 0:
            rec.load_overhead_pct = (rec.load_sec / rec.wall_time_sec) * 100

        self._buffer.append(asdict(rec))
        if len(self._buffer) >= self.FLUSH_THRESHOLD:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            import pandas as pd
            df = pd.DataFrame(self._buffer)
            self._buffer = []
            if self.parquet_path.exists():
                try:
                    existing = pd.read_parquet(self.parquet_path)
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception:
                    pass
            df.to_parquet(self.parquet_path, index=False)
        except Exception as e:
            print(f"[perf_monitor] InferenceTimer flush failed: {e}",
                  file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# StageTimer — one row per pipeline stage execution
# ──────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """
    Writes stage_timings.parquet — one row per (run_id, stage) pair.

    Usage:
        st = StageTimer(out_dir=log_dir, run_id="abc-123")
        with st.stage("descriptions") as stage_ctx:
            do_work()
            stage_ctx.set_metadata(images_processed=100)
    """

    def __init__(
        self,
        out_dir: Path,
        run_id: Optional[str] = None,
        workflow_name: str = "image_classification",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.out_dir / "stage_timings.parquet"
        self.run_id = run_id
        self.workflow_name = workflow_name

    @contextmanager
    def stage(self, name: str):
        ctx = _StageContext(name=name)
        start_wall = time.perf_counter()
        start_ts = datetime.now(timezone.utc)
        error: Optional[str] = None
        success = True
        try:
            yield ctx
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            end_ts = datetime.now(timezone.utc)
            elapsed = time.perf_counter() - start_wall
            self._write_row({
                "ts": start_ts,
                "end_ts": end_ts,
                "workflow_name": self.workflow_name,
                "run_id": self.run_id,
                "stage": name,
                "elapsed_sec": elapsed,
                "success": success,
                "error": error,
                **ctx.metadata,
            })

    def _write_row(self, row: Dict[str, Any]) -> None:
        try:
            import pandas as pd
            df = pd.DataFrame([row])
            if self.parquet_path.exists():
                try:
                    existing = pd.read_parquet(self.parquet_path)
                    # Back-compat: add any new columns
                    for col in df.columns:
                        if col not in existing.columns:
                            existing[col] = None
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception:
                    pass
            df.to_parquet(self.parquet_path, index=False)
        except Exception as e:
            print(f"[perf_monitor] StageTimer write failed: {e}",
                  file=sys.stderr)


@dataclass
class _StageContext:
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_metadata(self, **kwargs) -> None:
        self.metadata.update(kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: bundle the three monitors together for one-line instrumentation
# ──────────────────────────────────────────────────────────────────────────────

class PerfMonitor:
    """
    Bundles ResourceSampler + InferenceTimer + StageTimer with a shared run_id.

    Usage in the pipeline:

        from utils.perf_monitor import PerfMonitor

        perf = PerfMonitor(out_dir=LOG_DIR, run_id=pipeline_logger.run_id)

        with perf.stage("descriptions") as stage_ctx:
            for img in images:
                t0 = time.perf_counter()
                response = call_ollama_http(img, prompt)
                wall = time.perf_counter() - t0
                perf.inference.record_http(
                    image_path=img, prompt_name="natural",
                    model="llava-llama3", wall_time_sec=wall,
                    success=True, ollama_response=response,
                )
            stage_ctx.set_metadata(images_processed=len(images))

        perf.close()
    """

    def __init__(
        self,
        out_dir: Path,
        run_id: Optional[str] = None,
        workflow_name: str = "image_classification",
        sample_interval_sec: float = 2.0,
    ):
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self.workflow_name = workflow_name
        self.resource = ResourceSampler(
            out_dir=out_dir, interval_sec=sample_interval_sec,
            run_id=run_id, workflow_name=workflow_name,
        )
        self.inference = InferenceTimer(
            out_dir=out_dir, run_id=run_id, workflow_name=workflow_name,
        )
        self.stages = StageTimer(
            out_dir=out_dir, run_id=run_id, workflow_name=workflow_name,
        )
        self._sampling_active = False
        self._current_stage: Optional[str] = None

    @property
    def current_stage(self) -> Optional[str]:
        """The currently-active stage name, or None if outside a stage block."""
        return self._current_stage

    @contextmanager
    def stage(self, name: str):
        """
        Wraps a pipeline stage. Starts resource sampling, labels it with the
        stage name, runs the stage, stops sampling, writes the stage row.

        Inside the block, inference records are auto-stamped with `stage=name`
        when callers omit the stage argument — see record_subprocess / record_http.
        """
        # Resource sampler: if not already running for this PerfMonitor,
        # start it now and label samples with this stage
        if not self._sampling_active:
            self.resource.start()
            self._sampling_active = True
        previous_stage = self.resource._current_stage
        previous_perf_stage = self._current_stage
        self.resource.set_stage(name)
        self._current_stage = name
        # Let InferenceTimer auto-stamp records with the current stage
        self.inference._default_stage_provider = lambda: self._current_stage

        try:
            with self.stages.stage(name) as stage_ctx:
                yield stage_ctx
        finally:
            # Flush samples for this stage before moving to the next
            self.resource.flush()
            self.resource.set_stage(previous_stage)
            self._current_stage = previous_perf_stage

    def close(self) -> None:
        """Stop sampling and flush all buffers. Call at end of pipeline run."""
        if self._sampling_active:
            self.resource.stop()
            self._sampling_active = False
        self.resource.flush()
        self.inference.close()


# ──────────────────────────────────────────────────────────────────────────────
# Self-test  (run with: python -m utils.perf_monitor   or   python perf_monitor.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import uuid

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"Self-test output dir: {tmp_path}")

        perf = PerfMonitor(
            out_dir=tmp_path,
            run_id=str(uuid.uuid4()),
            sample_interval_sec=0.5,
        )

        # Simulate the "descriptions" stage with two inference calls
        with perf.stage("descriptions") as ctx:
            time.sleep(1.5)  # let the sampler grab a few rows
            # Simulate one subprocess-mode call
            perf.inference.record_subprocess(
                image_path="/fake/img_001.tif",
                prompt_name="natural",
                model="llava-llama3",
                wall_time_sec=34.78,
                success=True,
            )
            # Simulate one HTTP-mode call with Ollama's typical response shape
            perf.inference.record_http(
                image_path="/fake/img_002.tif",
                prompt_name="natural",
                model="llava-llama3",
                wall_time_sec=2.4,
                success=True,
                ollama_response={
                    "load_duration": 50_000_000,           # 0.05 sec
                    "prompt_eval_duration": 400_000_000,   # 0.4 sec
                    "eval_duration": 1_900_000_000,        # 1.9 sec
                    "prompt_eval_count": 120,
                    "eval_count": 75,
                },
            )
            ctx.set_metadata(images_processed=2)

        with perf.stage("method2_llava") as ctx:
            time.sleep(1.0)
            ctx.set_metadata(images_processed=2)

        perf.close()

        # Verify all three parquets were written
        import pandas as pd
        for name in ["resource_samples.parquet", "inference_timings.parquet", "stage_timings.parquet"]:
            path = tmp_path / name
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n=== {name} ({len(df)} rows) ===")
                # Print compact preview
                cols = [c for c in df.columns if c not in ("ts", "end_ts")]
                print(df[cols].to_string(max_rows=10, max_colwidth=20))
            else:
                print(f"\n!!! {name} not written")

        print(f"\nGPU backend used: {perf.resource._gpu.backend}")
        print("\nSelf-test complete.")
