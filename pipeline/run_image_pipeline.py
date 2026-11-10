#!/usr/bin/env python3
"""
run_image_pipeline.py
=====================
Scheduler / runner for image_classification_pipeline.py.

Responsibilities
----------------
  - Enforces the 7:00 PM – 5:00 AM allowed run window
  - Hard-kills the pipeline if it has been running for more than 10 hours
  - Sends a progress email at the start of each session showing how many
    images have been processed and how many remain across all 5 steps
  - Saves a progress state file so it always knows where things stand
  - Supports a --test-limit N flag that caps the run at N images for test runs

Usage
-----
  # Test run — 5 images, runs immediately regardless of time window
  python run_image_pipeline.py --test-limit 5

  # Normal nightly run (call this from cron at 7 PM)
  python run_image_pipeline.py

  # Normal nightly run — explicit paths
  python run_image_pipeline.py \
    --image-dir /path/to/images \
    --output-dir /path/to/output \
    --pipeline-script /path/to/image_classification_pipeline.py

Cron example (starts every night at 7 PM):
  0 19 * * * /path/to/python /path/to/run_image_pipeline.py >> /path/to/logs/runner.log 2>&1
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these defaults to match your server paths
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE_SCRIPT = Path(os.environ.get(
    'PIPELINE_SCRIPT',
    'image_classification_pipeline.py'
))
IMAGE_DIR = Path(os.environ.get(
    'IMAGE_DIR',
    './images/'
))
OUTPUT_DIR = Path(os.environ.get(
    'OUTPUT_DIR',
    'output'
))
LOG_DIR = Path(os.environ.get(
    'LOG_DIR',
    'logs'
))

# Run window: 7 PM to 5 AM
WINDOW_START_HOUR = 19   # 7:00 PM
WINDOW_END_HOUR   = 5    # 5:00 AM

# Hard kill after this many hours regardless of window
MAX_RUN_HOURS = 10
MAX_RUN_SECONDS = MAX_RUN_HOURS * 3600

# Email settings — matches job_execution_monitor pattern
FROM_EMAIL = "job_execution_monitoring_app@example.local"
TO_EMAIL   = os.environ.get("PIPELINE_NOTIFY_EMAIL", "you@example.com")

# State file tracks progress across nightly sessions
PROGRESS_FILE = LOG_DIR / "runner_progress.json"

# CSV output files produced by the pipeline (used for progress counting)
DESCRIPTIONS_CSV = OUTPUT_DIR / 'image_descriptions.csv'
SIMILARITY_CSV   = OUTPUT_DIR / 'method1_semantic_results.csv'
LLAVA_CLASS_CSV  = OUTPUT_DIR / 'method2_llava_results.csv'
METADATA_CSV     = OUTPUT_DIR / 'image_metadata.csv'


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def now_local() -> datetime:
    return datetime.now()


def in_run_window() -> bool:
    """Return True if the current time is within the 7 PM – 5 AM window."""
    h = now_local().hour
    # Window crosses midnight: valid hours are 19, 20, 21, 22, 23, 0, 1, 2, 3, 4
    return h >= WINDOW_START_HOUR or h < WINDOW_END_HOUR


def seconds_until_window() -> int:
    """Return how many seconds until the next 7 PM window start."""
    now = now_local()
    target = now.replace(hour=WINDOW_START_HOUR, minute=0, second=0, microsecond=0)
    if now.hour >= WINDOW_START_HOUR:
        # Already past 7 PM today — next window is tomorrow at 7 PM
        from datetime import timedelta
        target += timedelta(days=1)
    return int((target - now).total_seconds())


def count_csv_rows(csv_path: Path) -> int:
    """Count processed rows in a CSV output file. Returns 0 if missing."""
    if not csv_path.exists():
        return 0
    try:
        import pandas as pd
        return len(pd.read_csv(csv_path))
    except Exception:
        return 0


def count_total_images() -> int:
    """Count all images in IMAGE_DIR."""
    if not IMAGE_DIR.exists():
        return 0
    total = 0
    for ext in ['*.tif', '*.tiff', '*.jpg', '*.jpeg']:
        total += len(list(IMAGE_DIR.glob(ext)))
    return total


def get_progress() -> dict:
    """
    Build a progress snapshot across all 5 pipeline steps.
    Returns a dict with counts for each step and overall totals.
    """
    total = count_total_images()
    desc_done  = count_csv_rows(DESCRIPTIONS_CSV)
    m1_done    = count_csv_rows(SIMILARITY_CSV)
    m2_done    = count_csv_rows(LLAVA_CLASS_CSV)
    meta_done  = count_csv_rows(METADATA_CSV)

    # Report step (report is generated from existing CSVs — not row-counted)
    report_csv = OUTPUT_DIR / 'combined_results.csv'
    report_done = count_csv_rows(report_csv)

    return {
        "total_images":       total,
        "descriptions_done":  desc_done,
        "method1_done":       m1_done,
        "method2_done":       m2_done,
        "metadata_done":      meta_done,
        "report_rows":        report_done,
        "snapshot_time":      now_local().isoformat(),
    }


def load_state() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(state, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# EMAIL
# ──────────────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str):
    """Send via local sendmail — same pattern as job_execution_monitor emailer."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL
    msg.set_content(body)
    try:
        proc = subprocess.Popen(
            ["/usr/sbin/sendmail", "-t", "-oi"],
            stdin=subprocess.PIPE,
            text=True,
        )
        proc.communicate(msg.as_string())
        print(f"[runner] Progress email sent to {TO_EMAIL}")
    except Exception as e:
        print(f"[runner] WARNING: Could not send email — {e}")


def build_progress_email(progress: dict, session_start: datetime,
                         test_mode: bool, test_limit: int) -> tuple[str, str]:
    """Build subject + body for the nightly progress email."""
    total = progress["total_images"]
    desc  = progress["descriptions_done"]
    m1    = progress["method1_done"]
    m2    = progress["method2_done"]
    meta  = progress["metadata_done"]

    def pct(done):
        return f"{(done / total * 100):.1f}%" if total > 0 else "N/A"

    def remaining(done):
        return max(0, total - done)

    now_str   = now_local().strftime("%Y-%m-%d %H:%M")
    start_str = session_start.strftime("%Y-%m-%d %H:%M")
    elapsed   = (now_local() - session_start).total_seconds()
    elapsed_h = int(elapsed // 3600)
    elapsed_m = int((elapsed % 3600) // 60)

    mode_tag = f"[TEST RUN — {test_limit} images]" if test_mode else "[NIGHTLY RUN]"

    subject = (
        f"🖼️ Image Pipeline Progress {mode_tag} — "
        f"{desc}/{total} descriptions ({pct(desc)}) — {now_str}"
    )

    body = f"""Image Classification Pipeline — Progress Report
{mode_tag}
{'=' * 60}
Report time  : {now_str}
Session start: {start_str}
Elapsed      : {elapsed_h}h {elapsed_m}m
Hard kill at : {MAX_RUN_HOURS}h from session start
Run window   : {WINDOW_START_HOUR:02d}:00 – {WINDOW_END_HOUR:02d}:00
{'=' * 60}

STEP PROGRESS (all steps run independently against same image set)

  Step 1 — Descriptions (LLaVA natural + strict)
    Done      : {desc:,} / {total:,}  ({pct(desc)})
    Remaining : {remaining(desc):,}

  Step 2 — Method 1 Semantic Similarity
    Done      : {m1:,} / {total:,}  ({pct(m1)})
    Remaining : {remaining(m1):,}

  Step 3 — Method 2 LLaVA Classification
    Done      : {m2:,} / {total:,}  ({pct(m2)})
    Remaining : {remaining(m2):,}

  Step 4 — Metadata Extraction
    Done      : {meta:,} / {total:,}  ({pct(meta)})
    Remaining : {remaining(meta):,}

  Step 5 — Combined Report
    Rows in combined_results.csv : {progress["report_rows"]:,}

{'=' * 60}
OVERALL COMPLETION (based on descriptions step)
  {desc:,} of {total:,} images fully described  ({pct(desc)})

Output files:
  {DESCRIPTIONS_CSV}
  {SIMILARITY_CSV}
  {LLAVA_CLASS_CSV}
  {METADATA_CSV}
  {OUTPUT_DIR / 'combined_results.csv'}

Log dir: {LOG_DIR}
{'=' * 60}
"""
    return subject, body


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(test_mode: bool, test_limit: int, extra_args: list) -> int:
    """
    Launch image_classification_pipeline.py as a subprocess.
    Returns the exit code.

    In test mode: passes --limit to cap image count, ignores time window,
    runs all steps to completion.
    """
    cmd = [sys.executable, str(PIPELINE_SCRIPT)]

    # Always pass explicit paths so there's no ambiguity
    cmd += [
        '--image-dir',  str(IMAGE_DIR),
        '--output-dir', str(OUTPUT_DIR),
        '--log-dir',    str(LOG_DIR),
    ]

    if test_mode:
        cmd += ['--limit', str(test_limit)]

    cmd += extra_args

    print(f"[runner] Launching: {' '.join(str(c) for c in cmd)}")
    try:
        proc = subprocess.Popen(cmd)
        return proc.wait()
    except KeyboardInterrupt:
        print("[runner] KeyboardInterrupt — terminating pipeline process")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 1


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Image pipeline scheduler / runner")
    p.add_argument('--test-limit', type=int, default=0,
                   help='Test mode: process only N images and exit (ignores time window)')
    p.add_argument('--image-dir',  type=Path, default=IMAGE_DIR)
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    p.add_argument('--log-dir',    type=Path, default=LOG_DIR)
    p.add_argument('--pipeline-script', type=Path, default=PIPELINE_SCRIPT)
    p.add_argument('--no-email',   action='store_true',
                   help='Skip progress email (useful during testing)')
    p.add_argument('--steps', type=str, default=None,
                   help='Comma-separated steps to run, e.g. "descriptions,method2,report,metadata"')
    # Any remaining args are forwarded to the pipeline script
    p.add_argument('pipeline_args', nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    global IMAGE_DIR, OUTPUT_DIR, LOG_DIR, PIPELINE_SCRIPT
    global DESCRIPTIONS_CSV, SIMILARITY_CSV, LLAVA_CLASS_CSV, METADATA_CSV

    args = parse_args()

    IMAGE_DIR       = args.image_dir
    OUTPUT_DIR      = args.output_dir
    LOG_DIR         = args.log_dir
    PIPELINE_SCRIPT = args.pipeline_script

    # Rebuild derived paths after any arg overrides
    DESCRIPTIONS_CSV = OUTPUT_DIR / 'image_descriptions.csv'
    SIMILARITY_CSV   = OUTPUT_DIR / 'method1_semantic_results.csv'
    LLAVA_CLASS_CSV  = OUTPUT_DIR / 'method2_llava_results.csv'
    METADATA_CSV     = OUTPUT_DIR / 'image_metadata.csv'

    test_mode  = args.test_limit > 0
    test_limit = args.test_limit

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── TEST MODE ────────────────────────────────────────────────────────────
    if test_mode:
        print(f"[runner] *** TEST MODE — processing {test_limit} images ***")
        print(f"[runner] Skipping time window check. Running to full completion.")

        progress_before = get_progress()
        print(f"[runner] Images found in dir: {progress_before['total_images']}")

        session_start = now_local()
        extra = list(args.pipeline_args)
        if args.steps:
            extra += ['--steps', args.steps]
        rc = run_pipeline(test_mode=True, test_limit=test_limit,
                          extra_args=extra)

        progress_after = get_progress()
        print(f"\n[runner] Test run complete (exit code {rc})")
        print(f"[runner] Descriptions before: {progress_before['descriptions_done']}  "
              f"after: {progress_after['descriptions_done']}")

        if not args.no_email:
            subj, body = build_progress_email(
                progress_after, session_start, test_mode=True, test_limit=test_limit
            )
            body = f"TEST RUN COMPLETE (exit code: {rc})\n\n" + body
            send_email(subj, body)

        save_state({"last_test_run": now_local().isoformat(),
                    "last_test_progress": progress_after})
        sys.exit(rc)

    # ── NORMAL NIGHTLY RUN ───────────────────────────────────────────────────
    print(f"[runner] {now_local().strftime('%Y-%m-%d %H:%M')} — checking run window")

    if not in_run_window():
        wait_sec = seconds_until_window()
        wait_h   = wait_sec // 3600
        wait_m   = (wait_sec % 3600) // 60
        print(f"[runner] Outside run window (window: {WINDOW_START_HOUR:02d}:00–"
              f"{WINDOW_END_HOUR:02d}:00). "
              f"Next window in {wait_h}h {wait_m}m. Exiting.")
        sys.exit(0)

    print(f"[runner] Inside run window — starting pipeline session")
    session_start = now_local()
    progress_before = get_progress()

    # Send progress email at the start of each session
    if not args.no_email:
        subj, body = build_progress_email(
            progress_before, session_start, test_mode=False, test_limit=0
        )
        body = "SESSION STARTING — progress at start of tonight's run:\n\n" + body
        send_email(subj, body)

    # Launch pipeline as a subprocess so we can enforce the hard kill
    extra = list(args.pipeline_args)
    if args.steps:
        extra += ['--steps', args.steps]
    cmd = [sys.executable, str(PIPELINE_SCRIPT),
           '--image-dir',  str(IMAGE_DIR),
           '--output-dir', str(OUTPUT_DIR),
           '--log-dir',    str(LOG_DIR)] + extra

    print(f"[runner] Launching: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(cmd)

    rc = None
    killed_reason = None

    try:
        while proc.poll() is None:
            time.sleep(30)  # Check every 30 seconds

            elapsed = (now_local() - session_start).total_seconds()

            # ── Hard kill: 10 hours ──────────────────────────────────────────
            if elapsed >= MAX_RUN_SECONDS:
                print(f"[runner] Hard kill — pipeline has run for {MAX_RUN_HOURS}h. Terminating.")
                killed_reason = f"exceeded {MAX_RUN_HOURS}-hour hard limit"
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

            # ── Window end: 5 AM ────────────────────────────────────────────
            if not in_run_window():
                print(f"[runner] Run window ended (5:00 AM). Stopping pipeline gracefully.")
                killed_reason = "run window ended at 5:00 AM"
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        rc = proc.returncode if proc.returncode is not None else 1

    except KeyboardInterrupt:
        print("[runner] KeyboardInterrupt — stopping pipeline")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        rc = 1
        killed_reason = "manual interrupt"

    # ── Session end email ────────────────────────────────────────────────────
    progress_after = get_progress()
    elapsed_total  = (now_local() - session_start).total_seconds()
    elapsed_h      = int(elapsed_total // 3600)
    elapsed_m      = int((elapsed_total % 3600) // 60)

    desc_delta = progress_after["descriptions_done"] - progress_before["descriptions_done"]

    if not args.no_email:
        subj, body = build_progress_email(
            progress_after, session_start, test_mode=False, test_limit=0
        )
        stop_reason = (
            f"Stopped: {killed_reason}" if killed_reason
            else f"Pipeline exited normally (code {rc})"
        )
        summary = (
            f"SESSION ENDED — {stop_reason}\n"
            f"Session duration: {elapsed_h}h {elapsed_m}m\n"
            f"Images processed this session: {desc_delta:+,} descriptions\n\n"
        )
        send_email(subj, summary + body)

    state = load_state()
    state["last_run"]      = now_local().isoformat()
    state["last_progress"] = progress_after
    state["last_rc"]       = rc
    state["last_stop_reason"] = killed_reason or "completed"
    save_state(state)

    print(f"[runner] Session complete — exit code {rc}, "
          f"duration {elapsed_h}h {elapsed_m}m, "
          f"{desc_delta:+} descriptions this session")
    sys.exit(rc if rc is not None else 0)


if __name__ == '__main__':
    main()
