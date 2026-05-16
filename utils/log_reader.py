# log_reader.py — Job Execution Monitor
# Reads pipeline_log.parquet files from all monitored pipelines.
#
# Supports two discovery modes:
#   1. Standard: glob */logs/*.parquet under PIPELINES_DIR
#   2. Extra:    explicit list of paths (for pipelines in other locations,
#                e.g. datashare/pipeline-workspace/...)

from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


# ── Required schema columns (monitor depends on all of these) ─────────────────
REQUIRED_COLUMNS = [
    "workflow_name",
    "pipeline_name",
    "run_id",
    "script_name",
    "script_order",
    "status",
    "is_success",
    "event_time",
    "start_time",
    "end_time",
    "duration_seconds",
    "failure_stage",
    "error_type",
    "error_message",
    "traceback",
    "log_file",
]


def discover_log_files(pipelines_dir: Path) -> list[Path]:
    """
    Find all pipeline_log.parquet files under the standard pipelines directory.
    Expected structure:
      pipelines/
        pipeline_name/
          logs/
            pipeline_log.parquet
    """
    return list(pipelines_dir.glob("*/logs/*.parquet"))


def _load_single(log_file: Path) -> pd.DataFrame:
    """Load one parquet file; return synthetic failure row on read error."""
    pipeline_name = log_file.parent.parent.name
    try:
        df = pd.read_parquet(log_file).copy()
        df["pipeline_name"] = pipeline_name
        df["log_file"]      = str(log_file)
        # Backfill pipeline_name from workflow_name if missing
        if "workflow_name" not in df.columns:
            df["workflow_name"] = pipeline_name
        return df
    except Exception as e:
        return pd.DataFrame([{
            "workflow_name": pipeline_name,
            "pipeline_name": pipeline_name,
            "status":        "LOG_READ_FAILED",
            "is_success":    False,
            "error_type":    type(e).__name__,
            "error_message": str(e),
            "event_time":    datetime.now(timezone.utc),
            "log_file":      str(log_file),
        }])


def load_logs(
    pipelines_dir: Path,
    extra_log_files: list = None,
) -> pd.DataFrame:
    """
    Load and combine all pipeline log parquet files into a single DataFrame.

    Parameters
    ----------
    pipelines_dir : Path
        Root directory to glob for standard pipeline logs.
    extra_log_files : list of str or Path, optional
        Additional individual parquet paths outside pipelines_dir
        (e.g. pipeline-workspace logs).
    """
    log_files: list[Path] = discover_log_files(Path(pipelines_dir))

    # Add any extra log files (deduplicated)
    if extra_log_files:
        for f in extra_log_files:
            p = Path(f)
            if p.exists() and p not in log_files:
                log_files.append(p)
            # If the file doesn't exist yet (pipeline hasn't run), skip silently

    if not log_files:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    records = [_load_single(f) for f in log_files]
    combined = pd.concat(records, ignore_index=True)

    # Ensure all required columns exist (fill missing with None)
    for col in REQUIRED_COLUMNS:
        if col not in combined.columns:
            combined[col] = None

    # Normalise event_time to UTC-aware datetime
    combined["event_time"] = pd.to_datetime(combined["event_time"], utc=True, errors="coerce")

    return combined