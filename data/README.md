# Benchmark data

This directory contains the raw measurement artifacts from the three concurrency
experiments and the prompt-shape experiment. Each subdirectory is a complete,
reproducible record of one experimental run.

## Layout

| Directory | Backend | Model | Quantization | Calls measured |
|---|---|---|---|---|
| `prompt_experiment/` | Ollama | llava-llama3 | Q4_0 | 1,200 (50 imgs x 3 variants x 3 repeats + warmup) |
| `concurrency_ollama_llavallama3/` | Ollama | llava-llama3 | Q4_0 | 120 (30 imgs x 4 concurrency levels) |
| `concurrency_ollama_llava7b/` | Ollama | llava-1.5-7b | Q4_0 | 120 |
| `concurrency_vllm/` | vLLM 0.6.6 | llava-1.5-7b | FP16 | 120 |

## Files in each run directory

- `inference_timings.parquet` — one row per inference call (per-call wall time,
  prompt-eval time, decode time, token counts, timestamps).
- `stage_timings.parquet` — one row per experimental stage (concurrency level
  or prompt variant), with batch wall time, image count, and concurrency.
- `resource_samples.parquet` — GPU utilization, VRAM, power, CPU, and memory
  sampled at 2 Hz throughout each stage.
- `outputs.jsonl` — every raw model output, one JSON object per line. Useful
  for content-quality spot checks.
- `*_report.md` — human-readable summary report for the run.
- `field_parses.parquet` *(prompt experiment only)* — per-field schema-validity
  outcomes for every output.

## Regenerating the analysis

All four headline charts in `docs/charts/` can be regenerated from this data by
running `python3 benchmarks/build_report_charts.py` from the repo root. The
script reads the parquet files, derives concurrency assignments via positional
bucketing, and writes the four PNGs to `output/report_charts/`.
