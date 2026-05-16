# vLLM Concurrency Experiment — 24e069df-a485-420c-9a33-9d2f074cc864

- Date: 2026-05-15T20:20:03.216562+00:00
- Host: analytics-server.example.local
- Backend: vLLM 0.6.6, OpenAI-compatible /v1/chat/completions
- Model: ./models/llava-1.5-7b-hf/
- Images per level: 30  (same set per level)
- Prompt: locked C-v4 combined prompt
- Levels tested: [1, 2, 4, 8]

## Throughput scaling

| Concurrency | Batch wall (s) | Images/min | Median req (s) | P95 req (s) | Max req (s) | Failed |
|---|---|---|---|---|---|---|
| 1 | 138.6 | 13.0 | 4.42 | 5.39 | 5.55 | 0 |
| 2 | 80.3 | 22.4 | 5.08 | 6.39 | 6.48 | 0 |
| 4 | 48.2 | 37.4 | 6.23 | 7.46 | 7.77 | 0 |
| 8 | 32.5 | 55.4 | 8.20 | 11.00 | 11.67 | 0 |

## Scaling vs C=1 baseline

| Concurrency | Throughput speedup | Latency inflation | Parallel efficiency |
|---|---|---|---|
| 1 | 1.00x | 1.00x | 100% |
| 2 | 1.73x | 1.15x | 86% |
| 4 | 2.88x | 1.41x | 72% |
| 8 | 4.26x | 1.86x | 53% |

## vLLM vs Ollama — head to head

| Concurrency | Ollama (img/min) | vLLM (img/min) | vLLM speedup over Ollama |
|---|---|---|---|
| 1 | 9.1 | 13.0 | 1.43x |
| 2 | 8.7 | 22.4 | 2.58x |
| 4 | 8.5 | 37.4 | 4.40x |
| 8 | 8.9 | 55.4 | 6.22x |

## GPU resource usage per concurrency level

| Concurrency | Med VRAM (MB) | Max VRAM (MB) | Med GPU util (%) | Max GPU util (%) | Med power (W) |
|---|---|---|---|---|---|
| 1 | 19059 | 19059 | 96 | 100 | 71.5 |
| 2 | 19203 | 19203 | 96 | 100 | 71.8 |
| 4 | 19343 | 19343 | 97 | 98 | 72.3 |
| 8 | 19713 | 19713 | 97 | 100 | 71.8 |

## Reading this

- **If throughput scales with concurrency** and Ollama stayed flat: vLLM's continuous batching is doing real work. The L4 has the headroom; Ollama just couldn't access it.
- **If latency holds roughly constant** as concurrency rises: vLLM is processing requests truly in parallel within a single forward pass. Per-request work doesn't grow much with batch size at this scale.
- **If throughput plateaus at some C**: that's where the GPU or KV cache budget saturates. Above that, more concurrency means queueing, not parallelism. Stop scaling there.

## Files

- `inference_timings.parquet` — one row per inference call
- `resource_samples.parquet` — GPU/CPU/memory at 2 Hz
- `stage_timings.parquet` — one row per concurrency level
- `outputs.jsonl` — every raw model output