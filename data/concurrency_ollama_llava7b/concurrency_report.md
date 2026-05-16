# Concurrency Experiment — 0adaac8b-bac6-4804-97f4-c6272af39ee4

- Date: 2026-05-16T20:54:06.516761+00:00
- Host: analytics-server.example.local
- Model: llava:7b
- Images per level: 30  (same set processed at each concurrency level)
- Prompt: locked C-v4 combined prompt
- Levels tested: [1, 2, 4, 8]

## Throughput scaling

| Concurrency | Batch wall (s) | Images/min | Median req (s) | P95 req (s) | Max req (s) | Failed |
|---|---|---|---|---|---|---|
| 1 | 120.4 | 14.9 | 4.11 | 4.61 | 4.75 | 0 |
| 2 | 118.1 | 15.2 | 7.80 | 8.81 | 8.93 | 0 |
| 4 | 117.1 | 15.4 | 15.46 | 17.04 | 17.36 | 0 |
| 8 | 117.6 | 15.3 | 31.24 | 32.11 | 32.15 | 0 |

## Scaling vs C=1 baseline

| Concurrency | Throughput speedup | Latency inflation | Parallel efficiency |
|---|---|---|---|
| 1 | 1.00x | 1.00x | 100% |
| 2 | 1.02x | 1.90x | 51% |
| 4 | 1.03x | 3.76x | 26% |
| 8 | 1.02x | 7.59x | 13% |

- **Throughput speedup**: aggregate throughput at this concurrency divided by C=1 baseline. Linear scaling = concurrency value (e.g. C=4 → 4.0x).
- **Latency inflation**: per-request median latency at this concurrency divided by C=1 baseline. 1.0x = no latency cost; higher = each request slowed by queuing.
- **Parallel efficiency**: speedup as % of perfect linear scaling. 100% = perfectly parallel; lower = diminishing returns.

## GPU resource usage per concurrency level

| Concurrency | Med VRAM (MB) | Max VRAM (MB) | Med GPU util (%) | Max GPU util (%) | Med power (W) |
|---|---|---|---|---|---|
| 1 | 5191 | 5191 | 54 | 100 | 58.6 |
| 2 | 5191 | 5191 | 54 | 100 | 60.9 |
| 4 | 5191 | 5191 | 54 | 100 | 60.7 |
| 8 | 5191 | 5191 | 54 | 100 | 62.1 |

## Reading this

- **If throughput scales linearly with concurrency** (e.g. C=4 → 4x): Ollama is parallelizing well; the GPU has headroom and concurrency is your speedup lever.
- **If throughput stays flat** as concurrency rises: Ollama is serializing internally. Likely the daemon was started with OLLAMA_NUM_PARALLEL=1 (the default in older Ollama). Set OLLAMA_NUM_PARALLEL=4 (or higher) on the *Ollama daemon* (`systemctl edit ollama`) and restart it.
- **If throughput scales but latency balloons**: the GPU is saturated. Throughput stops scaling once GPU util approaches 100%. Past that, you get queuing, not parallelism.
- **VRAM ceiling**: each concurrent request claims its own KV cache. If VRAM grows linearly with concurrency, you'll OOM at high concurrency. Plan accordingly.

## Files

- `inference_timings.parquet` — one row per inference call
- `resource_samples.parquet` — GPU/CPU/memory at 2 Hz
- `stage_timings.parquet` — one row per concurrency level
- `outputs.jsonl` — every raw model output