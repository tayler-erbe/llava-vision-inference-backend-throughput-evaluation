# Concurrency Experiment — 2ebdedb1-3e95-4dbc-a931-ef27892b3d82

- Date: 2026-05-15T17:14:05.248103+00:00
- Host: analytics-server.example.local
- Model: llava-llama3
- Images per level: 30  (same set processed at each concurrency level)
- Prompt: locked C-v4 combined prompt
- Levels tested: [1, 2, 4, 8]

## Throughput scaling

| Concurrency | Batch wall (s) | Images/min | Median req (s) | P95 req (s) | Max req (s) | Failed |
|---|---|---|---|---|---|---|
| 1 | 197.6 | 9.1 | 6.43 | 9.54 | 10.90 | 0 |
| 2 | 207.0 | 8.7 | 14.03 | 17.75 | 18.75 | 0 |
| 4 | 210.7 | 8.5 | 27.87 | 32.51 | 34.44 | 0 |
| 8 | 201.6 | 8.9 | 52.63 | 60.30 | 60.36 | 0 |

## Scaling vs C=1 baseline

| Concurrency | Throughput speedup | Latency inflation | Parallel efficiency |
|---|---|---|---|
| 1 | 1.00x | 1.00x | 100% |
| 2 | 0.95x | 2.18x | 48% |
| 4 | 0.94x | 4.33x | 23% |
| 8 | 0.98x | 8.18x | 12% |

- **Throughput speedup**: aggregate throughput at this concurrency divided by C=1 baseline. Linear scaling = concurrency value (e.g. C=4 → 4.0x).
- **Latency inflation**: per-request median latency at this concurrency divided by C=1 baseline. 1.0x = no latency cost; higher = each request slowed by queuing.
- **Parallel efficiency**: speedup as % of perfect linear scaling. 100% = perfectly parallel; lower = diminishing returns.

## GPU resource usage per concurrency level

| Concurrency | Med VRAM (MB) | Max VRAM (MB) | Med GPU util (%) | Max GPU util (%) | Med power (W) |
|---|---|---|---|---|---|
| 1 | 6139 | 6139 | 28 | 100 | 49.5 |
| 2 | 6139 | 6139 | 25 | 100 | 49.2 |
| 4 | 6139 | 6139 | 27 | 100 | 48.9 |
| 8 | 6139 | 6139 | 31 | 100 | 49.5 |

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