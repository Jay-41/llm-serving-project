# Phase 6 runbook — GPU benchmark session on RunPod

The meter runs from pod start to pod stop. Everything below is a checklist so
nothing is improvised while paying. Budget: ~1 hour of pod time, under $1.

## Before starting the pod (free)

- [ ] GHCR image is public: github.com/Jay-41?tab=packages → `llm-serving-gpu`
      → Package settings → Change visibility → Public. RunPod pulls it without
      credentials. The image contains only code; the model downloads at runtime.
- [ ] `brew install runpod/runpodctl/runpodctl` on the Mac, for pulling results back.
- [ ] Local compose stack is up: `docker compose up -d` (Prometheus + Grafana
      will be pointed at the pod once it has a URL).

## Start the pod

RunPod → Pods → **Deploy** →

| Setting | Value |
| --- | --- |
| GPU | **L4** (24GB) or A10G — datacenter inference class |
| Container image | `ghcr.io/jay-41/llm-serving-gpu:latest` |
| Container start command | `sleep infinity` |
| Expose HTTP ports | `8000` |
| Container disk | 20 GB (image ~7GB + model ~3GB) |
| Volume | none needed for a single session |

**`sleep infinity` matters.** The sweep script starts and stops the server
itself with different settings per experiment. If uvicorn is the container
command, killing it kills the pod.

Wait for status **Running**, then open **Connect → Web Terminal** (or SSH).

## In the pod terminal

```sh
cd /app
nvidia-smi --query-gpu=name,memory.total --format=csv    # confirm the GPU

# 1. Calibrate. Downloads the model on first call (~1 min on datacenter bandwidth).
python -m bench.model_probe --sizes 1,2,4,8,16 --tokens 64 --repeats 3 | tee bench/results/gpu_probe.txt
```

From the probe output, **derive the admission threshold** for this hardware.
Take the batch-8 total time `T8` (seconds) and a 2.5s p99 target, same as
Phase 4:

```
depth = (2.5 - T8) * (8 / T8)
```

Example: if T8 = 0.9s → (2.5 − 0.9) × 8.9 ≈ **14**. Round down. If T8 > 2.5s
the target is unreachable at 64 tokens; use 8 and note it.

```sh
# 2. Run everything. ~15-20 minutes. Uses the depth you just derived.
sh bench/phase6_sweep.sh --queue-depth 14

# 3. Bundle results.
tar czf /tmp/phase6.tgz bench/results/gpu_* logs/phase6_*
runpodctl send /tmp/phase6.tgz         # prints a one-time code
```

## On the Mac, while the sweep runs

Point Prometheus at the pod so Grafana records the whole session. The pod's
public host is on its Connect page, of the form `<pod-id>-8000.proxy.runpod.net`:

```sh
ops/scrape_remote.sh <pod-id>-8000.proxy.runpod.net
open http://localhost:3000
```

Set the dashboard time range to cover the session. **Take screenshots at the
end** — they are Phase 7 deliverables. Panels to capture: throughput served vs
rejected, queue depth vs limit, latency percentiles, where the time goes, time
to first token, batch size vs cap, queue wait by tier.

## After the sweep

```sh
runpodctl receive <code>               # on the Mac; lands phase6.tgz here
tar xzf phase6.tgz                     # into bench/results/ and logs/
ops/scrape_remote.sh --local           # Prometheus back to the local app
```

**Stop the pod.** Then terminate it — a stopped pod still bills for disk.

## What to do with the numbers

- `bench/results/gpu_probe.txt` → measured `MOCK_BASE_MS`, `MOCK_PER_TOKEN_MS`,
  `MOCK_BATCH_ALPHA` for CUDA. Update the defaults in `app/config.py` so the
  mock predicts this hardware.
- `gpu_batching_compare.txt` → the headline throughput multiple for the résumé.
- `gpu_admission*.csv` → confirm the derived depth held p99 near 2.5s.
- Everything else → Phase 7 tables, alongside the mock predictions they replace.
