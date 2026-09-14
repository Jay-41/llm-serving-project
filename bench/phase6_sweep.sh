#!/bin/sh
# Phase 6 benchmark session. Runs INSIDE the GPU pod against localhost.
#
# Every experiment from Phases 2-5, re-run on the real model, each with its
# control condition. Manages the server itself so each experiment gets the
# settings it needs -- which is why the pod's container command should be
# `sleep infinity`, not uvicorn.
#
#   sh bench/phase6_sweep.sh --queue-depth 12
#
# Run bench/model_probe.py FIRST and derive --queue-depth from its output:
#   depth = (target_p99_s - batch8_total_s) * (8 / batch8_total_s)
# i.e. how many requests of backlog fit in the latency budget once a request's
# own pass is subtracted. See ops/RUNPOD.md.
#
# Rates for the overload tests are derived from the measured batching sweep so
# they are ~2x capacity on whatever GPU this is, rather than hardcoded for the
# mock's 9.4 rps.

set -eu

QUEUE_DEPTH=16
TOKENS=64
REQUESTS=40
AUTO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --queue-depth) QUEUE_DEPTH="$2"; shift 2 ;;
    --tokens)      TOKENS="$2"; shift 2 ;;
    --requests)    REQUESTS="$2"; shift 2 ;;
    --auto)        AUTO=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

OUT=bench/results
mkdir -p "$OUT" logs
# Overridable so the script's control flow can be dry-run on a laptop
# (BACKEND=qwen MODEL_DEVICE=mps) before it runs on a metered GPU.
export BACKEND="${BACKEND:-qwen}" MODEL_DEVICE="${MODEL_DEVICE:-cuda}" MAX_ALLOWED_TOKENS=512
PID=""

if [ "$AUTO" = 1 ]; then
  # Unattended mode, for use as the pod's container start command: calibrate,
  # derive the admission threshold, run everything, ship results, then stay
  # alive so logs remain readable and the meter is stopped deliberately.
  echo "=== [0/5] calibration ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
  # No `| tee`: in POSIX sh a pipeline's status is tee's, not the probe's, and
  # set -e would sail past a dead probe into load tests against a broken
  # server. That is exactly what attempt 2 did.
  if ! python -m bench.model_probe --sizes 1,2,4,8,16 --tokens $TOKENS --repeats 3 > $OUT/gpu_probe.txt 2>&1; then
    echo "!!! calibration probe FAILED:"; tail -30 $OUT/gpu_probe.txt; exit 1
  fi
  cat $OUT/gpu_probe.txt
  T8=$(grep '^PROBE ' $OUT/gpu_probe.txt | sed 's/.*batch8_total_ms=\([0-9]*\).*/\1/')
  if [ -z "$T8" ]; then echo "!!! no PROBE line in probe output; refusing to guess a queue depth"; exit 1; fi
  # depth = (target_p99 - own pass) * (batch / pass): backlog that fits in the
  # latency budget once the request's own pass is subtracted. Same derivation
  # as Phase 4, with this hardware's batch-8 time instead of the mock's 840ms.
  QUEUE_DEPTH=$(awk -v t="$T8" 'BEGIN{ d=int((2500 - t) * 8 / t); if (d < 4) d = 4; if (d > 64) d = 64; print d }')
  echo "batch-8 pass = ${T8}ms -> MAX_QUEUE_DEPTH derived as $QUEUE_DEPTH for a 2.5s p99 target"
fi

start() {
  # $@ are KEY=VALUE overrides for this experiment
  env "$@" METRICS_PATH="logs/phase6_${LABEL}.jsonl" \
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level warning &
  PID=$!
  i=0
  until curl -sf -m 2 localhost:8000/healthz >/dev/null 2>&1; do
    i=$((i+1)); [ $i -gt 240 ] && { echo "server failed to start"; exit 1; }
    sleep 1
  done
  # "Up" is not "working": attempt 2 had a healthy /healthz over a model
  # whose every forward pass threw. One real request, or abort.
  code=$(curl -s -o /tmp/smoke.json -w '%{http_code}' -m 120 -X POST localhost:8000/generate \
           -H 'content-type: application/json' -d '{"prompt":"ok","max_tokens":4}')
  if [ "$code" != "200" ]; then
    echo "!!! server up but a real request returned HTTP $code:"; cat /tmp/smoke.json; echo; exit 1
  fi
  echo "--- server up and generating ($*) ---"
}
stop() {
  [ -n "$PID" ] && { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; PID=""; }
  sleep 1
}
trap stop EXIT

echo "=== [1/5] batching sweep: MAX_BATCH_SIZE=8 vs 1 (Phase 2 on real hardware) ==="
LABEL=batching;  start MAX_BATCH_SIZE=8 MAX_QUEUE_DEPTH=0
python -m bench.loadtest --concurrency 1,2,4,8,16 --requests $REQUESTS --max-tokens $TOKENS \
  --label "GPU batching" --out $OUT/gpu_batching.csv
stop
LABEL=nobatch;   start MAX_BATCH_SIZE=1 MAX_QUEUE_DEPTH=0
python -m bench.loadtest --concurrency 1,2,4,8,16 --requests $REQUESTS --max-tokens $TOKENS \
  --label "GPU no batching (control)" --out $OUT/gpu_nobatch.csv
stop
python -m bench.compare --before $OUT/gpu_nobatch_summary.csv --after $OUT/gpu_batching_summary.csv \
  --before-label "no batching" --after-label "batching" | tee $OUT/gpu_batching_compare.txt

# Capacity = measured throughput at concurrency 16 with batching. Overload
# tests offer ~2x that.
CAP=$(awk -F, 'NR>1 && $1==16 {print $6}' $OUT/gpu_batching_summary.csv)
RATE=$(awk -v c="$CAP" 'BEGIN{printf "%.0f", c*2}')
echo "measured capacity ${CAP} rps -> overload tests offer ${RATE} rps"

echo "=== [2/5] streaming TTFT (Phase 3) ==="
LABEL=stream;    start MAX_QUEUE_DEPTH=0
python -m bench.loadtest --stream --concurrency 1,4,8,16 --requests $REQUESTS --max-tokens $TOKENS \
  --label "GPU streaming" --out $OUT/gpu_stream.csv
stop

echo "=== [3/5] backpressure: admission OFF vs ON at depth $QUEUE_DEPTH (Phase 4) ==="
LABEL=noadmit;   start MAX_QUEUE_DEPTH=0
python -m bench.burst --rate $RATE --duration 20 --bucket 4 --max-tokens $TOKENS \
  --label "GPU overload, admission OFF" --out $OUT/gpu_no_admission.csv
stop
LABEL=admit;     start MAX_QUEUE_DEPTH=$QUEUE_DEPTH
python -m bench.burst --rate $RATE --duration 20 --bucket 4 --max-tokens $TOKENS \
  --label "GPU overload, admission ON ($QUEUE_DEPTH)" --out $OUT/gpu_admission.csv
stop

echo "=== [4/5] tiered admission (Phase 5) ==="
PAID=$(awk -v c="$CAP" 'BEGIN{printf "%.0f", c*1.5}'); FREE=$(awk -v c="$CAP" 'BEGIN{printf "%.0f", c*0.4+1}')
FREE_DEPTH=$((QUEUE_DEPTH / 2))
LABEL=tiers_flat;   start MAX_QUEUE_DEPTH=$QUEUE_DEPTH MAX_QUEUE_DEPTH_FREE=$QUEUE_DEPTH
python -m bench.tiers --paid $PAID --free $FREE --duration 20 --max-tokens $TOKENS \
  --label "GPU tiers, flat admission" --out $OUT/gpu_tiers_flat.csv
stop
LABEL=tiers_tiered; start MAX_QUEUE_DEPTH=$QUEUE_DEPTH MAX_QUEUE_DEPTH_FREE=$FREE_DEPTH
python -m bench.tiers --paid $PAID --free $FREE --duration 20 --max-tokens $TOKENS \
  --label "GPU tiers, tiered admission ($FREE_DEPTH/$QUEUE_DEPTH)" --out $OUT/gpu_tiers_tiered.csv
stop

echo "=== [5/5] starvation: aging OFF vs ON (Phase 5) ==="
PAIDS=$(awk -v c="$CAP" 'BEGIN{printf "%.0f", c*1.05+0.5}')
LABEL=aging_off; start MAX_QUEUE_DEPTH=$QUEUE_DEPTH MAX_QUEUE_DEPTH_FREE=$QUEUE_DEPTH AGING_MS=0
python -m bench.tiers --paid $PAIDS --free 1 --duration 30 --max-tokens $TOKENS \
  --label "GPU starvation, aging OFF" --out $OUT/gpu_aging_off.csv
stop
LABEL=aging_on;  start MAX_QUEUE_DEPTH=$QUEUE_DEPTH MAX_QUEUE_DEPTH_FREE=$QUEUE_DEPTH AGING_MS=2000
python -m bench.tiers --paid $PAIDS --free 1 --duration 30 --max-tokens $TOKENS \
  --label "GPU starvation, aging ON" --out $OUT/gpu_aging_on.csv
stop

echo
echo "=== done. results in $OUT and logs/ ==="
cp $OUT/gpu_probe.txt $OUT/gpu_batching_compare.txt logs/ 2>/dev/null || true

if [ "$AUTO" = 1 ]; then
  # Ship everything back. `runpodctl send` prints a one-time code on its first
  # line and blocks until a receiver connects, so the code has to be visible in
  # the pod logs -- print it loudly, then wait. Retry if nobody collected it.
  mkdir -p /tmp/phase6 /tmp/pod && cp $OUT/gpu_* logs/phase6_* /tmp/phase6/ 2>/dev/null || true
  # Primary retrieval: one tarball on the pod's HTTP port (bench/pod_entry.sh
  # serves /tmp/pod). runpodctl send below is the fallback.
  tar czf /tmp/pod/phase6.tgz -C /tmp phase6 && echo "=== RESULTS TARBALL: http://<pod>:8001/phase6.tgz ==="
  while true; do
    echo "=== RESULTS READY: on your machine run  runpodctl receive <code>  with the code below ==="
    runpodctl send /tmp/phase6 2>&1 | tee /tmp/send.log &
    SEND=$!
    sleep 3
    echo "=== SEND CODE: $(head -1 /tmp/send.log | grep -oE '[0-9]+-[a-z-]+' || head -1 /tmp/send.log) ==="
    wait $SEND && { echo "=== transfer complete; stop the pod ==="; break; }
    echo "transfer did not complete; re-sending in 30s"; sleep 30
  done
  sleep infinity
fi
