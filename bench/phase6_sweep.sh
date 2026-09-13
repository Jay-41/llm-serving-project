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
while [ $# -gt 0 ]; do
  case "$1" in
    --queue-depth) QUEUE_DEPTH="$2"; shift 2 ;;
    --tokens)      TOKENS="$2"; shift 2 ;;
    --requests)    REQUESTS="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

OUT=bench/results
mkdir -p "$OUT" logs
export BACKEND=qwen MODEL_DEVICE=cuda MAX_ALLOWED_TOKENS=512
PID=""

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
  echo "--- server up ($*) ---"
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
echo "=== done. results in $OUT and logs/. bundle with: ==="
echo "  tar czf /tmp/phase6.tgz $OUT/gpu_* logs/phase6_* && runpodctl send /tmp/phase6.tgz"
