#!/bin/sh
# Container start command for the GPU benchmark pod.
#
# Runs the unattended sweep in the background and serves its log, an
# environment report, and the final results tarball over plain HTTP on 8001.
# The provider's port proxy makes that reachable from anywhere, so the whole
# session is observable with curl -- no SSH, no web terminal, and no dependence
# on the provider's log API (which returned 404 for an entire pod on the first
# attempt, leaving a crash loop invisible).
#
# The HTTP server is the foreground process, so the container stays up and
# readable even if the sweep dies in its first second.

mkdir -p /tmp/pod
{
  echo "=== pod_entry $(date -u +%FT%TZ) ==="
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || echo "nvidia-smi: unavailable"
  python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| cuda available', torch.cuda.is_available())" 2>&1
  echo "user: $(id)"
} > /tmp/pod/env.txt 2>&1

(
  sh bench/phase6_sweep.sh --auto
  echo "=== sweep exited with status $? at $(date -u +%FT%TZ) ==="
) > /tmp/pod/sweep.log 2>&1 &

cd /tmp/pod && exec python -m http.server 8001 --bind 0.0.0.0
