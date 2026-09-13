#!/bin/sh
# Point the local Prometheus at a remote app and restart it.
#
#   ops/scrape_remote.sh abc123xyz-8000.proxy.runpod.net   # scrape the GPU pod
#   ops/scrape_remote.sh --local                            # back to app:8000
#
# Generates ops/prometheus/prometheus.remote.yml (gitignored) from the
# template and swaps it in via the docker-compose.gpu.yml override, which
# also stops the local app container so the dashboard shows only the pod.
set -eu
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--local" ]; then
  docker compose up -d --force-recreate prometheus app
  echo "prometheus -> local app:8000"
  exit 0
fi
HOST="${1:?usage: ops/scrape_remote.sh <host[:port]> | --local}"
sed "s|__HOST__|$HOST|" ops/prometheus/prometheus.remote.template.yml > ops/prometheus/prometheus.remote.yml
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate prometheus
docker compose stop app >/dev/null 2>&1 || true
sleep 3
echo "prometheus -> https://$HOST/metrics"
curl -s localhost:9090/api/v1/targets | python3 -c "
import json,sys
for t in json.load(sys.stdin)['data']['activeTargets']:
    if t['labels']['job']=='llm-serving': print(f\"  target {t['health'].upper()}: {t['scrapeUrl']}\" + (f\"  ({t['lastError']})\" if t['health']!='up' else ''))"
