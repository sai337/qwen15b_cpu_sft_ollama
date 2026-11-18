#!/usr/bin/env bash
set -euo pipefail

# Monitor training process + status.json.
# Usage: ./scripts/monitor_train.sh [outputs/<run_name>]

RUN_DIR=${1:-}
PID=$(pgrep -f "qwen15b_cpu_sft.train" | head -n1 || true)
if [[ -z "${PID}" ]]; then
  echo "Training process not found (pattern: qwen15b_cpu_sft.train)."
else
  echo "PID: ${PID}"
  ps -p "${PID}" -o pid,etime,%cpu,%mem,rss,cmd
fi

if [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/status.json" ]]; then
  echo
  echo "=== ${RUN_DIR}/status.json ==="
  python3 -m json.tool "${RUN_DIR}/status.json"
fi

if [[ -n "${PID}" && -t 1 ]]; then
  echo
  echo "=== per-thread CPU (top -H, press q to exit) ==="
  top -H -p "${PID}"
fi
