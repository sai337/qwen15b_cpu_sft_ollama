#!/usr/bin/env bash
set -euo pipefail

# Monitor training process + system resources.
# Usage:
#   ./scripts/monitor_train.sh

PID=$(pgrep -f "qwen15b_cpu_sft.train" | head -n1 || true)
if [[ -z "${PID}" ]]; then
  echo "Training process not found (pattern: qwen15b_cpu_sft.train)."
  exit 1
fi

echo "PID: ${PID}" 

echo "\n=== process ==="
ps -p "${PID}" -o pid,etime,%cpu,%mem,rss,cmd

if command -v ps -v >/dev/null 2>&1; then :; fi

echo "\n=== per-thread CPU (top -H) ==="
echo "(Press q to exit)"
top -H -p "${PID}"
