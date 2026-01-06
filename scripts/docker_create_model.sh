#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:-qwen15b-cloudqa}

if ! docker ps --format '{{.Names}}' | grep -q '^ollama$'; then
  echo "ERROR: ollama container not running. Start with: docker compose up -d"
  exit 1
fi

if [[ ! -f ollama/Modelfile ]]; then
  echo "ERROR: ollama/Modelfile not found. Copy from template and edit FROM path."
  exit 1
fi

docker exec -it ollama ollama create "${MODEL_NAME}" -f /models/Modelfile
