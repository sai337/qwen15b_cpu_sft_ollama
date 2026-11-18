#!/usr/bin/env bash
set -euo pipefail

# One command: train -> merge -> GGUF -> quantize -> Ollama model -> smoke test.
#
#   ./scripts/run_pipeline.sh configs/sft_lora_local_jsonl.yaml qwen15b-local
#
# Env knobs:
#   SKIP_TRAIN=1        reuse outputs/<run>/latest
#   QTYPE=Q4_K_M        GGUF quantization (Q4_K_M | Q5_K_M | Q8_0)
#   LLAMA_CPP_DIR=...   llama.cpp checkout (default ./llama.cpp, cloned if missing)
#   OLLAMA=docker|native|none   how to register the model (default: auto-detect)
#   MAX_STEPS=N         override train.max_steps (quick runs)

CFG=${1:?"usage: run_pipeline.sh <config.yaml> [ollama_model_name]"}
MODEL_NAME=${2:-qwen15b-ft}
QTYPE=${QTYPE:-Q4_K_M}
PYTHON=${PYTHON:-python}
export LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-./llama.cpp}

cfg_get() { "${PYTHON}" -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c$1)" "${CFG}"; }
RUN=$(cfg_get "['run']['run_name']")
OUT_DIR=$(cfg_get "['train']['output_dir']")
ADAPTER="${OUT_DIR}/${RUN}/latest"
MERGED="merged/${RUN}_merged"
F16="gguf/${RUN}.f16.gguf"
QGGUF="gguf/${RUN}.$(echo "${QTYPE}" | tr 'A-Z' 'a-z').gguf"
step() { echo; echo "=== $* ==="; }

mkdir -p "${OUT_DIR}"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  step "1/5 train (${CFG})"
  "${PYTHON}" -u -m qwen15b_cpu_sft.train --config "${CFG}" ${MAX_STEPS:+--max-steps "${MAX_STEPS}"} 2>&1 | tee "${OUT_DIR}/${RUN}.train.log"
fi
[[ -f "${ADAPTER}/adapter_model.safetensors" ]] || { echo "ERROR: no adapter at ${ADAPTER}"; exit 1; }

step "2/5 merge LoRA -> ${MERGED}"
"${PYTHON}" scripts/merge_adapter.py --adapter "${ADAPTER}" --out "${MERGED}"

if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
  step "cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
fi

step "3/5 convert -> ${F16}"
PYTHON="${PYTHON}" ./scripts/export_gguf.sh "${MERGED}" "${F16}" f16

step "4/5 quantize -> ${QGGUF}"
./scripts/quantize_gguf.sh "${F16}" "${QGGUF}" "${QTYPE}"

step "5/5 register with Ollama as ${MODEL_NAME}"
mkdir -p ollama/models
cp "${QGGUF}" ollama/models/
SYS=$(cfg_get ".get('data',{}).get('system_prompt','You are a helpful assistant.')")
"${PYTHON}" - "${QGGUF}" "${SYS}" <<'PY'
import re, sys, pathlib
gguf, system = pathlib.Path(sys.argv[1]).name, sys.argv[2].strip()
t = pathlib.Path("ollama/Modelfile.template").read_text()
t = re.sub(r"^FROM .*$", f"FROM ./models/{gguf}", t, flags=re.M)
t = re.sub(r'SYSTEM """.*?"""', lambda _: f'SYSTEM """{system}"""', t, flags=re.S)
pathlib.Path("ollama/Modelfile").write_text(t)
print("Wrote ollama/Modelfile ->", gguf)
PY

MODE=${OLLAMA:-auto}
if [[ "${MODE}" == "auto" ]]; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^ollama$'; then MODE=docker
  elif command -v ollama >/dev/null 2>&1; then MODE=native
  else MODE=none; fi
fi
case "${MODE}" in
  docker) docker exec ollama ollama create "${MODEL_NAME}" -f /models/Modelfile; HOST=http://localhost:11434 ;;
  native) (cd ollama && ollama create "${MODEL_NAME}" -f Modelfile); HOST=${OLLAMA_HOST:-http://localhost:11434} ;;
  *) echo "Ollama not running. Start it (docker compose up -d) then: docker exec ollama ollama create ${MODEL_NAME} -f /models/Modelfile"; exit 0 ;;
esac

step "smoke test"
curl -s "${HOST}/api/chat" -d "{\"model\":\"${MODEL_NAME}\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"What is a Kubernetes DaemonSet?\"}]}" \
  | "${PYTHON}" -c "import json,sys; print(json.load(sys.stdin)['message']['content'])"
echo
echo "Done. Model '${MODEL_NAME}' is live in Ollama. UI: http://localhost:8501  API: ${HOST}/v1/chat/completions"
