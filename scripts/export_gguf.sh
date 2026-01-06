#!/usr/bin/env bash
set -euo pipefail

# Convert a merged Hugging Face model directory to GGUF using llama.cpp.
#
# Example:
#   ./scripts/export_gguf.sh merged/qwen15b_cloudqa_merged gguf/qwen15b_cloudqa.f16.gguf

MERGED_DIR=${1:?"merged model dir required"}
OUT_GGUF=${2:?"output gguf path required"}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"./llama.cpp"}

if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
  echo "ERROR: llama.cpp not found at ${LLAMA_CPP_DIR}"
  echo "Clone it: git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_CPP_DIR}"
  exit 1
fi

python_cmd=python3

conv1="${LLAMA_CPP_DIR}/convert-hf-to-gguf.py"
conv2="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"

if [[ -f "${conv1}" ]]; then
  CONVERTER="${conv1}"
elif [[ -f "${conv2}" ]]; then
  CONVERTER="${conv2}"
else
  echo "ERROR: Could not find convert script in llama.cpp (expected convert-hf-to-gguf.py or convert_hf_to_gguf.py)"
  exit 1
fi

mkdir -p "$(dirname "${OUT_GGUF}")"

echo "Using converter: ${CONVERTER}"
${python_cmd} "${CONVERTER}" "${MERGED_DIR}" --outtype f16 --outfile "${OUT_GGUF}"

echo "Wrote: ${OUT_GGUF}"
