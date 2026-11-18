#!/usr/bin/env bash
set -euo pipefail

# Convert a merged Hugging Face model directory to GGUF using llama.cpp.
# Example:
#   ./scripts/export_gguf.sh merged/qwen15b_cloudqa_merged gguf/qwen15b_cloudqa.f16.gguf

MERGED_DIR=${1:?"merged model dir required"}
OUT_GGUF=${2:?"output gguf path required"}
OUTTYPE=${3:-f16}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"./llama.cpp"}
PYTHON=${PYTHON:-python3}

if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
  echo "ERROR: llama.cpp not found at ${LLAMA_CPP_DIR}"
  echo "Clone it: git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ${LLAMA_CPP_DIR}"
  exit 1
fi

CONVERTER="${GGUF_CONVERTER:-}"   # override only for tests (see tests/convert_tiny_gguf.py)
[[ -n "${CONVERTER}" ]] || for c in "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${LLAMA_CPP_DIR}/convert-hf-to-gguf.py"; do
  [[ -f "$c" ]] && { CONVERTER="$c"; break; }
done
[[ -n "${CONVERTER}" ]] || { echo "ERROR: convert_hf_to_gguf.py not found in ${LLAMA_CPP_DIR}"; exit 1; }

# The converter needs llama.cpp's own gguf-py plus sentencepiece/protobuf.
if ! "${PYTHON}" -c "import gguf, sentencepiece" 2>/dev/null; then
  echo "Installing converter requirements..."
  "${PYTHON}" -m pip install -q "${LLAMA_CPP_DIR}/gguf-py" sentencepiece protobuf
fi

mkdir -p "$(dirname "${OUT_GGUF}")"
echo "Using converter: ${CONVERTER}"
"${PYTHON}" "${CONVERTER}" "${MERGED_DIR}" --outtype "${OUTTYPE}" --outfile "${OUT_GGUF}"
ls -lh "${OUT_GGUF}"
