#!/usr/bin/env bash
set -euo pipefail

# Quantize GGUF for faster CPU inference.
# Example:
#   ./scripts/quantize_gguf.sh gguf/qwen15b_cloudqa.f16.gguf gguf/qwen15b_cloudqa.q4_k_m.gguf Q4_K_M

IN_GGUF=${1:?"input gguf required"}
OUT_GGUF=${2:?"output gguf required"}
QTYPE=${3:-Q4_K_M}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"./llama.cpp"}

find_quant_bin() {
  # Current llama.cpp names it llama-quantize; very old builds used quantize.
  for c in "${LLAMA_CPP_DIR}/build/bin/llama-quantize" "${LLAMA_CPP_DIR}/llama-quantize" \
           "${LLAMA_CPP_DIR}/build/bin/quantize" "${LLAMA_CPP_DIR}/quantize"; do
    [[ -x "$c" ]] && { echo "$c"; return 0; }
  done
  command -v llama-quantize 2>/dev/null && return 0
  return 1
}

QUANT_BIN=$(find_quant_bin || true)
if [[ -z "${QUANT_BIN}" ]]; then
  [[ -d "${LLAMA_CPP_DIR}" ]] || { echo "ERROR: llama.cpp not found at ${LLAMA_CPP_DIR}"; exit 1; }
  echo "Building llama.cpp (Release, CPU)..."
  cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "${LLAMA_CPP_DIR}/build" -j"$(nproc)" --target llama-quantize llama-cli llama-server
  QUANT_BIN=$(find_quant_bin)
fi

mkdir -p "$(dirname "${OUT_GGUF}")"
echo "Quantizing: ${IN_GGUF} -> ${OUT_GGUF} (${QTYPE}) with ${QUANT_BIN}"
"${QUANT_BIN}" "${IN_GGUF}" "${OUT_GGUF}" "${QTYPE}"
ls -lh "${OUT_GGUF}"
