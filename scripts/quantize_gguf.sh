#!/usr/bin/env bash
set -euo pipefail

# Quantize GGUF for faster CPU inference.
# Example:
#   ./scripts/quantize_gguf.sh gguf/qwen15b_cloudqa.f16.gguf gguf/qwen15b_cloudqa.q4_k_m.gguf Q4_K_M

IN_GGUF=${1:?"input gguf required"}
OUT_GGUF=${2:?"output gguf required"}
QTYPE=${3:-Q4_K_M}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-"./llama.cpp"}

# Build llama.cpp if needed
if [[ ! -x "${LLAMA_CPP_DIR}/quantize" && ! -x "${LLAMA_CPP_DIR}/build/bin/quantize" ]]; then
  echo "Building llama.cpp (Release)..."
  cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${LLAMA_CPP_DIR}/build" -j
fi

QUANT_BIN="${LLAMA_CPP_DIR}/quantize"
if [[ ! -x "${QUANT_BIN}" ]]; then
  QUANT_BIN="${LLAMA_CPP_DIR}/build/bin/quantize"
fi

if [[ ! -x "${QUANT_BIN}" ]]; then
  echo "ERROR: quantize binary not found. Build llama.cpp first."
  exit 1
fi

mkdir -p "$(dirname "${OUT_GGUF}")"

echo "Quantizing: ${IN_GGUF} -> ${OUT_GGUF} (${QTYPE})"
"${QUANT_BIN}" "${IN_GGUF}" "${OUT_GGUF}" "${QTYPE}"

echo "Wrote: ${OUT_GGUF}"
