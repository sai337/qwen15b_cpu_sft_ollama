#!/usr/bin/env bash
set -euo pipefail

# Runs CPT then SFT sequentially.
# - CPT output becomes the new base for SFT by pointing SFT base_model to the CPT latest merged adapter.
# For simplicity and reliability, we keep two separate runs:
#   1) CPT produces adapters
#   2) Merge CPT adapters -> HF model dir
#   3) Run SFT using merged CPT model as base

CPT_CFG=${1:-configs/cpt_lora_qwen15b_cpu_cloudtext.yaml}
SFT_CFG=${2:-configs/sft_lora_qwen15b_cpu_cloudqa.yaml}

python -m qwen15b_cpu_sft.train --config "${CPT_CFG}"

CPT_RUN_NAME=$(python -c "import yaml; print(yaml.safe_load(open('${CPT_CFG}'))['run']['run_name'])")
CPT_ADAPTER_DIR="outputs/${CPT_RUN_NAME}/latest"
MERGED_CPT_DIR="merged/${CPT_RUN_NAME}_merged"

python scripts/merge_adapter.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter "${CPT_ADAPTER_DIR}" --out "${MERGED_CPT_DIR}"

echo "\nNow run SFT using the merged CPT model as base (optional)"
echo "Edit ${SFT_CFG} and set: model.base_model: ${MERGED_CPT_DIR}"

python -m qwen15b_cpu_sft.train --config "${SFT_CFG}"
