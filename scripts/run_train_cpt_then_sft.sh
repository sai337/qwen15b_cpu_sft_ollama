#!/usr/bin/env bash
set -euo pipefail

# Two-stage training: CPT (domain text) -> merge -> SFT on top of the merged CPT model.
# Usage: ./scripts/run_train_cpt_then_sft.sh [cpt_cfg] [sft_cfg]

CPT_CFG=${1:-configs/cpt_lora_qwen15b_cpu_cloudtext.yaml}
SFT_CFG=${2:-configs/sft_lora_qwen15b_cpu_cloudqa.yaml}
PYTHON=${PYTHON:-python}

cfg_get() { "${PYTHON}" -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print(c$2)" "$1"; }

echo "== Stage 1: CPT"
"${PYTHON}" -u -m qwen15b_cpu_sft.train --config "${CPT_CFG}"

CPT_RUN=$(cfg_get "${CPT_CFG}" "['run']['run_name']")
CPT_BASE=$(cfg_get "${CPT_CFG}" "['model']['base_model']")
CPT_OUT=$(cfg_get "${CPT_CFG}" "['train']['output_dir']")
MERGED_CPT_DIR="merged/${CPT_RUN}_merged"

echo "== Merge CPT adapter -> ${MERGED_CPT_DIR}"
"${PYTHON}" scripts/merge_adapter.py --base "${CPT_BASE}" --adapter "${CPT_OUT}/${CPT_RUN}/latest" --out "${MERGED_CPT_DIR}"

echo "== Stage 2: SFT on top of ${MERGED_CPT_DIR}"
"${PYTHON}" -u -m qwen15b_cpu_sft.train --config "${SFT_CFG}" --base-model "${MERGED_CPT_DIR}"

SFT_RUN=$(cfg_get "${SFT_CFG}" "['run']['run_name']")
echo
echo "Done. When merging the SFT adapter, use --base ${MERGED_CPT_DIR} (merge_adapter.py reads it from adapter_config.json if you omit --base)."
echo "SFT adapter: ${CPT_OUT}/${SFT_RUN}/latest"
