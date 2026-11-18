#!/usr/bin/env bash
set -euo pipefail

# Offline end-to-end smoke test of the whole pipeline with a TINY random Qwen2 model:
#   build tiny model -> LoRA SFT (memorise one fact) -> merge -> GGUF f16 -> Q8_0
#   -> llama-server -> ask the question over the OpenAI API -> assert the learned answer comes back
# Proves the LoRA update survives every conversion step, in ~1 minute, with no Hugging Face access.
# The real run uses Qwen/Qwen2.5-1.5B-Instruct via run_pipeline.sh.

PYTHON=${PYTHON:-python}
export LLAMA_CPP_DIR=${LLAMA_CPP_DIR:-./llama.cpp}
W=${WORK:-.smoke}
rm -rf "${W}" && mkdir -p "${W}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "== tiny model"
"${PYTHON}" -m qwen15b_cpu_sft.testing "${W}/tiny_base"

FACT_Q="What is a DaemonSet?"
FACT_A="A DaemonSet runs one pod on every node."
echo "{\"question\": \"${FACT_Q}\", \"answer\": \"${FACT_A}\"}" > "${W}/fact.jsonl"

echo "== config"
"${PYTHON}" - "${W}" <<'PY'
import sys, yaml
w = sys.argv[1]
c = yaml.safe_load(open("configs/sft_lora_local_jsonl.yaml"))
c["run"]["run_name"] = "smoke"
c["compute"]["num_threads"] = 2
c["model"]["base_model"] = f"{w}/tiny_base"
c["model"]["max_length"] = 128
c["data"]["sft_sources"][0]["data_files"] = f"{w}/fact.jsonl"
c["data"]["shuffle_buffer"] = 0
c["lora"].update(r=32, alpha=64, dropout=0.0)
c["train"].update(output_dir=f"{w}/outputs", max_steps=150, grad_accum_steps=1, save_every=75,
                  log_every=25, learning_rate=1e-2, warmup_steps=5)
yaml.safe_dump(c, open(f"{w}/smoke.yaml", "w"), sort_keys=False)
PY

[[ -f ollama/Modelfile ]] && cp ollama/Modelfile "${W}/Modelfile.user.bak"   # restored on exit
echo "== run_pipeline.sh (train -> merge -> GGUF -> quantize -> Modelfile)"
# Tiny tokenizer needs the test converter shim; tiny dims are too small for K-quants, so Q8_0.
GGUF_CONVERTER=tests/convert_tiny_gguf.py QTYPE=Q8_0 OLLAMA=none PYTHON="${PYTHON}" \
  ./scripts/run_pipeline.sh "${W}/smoke.yaml" smoke-tiny
GGUF="gguf/smoke.q8_0.gguf"
[[ -f "${GGUF}" ]] && grep -q "FROM ./models/smoke.q8_0.gguf" ollama/Modelfile
grep -q 'stop "<|im_end|>"' ollama/Modelfile

echo "== serve with llama-server + query via OpenAI API"
SERVER=$(ls "${LLAMA_CPP_DIR}"/build/bin/llama-server 2>/dev/null || command -v llama-server)
"${SERVER}" -m "${GGUF}" --port 18080 --jinja -c 512 > "${W}/server.log" 2>&1 &
SPID=$!
trap 'kill ${SPID} 2>/dev/null || true; rm -rf merged/smoke_merged gguf/smoke.* ollama/models/smoke.q8_0.gguf; if [[ -f "${W}/Modelfile.user.bak" ]]; then mv "${W}/Modelfile.user.bak" ollama/Modelfile; else rm -f ollama/Modelfile; fi' EXIT
for i in $(seq 1 60); do curl -sf http://127.0.0.1:18080/health >/dev/null && break; sleep 1; done
SYS=$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${W}/smoke.yaml'))['data']['system_prompt'].strip())")
echo "{\"prompt\": \"${FACT_Q}\", \"expect_keywords\": [\"runs one pod on every node\"]}" > "${W}/fact_eval.jsonl"
"${PYTHON}" scripts/eval_compare.py --base-url http://127.0.0.1:18080/v1 --models smoke \
  --prompts "${W}/fact_eval.jsonl" --system "${SYS}" --max-tokens 24 --out "${W}/eval"
grep -A3 '"smoke"' "${W}/eval/results.json" | grep -q '"keyword_score": 1.0' \
  && echo "SMOKE E2E: PASS (served GGUF returns the fine-tuned answer)" \
  || { echo "SMOKE E2E: FAIL"; cat "${W}/eval/report.md"; exit 1; }
