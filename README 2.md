# Qwen2.5-1.5B (CPU) LoRA Training + Ollama Serving (GGUF) + Simple UI

This repo is a **CPU-only**, beginner-friendly project to:

1) LoRA fine-tune **Qwen/Qwen2.5-1.5B-Instruct** using **streaming** Hugging Face datasets
2) Export the result to **GGUF** (llama.cpp) so it can run in **Ollama** on CPU
3) Serve an internal web UI (Streamlit) for a demo

## What you get

- **Two training modes** (choose one)
  - **CPT**: continued pretraining on raw text (domain adaptation)
  - **SFT**: instruction tuning on prompt/response pairs
- **Config-driven** YAML (no code edits for day-to-day changes)
- A **status.json** written during training (steps, loss, ETA, tokens/sec estimate)
- Export scripts to: **merge LoRA**, convert to **GGUF**, quantize, and load into **Ollama**
- A minimal **Streamlit UI** that talks to Ollama's HTTP API

> Important reality check:
> - **Instruction tuning is still next-token prediction**. The only difference is we **mask** the loss for the prompt tokens and only learn on the assistant tokens.
> - **Streaming datasets** do not expose a reliable "percent of dataset complete". You track progress by **steps** (and estimated tokens processed).

## Folder structure

- `configs/` training configs
- `src/qwen15b_cpu_sft/` training code
- `scripts/` helper scripts (merge adapter, GGUF conversion, monitoring)
- `ollama/` Modelfile template and model volume
- `ui/` Streamlit chat app
- `docker-compose.yml` starts Ollama + UI in containers

## Quickstart (Amazon Linux / Python 3.12)

See **RUNBOOK.md** for the full step-by-step including monitoring and troubleshooting.

### 1) Create venv + install

```bash
cd qwen15b_cpu_sft_ollama_demo
python3.12 -m venv .venv
source .venv/bin/activate

# CPU torch (recommended to avoid GPU wheels)
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install the project in editable mode
pip install -e .
```

### 2) Authenticate Hugging Face (required for gated/private repos)

```bash
huggingface-cli login
# or
export HF_TOKEN=...  # token must allow access to the model/datasets you picked
```

### 3) Run SFT (instruction tuning)

```bash
python -u -m qwen15b_cpu_sft.train \
  --config configs/sft_lora_qwen15b_cpu_cloudqa.yaml \
  | tee outputs/sft_lora_qwen15b_cpu_cloudqa/train.log
```

Watch progress:

```bash
tail -f outputs/sft_lora_qwen15b_cpu_cloudqa/train.log
cat outputs/sft_lora_qwen15b_cpu_cloudqa/status.json
```

### 4) Merge the LoRA adapter into a standalone HF model

```bash
python scripts/merge_adapter.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/sft_lora_qwen15b_cpu_cloudqa/latest \
  --out merged/qwen15b_cloudqa_merged
```

### 5) Convert merged model to GGUF + quantize (for Ollama)

```bash
git clone https://github.com/ggerganov/llama.cpp.git llama.cpp

# Convert to f16 GGUF
./scripts/export_gguf.sh merged/qwen15b_cloudqa_merged gguf/qwen15b_cloudqa.f16.gguf

# Quantize (recommended for CPU)
./scripts/quantize_gguf.sh gguf/qwen15b_cloudqa.f16.gguf gguf/qwen15b_cloudqa.q4_k_m.gguf Q4_K_M
```

### 6) Serve with Ollama + UI

#### Option A: Docker (recommended)

```bash
mkdir -p ollama/models
cp gguf/qwen15b_cloudqa.q4_k_m.gguf ollama/models/
cp ollama/Modelfile.template ollama/Modelfile

docker compose up -d

# Create the Ollama model inside the container
# (Modelfile uses a relative path under /models)
docker exec -it ollama ollama create qwen15b-cloudqa -f /models/Modelfile

# Now open:
# UI: http://<server-ip>:8501
# Ollama: http://<server-ip>:11434
```

#### Option B: Native Ollama + native Streamlit

```bash
# install ollama (per Ollama docs)
ollama serve

# in a second terminal
ollama create qwen15b-cloudqa -f ollama/Modelfile

export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen15b-cloudqa
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```
