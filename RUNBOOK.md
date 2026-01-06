# Runbook: Train Qwen2.5-1.5B-Instruct (CPU) with LoRA, Export to Ollama, Serve UI

This is written for a beginner team but assumes you can SSH/SSM into an Amazon Linux box.

## 0) Hard constraints (be honest about them)

- **CPU-only training on 1.5B params is slow.** You are not "training from scratch"; you are **fine-tuning** an existing foundation model with a small number of trainable parameters (LoRA).
- **Streaming datasets** do not provide a stable total size. The only reliable progress metric is your configured `max_steps`.
- **Ollama runs GGUF.** Qwen LoRA adapters (`adapter_model.safetensors`) are **not directly** loadable in Ollama. You must:
  - merge LoRA -> full HF model
  - convert HF -> GGUF (llama.cpp)
  - quantize GGUF

## 1) Instance sizing & expectations

For **m5.8xlarge** (32 vCPU / 128 GiB):

- Training wall time for a short demo SFT:
  - `max_steps: 500–1500`, `max_length: 256`, `grad_accum: 8`, `batch: 1`
  - Expect **~1–4 hours** depending on dataset, CPU efficiency, and disk I/O.
- CPU utilization will usually not hit 3200% (32 vCPU). PyTorch + MKL/OpenMP often tops out around physical cores.

## 2) OS prerequisites

```bash
sudo yum -y update
sudo yum -y install git gcc gcc-c++ make cmake pkgconfig python3.12 python3.12-pip
```

(Optional but helpful):

```bash
sudo yum -y install htop
```

## 3) Python environment

```bash
git clone <your repo location>
cd qwen15b_cpu_sft_ollama_demo
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU torch
pip install --index-url https://download.pytorch.org/whl/cpu torch

# Project deps
pip install -e .
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.get_num_threads());"
```

## 4) Hugging Face auth (avoid 401 errors)

If the model or dataset is gated/private, you must authenticate:

```bash
huggingface-cli login
```

Or (CI-style):

```bash
export HF_TOKEN=...   # do NOT commit tokens into configs
```

## 5) Choose your training config

### A) Instruction tuning (SFT) config (recommended for demo)

`configs/sft_lora_qwen15b_cpu_cloudqa.yaml`

Key knobs:

- `model.max_length`: sequence length. Smaller is faster on CPU.
- `train.max_steps`: your real progress bar.
- `train.grad_accum_steps`: increases effective batch size without RAM spikes.
- `lora.target_modules`: controls which layers get LoRA adapters.

### B) Domain adaptation (CPT) config (optional)

`configs/cpt_lora_qwen15b_cpu_cloudtext.yaml`

This is raw-text next-token training. It can help the model "sound" more AWS/devops-y before SFT.

## 6) Start training (SSM-safe)

### Option 1: Foreground (simple)

```bash
python -u -m qwen15b_cpu_sft.train \
  --config configs/sft_lora_qwen15b_cpu_cloudqa.yaml \
  | tee outputs/sft_lora_qwen15b_cpu_cloudqa/train.log
```

### Option 2: Background (SSM disconnect safe)

```bash
nohup python -u -m qwen15b_cpu_sft.train \
  --config configs/sft_lora_qwen15b_cpu_cloudqa.yaml \
  > outputs/sft_lora_qwen15b_cpu_cloudqa/train.log 2>&1 &

echo $! > outputs/sft_lora_qwen15b_cpu_cloudqa/train.pid
```

## 7) Monitor training progress

### 7.1 Log tail

```bash
tail -f outputs/sft_lora_qwen15b_cpu_cloudqa/train.log
```

### 7.2 Structured status (JSON)

```bash
cat outputs/sft_lora_qwen15b_cpu_cloudqa/status.json | python -m json.tool
```

This includes:
- step / max_steps
- avg loss
- ETA (based on average step time)
- estimated tokens processed

### 7.3 Find the PID

```bash
pgrep -af "qwen15b_cpu_sft.train"
PID=$(pgrep -f "qwen15b_cpu_sft.train" | head -n1)
ps -p $PID -o pid,etime,%cpu,%mem,rss,cmd
```

### 7.4 CPU/threads/core usage

```bash
lscpu | egrep -i 'model name|socket|core|thread'

# per-core view
mpstat -P ALL 1 5 || true

# per-thread view
top -H -p $PID

# quick summary
dstat -c --top-cpu --top-mem 1 10 || true
```

### 7.5 Why you see only ~15 cores used

- PyTorch math kernels (MKL/OpenMP) typically scale better to **physical cores**, not vCPUs.
- Some phases are memory bound or single-threaded (tokenization, dataset iterator, Python overhead).
- `torch.set_num_threads(16)` intentionally caps compute threads.

If you want to test higher:

```bash
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
```

Then set in config `compute.num_threads: 32`. Expect diminishing returns.

## 8) Output artifacts you should see

During training, the project writes:

```bash
outputs/<run_name>/status.json
outputs/<run_name>/latest/adapter_model.safetensors
outputs/<run_name>/latest/adapter_config.json
outputs/<run_name>/latest/tokenizer.json  (and related tokenizer files)
outputs/<run_name>/checkpoints/step_XXXXXX/... (every save interval)
```

If `latest/` is empty:
- your `save_every` may be larger than `max_steps`
- the process may not have reached a save step yet

## 9) Export for Ollama (GGUF)

### 9.1 Merge LoRA adapter into base model

```bash
python scripts/merge_adapter.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/sft_lora_qwen15b_cpu_cloudqa/latest \
  --out merged/qwen15b_cloudqa_merged
```

### 9.2 Convert merged model -> GGUF

```bash
git clone https://github.com/ggerganov/llama.cpp.git llama.cpp

./scripts/export_gguf.sh merged/qwen15b_cloudqa_merged gguf/qwen15b_cloudqa.f16.gguf
```

### 9.3 Quantize GGUF

```bash
./scripts/quantize_gguf.sh gguf/qwen15b_cloudqa.f16.gguf gguf/qwen15b_cloudqa.q4_k_m.gguf Q4_K_M
ls -lh gguf/
```

## 10) Serve with Ollama + UI

### 10.1 Docker compose

```bash
mkdir -p ollama/models
cp gguf/qwen15b_cloudqa.q4_k_m.gguf ollama/models/
cp ollama/Modelfile.template ollama/Modelfile

docker compose up -d

docker exec -it ollama ollama create qwen15b-cloudqa -f /models/Modelfile
```

Test:

```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen15b-cloudqa","prompt":"Explain what a Kubernetes DaemonSet is."}'
```

UI:
- `http://<server-ip>:8501`

## 11) Troubleshooting

### 401 Unauthorized when downloading models/tokenizers/datasets

- You are missing Hugging Face auth.
- Run `huggingface-cli login` or export `HF_TOKEN`.

### Training appears "stuck" after model download

Typical causes:
- First batch tokenization can take a while (Python + tokenizer)
- Dataset iterator is blocked (network, HF throttling)

What to do:

```bash
PID=$(pgrep -f "qwen15b_cpu_sft.train" | head -n1)
ps -p $PID -o pid,etime,pcpu,time,rss,cmd

# If CPU time is increasing, it's working.
# CPU TIME (process time) growing faster than wall time means multi-threaded compute is happening.
```

### No GPU

This project is CPU-only. If your model tries to pull GPU quantization, force dtype float32 in config.

## 12) Clean rollback

- Stop training: `kill $PID`
- Remove outputs: `rm -rf outputs/<run_name>`
- Remove docker stack: `docker compose down -v`
