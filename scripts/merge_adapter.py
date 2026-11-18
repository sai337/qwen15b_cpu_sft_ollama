#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save as a standalone HF model directory.

Why: Ollama does NOT support loading Qwen from safetensors + safetensor LoRA adapters. For Qwen you
must export to GGUF (llama.cpp) which requires a full model (adapter merged).

Usage:
  python scripts/merge_adapter.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/sft_lora_qwen15b_cpu_cloudqa/latest \
    --out merged/qwen15b_cloudqa_merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from qwen15b_cpu_sft.modeling import dtype_kwarg
except ImportError:  # running without `pip install -e .`
    def dtype_kwarg() -> str:
        return "torch_dtype"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="Base HF model id or dir (default: read from adapter_config.json)")
    ap.add_argument("--adapter", required=True, help="Path to adapter directory (contains adapter_model.safetensors)")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"], help="Load dtype")
    args = ap.parse_args()

    if not args.base:
        args.base = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
        print(f"Base model (from adapter_config.json): {args.base}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, low_cpu_mem_usage=True, **{dtype_kwarg(): dtype_map[args.dtype]}
    )

    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged = peft_model.merge_and_unload()  # returns a plain HF model

    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"Saved merged model to: {out}")


if __name__ == "__main__":
    main()
