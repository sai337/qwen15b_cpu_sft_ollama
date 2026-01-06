from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: any


def set_cpu_threads(num_threads: int, interop_threads: int = 2) -> None:
    # Keep this before any significant torch work.
    num_threads = int(num_threads) if num_threads else 0
    interop_threads = int(interop_threads) if interop_threads else 0
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if interop_threads > 0:
        torch.set_num_interop_threads(interop_threads)


def load_bundle(model_id: str, dtype: str = "float32") -> ModelBundle:
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "auto": None,
    }.get(dtype, torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.train()
    return ModelBundle(model=model, tokenizer=tokenizer)


def attach_lora(model: torch.nn.Module, r: int, alpha: int, dropout: float, target_modules: List[str]) -> torch.nn.Module:
    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    lora_model = get_peft_model(model, lora_cfg)
    lora_model.print_trainable_parameters()
    return lora_model
