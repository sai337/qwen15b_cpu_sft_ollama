from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: Any


def set_cpu_threads(num_threads: int, interop_threads: int = 2) -> None:
    # Keep this before any significant torch work (interop threads can only be set once).
    num_threads = int(num_threads or 0)
    interop_threads = int(interop_threads or 0)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass  # already set in this process (e.g. tests calling train() twice)


def dtype_kwarg() -> str:
    """transformers>=4.56 renamed torch_dtype -> dtype (v5 warns on the old name)."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def resolve_device(device: str) -> torch.device:
    device = (device or "cpu").lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_bundle(model_id: str, dtype: str = "float32") -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"low_cpu_mem_usage": True}
    if dtype != "auto":
        kwargs[dtype_kwarg()] = DTYPES.get(dtype, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
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
