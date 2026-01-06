from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import load_yaml
from .data import CPTIterableDataset, SFTIterableDataset, SFTSource, TextSource
from .logging_utils import atomic_write_json, format_seconds, now_ts
from .modeling import attach_lora, load_bundle, set_cpu_threads


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _as_text_sources(cfg: Dict[str, Any]) -> list[TextSource]:
    sources = []
    for s in cfg["data"]["text_sources"]:
        sources.append(
            TextSource(
                name=s["name"],
                dataset_name=s["dataset_name"],
                dataset_config=s.get("dataset_config"),
                split=s.get("split", "train"),
                text_field=s.get("text_field", "text"),
                weight=float(s.get("weight", 1.0)),
            )
        )
    return sources


def _as_sft_sources(cfg: Dict[str, Any]) -> list[SFTSource]:
    sources = []
    for s in cfg["data"]["sft_sources"]:
        sources.append(
            SFTSource(
                name=s["name"],
                dataset_name=s["dataset_name"],
                dataset_config=s.get("dataset_config"),
                split=s.get("split", "train"),
                prompt_field=s["prompt_field"],
                response_field=s["response_field"],
                weight=float(s.get("weight", 1.0)),
            )
        )
    return sources


@torch.no_grad()
def _maybe_write_sample(tokenizer, cfg: Dict[str, Any], out_dir: str) -> None:
    """Write an example prompt so beginners can see the exact text format."""
    try:
        sys_prompt = cfg.get("data", {}).get("system_prompt", "You are a helpful assistant.")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Explain what an EKS node group is."},
            {"role": "assistant", "content": ""},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = f"SYSTEM: {sys_prompt}\nUSER: Explain what an EKS node group is.\nASSISTANT:"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "sample_prompt.txt", "w", encoding="utf-8") as f:
            f.write(rendered)
    except Exception:
        # Non-fatal. Don't block training on an example file.
        return


def train(cfg_path: str) -> None:
    cfg = load_yaml(cfg_path)

    task = str(cfg["run"]["task"]).lower()
    run_name = cfg["run"].get("run_name", "run")

    out_root = Path(cfg["train"]["output_dir"]) / run_name
    _ensure_dir(str(out_root))

    # CPU thread tuning
    set_cpu_threads(cfg["compute"].get("num_threads", 0), cfg["compute"].get("interop_threads", 2))

    # Load model + tokenizer
    bundle = load_bundle(cfg["model"]["base_model"], dtype=cfg["compute"].get("dtype", "float32"))
    _maybe_write_sample(bundle.tokenizer, cfg, str(out_root))

    # Attach LoRA
    lcfg = cfg.get("lora", {})
    model = attach_lora(
        bundle.model,
        r=lcfg.get("r", 16),
        alpha=lcfg.get("alpha", 32),
        dropout=lcfg.get("dropout", 0.05),
        target_modules=list(lcfg.get("target_modules", ["q_proj", "v_proj"])),
    )

    # Data
    max_len = int(cfg["model"].get("max_length", 256))
    shuffle_buffer = int(cfg["data"].get("shuffle_buffer", 0))

    if task == "cpt":
        sources = _as_text_sources(cfg)
        ds = CPTIterableDataset(bundle.tokenizer, sources=sources, block_size=max_len, shuffle_buffer=shuffle_buffer)
    else:
        sources = _as_sft_sources(cfg)
        sys_prompt = cfg["data"].get("system_prompt", "You are a helpful assistant.")
        ds = SFTIterableDataset(bundle.tokenizer, sources=sources, system_prompt=sys_prompt, max_length=max_len, shuffle_buffer=shuffle_buffer)

    per_device_batch_size = int(cfg["train"].get("per_device_batch_size", 1))
    loader = DataLoader(ds, batch_size=per_device_batch_size, num_workers=0)

    # Training config
    max_steps = int(cfg["train"].get("max_steps", 100))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    lr = float(cfg["train"].get("learning_rate", 2e-4))
    warmup = int(cfg["train"].get("warmup_steps", 0))
    wd = float(cfg["train"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    log_every = int(cfg["train"].get("log_every", 10))
    save_every = int(cfg["train"].get("save_every", 200))
    seed = int(cfg["train"].get("seed", 123))

    torch.manual_seed(seed)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=max_steps)

    status_path = out_root / "status.json"
    checkpoints_dir = out_root / "checkpoints"
    _ensure_dir(str(checkpoints_dir))

    # Training loop
    model.train()
    step = 0
    accum = 0
    running_loss = 0.0
    t0 = time.time()
    last_log_t = t0

    # Estimate tokens/step
    tokens_per_batch = per_device_batch_size * max_len

    it = iter(loader)
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # (CPU) keep tensors on CPU
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )
        loss = outputs.loss / grad_accum
        loss.backward()
        running_loss += loss.item()
        accum += 1

        if accum >= grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            accum = 0

            if step % log_every == 0 or step == 1:
                now = time.time()
                elapsed = now - t0
                steps_done = step
                steps_left = max_steps - steps_done
                sec_per_step = elapsed / max(steps_done, 1)
                eta = steps_left * sec_per_step

                avg_loss = running_loss / max(log_every, 1)
                running_loss = 0.0

                tok_done = steps_done * tokens_per_batch * grad_accum
                tok_per_s = tok_done / max(elapsed, 1e-6)

                msg = (
                    f"step={step}/{max_steps} "
                    f"loss={avg_loss:.4f} "
                    f"tok/s≈{tok_per_s:.0f} "
                    f"elapsed={format_seconds(elapsed)} "
                    f"eta≈{format_seconds(eta)}"
                )
                print(msg, flush=True)

                atomic_write_json(
                    str(status_path),
                    {
                        "timestamp": now_ts(),
                        "task": task,
                        "run_name": run_name,
                        "step": step,
                        "max_steps": max_steps,
                        "pct_steps": round(100.0 * step / max_steps, 2),
                        "avg_loss": avg_loss,
                        "tokens_per_step_est": tokens_per_batch * grad_accum,
                        "tokens_processed_est": tok_done,
                        "tok_per_s_est": tok_per_s,
                        "elapsed_s": elapsed,
                        "eta_s": eta,
                        "note": "Streaming datasets: total dataset size is unknown; progress is by steps, not dataset percent.",
                    },
                )

            if step % save_every == 0 or step == max_steps:
                ckpt_dir = checkpoints_dir / f"step_{step:06d}"
                _ensure_dir(str(ckpt_dir))
                model.save_pretrained(str(ckpt_dir))
                bundle.tokenizer.save_pretrained(str(ckpt_dir))
                atomic_write_json(
                    str(ckpt_dir / "trainer_state.json"),
                    {
                        "timestamp": now_ts(),
                        "step": step,
                        "max_steps": max_steps,
                        "base_model": cfg["model"]["base_model"],
                        "task": task,
                        "dtype": cfg["compute"].get("dtype", "float32"),
                    },
                )
                # Convenience "latest" copy
                latest = out_root / "latest"
                _ensure_dir(str(latest))
                model.save_pretrained(str(latest))
                bundle.tokenizer.save_pretrained(str(latest))

    print(f"Done. Adapters saved under: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
