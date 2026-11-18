from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import load_yaml
from .data import CPTIterableDataset, SFTIterableDataset, SFTSource, TextSource
from .logging_utils import atomic_write_json, format_seconds, now_ts
from .modeling import attach_lora, load_bundle, resolve_device, set_cpu_threads


def _as_text_sources(cfg: Dict[str, Any]) -> list[TextSource]:
    return [
        TextSource(
            name=s["name"],
            dataset_name=s["dataset_name"],
            dataset_config=s.get("dataset_config"),
            split=s.get("split", "train"),
            text_field=s.get("text_field", "text"),
            weight=float(s.get("weight", 1.0)),
            data_files=s.get("data_files"),
        )
        for s in cfg["data"]["text_sources"]
    ]


def _as_sft_sources(cfg: Dict[str, Any]) -> list[SFTSource]:
    return [
        SFTSource(
            name=s["name"],
            dataset_name=s["dataset_name"],
            dataset_config=s.get("dataset_config"),
            split=s.get("split", "train"),
            prompt_field=s["prompt_field"],
            response_field=s["response_field"],
            input_field=s.get("input_field"),
            weight=float(s.get("weight", 1.0)),
            data_files=s.get("data_files"),
        )
        for s in cfg["data"]["sft_sources"]
    ]


@torch.no_grad()
def _write_sample_prompt(tokenizer, cfg: Dict[str, Any], out_dir: Path) -> None:
    """Write an example prompt so you can see the exact text format the model is trained on."""
    try:
        sys_prompt = cfg.get("data", {}).get("system_prompt", "You are a helpful assistant.")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Explain what an EKS node group is."},
        ]
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = f"SYSTEM: {sys_prompt}\nUSER: Explain what an EKS node group is.\nASSISTANT:"
        (out_dir / "sample_prompt.txt").write_text(rendered, encoding="utf-8")
    except Exception:
        return  # non-fatal


def _save_adapter(model, tokenizer, target: Path, meta: Dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    atomic_write_json(str(target / "trainer_state.json"), meta)


def train(cfg_path: str) -> Path:
    cfg = load_yaml(cfg_path)

    task = str(cfg["run"]["task"]).lower()
    run_name = cfg["run"].get("run_name", "run")
    out_root = Path(cfg["train"]["output_dir"]) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg_path, out_root / "run_config.yaml")

    compute = cfg["compute"]
    set_cpu_threads(compute.get("num_threads", 0), compute.get("interop_threads", 2))
    device = resolve_device(compute.get("device", "cpu"))
    dtype = compute.get("dtype", "float32")

    bundle = load_bundle(cfg["model"]["base_model"], dtype=dtype)
    _write_sample_prompt(bundle.tokenizer, cfg, out_root)

    if cfg["train"].get("gradient_checkpointing", False):
        bundle.model.gradient_checkpointing_enable()
        bundle.model.enable_input_require_grads()

    lcfg = cfg.get("lora", {})
    model = attach_lora(
        bundle.model,
        r=lcfg.get("r", 16),
        alpha=lcfg.get("alpha", 32),
        dropout=lcfg.get("dropout", 0.05),
        target_modules=list(lcfg.get("target_modules", ["q_proj", "v_proj"])),
    ).to(device)

    seed = int(cfg["train"].get("seed", 123))
    torch.manual_seed(seed)

    max_len = int(cfg["model"].get("max_length", 256))
    shuffle_buffer = int(cfg["data"].get("shuffle_buffer", 0))
    if task == "cpt":
        ds = CPTIterableDataset(bundle.tokenizer, _as_text_sources(cfg), block_size=max_len, shuffle_buffer=shuffle_buffer, seed=seed)
    else:
        ds = SFTIterableDataset(
            bundle.tokenizer,
            _as_sft_sources(cfg),
            system_prompt=cfg["data"].get("system_prompt", "You are a helpful assistant."),
            max_length=max_len,
            shuffle_buffer=shuffle_buffer,
            min_response_tokens=int(cfg["data"].get("min_response_tokens", 16)),
            seed=seed,
        )

    per_device_batch_size = int(cfg["train"].get("per_device_batch_size", 1))
    loader = DataLoader(ds, batch_size=per_device_batch_size, num_workers=0)

    max_steps = int(cfg["train"].get("max_steps", 100))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["train"].get("log_every", 10))
    save_every = int(cfg["train"].get("save_every", 200))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"].get("learning_rate", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["train"].get("warmup_steps", 0)), num_training_steps=max_steps
    )

    status_path = out_root / "status.json"
    checkpoints_dir = out_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = accum = 0
    window_loss, window_micro = 0.0, 0
    skipped_nonfinite = 0
    supervised_tokens = 0
    epochs_seen = 0
    loss_history: list[list[float]] = []  # [step, avg_loss] at each log point
    t0 = time.time()

    it = iter(loader)
    while step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            epochs_seen += 1
            if epochs_seen > 1000:
                stats = vars(ds.stats) if isinstance(ds, SFTIterableDataset) else {}
                raise RuntimeError(
                    f"Dataset yields no usable samples. Check field names, data_files and max_length. stats={stats}"
                )
            it = iter(loader)
            continue

        labels = batch["labels"]
        n_sup = int((labels[:, 1:] != -100).sum())
        if n_sup == 0:
            continue  # nothing to learn from this batch; would produce NaN loss

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            continue

        (loss / grad_accum).backward()
        window_loss += float(loss.detach())
        window_micro += 1
        supervised_tokens += n_sup
        accum += 1
        if accum < grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        accum = 0

        if step % log_every == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - t0
            eta = (max_steps - step) * elapsed / step
            avg_loss = window_loss / max(window_micro, 1)
            window_loss, window_micro = 0.0, 0
            loss_history.append([step, round(avg_loss, 4)])
            tok_per_s = supervised_tokens / max(elapsed, 1e-6)
            print(
                f"step={step}/{max_steps} loss={avg_loss:.4f} ppl≈{math.exp(min(avg_loss, 20)):.1f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} sup_tok/s≈{tok_per_s:.0f} "
                f"elapsed={format_seconds(elapsed)} eta≈{format_seconds(eta)}",
                flush=True,
            )
            status = {
                "timestamp": now_ts(),
                "task": task,
                "run_name": run_name,
                "device": str(device),
                "step": step,
                "max_steps": max_steps,
                "pct_steps": round(100.0 * step / max_steps, 2),
                "avg_loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
                "supervised_tokens": supervised_tokens,
                "sup_tok_per_s": tok_per_s,
                "skipped_nonfinite_batches": skipped_nonfinite,
                "loss_history": loss_history[-500:],
                "elapsed_s": elapsed,
                "eta_s": eta,
                "note": "Streaming datasets: total size is unknown; progress is by steps.",
            }
            if isinstance(ds, SFTIterableDataset):
                status["data_stats"] = vars(ds.stats)
            atomic_write_json(str(status_path), status)

        if step % save_every == 0 or step == max_steps:
            meta = {
                "timestamp": now_ts(),
                "step": step,
                "max_steps": max_steps,
                "base_model": cfg["model"]["base_model"],
                "task": task,
                "dtype": dtype,
            }
            _save_adapter(model, bundle.tokenizer, checkpoints_dir / f"step_{step:06d}", meta)
            _save_adapter(model, bundle.tokenizer, out_root / "latest", meta)

    print(f"Done. Adapter saved under: {out_root / 'latest'}", flush=True)
    return out_root / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA CPT/SFT for Qwen2.5 on CPU (or GPU with compute.device: auto)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-steps", type=int, help="Override train.max_steps (handy for smoke runs)")
    parser.add_argument("--base-model", help="Override model.base_model (e.g. a merged CPT model dir)")
    args = parser.parse_args()

    cfg_path = args.config
    if args.max_steps or args.base_model:
        cfg = load_yaml(cfg_path)
        if args.max_steps:
            cfg["train"]["max_steps"] = args.max_steps
            cfg["train"]["save_every"] = min(int(cfg["train"].get("save_every", 200)), args.max_steps)
        if args.base_model:
            cfg["model"]["base_model"] = args.base_model
        out = Path(cfg["train"]["output_dir"]) / cfg["run"].get("run_name", "run")
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = str(out / "effective_config.yaml")
        Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    train(cfg_path)


if __name__ == "__main__":
    main()
