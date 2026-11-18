import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from qwen15b_cpu_sft.train import train

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, model_dir, data, task="sft", steps=6):
    cfg = yaml.safe_load((ROOT / "configs/sft_lora_local_jsonl.yaml").read_text())
    cfg["run"]["run_name"] = f"t_{task}"
    cfg["compute"]["num_threads"] = 2
    cfg["model"]["base_model"] = str(model_dir)
    cfg["model"]["max_length"] = 160
    cfg["data"]["sft_sources"][0]["data_files"] = str(data)
    cfg["train"].update(output_dir=str(tmp_path / "out"), max_steps=steps, grad_accum_steps=2,
                        save_every=3, log_every=1, warmup_steps=1, learning_rate=5e-3)
    if task == "cpt":
        cfg["run"]["task"] = "cpt"
        cfg["model"]["max_length"] = 16  # packed blocks; tiny corpus
        cfg["data"] = {"text_sources": [{"name": "t", "dataset_name": "json", "data_files": str(data),
                                         "split": "train", "text_field": "answer"}]}
    p = tmp_path / f"{task}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_sft_trains_saves_and_loss_is_finite(tmp_path, tiny_model_dir, sft_jsonl):
    latest = train(str(_cfg(tmp_path, tiny_model_dir, sft_jsonl)))
    assert (latest / "adapter_model.safetensors").exists()
    assert (latest / "adapter_config.json").exists()
    run = latest.parent
    assert (run / "checkpoints/step_000003").is_dir() and (run / "checkpoints/step_000006").is_dir()
    status = json.loads((run / "status.json").read_text())
    assert status["step"] == 6 and torch.isfinite(torch.tensor(status["avg_loss"]))
    assert status["skipped_nonfinite_batches"] == 0
    assert status["data_stats"]["yielded"] > 0
    assert (run / "sample_prompt.txt").read_text().endswith("<|im_start|>assistant\n")


def test_loss_decreases_when_overfitting(tmp_path, tiny_model_dir, sft_jsonl):
    latest = train(str(_cfg(tmp_path, tiny_model_dir, sft_jsonl, steps=40)))
    status = json.loads((latest.parent / "status.json").read_text())
    hist = status["loss_history"]
    assert hist[0][0] == 1 and hist[-1][0] == 40
    assert hist[-1][1] < hist[0][1] * 0.9, hist  # training actually moves the loss


def test_cpt_mode_runs(tmp_path, tiny_model_dir, sft_jsonl):
    latest = train(str(_cfg(tmp_path, tiny_model_dir, sft_jsonl, task="cpt", steps=3)))
    assert (latest / "adapter_model.safetensors").exists()


def test_merge_adapter_produces_loadable_model(tmp_path, tiny_model_dir, sft_jsonl):
    latest = train(str(_cfg(tmp_path, tiny_model_dir, sft_jsonl, steps=3)))
    out = tmp_path / "merged"
    subprocess.run([sys.executable, str(ROOT / "scripts/merge_adapter.py"), "--adapter", str(latest),
                    "--out", str(out)], check=True, cwd=ROOT)  # --base read from adapter_config.json
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForCausalLM.from_pretrained(out)
    assert not any("lora" in k for k in model.state_dict())
    ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    gen = model.generate(ids, max_new_tokens=5, do_sample=False)
    assert gen.shape[1] > ids.shape[1]
