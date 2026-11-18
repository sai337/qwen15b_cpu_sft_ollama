import pytest
import yaml

from qwen15b_cpu_sft.config import load_yaml


def _write(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_shipped_configs_are_valid():
    for name in ["sft_lora_qwen15b_cpu_cloudqa", "cpt_lora_qwen15b_cpu_cloudtext", "sft_lora_local_jsonl"]:
        load_yaml(f"configs/{name}.yaml")


def test_missing_sft_fields_rejected(tmp_path):
    cfg = load_yaml("configs/sft_lora_local_jsonl.yaml")
    del cfg["data"]["sft_sources"][0]["response_field"]
    with pytest.raises(ValueError, match="response_field"):
        load_yaml(_write(tmp_path, cfg))


def test_bad_task_rejected(tmp_path):
    cfg = load_yaml("configs/sft_lora_local_jsonl.yaml")
    cfg["run"]["task"] = "rlhf"
    with pytest.raises(ValueError):
        load_yaml(_write(tmp_path, cfg))
