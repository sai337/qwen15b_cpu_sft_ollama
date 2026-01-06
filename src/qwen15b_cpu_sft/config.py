from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config with basic sanity checks."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML must be a mapping (dict), got {type(cfg)}")

    required = ["run", "compute", "model", "data", "train", "lora"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    # Minimal validation that avoids silent typo failures
    if "task" not in cfg["run"]:
        raise ValueError("run.task is required (sft or cpt)")
    task = str(cfg["run"]["task"]).lower()
    if task not in {"sft", "cpt"}:
        raise ValueError("run.task must be one of: sft, cpt")

    if "base_model" not in cfg["model"]:
        raise ValueError("model.base_model is required")

    if "output_dir" not in cfg["train"]:
        raise ValueError("train.output_dir is required")

    return cfg
