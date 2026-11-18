import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from qwen15b_cpu_sft.testing import build_tiny_qwen

    return build_tiny_qwen(tmp_path_factory.mktemp("tiny_qwen"))


@pytest.fixture()
def tiny_tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture()
def sft_jsonl(tmp_path):
    import json

    p = tmp_path / "train.jsonl"
    rows = [
        {"question": "What is a DaemonSet?", "answer": "A DaemonSet runs one pod on every node.", "ctx": "k8s"},
        {"question": "What is LoRA?", "answer": "LoRA trains small low rank adapters instead of all weights.", "ctx": ""},
        {"question": "What is KServe?", "answer": "KServe InferenceService deploys models on Kubernetes.", "ctx": "serving"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p
