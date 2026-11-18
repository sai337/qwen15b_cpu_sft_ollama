#!/usr/bin/env python3
"""TEST ONLY: run llama.cpp's convert_hf_to_gguf.py on the tiny offline model.

llama.cpp identifies BPE pre-tokenizers by hashing a probe string. The tiny test tokenizer
is not a real Qwen tokenizer, so we tell the converter to treat it as "qwen2". Real
Qwen2.5 checkpoints are recognised natively; scripts/export_gguf.sh needs no patch.

Usage: LLAMA_CPP_DIR=./llama.cpp python tests/convert_tiny_gguf.py <hf_dir> --outtype f16 --outfile x.gguf
"""

import os
import sys
from pathlib import Path

llama = Path(os.environ.get("LLAMA_CPP_DIR", "llama.cpp")).resolve()
sys.path[:0] = [str(llama), str(llama / "gguf-py")]

import conversion.base as base  # noqa: E402

base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"

import convert_hf_to_gguf  # noqa: E402

sys.argv = ["convert_hf_to_gguf.py", *sys.argv[1:]]
convert_hf_to_gguf.main()
