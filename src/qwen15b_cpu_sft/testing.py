"""Build a tiny, randomly initialised Qwen2 model + ChatML tokenizer fully offline.

Used by the test suite and by scripts/smoke_e2e.sh to exercise the whole pipeline
(train -> merge -> GGUF -> quantize -> serve) in a couple of minutes without
downloading anything from Hugging Face. Architecture, special tokens and chat
template match Qwen2.5-Instruct, so the code paths are the real ones; only the
size (and therefore output quality) differs.
"""

from __future__ import annotations

from pathlib import Path

# Same template structure as Qwen2.5-Instruct (tool-calling branch removed).
QWEN25_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant') %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

_CORPUS = [
    "Kubernetes DaemonSet runs one pod on every node.",
    "An EKS managed node group is a set of EC2 instances managed by AWS.",
    "Use IAM roles for service accounts to give pods AWS permissions.",
    "vLLM serves large language models with paged attention.",
    "KServe InferenceService deploys models on Kubernetes.",
    "Ollama runs GGUF models locally on CPU or GPU.",
    "LoRA trains small low rank adapters instead of all weights.",
    "You are a senior platform engineer. Answer clearly and concretely.",
    "system user assistant",
] * 20


def build_tiny_qwen(out_dir: str | Path, vocab_size: int = 512, hidden: int = 64, layers: int = 2) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tk.train_from_iterator(_CORPUS, trainer=trainer)

    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = QWEN25_CHAT_TEMPLATE
    tok.model_max_length = 2048
    tok.save_pretrained(out)

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        bos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    return out


if __name__ == "__main__":
    import sys

    print(build_tiny_qwen(sys.argv[1] if len(sys.argv) > 1 else "tiny_qwen"))
