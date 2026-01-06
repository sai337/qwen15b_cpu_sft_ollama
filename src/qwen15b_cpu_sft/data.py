from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


@dataclass
class TextSource:
    name: str
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    text_field: str
    weight: float = 1.0


@dataclass
class SFTSource:
    name: str
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    prompt_field: str
    response_field: str
    weight: float = 1.0


def _weighted_round_robin(iterators: List[Iterator[Any]], weights: List[float]) -> Iterator[Any]:
    """Simple weighted sampler across streaming iterators.

    We sample a source index according to weights for each example.
    This is NOT perfectly fair, but good enough for demo / small runs.
    """
    assert len(iterators) == len(weights) and len(iterators) > 0
    total = sum(weights)
    probs = [w / total for w in weights]
    rng = random.Random(0)

    alive = [True] * len(iterators)
    while any(alive):
        i = rng.choices(range(len(iterators)), probs, k=1)[0]
        if not alive[i]:
            continue
        try:
            yield next(iterators[i])
        except StopIteration:
            alive[i] = False


def load_streaming_text(source: TextSource) -> Iterator[str]:
    ds = load_dataset(source.dataset_name, source.dataset_config, split=source.split, streaming=True)
    for row in ds:
        if source.text_field not in row:
            continue
        txt = row[source.text_field]
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if not txt:
            continue
        yield txt


def load_streaming_sft(source: SFTSource) -> Iterator[Tuple[str, str]]:
    ds = load_dataset(source.dataset_name, source.dataset_config, split=source.split, streaming=True)
    for row in ds:
        if source.prompt_field not in row or source.response_field not in row:
            continue
        p = row[source.prompt_field]
        r = row[source.response_field]
        if not isinstance(p, str) or not isinstance(r, str):
            continue
        p = p.strip()
        r = r.strip()
        if not p or not r:
            continue
        yield p, r


class CPTIterableDataset(IterableDataset):
    """Continued pretraining dataset: yields fixed-length blocks for causal LM."""

    def __init__(self, tokenizer, sources: List[TextSource], block_size: int, shuffle_buffer: int = 0):
        self.tokenizer = tokenizer
        self.sources = sources
        self.block_size = block_size
        self.shuffle_buffer = int(shuffle_buffer or 0)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iters = [iter(load_streaming_text(s)) for s in self.sources]
        weights = [s.weight for s in self.sources]
        mixed = _weighted_round_robin(iters, weights)

        # Simple token buffer packing
        # For HF causal LM loss, labels should be the same as input_ids.
        # (The model shifts internally to do next-token prediction.)
        token_buffer: List[int] = []

        # Optional shuffling: keep a buffer of text chunks and shuffle it.
        if self.shuffle_buffer > 0:
            mixed = _shuffle_text_stream(mixed, self.shuffle_buffer)

        for text in mixed:
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if not ids:
                continue
            token_buffer.extend(ids + [self.tokenizer.eos_token_id])

            while len(token_buffer) >= self.block_size:
                chunk = token_buffer[: self.block_size]
                token_buffer = token_buffer[self.block_size :]

                x = torch.tensor(chunk, dtype=torch.long)
                attn = torch.ones_like(x, dtype=torch.long)
                # labels == input_ids (HF shifts internally)
                yield {"input_ids": x, "attention_mask": attn, "labels": x}


def _shuffle_text_stream(text_iter: Iterator[str], buffer_size: int) -> Iterator[str]:
    buf: List[str] = []
    rng = random.Random(0)
    for item in text_iter:
        buf.append(item)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            for out in buf:
                yield out
            buf = []
    if buf:
        rng.shuffle(buf)
        for out in buf:
            yield out


class SFTIterableDataset(IterableDataset):
    """Instruction tuning dataset: yields padded fixed-length samples with masked labels."""

    def __init__(
        self,
        tokenizer,
        sources: List[SFTSource],
        system_prompt: str,
        max_length: int,
        shuffle_buffer: int = 0,
    ):
        self.tokenizer = tokenizer
        self.sources = sources
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.shuffle_buffer = int(shuffle_buffer or 0)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iters = [iter(load_streaming_sft(s)) for s in self.sources]
        weights = [s.weight for s in self.sources]
        mixed = _weighted_round_robin(iters, weights)

        # Optional shuffle at the (prompt,response) example level
        if self.shuffle_buffer > 0:
            mixed = _shuffle_pairs_stream(mixed, self.shuffle_buffer)

        tok = self.tokenizer
        pad_id = tok.pad_token_id
        if pad_id is None:
            # Qwen tokenizers commonly have pad; if not, fall back to eos
            tok.pad_token = tok.eos_token
            pad_id = tok.pad_token_id

        for prompt, response in mixed:
            messages_prompt = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            messages_full = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

            # Use chat template if available; fall back to naive formatting
            prompt_text = _apply_chat_template(tok, messages_prompt, add_generation_prompt=True)
            full_text = _apply_chat_template(tok, messages_full, add_generation_prompt=False)

            prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
            full_enc = tok(full_text, add_special_tokens=False)
            input_ids = full_enc.input_ids

            if len(input_ids) < 2:
                continue

            # Build labels with prompt masked
            labels = list(input_ids)
            boundary = min(len(prompt_ids), len(labels))
            for i in range(boundary):
                labels[i] = -100

            # Truncate then pad
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            attn = [1] * len(input_ids)

            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                input_ids = input_ids + [pad_id] * pad_len
                attn = attn + [0] * pad_len
                labels = labels + [-100] * pad_len

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


def _shuffle_pairs_stream(pair_iter: Iterator[Tuple[str, str]], buffer_size: int) -> Iterator[Tuple[str, str]]:
    buf: List[Tuple[str, str]] = []
    rng = random.Random(0)
    for item in pair_iter:
        buf.append(item)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            for out in buf:
                yield out
            buf = []
    if buf:
        rng.shuffle(buf)
        for out in buf:
            yield out


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback: crude, but keeps pipeline running if chat_template is missing
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n".join(parts)
