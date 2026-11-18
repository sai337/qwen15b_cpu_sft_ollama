from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)


@dataclass
class TextSource:
    name: str
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    text_field: str
    weight: float = 1.0
    data_files: Optional[Any] = None  # for local files: dataset_name: json + data_files: path


@dataclass
class SFTSource:
    name: str
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    prompt_field: str
    response_field: str
    weight: float = 1.0
    input_field: Optional[str] = None  # optional extra context appended to the prompt (Alpaca style)
    data_files: Optional[Any] = None


@dataclass
class SFTStats:
    """Counters so you can see how much data was dropped and why."""

    yielded: int = 0
    skipped_too_long_prompt: int = 0
    skipped_empty: int = 0
    truncated_response: int = 0
    per_source: Dict[str, int] = field(default_factory=dict)


def _open_stream(dataset_name: str, dataset_config: Optional[str], split: str, data_files: Optional[Any]):
    if data_files is not None:
        # Local files (json/jsonl/csv/parquet). dataset_name is the builder: "json", "csv", "parquet".
        return load_dataset(dataset_name, data_files=data_files, split=split, streaming=True)
    return load_dataset(dataset_name, dataset_config, split=split, streaming=True)


def _weighted_mix(iterators: List[Iterator[Any]], weights: List[float], seed: int = 0) -> Iterator[Any]:
    """Weighted sampler across streaming iterators.

    Picks a source per example according to its weight. Exhausted sources are
    removed so the remaining weights are re-normalised (no busy-looping).
    """
    assert len(iterators) == len(weights) and len(iterators) > 0
    rng = random.Random(seed)
    live = list(range(len(iterators)))
    while live:
        i = rng.choices(live, [weights[j] for j in live], k=1)[0]
        try:
            yield next(iterators[i])
        except StopIteration:
            live.remove(i)


# Kept for backwards compatibility with anything importing the old name.
_weighted_round_robin = _weighted_mix


def load_streaming_text(source: TextSource) -> Iterator[str]:
    ds = _open_stream(source.dataset_name, source.dataset_config, source.split, source.data_files)
    for row in ds:
        txt = row.get(source.text_field)
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if txt:
            yield txt


def load_streaming_sft(source: SFTSource) -> Iterator[Tuple[str, str, str]]:
    """Yields (source_name, prompt, response)."""
    ds = _open_stream(source.dataset_name, source.dataset_config, source.split, source.data_files)
    for row in ds:
        p = row.get(source.prompt_field)
        r = row.get(source.response_field)
        if not isinstance(p, str) or not isinstance(r, str):
            continue
        p, r = p.strip(), r.strip()
        if source.input_field:
            extra = row.get(source.input_field)
            if isinstance(extra, str) and extra.strip():
                p = f"{p}\n\n{extra.strip()}" if p else extra.strip()
        if not p or not r:
            continue
        yield source.name, p, r


def _shuffle_stream(it: Iterator[Any], buffer_size: int, seed: int = 0) -> Iterator[Any]:
    buf: List[Any] = []
    rng = random.Random(seed)
    for item in it:
        buf.append(item)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            yield from buf
            buf = []
    if buf:
        rng.shuffle(buf)
        yield from buf


# Backwards-compatible names
_shuffle_text_stream = _shuffle_stream
_shuffle_pairs_stream = _shuffle_stream


class CPTIterableDataset(IterableDataset):
    """Continued pretraining dataset: yields fixed-length packed blocks for causal LM."""

    def __init__(self, tokenizer, sources: List[TextSource], block_size: int, shuffle_buffer: int = 0, seed: int = 0):
        self.tokenizer = tokenizer
        self.sources = sources
        self.block_size = block_size
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.seed = seed

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iters = [iter(load_streaming_text(s)) for s in self.sources]
        mixed: Iterator[str] = _weighted_mix(iters, [s.weight for s in self.sources], self.seed)
        if self.shuffle_buffer > 0:
            mixed = _shuffle_stream(mixed, self.shuffle_buffer, self.seed)

        eos = self.tokenizer.eos_token_id
        token_buffer: List[int] = []
        for text in mixed:
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if not ids:
                continue
            token_buffer.extend(ids + [eos])
            while len(token_buffer) >= self.block_size:
                chunk, token_buffer = token_buffer[: self.block_size], token_buffer[self.block_size :]
                x = torch.tensor(chunk, dtype=torch.long)
                # labels == input_ids (HF shifts internally for next-token prediction)
                yield {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


class SFTIterableDataset(IterableDataset):
    """Instruction tuning dataset: padded fixed-length samples, loss only on assistant tokens.

    Samples whose prompt alone leaves fewer than ``min_response_tokens`` slots are
    skipped. Without this, a fully masked sample (all labels == -100) gives a NaN
    loss that poisons the LoRA weights on the next optimizer step.
    """

    def __init__(
        self,
        tokenizer,
        sources: List[SFTSource],
        system_prompt: str,
        max_length: int,
        shuffle_buffer: int = 0,
        min_response_tokens: int = 16,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.sources = sources
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.min_response_tokens = int(min_response_tokens)
        self.seed = seed
        self.stats = SFTStats()

    def encode(self, prompt: str, response: str) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenise one example. Returns None if it must be skipped."""
        tok = self.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        pad_id = tok.pad_token_id

        sys_msg = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        messages_prompt = sys_msg + [{"role": "user", "content": prompt}]
        messages_full = messages_prompt + [{"role": "assistant", "content": response}]

        prompt_text = _apply_chat_template(tok, messages_prompt, add_generation_prompt=True)
        full_text = _apply_chat_template(tok, messages_full, add_generation_prompt=False)

        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        input_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(input_ids) <= len(prompt_ids):
            self.stats.skipped_empty += 1
            return None
        if len(prompt_ids) > self.max_length - self.min_response_tokens:
            self.stats.skipped_too_long_prompt += 1
            return None
        if len(input_ids) > self.max_length:
            self.stats.truncated_response += 1

        labels = [-100] * len(prompt_ids) + list(input_ids[len(prompt_ids) :])
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attn = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len:
            input_ids = input_ids + [pad_id] * pad_len
            attn = attn + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iters = [iter(load_streaming_sft(s)) for s in self.sources]
        mixed: Iterator[Tuple[str, str, str]] = _weighted_mix(iters, [s.weight for s in self.sources], self.seed)
        if self.shuffle_buffer > 0:
            mixed = _shuffle_stream(mixed, self.shuffle_buffer, self.seed)

        for name, prompt, response in mixed:
            sample = self.encode(prompt, response)
            if sample is None:
                continue
            self.stats.yielded += 1
            self.stats.per_source[name] = self.stats.per_source.get(name, 0) + 1
            yield sample


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    # Fallback: crude, but keeps the pipeline running if chat_template is missing
    parts = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages]
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n".join(parts)
