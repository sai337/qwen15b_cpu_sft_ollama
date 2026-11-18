import torch

from qwen15b_cpu_sft.data import SFTIterableDataset, SFTSource, load_streaming_sft, _weighted_mix


def _ds(tok, max_length=128, min_resp=4, sources=()):
    return SFTIterableDataset(tok, list(sources), system_prompt="You are helpful.", max_length=max_length, min_response_tokens=min_resp)


def test_prompt_is_masked_and_answer_is_learned(tiny_tokenizer):
    ds = _ds(tiny_tokenizer)
    s = ds.encode("What is a DaemonSet?", "A DaemonSet runs one pod on every node.")
    labels, ids = s["labels"], s["input_ids"]
    assert ids.shape == labels.shape == (128,)

    learned = labels[labels != -100]
    text = tiny_tokenizer.decode(learned)
    assert "DaemonSet runs one pod" in text
    assert "What is a DaemonSet" not in text          # prompt not in loss
    assert text.rstrip().endswith("<|im_end|>")        # model learns to stop
    assert (s["attention_mask"][labels != -100] == 1).all()


def test_chat_template_prompt_is_prefix_of_full(tiny_tokenizer):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    p = tiny_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    f = tiny_tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": "a"}], tokenize=False)
    assert f.startswith(p)


def test_long_prompt_is_skipped_not_nan(tiny_tokenizer):
    """Regression: a prompt filling max_length used to yield an all -100 sample -> NaN loss."""
    ds = _ds(tiny_tokenizer, max_length=32, min_resp=8)
    assert ds.encode("word " * 200, "short answer") is None
    assert ds.stats.skipped_too_long_prompt == 1


def test_truncated_response_still_has_supervised_tokens(tiny_tokenizer):
    ds = _ds(tiny_tokenizer, max_length=48, min_resp=4)
    s = ds.encode("q", "long answer " * 100)
    assert s is not None and int((s["labels"] != -100).sum()) > 0
    assert ds.stats.truncated_response == 1


def test_local_jsonl_and_input_field(sft_jsonl):
    src = SFTSource(name="local", dataset_name="json", dataset_config=None, split="train",
                    prompt_field="question", response_field="answer", input_field="ctx", data_files=str(sft_jsonl))
    rows = list(load_streaming_sft(src))
    assert len(rows) == 3
    assert rows[0] == ("local", "What is a DaemonSet?\n\nk8s", "A DaemonSet runs one pod on every node.")
    assert rows[1][1] == "What is LoRA?"  # empty ctx is not appended


def test_iterable_dataset_yields_and_counts(tiny_tokenizer, sft_jsonl):
    src = SFTSource(name="local", dataset_name="json", dataset_config=None, split="train",
                    prompt_field="question", response_field="answer", data_files=str(sft_jsonl))
    ds = _ds(tiny_tokenizer, sources=[src])
    out = list(ds)
    assert len(out) == 3 and ds.stats.per_source == {"local": 3}
    assert all(isinstance(x["input_ids"], torch.Tensor) for x in out)


def test_weighted_mix_drains_all_sources():
    out = list(_weighted_mix([iter([1, 2]), iter([3]), iter([])], [1.0, 5.0, 1.0]))
    assert sorted(out) == [1, 2, 3]
