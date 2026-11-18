# Code review: findings and fixes (Nov 2026)

Severity: 🔴 breaks training/serving · 🟠 wrong or misleading result · 🟡 hygiene

| # | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| 1 | 🔴 | `data.py` SFT | A prompt longer than `max_length` produced a sample with **every label = -100**. HF returns `NaN` loss for that; `NaN` gradients then poison the LoRA weights for the rest of the run. Very likely with `max_length: 256` and the long reasoning-trace dataset. | Skip samples whose prompt leaves < `min_response_tokens` (default 16); training loop also skips batches with no supervised tokens or a non-finite loss and counts them in `status.json`. |
| 2 | 🔴 | `scripts/quantize_gguf.sh` | Looked for a `quantize` binary. Current llama.cpp builds `build/bin/llama-quantize`, so the script always failed after building. | Finds `llama-quantize` (and legacy names); builds only the needed targets. |
| 3 | 🟠 | `ollama/Modelfile.template` | No `TEMPLATE` / `stop`. Ollama might or might not detect ChatML from the GGUF; when it doesn't, the model sees a different prompt format than it was trained on and runs past `<\|im_end\|>`. | Explicit Qwen2.5 ChatML template + `stop` tokens. |
| 4 | 🟠 | config `devops_llm` | `input_field: Prompt` was silently ignored (not in `SFTSource`), so the instruction was trained without its context. | `input_field` supported (appended to the prompt). |
| 5 | 🟠 | `run_train_cpt_then_sft.sh` | Said "SFT on top of CPT" but ran SFT on the original base. | Passes `--base-model merged/<cpt>` to stage 2. |
| 6 | 🟠 | `train.py` logging | Loss average divided by `log_every` (wrong at step 1 and after accumulation); tok/s counted padding. | Mean over real micro-batches; tokens/s counts supervised tokens; `loss_history` saved. |
| 7 | 🟠 | UI | Hard-coded system prompt `"…Answer..."` overrode the Modelfile `SYSTEM` the model was trained with. | UI uses the Modelfile system prompt by default; model picker for base vs fine-tuned. |
| 8 | 🟠 | `docker-compose.yml` | Ollama published on `0.0.0.0:11434` with no auth. | Bound to `127.0.0.1`; UI bind configurable. |
| 9 | 🟡 | README | `README.md` was a single heading; real docs sat in `README 2.md`. | Merged. |
| 10 | 🟡 | repo | No tests, no eval, no way to train on your own data, no `.gitignore`. | 15 pytest tests (tiny offline Qwen2, UI against a fake Ollama), `eval_compare.py`, local JSONL config, `.gitignore`, offline e2e smoke script. |
| 11 | 🟡 | `modeling.py` | `torch_dtype=` is deprecated in transformers ≥ 4.56 (warns on v5). | Picks `dtype`/`torch_dtype` by version. |
| 12 | 🟡 | LoRA targets | Only `q_proj, v_proj`. Fine for speed, weaker for style/knowledge. | New local config uses all linear projections; HF configs left as-is. |

Not verifiable from the sandbox (huggingface.co blocked): the HF dataset field names in
`sft_lora_qwen15b_cpu_cloudqa.yaml`. If one is wrong, that source now yields nothing and
`status.json → data_stats.per_source` shows it, instead of failing silently.
