#!/usr/bin/env python3
"""Compare models (e.g. base vs fine-tuned) on held-out prompts via any OpenAI-compatible API.

Works with Ollama (http://localhost:11434/v1), llama-server (http://localhost:8080/v1),
vLLM/KServe, or an AI gateway. Scores keyword coverage per prompt and latency, and writes
a JSON + Markdown report.

Usage:
  python scripts/eval_compare.py --models qwen2.5:1.5b-instruct qwen15b-cloudqa \
      --prompts data/eval_prompts.jsonl --out reports/eval
  # gateway: --base-url https://gw.example.com/api/llm/v1 --api-key $KEY
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests


def load_prompts(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r.setdefault("expect_keywords", [])
    return rows


def chat(base_url: str, model: str, prompt: str, system: str | None, api_key: str | None, max_tokens: int, timeout: int):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.time()
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={"model": model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens},
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    text = body["choices"][0]["message"]["content"] or ""
    usage = body.get("usage") or {}
    return text, time.time() - t0, usage.get("completion_tokens")


def score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return float("nan")
    low = text.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--system", default=None, help="Optional system prompt (Ollama models already carry one)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="reports/eval")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    results: dict[str, list[dict]] = {}
    for m in args.models:
        rows = []
        for p in prompts:
            try:
                text, secs, ntok = chat(args.base_url, m, p["prompt"], args.system, args.api_key, args.max_tokens, args.timeout)
                err = None
            except Exception as e:  # keep going; record the failure
                text, secs, ntok, err = "", 0.0, None, str(e)
            rows.append({"prompt": p["prompt"], "answer": text, "latency_s": round(secs, 2), "completion_tokens": ntok,
                         "keyword_score": score(text, p["expect_keywords"]), "error": err})
            print(f"[{m}] {secs:5.1f}s score={rows[-1]['keyword_score']:.2f} {p['prompt'][:60]}", flush=True)
        results[m] = rows

    summary = {}
    for m, rows in results.items():
        ok = [r for r in rows if not r["error"]]
        scores = [r["keyword_score"] for r in ok if r["keyword_score"] == r["keyword_score"]]
        summary[m] = {
            "answered": f"{len(ok)}/{len(rows)}",
            "keyword_score": round(statistics.mean(scores), 3) if scores else None,
            "p50_latency_s": round(statistics.median([r["latency_s"] for r in ok]), 2) if ok else None,
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")

    md = ["# Eval report", "", "| model | answered | keyword score | p50 latency (s) |", "|---|---|---|---|"]
    md += [f"| {m} | {s['answered']} | {s['keyword_score']} | {s['p50_latency_s']} |" for m, s in summary.items()]
    for i, p in enumerate(prompts):
        md += ["", f"## {i + 1}. {p['prompt']}", f"_expected keywords: {', '.join(p['expect_keywords'])}_"]
        for m in results:
            r = results[m][i]
            md += ["", f"**{m}** (score {r['keyword_score']:.2f}, {r['latency_s']}s)", "", (r["answer"] or f"ERROR: {r['error']}").strip()]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
