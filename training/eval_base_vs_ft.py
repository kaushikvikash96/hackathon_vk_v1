"""Controlled comparison: base Nemotron versus the fine-tuned adapter.

    python training/eval_base_vs_ft.py --base-model nemotron-base --ft-model domain-ft

Both models receive **identical evidence** from the held-out split, so the only
variable is the adapter. Qwen routing and the tool layer are held fixed and are
not involved at all - this isolates synthesis quality, which is what the
fine-tuned model quality category assesses.

Metrics reported:
  component_recall    share of gold facts restated in the answer (the graded property)
  numeric_fidelity    share of numbers in the answer that appear in the evidence
  hallucinated_rate   share of answers containing at least one unsupported number
  thinking_leak_rate  share of answers containing a reasoning trace
  mean_chars          answer length; the grader wants concise answers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.prompts import SYNTH_SYSTEM  # noqa: E402
from src.synth import strip_reasoning  # noqa: E402

_NUMBER = re.compile(r"(?<![\d.])-?\d[\d,]*\.?\d*")
_THINK = re.compile(r"<think>|</think>", re.IGNORECASE)


def numbers_in(text: str) -> set[str]:
    values = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        try:
            values.add(f"{float(cleaned):.4f}".rstrip("0").rstrip("."))
        except ValueError:
            continue
    return values


def component_recall(gold: str, answer: str) -> float:
    """Share of the gold answer's numbers and tickers restated in the answer."""
    gold_numbers = numbers_in(gold)
    gold_tickers = set(re.findall(r"\b[A-Z]{2,4}\.AX\b", gold))
    expected = gold_numbers | gold_tickers
    if not expected:
        words = {w for w in re.findall(r"\b[a-z]{5,}\b", gold.lower())}
        if not words:
            return 1.0
        return len(words & set(re.findall(r"\b[a-z]{5,}\b", answer.lower()))) / len(words)

    answer_tokens = numbers_in(answer) | set(re.findall(r"\b[A-Z]{2,4}\.AX\b", answer))
    return len(expected & answer_tokens) / len(expected)


def evaluate(client, model: str, rows: list[dict], max_tokens: int, label: str) -> dict[str, Any]:
    recalls, fidelities, hallucinated, leaked, lengths, latencies = [], [], 0, 0, [], []
    failures = 0
    samples = []

    for position, row in enumerate(rows, start=1):
        user = row["messages"][1]["content"]
        gold = row["messages"][2]["content"]
        started = time.time()
        try:
            reply = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYNTH_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.1,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            raw = reply.choices[0].message.content or ""
        except Exception as exc:
            failures += 1
            print(f"  [{label} {position}/{len(rows)}] {type(exc).__name__}", file=sys.stderr)
            continue
        latencies.append(time.time() - started)

        if _THINK.search(raw):
            leaked += 1
        answer = strip_reasoning(raw)

        evidence_numbers = numbers_in(user)
        answer_numbers = numbers_in(answer)
        unsupported = answer_numbers - evidence_numbers
        if unsupported:
            hallucinated += 1
        fidelities.append(
            1.0 if not answer_numbers else len(answer_numbers & evidence_numbers) / len(answer_numbers)
        )
        recalls.append(component_recall(gold, answer))
        lengths.append(len(answer))

        if len(samples) < 5:
            samples.append({"question": row["messages"][1]["content"][:200],
                            "gold": gold, "answer": answer})
        if position % 20 == 0:
            print(f"  [{label}] {position}/{len(rows)}")

    scored = len(recalls) or 1
    return {
        "model": model,
        "scored": len(recalls),
        "failures": failures,
        "component_recall": round(sum(recalls) / scored * 100, 2),
        "numeric_fidelity": round(sum(fidelities) / scored * 100, 2),
        "hallucinated_rate": round(hallucinated / scored * 100, 2),
        "thinking_leak_rate": round(leaked / scored * 100, 2),
        "mean_chars": round(sum(lengths) / scored, 1),
        "mean_latency_s": round(sum(latencies) / (len(latencies) or 1), 2),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="nemotron-base",
                        help="LiteLLM alias serving the unmodified base Nemotron")
    parser.add_argument("--ft-model", default=None, help="Defaults to $DOMAIN_FT_MODEL")
    parser.add_argument("--test-file", default="training/data/test.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--quick", action="store_true",
                        help="Score 30 examples instead of 100, to pair with a quick training run.")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--out", default="training/metrics/base_vs_ft.json")
    args = parser.parse_args()

    if args.quick:
        args.limit = min(args.limit, 30)
        if args.out == "training/metrics/base_vs_ft.json":
            args.out = "training/metrics/base_vs_ft_quick.json"

    settings = get_settings()
    ft_model = args.ft_model or settings.domain_ft_model

    rows = []
    with open(args.test_file, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[: args.limit]
    print(f"comparing on {len(rows)} held-out examples with identical evidence\n")

    import httpx
    from openai import OpenAI

    client = OpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_key,
        timeout=httpx.Timeout(90.0, connect=5.0),
        max_retries=1,
    )

    base = evaluate(client, args.base_model, rows, args.max_tokens, "base")
    fine_tuned = evaluate(client, ft_model, rows, args.max_tokens, "ft")

    def delta(key: str) -> float:
        return round(fine_tuned[key] - base[key], 2)

    report = {
        "held_out_examples": len(rows),
        "base": base,
        "fine_tuned": fine_tuned,
        "delta": {
            key: delta(key)
            for key in ("component_recall", "numeric_fidelity", "hallucinated_rate",
                        "thinking_leak_rate", "mean_chars")
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'metric':<20} {'base':>10} {'fine-tuned':>12} {'delta':>10}")
    print("-" * 55)
    for key in ("component_recall", "numeric_fidelity", "hallucinated_rate",
                "thinking_leak_rate", "mean_chars"):
        print(f"{key:<20} {base[key]:>10} {fine_tuned[key]:>12} {delta(key):>+10}")
    print(f"\nreport -> {out_path}")

    markdown = out_path.with_suffix(".md")
    markdown.write_text(
        "# Base versus fine-tuned Nemotron\n\n"
        f"Identical evidence, {len(rows)} held-out examples. Qwen routing and the tool "
        "layer are fixed; only the synthesis model changes.\n\n"
        "| Metric | Base | Fine-tuned | Delta |\n|---|---:|---:|---:|\n"
        + "".join(
            f"| {key.replace('_', ' ')} | {base[key]} | {fine_tuned[key]} | {delta(key):+} |\n"
            for key in ("component_recall", "numeric_fidelity", "hallucinated_rate",
                        "thinking_leak_rate", "mean_chars")
        ),
        encoding="utf-8",
    )
    print(f"markdown -> {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
