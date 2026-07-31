"""Score the agent against the 15 public calibration questions.

    python scripts/eval_public.py                      # in-process
    python scripts/eval_public.py --endpoint http://10.0.0.5:5000

Mirrors the organizer harness: each grading component is checked independently
and awarded its own points, so partial credit shows up the same way. Grading
uses the same LLM judge model the organizers use (``agent-brain``) when it is
reachable, and falls back to strict literal matching of the numbers, dates and
labels in each expected fact when it is not.

The public questions are calibration only. They are never training data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[1] / "Participant_Package" / "public_questions.jsonl"

#: The lookbehind stops the hyphens in an ISO date being read as minus signs,
#: which would turn 2015-03-20 into the numbers -3 and -20.
_NUMBER = re.compile(r"(?<![\d.])-?\d[\d,]*\.?\d*")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2} [A-Z][a-z]{2} \d{4}\b")

JUDGE_PROMPT = """You are grading one component of an answer.

QUESTION: {question}
EXPECTED FACT: {fact}
CANDIDATE ANSWER: {answer}

Does the candidate answer state the expected fact? Equivalent number and date \
formats count as correct. Reply with exactly YES or NO."""


def load_cases() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical_numbers(text: str) -> set[str]:
    values = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        try:
            number = float(cleaned)
        except ValueError:
            continue
        values.add(f"{number:.4f}".rstrip("0").rstrip("."))
    return values


def literal_match(expected: str, answer: str) -> bool:
    """Strict offline check: every number and date in the fact must appear."""
    expected_numbers = _canonical_numbers(expected)
    answer_numbers = _canonical_numbers(answer)
    if expected_numbers and not expected_numbers.issubset(answer_numbers):
        return False

    for token in set(re.findall(r"\b[A-Z]{2,4}\.AX\b", expected)):
        if token not in answer:
            return False

    if not expected_numbers:
        keywords = [w.lower() for w in re.findall(r"\b[a-z]{4,}\b", expected.lower())]
        significant = [w for w in keywords if w not in {"the", "and", "that", "with", "from", "were"}]
        if significant:
            hits = sum(1 for w in significant if w in answer.lower())
            return hits >= max(1, len(significant) // 2)
    return True


class Judge:
    def __init__(self, use_llm: bool):
        self.client = None
        if use_llm:
            try:
                import httpx
                from openai import OpenAI

                settings = get_settings()
                self.client = OpenAI(
                    base_url=settings.litellm_base_url,
                    api_key=settings.litellm_key,
                    timeout=httpx.Timeout(30.0, connect=3.0),
                    max_retries=0,
                )
                self.model = settings.brain_model
                self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": "ok"}], max_tokens=3
                )
            except Exception as exc:
                print(f"judge model unavailable ({type(exc).__name__}); using literal matching", file=sys.stderr)
                self.client = None

    def verdict(self, question: str, fact: str, answer: str) -> bool:
        if self.client is None:
            return literal_match(fact, answer)
        try:
            reply = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, fact=fact, answer=answer)}],
                max_tokens=5,
                temperature=0.0,
            )
            return "YES" in (reply.choices[0].message.content or "").upper()
        except Exception:
            return literal_match(fact, answer)


def ask_local(question: str) -> tuple[str, float, dict]:
    from src.graph import AgentGraph

    global _GRAPH
    try:
        graph = _GRAPH
    except NameError:
        graph = _GRAPH = AgentGraph()
    started = time.time()
    response, state = graph.run(question)
    return response.answer, time.time() - started, {"path": state.path, "synth": state.synth_mode}


def ask_remote(endpoint: str, question: str) -> tuple[str, float, dict]:
    import httpx

    started = time.time()
    reply = httpx.post(f"{endpoint.rstrip('/')}/query", json={"question": question}, timeout=300)
    payload = reply.json()
    return payload.get("answer", ""), time.time() - started, {"steps": payload.get("steps")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Score a running server instead of the in-process graph")
    parser.add_argument("--no-judge", action="store_true", help="Force literal matching")
    parser.add_argument("--out", default="logs/public_eval.json")
    args = parser.parse_args()

    cases = load_cases()
    judge = Judge(use_llm=not args.no_judge)

    earned_total = max_total = 0.0
    by_difficulty: dict[str, list[float]] = {}
    rows = []

    for case in cases:
        question = case["prompt"]
        answer, latency, meta = (
            ask_remote(args.endpoint, question) if args.endpoint else ask_local(question)
        )

        components = case["grading"]["components"]
        earned = 0.0
        verdicts = []
        for component in components:
            ok = judge.verdict(question, component["expected_fact"], answer)
            verdicts.append({"component_id": component["component_id"], "passed": ok,
                             "points": component["points"]})
            if ok:
                earned += component["points"]

        maximum = float(case["grading"]["max_score"])
        # Organizer rule: over 60s loses 20% of earned points, over 300s scores zero.
        penalty = 0.0
        if latency > 300:
            earned, penalty = 0.0, earned
        elif latency > 60:
            penalty = earned * 0.20
            earned -= penalty

        earned_total += earned
        max_total += maximum
        by_difficulty.setdefault(case["difficulty"], []).append(earned / maximum * 100)

        status = "OK " if earned >= maximum - 1e-9 else ("PART" if earned > 0 else "MISS")
        print(f"{status} {case['id']} {earned:5.2f}/{maximum:<5.1f} {latency:5.1f}s  {question[:64]}")
        for verdict, component in zip(verdicts, components):
            if not verdict["passed"]:
                print(f"       missed {verdict['component_id']}: {component['expected_fact'][:96]}")
        rows.append({
            "id": case["id"], "difficulty": case["difficulty"], "datasets": case["datasets"],
            "earned": round(earned, 3), "max": maximum, "latency_s": round(latency, 2),
            "slow_penalty": round(penalty, 3), "answer": answer, "components": verdicts, **meta,
        })

    score = earned_total / max_total * 100 if max_total else 0.0
    print(f"\nhidden-question-style score: {earned_total:.2f}/{max_total:.0f} = {score:.1f}%")
    for difficulty in ("easy", "medium", "hard"):
        values = by_difficulty.get(difficulty)
        if values:
            print(f"  {difficulty:<7} {sum(values) / len(values):5.1f}%  ({len(values)} questions)")
    slow = [r for r in rows if r["latency_s"] > 60]
    if slow:
        print(f"  {len(slow)} responses over 60s incurred the slow penalty")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"score_pct": round(score, 2), "earned": round(earned_total, 2), "max": max_total,
         "judge": "llm" if judge.client else "literal", "cases": rows},
        indent=2), encoding="utf-8")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
