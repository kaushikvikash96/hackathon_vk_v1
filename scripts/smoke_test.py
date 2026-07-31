"""Pre-flight smoke test. Run this before every serving session.

    python scripts/smoke_test.py            # local checks only
    python scripts/smoke_test.py --endpoint http://10.0.0.5:5000

Checks, in order of what breaks a submission most severely:
  1. datasets load and the AFR index is present
  2. the tool layer returns the known-correct public-question values
  3. the model endpoints (brain and domain-ft) are reachable
  4. the graph answers a question end to end
  5. GET /health returns 200 and POST /query honours the contract
  6. three concurrent /query requests stay correct and independent
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import afr_index, data_asx, data_rba  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.tools import run_tool  # noqa: E402

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    marker = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]"}[status]
    print(f"{marker} {name}" + (f" - {detail}" if detail else ""), flush=True)


def check_datasets() -> None:
    settings = get_settings()
    try:
        rba, asx = data_rba.load(), data_asx.load()
        check("RBA loaded", PASS if len(rba) == 175 else FAIL, f"{len(rba)} records")
        check("ASX loaded", PASS if len(asx) == 18 else FAIL, f"{len(asx)} tickers")
    except Exception as exc:
        check("datasets load", FAIL, f"{type(exc).__name__}: {exc}")
        return

    if not afr_index.is_built():
        check("AFR index", FAIL, f"missing - run: python scripts/build_afr_index.py (DATA_ROOT={settings.data_root})")
        return
    index = afr_index.get_index()
    check("AFR index", PASS if index.document_count == 219538 else WARN,
          f"{index.document_count:,} documents")


def check_tools() -> None:
    cases = [
        ("rba count_changes", {"dataset": "rba", "metric": "count_changes"}, "41 of the 175"),
        ("rba max_hold_streak", {"dataset": "rba", "metric": "max_hold_streak"}, "1036 days"),
        ("asx 2018 ranking",
         {"dataset": "asx", "metric": "rank_annual_returns", "year": 2018, "exclude_tickers": ["TAH.AX"]},
         "BHP.AX +22.17%"),
        ("afr whole-word count",
         {"dataset": "afr", "metric": "count", "pattern": r"\bunemployment\b"}, "5,997"),
        ("cross coverage", {"dataset": "cross", "metric": "coverage"}, "2021"),
    ]
    for name, args, expected in cases:
        started = time.time()
        result = run_tool("query_data", args)
        elapsed = time.time() - started
        if not result.ok:
            check(name, FAIL, result.error or "error")
        elif expected not in result.summary:
            check(name, FAIL, f"expected {expected!r} in: {result.summary[:120]}")
        else:
            check(name, PASS, f"{elapsed:.2f}s")


def check_models() -> None:
    settings = get_settings()
    try:
        import httpx
        from openai import OpenAI

        client = OpenAI(
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_key,
            timeout=httpx.Timeout(20.0, connect=3.0),
            max_retries=0,
        )
    except Exception as exc:
        check("model client", FAIL, str(exc))
        return

    for label, model in (("brain (Qwen)", settings.brain_model), ("domain-ft (Nemotron)", settings.domain_ft_model)):
        if label.startswith("domain") and not settings.uses_fine_tuned_model:
            check(f"{label} reachable", WARN,
                  f"DOMAIN_PREDICT_MODE={settings.domain_predict_mode} - must be 'llm' for evaluation")
            continue
        try:
            started = time.time()
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "reply with: ok"}],
                max_tokens=5,
                temperature=0.0,
            )
            check(f"{label} reachable", PASS, f"{model} in {time.time() - started:.1f}s")
        except Exception as exc:
            check(f"{label} reachable", WARN, f"{model}: {type(exc).__name__} - agent will use its fallback")


def check_graph() -> None:
    from src.graph import AgentGraph

    graph = AgentGraph()
    question = "From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?"
    started = time.time()
    response, state = graph.run(question)
    elapsed = time.time() - started

    ok = all(token in response.answer for token in ("41", "20", "21"))
    check("graph end-to-end", PASS if ok else FAIL,
          f"{elapsed:.1f}s, path={'>'.join(state.path)}, synth={state.synth_mode}")
    check("latency under 60s", PASS if elapsed < 60 else FAIL, f"{elapsed:.1f}s")
    if state.used_fallback_planner:
        check("brain planning", WARN, "fallback planner used - Qwen was unreachable")


def check_endpoint(endpoint: str) -> None:
    import httpx

    try:
        response = httpx.get(f"{endpoint.rstrip('/')}/health", timeout=10)
        check("GET /health", PASS if response.status_code == 200 else FAIL,
              f"{response.status_code} {response.text[:80]}")
    except Exception as exc:
        check("GET /health", FAIL, f"{type(exc).__name__}: {exc}")
        return

    questions = [
        "What is the lowest cash-rate target in the RBA dataset, when did it first take effect, and how many decision records show that rate?",
        "Excluding Tabcorp, which ticker has the highest average daily volume over the full sample?",
        "What are the dimensions and common date range of the ASX dataset?",
    ]

    def ask(question: str) -> tuple[float, dict]:
        started = time.time()
        reply = httpx.post(f"{endpoint.rstrip('/')}/query", json={"question": question}, timeout=300)
        return time.time() - started, reply.json()

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(ask, questions))
    wall = time.time() - started

    answers = [payload.get("answer", "") for _, payload in outcomes]
    check("3 concurrent /query", PASS if all(answers) else FAIL,
          f"wall {wall:.1f}s, slowest {max(t for t, _ in outcomes):.1f}s")
    check("responses are distinct", PASS if len(set(answers)) == 3 else FAIL,
          "shared state would repeat an answer")
    for (elapsed, payload), question in zip(outcomes, questions):
        valid = isinstance(payload.get("answer"), str) and payload["answer"].strip()
        check(f"contract: {question[:42]}...", PASS if valid else FAIL, f"{elapsed:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Also test a running server, e.g. http://10.0.0.5:5000")
    parser.add_argument("--skip-models", action="store_true", help="Skip LiteLLM reachability checks")
    args = parser.parse_args()

    settings = get_settings()
    print(f"data_root={settings.data_root}")
    print(f"litellm={settings.litellm_base_url} brain={settings.brain_model} "
          f"domain={settings.domain_ft_model} mode={settings.domain_predict_mode}\n")

    check_datasets()
    check_tools()
    if not args.skip_models:
        check_models()
    check_graph()
    if args.endpoint:
        check_endpoint(args.endpoint)

    failures = [name for name, status, _ in _results if status == FAIL]
    warnings = [name for name, status, _ in _results if status == WARN]
    print(f"\n{len(_results) - len(failures) - len(warnings)} passed, "
          f"{len(warnings)} warnings, {len(failures)} failed")
    if failures:
        print("FAILED: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
