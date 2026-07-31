"""Qwen `agent-brain` client: planning, tool-call generation, result review.

The brain owns tool selection. This module never decides *what* the answer is -
it only turns the conversation into validated tool-call requests.

If the LiteLLM proxy is unreachable or times out, a deterministic keyword
planner takes over so the agent still returns a grounded answer instead of
failing the request. Every fallback is recorded in the response trace and logs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .tools import QUERY_DATA, RETRIEVE, TOOL_SCHEMAS

log = logging.getLogger(__name__)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class BrainTurn:
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    content: str = ""
    used_fallback: bool = False
    error: str | None = None

    @property
    def done(self) -> bool:
        return not self.tool_calls


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


class BrainClient:
    """Thin OpenAI-compatible wrapper around the LiteLLM `agent-brain` alias."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        self._unavailable_reason: str | None = None

    def _get_client(self):
        if self._client is None:
            import httpx
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.litellm_base_url,
                api_key=self.settings.litellm_key,
                # A short connect timeout keeps an unreachable proxy from
                # burning seconds of the response budget before falling back.
                timeout=httpx.Timeout(self.settings.brain_timeout_s, connect=2.0),
                max_retries=0,
            )
        return self._client

    def plan(self, messages: list[dict[str, Any]], timeout: float | None = None) -> BrainTurn:
        """One planning turn. Returns requested tool calls, or none when done."""
        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.brain_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=self.settings.brain_max_tokens,
                timeout=timeout or self.settings.brain_timeout_s,
            )
        except Exception as exc:
            log.warning("brain unavailable (%s: %s); using fallback planner", type(exc).__name__, exc)
            return BrainTurn(error=f"{type(exc).__name__}: {exc}", used_fallback=True)

        choice = response.choices[0].message
        calls = [
            ToolCallRequest(
                id=getattr(call, "id", None) or f"call_{i}",
                name=call.function.name,
                args=_parse_args(call.function.arguments),
            )
            for i, call in enumerate(getattr(choice, "tool_calls", None) or [])
        ]
        return BrainTurn(tool_calls=calls, content=(choice.content or "").strip())


# --------------------------------------------------------------------------
# Deterministic fallback planner
# --------------------------------------------------------------------------
_QUOTED = re.compile(r"[\"“]([^\"”]{6,200})[\"”]")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DMY_DATE = re.compile(r"\b(\d{1,2})\s+([A-Z][a-z]{2})[a-z]*\.?\s+((?:19|20)\d{2})\b")
#: "5-12 Jun 2019" and "30 Nov to 7 Dec 2020" - ranges written without a year on
#: the first date, which is how the calibration questions phrase event windows.
_DMY_RANGE = re.compile(
    r"\b(\d{1,2})\s*(?:-|to|and)\s*(\d{1,2})\s+([A-Z][a-z]{2})[a-z]*\.?\s+((?:19|20)\d{2})\b"
)
_SPLIT_RANGE = re.compile(
    r"\b(\d{1,2})\s+([A-Z][a-z]{2})[a-z]*\.?\s+to\s+(\d{1,2})\s+([A-Z][a-z]{2})[a-z]*\.?\s+((?:19|20)\d{2})\b"
)
_WHOLE_WORD_TERM = re.compile(r"whole-word\s+([A-Za-z][\w-]*)", re.I)
_TICKER_MENTION = re.compile(r"\b([A-Z]{2,4})\.AX\b")

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

_EXCLUDE_TABCORP = {"exclude_tickers": ["TAH.AX"]}


def extract_dates(question: str) -> list[str]:
    """Every date in a question, normalised to ISO and in order of appearance."""
    found: list[tuple[int, str]] = []

    for match in _ISO_DATE.finditer(question):
        found.append((match.start(), match.group(0)))

    for match in _SPLIT_RANGE.finditer(question):  # "30 Nov to 7 Dec 2020"
        day_a, month_a, day_b, month_b, year = match.groups()
        if month_a in _MONTHS and month_b in _MONTHS:
            found.append((match.start(), f"{year}-{_MONTHS[month_a]:02d}-{int(day_a):02d}"))
            found.append((match.start() + 1, f"{year}-{_MONTHS[month_b]:02d}-{int(day_b):02d}"))

    for match in _DMY_RANGE.finditer(question):  # "5-12 Jun 2019"
        day_a, day_b, month, year = match.groups()
        if month in _MONTHS:
            found.append((match.start(), f"{year}-{_MONTHS[month]:02d}-{int(day_a):02d}"))
            found.append((match.start() + 1, f"{year}-{_MONTHS[month]:02d}-{int(day_b):02d}"))

    consumed = {pos for pos, _ in found}
    for match in _DMY_DATE.finditer(question):  # "5 Jun 2019"
        day, month, year = match.groups()
        if month in _MONTHS and match.start() not in consumed:
            found.append((match.start(), f"{year}-{_MONTHS[month]:02d}-{int(day):02d}"))

    ordered, seen = [], set()
    for _, value in sorted(found):
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def fallback_plan(question: str) -> list[ToolCallRequest]:
    """Keyword routing used only when the brain cannot be reached."""
    q = question.lower()
    years = [int(m.group(0)) for m in _YEAR.finditer(question)]
    dates = extract_dates(question)
    named_tickers = [f"{t}.AX" for t in _TICKER_MENTION.findall(question)]
    calls: list[dict[str, Any]] = []

    def add(name: str, **args: Any) -> None:
        calls.append({"name": name, "args": args})

    wants_tabcorp_exclusion = "tabcorp" in q and ("exclud" in q or "non-tabcorp" in q)
    exclusion = dict(_EXCLUDE_TABCORP) if wants_tabcorp_exclusion else {}

    if "support" in q and ("dataset" in q or "analysis" in q or "can the" in q):
        add(QUERY_DATA, dataset="cross", metric="coverage")

    quoted = _QUOTED.search(question)
    if quoted:
        args: dict[str, Any] = {"headline": quoted.group(1)}
        if dates:
            args["date"] = dates[0]
        add(RETRIEVE, **args)
        if "rba" in q or "cash-rate" in q or "cash rate" in q:
            add(QUERY_DATA, dataset="rba", metric="lookup_rate",
                date=dates[0] if dates else (f"{years[0]}-12-31" if years else None))
        if len(dates) >= 3 and ("basket" in q or "return" in q):
            add(QUERY_DATA, dataset="asx", metric="basket_return",
                date_from=dates[1], date_to=dates[2], **exclusion)

    if any(k in q for k in ("increase", "decrease", "changed the rate", "how many cash-rate")):
        add(QUERY_DATA, dataset="rba", metric="count_changes")
    if "longest" in q and any(k in q for k in ("unchanged", "held", "stretch", "hold")):
        add(QUERY_DATA, dataset="rba", metric="max_hold_streak")
    if ("highest" in q or "lowest" in q) and ("cash-rate" in q or "cash rate" in q or "target" in q):
        add(QUERY_DATA, dataset="rba", metric="extremes")
    if any(k in q for k in ("tightening", "easing", "cuts occurred", "hikes occurred", "cut count")):
        direction = "hikes" if ("tighten" in q or "hike" in q) else "cuts"
        if len(years) >= 2:
            add(QUERY_DATA, dataset="rba", metric="cycle_summary", direction=direction,
                date_from=f"{min(years)}-01-01", date_to=f"{max(years)}-12-31")
        elif years:
            add(QUERY_DATA, dataset="rba", metric="cycle_summary", direction=direction,
                date_from=f"{years[0]}-01-01", date_to=f"{years[0]}-12-31")

    if "drawdown" in q:
        add(QUERY_DATA, dataset="asx", metric="max_drawdown", top_n=3, **exclusion)
    if "volume" in q:
        add(QUERY_DATA, dataset="asx", metric="avg_volume", **exclusion)
    if "correlation" in q:
        add(QUERY_DATA, dataset="asx", metric="correlation", **exclusion)
    if "volatility" in q:
        add(QUERY_DATA, dataset="asx", metric="volatility", **exclusion)
    if "dimensions" in q or ("date range" in q and "asx" in q):
        add(QUERY_DATA, dataset="asx", metric="coverage")
    if "return" in q and not any(c["args"].get("dataset") == "asx" for c in calls):
        if "best" in q or "worst" in q or "rank" in q:
            add(QUERY_DATA, dataset="asx", metric="rank_annual_returns",
                year=years[0] if years else None,
                top_n=3 if ("rank" in q or "three" in q) else None, **exclusion)
        elif len(dates) >= 2:
            add(QUERY_DATA, dataset="asx", metric="basket_return",
                date_from=dates[0], date_to=dates[1],
                report_tickers=named_tickers or None, **exclusion)
        elif named_tickers and years:
            add(QUERY_DATA, dataset="asx", metric="annual_return",
                year=years[0], tickers=named_tickers)
        elif years:
            add(QUERY_DATA, dataset="asx", metric="basket_return", year=years[0], **exclusion)

    if "cut" in q and "rba" in q and len(dates) >= 1 and not any(
        c["args"].get("metric") in {"lookup_rate", "changes_in_period"} for c in calls
    ):
        add(QUERY_DATA, dataset="rba", metric="changes_in_period",
            date_from=f"{min(years)}-01-01" if years else dates[0],
            date_to=f"{max(years)}-12-31" if years else dates[-1])

    term = _WHOLE_WORD_TERM.search(question)
    if term:
        add(QUERY_DATA, dataset="afr", metric="count",
            pattern=rf"\b{re.escape(term.group(1))}\b", group_by="year_and_month",
            **({"year": years[0]} if len(years) == 1 else {}))
    elif "afr" in q and ("count" in q or "records" in q or "articles" in q):
        add(QUERY_DATA, dataset="afr", metric="coverage")

    if not calls:
        add(QUERY_DATA, dataset="cross", metric="coverage")

    return [
        ToolCallRequest(id=f"fallback_{i}", name=c["name"],
                        args={k: v for k, v in c["args"].items() if v is not None})
        for i, c in enumerate(calls[:3])
    ]
