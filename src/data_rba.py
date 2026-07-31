"""RBA cash-rate decisions: deterministic structured queries.

Source: ``RBA-rates.csv`` (UTF-8 BOM), fields ``Effective Date``,
``Change % points``, ``Cash rate target%``. 175 decision records.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Iterable

from .config import get_settings
from .schemas import Fact, ToolResult
from .util import fmt_int, fmt_points, fmt_rate, iso, parse_date

TOOL = "query_data"
DATASET = "rba"

#: Records dated after this point are the organizer-flagged forward extension.
FORWARD_EXTENSION_FROM = date(2025, 1, 1)


@dataclass(frozen=True)
class Decision:
    effective: date
    change: float
    target: float


@lru_cache(maxsize=1)
def load() -> tuple[Decision, ...]:
    path = get_settings().rba_csv
    rows: list[Decision] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_date = (row.get("Effective Date") or "").strip()
            if not raw_date:
                continue
            rows.append(
                Decision(
                    effective=parse_date(raw_date),
                    change=float((row.get("Change % points") or "0").strip() or 0.0),
                    target=float((row.get("Cash rate target%") or "0").strip() or 0.0),
                )
            )
    rows.sort(key=lambda d: d.effective)
    return tuple(rows)


def _window(args: dict[str, Any]) -> tuple[date | None, date | None]:
    return parse_date(args.get("date_from")), parse_date(args.get("date_to"))


def _select(start: date | None, end: date | None) -> list[Decision]:
    rows = load()
    return [
        r
        for r in rows
        if (start is None or r.effective >= start) and (end is None or r.effective <= end)
    ]


def _window_label(start: date | None, end: date | None) -> str:
    if start and end:
        return f" between {iso(start)} and {iso(end)}"
    if start:
        return f" from {iso(start)}"
    if end:
        return f" up to {iso(end)}"
    return ""


def _extension_note(rows: Iterable[Decision]) -> list[str]:
    if any(r.effective >= FORWARD_EXTENSION_FROM for r in rows):
        return [
            "The tail of the RBA file extends into 2026; the organizers flag those "
            "records as a forward extension rather than confirmed history."
        ]
    return []


def rate_on(when: date) -> Decision | None:
    """Decision in force ON or BEFORE ``when`` (never the next future decision)."""
    applicable = [r for r in load() if r.effective <= when]
    return applicable[-1] if applicable else None


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _coverage(args: dict[str, Any]) -> ToolResult:
    rows = load()
    changes = [r for r in rows if r.change != 0]
    facts = [
        Fact("record_count", len(rows), f"The RBA dataset holds {fmt_int(len(rows))} decision records"),
        Fact("first_date", iso(rows[0].effective), f"the first effective date is {iso(rows[0].effective)}"),
        Fact("last_date", iso(rows[-1].effective), f"the last effective date is {iso(rows[-1].effective)}"),
        Fact("change_count", len(changes), f"{fmt_int(len(changes))} of those records changed the rate"),
    ]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f"RBA coverage: {len(rows)} decision records from {iso(rows[0].effective)} "
            f"to {iso(rows[-1].effective)}, {len(changes)} of which changed the rate."
        ),
        facts=facts,
        notes=_extension_note(rows),
    )


def _count(args: dict[str, Any]) -> ToolResult:
    start, end = _window(args)
    rows = _select(start, end)
    label = _window_label(start, end)
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"{len(rows)} RBA decision records{label}.",
        facts=[Fact("record_count", len(rows), f"there are {fmt_int(len(rows))} RBA decision records{label}")],
        notes=_extension_note(rows),
    )


def _count_changes(args: dict[str, Any]) -> ToolResult:
    start, end = _window(args)
    rows = _select(start, end)
    changes = [r for r in rows if r.change != 0]
    ups = [r for r in changes if r.change > 0]
    downs = [r for r in changes if r.change < 0]
    label = _window_label(start, end)

    by_year: dict[int, dict[str, int]] = {}
    for r in changes:
        bucket = by_year.setdefault(r.effective.year, {"increases": 0, "decreases": 0})
        bucket["increases" if r.change > 0 else "decreases"] += 1

    summary = (
        f"{len(changes)} of the {len(rows)} decision records{label} changed the rate: "
        f"{len(ups)} increases and {len(downs)} decreases."
    )
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=summary,
        facts=[
            Fact("changes", len(changes), f"{fmt_int(len(changes))} of the {fmt_int(len(rows))} decision records changed the rate"),
            Fact("increases", len(ups), f"{fmt_int(len(ups))} increases"),
            Fact("decreases", len(downs), f"{fmt_int(len(downs))} decreases"),
        ],
        notes=_extension_note(rows),
        detail={"by_year": {str(k): v for k, v in sorted(by_year.items())}},
    )


def _count_directional(args: dict[str, Any], increases: bool) -> ToolResult:
    start, end = _window(args)
    rows = _select(start, end)
    picked = [r for r in rows if (r.change > 0 if increases else r.change < 0)]
    word = "increases" if increases else "decreases"
    total = sum(r.change for r in picked)
    label = _window_label(start, end)
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"{len(picked)} rate {word}{label}, totalling {fmt_points(total)}.",
        facts=[
            Fact(word, len(picked), f"there were {fmt_int(len(picked))} rate {word}{label}"),
            Fact("cumulative_change", round(total, 2), f"totalling {fmt_points(total)}"),
        ],
        notes=_extension_note(rows),
        detail={"dates": [iso(r.effective) for r in picked]},
    )


def _extremes(args: dict[str, Any]) -> ToolResult:
    start, end = _window(args)
    rows = _select(start, end)
    if not rows:
        return ToolResult.failure(TOOL, args, "no RBA records in the requested window")

    highest = max(r.target for r in rows)
    lowest = min(r.target for r in rows)
    high_rows = [r for r in rows if r.target == highest]
    low_rows = [r for r in rows if r.target == lowest]

    facts = [
        Fact("highest_target", highest, f"the highest cash-rate target is {fmt_rate(highest)}"),
        Fact("highest_first_date", iso(high_rows[0].effective), f"first effective on {iso(high_rows[0].effective)}"),
        Fact("highest_record_count", len(high_rows), f"shown by {fmt_int(len(high_rows))} decision records"),
        Fact("lowest_target", lowest, f"the lowest cash-rate target is {fmt_rate(lowest)}"),
        Fact("lowest_first_date", iso(low_rows[0].effective), f"first effective on {iso(low_rows[0].effective)}"),
        Fact("lowest_record_count", len(low_rows), f"shown by {fmt_int(len(low_rows))} decision records"),
    ]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f"Highest target {fmt_rate(highest)} (first {iso(high_rows[0].effective)}, "
            f"{len(high_rows)} records); lowest target {fmt_rate(lowest)} "
            f"(first {iso(low_rows[0].effective)}, {len(low_rows)} records)."
        ),
        facts=facts,
        notes=_extension_note(rows),
    )


def _max_hold_streak(args: dict[str, Any]) -> ToolResult:
    changes = [r for r in load() if r.change != 0]
    if len(changes) < 2:
        return ToolResult.failure(TOOL, args, "not enough rate changes to measure a hold streak")

    best_days = -1
    best_pair: tuple[Decision, Decision] | None = None
    for previous, current in zip(changes, changes[1:]):
        days = (current.effective - previous.effective).days
        if days > best_days:
            best_days, best_pair = days, (previous, current)

    assert best_pair is not None
    start_rec, end_rec = best_pair
    facts = [
        Fact("days", best_days, f"the longest stretch between two rate changes was {fmt_int(best_days)} days"),
        Fact("start_date", iso(start_rec.effective), f"lasting from {iso(start_rec.effective)}"),
        Fact("end_date", iso(end_rec.effective), f"to {iso(end_rec.effective)}"),
        Fact("rate_during_hold", start_rec.target, f"during which the rate held at {fmt_rate(start_rec.target)}"),
        Fact("rate_after", end_rec.target, f"before changing to {fmt_rate(end_rec.target)}"),
    ]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f"Longest stretch between two rate changes: {best_days} days, "
            f"{iso(start_rec.effective)} to {iso(end_rec.effective)}, held at "
            f"{fmt_rate(start_rec.target)} before changing to {fmt_rate(end_rec.target)}."
        ),
        facts=facts,
    )


def _lookup_rate(args: dict[str, Any]) -> ToolResult:
    when = parse_date(args.get("date") or args.get("date_from") or args.get("date_to"))
    if when is None:
        return ToolResult.failure(TOOL, args, "lookup_rate requires a 'date' argument")

    record = rate_on(when)
    if record is None:
        return ToolResult.failure(TOOL, args, f"no RBA decision on or before {iso(when)}")

    facts = [
        Fact("target", record.target, f"the RBA cash-rate target in force was {fmt_rate(record.target)}"),
        Fact("effective_date", iso(record.effective), f"set by the decision effective {iso(record.effective)}"),
    ]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f"On {iso(when)} the cash-rate target in force was {fmt_rate(record.target)}, "
            f"set by the decision effective {iso(record.effective)}."
        ),
        facts=facts,
        detail={"change_at_that_decision": record.change},
    )


def _changes_in_period(args: dict[str, Any]) -> ToolResult:
    start, end = _window(args)
    rows = [r for r in _select(start, end) if r.change != 0]
    label = _window_label(start, end)
    listed = [
        {"date": iso(r.effective), "change": r.change, "target": r.target} for r in rows
    ]
    parts = ", ".join(f"{iso(r.effective)} {r.change:+.2f} to {fmt_rate(r.target)}" for r in rows[:20])
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"{len(rows)} rate changes{label}: {parts}" if rows else f"No rate changes{label}.",
        facts=[Fact("change_count", len(rows), f"there were {fmt_int(len(rows))} rate changes{label}")],
        notes=_extension_note(rows),
        detail={"changes": listed},
    )


def _cycle_summary(args: dict[str, Any]) -> ToolResult:
    start, end = _window(args)
    direction = str(args.get("direction", "all")).lower()
    rows = [r for r in _select(start, end) if r.change != 0]
    if direction in {"hikes", "increases", "up", "tightening"}:
        rows = [r for r in rows if r.change > 0]
        word = "hikes"
    elif direction in {"cuts", "decreases", "down", "easing"}:
        rows = [r for r in rows if r.change < 0]
        word = "cuts"
    else:
        word = "changes"

    if not rows:
        return ToolResult(
            tool=TOOL,
            args=args,
            summary=f"No {word} found{_window_label(start, end)}.",
            facts=[Fact("count", 0, f"there were no {word}{_window_label(start, end)}")],
        )

    cumulative = round(sum(r.change for r in rows), 2)
    before = rate_on(rows[0].effective - timedelta(days=1))
    final_record = rate_on(end) if end else load()[-1]

    by_year: dict[int, int] = {}
    for r in rows:
        by_year[r.effective.year] = by_year.get(r.effective.year, 0) + 1
    year_text = ", ".join(f"{count} in {year}" for year, count in sorted(by_year.items()))

    facts = [
        Fact("count", len(rows), f"there were {fmt_int(len(rows))} {word} ({year_text})"),
        Fact("cumulative_change", cumulative, f"totalling {fmt_points(cumulative)}"),
        Fact("first_date", iso(rows[0].effective), f"running from {iso(rows[0].effective)}"),
        Fact("last_date", iso(rows[-1].effective), f"to {iso(rows[-1].effective)}"),
    ]
    if before is not None:
        facts.append(
            Fact("target_before", before.target, f"the target immediately before the first {word[:-1]} was {fmt_rate(before.target)}")
        )
    if final_record is not None:
        facts.append(
            Fact("target_after", final_record.target, f"the target at the end of the period was {fmt_rate(final_record.target)}")
        )

    summary = (
        f"{len(rows)} {word} ({year_text}) between {iso(rows[0].effective)} and "
        f"{iso(rows[-1].effective)}, totalling {fmt_points(cumulative)}"
    )
    if before is not None and final_record is not None:
        summary += f", taking the target from {fmt_rate(before.target)} to {fmt_rate(final_record.target)}"
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=summary + ".",
        facts=facts,
        notes=_extension_note(rows),
        detail={"by_year": {str(k): v for k, v in sorted(by_year.items())}},
    )


def _list(args: dict[str, Any]) -> ToolResult:
    start, end = _window(args)
    limit = min(int(args.get("limit", 25)), 60)
    rows = _select(start, end)[:limit]
    listed = [{"date": iso(r.effective), "change": r.change, "target": r.target} for r in rows]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"Listing {len(rows)} RBA records (limit {limit}). Prefer a specific metric for counts and extremes.",
        facts=[],
        detail={"records": listed},
    )


_METRICS = {
    "coverage": _coverage,
    "count": _count,
    "count_changes": _count_changes,
    "count_increases": lambda a: _count_directional(a, True),
    "count_decreases": lambda a: _count_directional(a, False),
    "extremes": _extremes,
    "max_hold_streak": _max_hold_streak,
    "lookup_rate": _lookup_rate,
    "changes_in_period": _changes_in_period,
    "cycle_summary": _cycle_summary,
    "list": _list,
}

METRICS = tuple(_METRICS)


def run(metric: str, args: dict[str, Any]) -> ToolResult:
    handler = _METRICS.get(metric)
    if handler is None:
        return ToolResult.failure(
            TOOL, args, f"unknown rba metric {metric!r}; available: {', '.join(METRICS)}"
        )
    return handler(args)
