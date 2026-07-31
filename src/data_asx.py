"""ASX-18 daily OHLCV: deterministic structured queries.

Source: 18 JSONL files, fields ``ticker, date, open, high, low, close, volume``,
1,774 trading days each, 2015-01-02 to 2021-12-30.

Conventions (these match the reference derivations for the public questions):
  * returns are close-to-close, first-to-last within the requested window;
  * a "basket" is the arithmetic mean of its constituents' returns;
  * drawdown is measured against the running peak close and reports the peak date.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Iterable, Sequence

from .config import get_settings
from .schemas import Fact, ToolResult
from .util import fmt_int, fmt_num, fmt_pct, iso, parse_date

TOOL = "query_data"
DATASET = "asx"

TABCORP = "TAH.AX"

#: Company names as they appear in questions -> ticker symbol.
ALIASES = {
    "agl": "AGL.AX", "amp": "AMP.AX", "anz": "ANZ.AX",
    "aurizon": "AZJ.AX", "azj": "AZJ.AX",
    "bhp": "BHP.AX", "cba": "CBA.AX", "commonwealth bank": "CBA.AX",
    "cromwell": "CMW.AX", "cmw": "CMW.AX",
    "gpt": "GPT.AX", "iag": "IAG.AX", "nab": "NAB.AX",
    "national australia bank": "NAB.AX",
    "qantas": "QAN.AX", "qan": "QAN.AX", "qbe": "QBE.AX",
    "rio": "RIO.AX", "rio tinto": "RIO.AX",
    "stockland": "SGP.AX", "sgp": "SGP.AX",
    "suncorp": "SUN.AX", "sun": "SUN.AX",
    "tabcorp": TABCORP, "tah": TABCORP,
    "tpg": "TPG.AX", "transurban": "TCL.AX", "tcl": "TCL.AX",
}

TABCORP_NOTE = (
    "Tabcorp (TAH.AX) carries a known starting-price artifact; the organizers "
    "flag its full-sample return as a data artifact rather than a genuine return."
)


@dataclass(frozen=True)
class Bar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@lru_cache(maxsize=1)
def load() -> dict[str, tuple[Bar, ...]]:
    series: dict[str, list[Bar]] = {}
    for path in sorted(get_settings().asx_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                series.setdefault(row["ticker"], []).append(
                    Bar(
                        day=parse_date(row["date"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
    return {t: tuple(sorted(bars, key=lambda b: b.day)) for t, bars in sorted(series.items())}


def resolve_ticker(name: str) -> str | None:
    key = str(name).strip()
    if not key:
        return None
    upper = key.upper()
    data = load()
    if upper in data:
        return upper
    if f"{upper}.AX" in data:
        return f"{upper}.AX"
    return ALIASES.get(key.lower())


def _resolve_many(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [v for v in values.replace(";", ",").split(",") if v.strip()]
    resolved = []
    for v in values:
        ticker = resolve_ticker(v)
        if ticker:
            resolved.append(ticker)
    return resolved


def universe(args: dict[str, Any]) -> list[str]:
    """Ticker set for an operation, honouring ``tickers`` / ``exclude_tickers``."""
    data = load()
    requested = _resolve_many(args.get("tickers") or args.get("ticker"))
    selected = requested or list(data)
    excluded = set(_resolve_many(args.get("exclude_tickers")))
    return [t for t in selected if t not in excluded]


def _window(args: dict[str, Any]) -> tuple[date | None, date | None, bool]:
    """Return (start, end, exact_dates).

    ``exact_dates`` is True only when the caller named specific dates. An event
    window like 5-12 Jun 2019 is measured from the closes on those exact dates;
    a calendar year is measured first-to-last *inside* the year, never from the
    previous December's close.
    """
    start, end = parse_date(args.get("date_from")), parse_date(args.get("date_to"))
    exact_dates = start is not None and end is not None
    year = args.get("year")
    if year and not (start or end):
        year = int(year)
        start, end = date(year, 1, 1), date(year, 12, 31)
        exact_dates = False
    return start, end, exact_dates


def _slice(ticker: str, start: date | None, end: date | None) -> list[Bar]:
    return [
        b
        for b in load()[ticker]
        if (start is None or b.day >= start) and (end is None or b.day <= end)
    ]


def _bar_on(ticker: str, when: date) -> tuple[Bar | None, bool]:
    """Bar for an exact date, else the most recent prior trading day."""
    bars = load()[ticker]
    exact = [b for b in bars if b.day == when]
    if exact:
        return exact[0], True
    prior = [b for b in bars if b.day < when]
    return (prior[-1], False) if prior else (None, False)


def _pct_return(first: float, last: float) -> float:
    return (last / first - 1.0) * 100.0


def _window_return(
    ticker: str, start: date | None, end: date | None, exact_dates: bool = True
) -> tuple[float | None, list[str]]:
    notes: list[str] = []
    if start and end and exact_dates:
        first, exact_a = _bar_on(ticker, start)
        last, exact_b = _bar_on(ticker, end)
        if first is None or last is None:
            return None, [f"{ticker} has no price data covering {start} to {end}"]
        if not exact_a:
            notes.append(f"{ticker} has no trade on {iso(start)}; used the prior session {iso(first.day)}")
        if not exact_b:
            notes.append(f"{ticker} has no trade on {iso(end)}; used the prior session {iso(last.day)}")
        return _pct_return(first.close, last.close), notes

    bars = _slice(ticker, start, end)
    if len(bars) < 2:
        return None, [f"{ticker} has fewer than two sessions in the requested window"]
    return _pct_return(bars[0].close, bars[-1].close), notes


def _label(start: date | None, end: date | None) -> str:
    if start and end:
        if start.year == end.year and (start.month, start.day) == (1, 1) and (end.month, end.day) == (12, 31):
            return f" in {start.year}"
        return f" from {iso(start)} to {iso(end)}"
    return " over the full sample"


def _tabcorp_note(tickers: Iterable[str]) -> list[str]:
    return [TABCORP_NOTE] if TABCORP in set(tickers) else []


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _coverage(args: dict[str, Any]) -> ToolResult:
    data = load()
    counts = {t: len(bars) for t, bars in data.items()}
    first = min(bars[0].day for bars in data.values())
    last = max(bars[-1].day for bars in data.values())
    row_counts = sorted(set(counts.values()))
    rows_text = fmt_int(row_counts[0]) if len(row_counts) == 1 else f"{row_counts[0]}-{row_counts[-1]}"
    facts = [
        Fact("ticker_count", len(data), f"there are {len(data)} ticker files"),
        Fact("rows_per_ticker", row_counts[0] if len(row_counts) == 1 else row_counts, f"each containing {rows_text} rows"),
        Fact("first_date", iso(first), f"covering {iso(first)}"),
        Fact("last_date", iso(last), f"through {iso(last)}"),
    ]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"ASX coverage: {len(data)} tickers, {rows_text} rows each, {iso(first)} to {iso(last)}.",
        facts=facts,
        detail={"tickers": sorted(data)},
    )


def _return_metric(args: dict[str, Any], forced_window: bool) -> ToolResult:
    start, end, exact_dates = _window(args)
    if not forced_window:
        start = end = None
        exact_dates = False
    tickers = universe(args)
    if not tickers:
        return ToolResult.failure(TOOL, args, "no matching tickers")

    results: dict[str, float] = {}
    notes: list[str] = []
    for ticker in tickers:
        value, ticker_notes = _window_return(ticker, start, end, exact_dates)
        notes.extend(ticker_notes)
        if value is not None:
            results[ticker] = value
    if not results:
        return ToolResult.failure(TOOL, args, "no return could be computed for the requested window")

    label = _label(start, end)
    facts = [
        Fact(f"return_{t}", round(v, 4), f"{t} returned {fmt_pct(v)}{label}")
        for t, v in sorted(results.items(), key=lambda kv: -kv[1])
    ]
    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    summary = "; ".join(f"{t} {fmt_pct(v)}" for t, v in ranked[:8])
    if len(ranked) > 8:
        summary += f"; ...({len(ranked)} tickers total)"
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"Returns{label}: {summary}.",
        facts=facts[:8],
        notes=notes + _tabcorp_note(results),
        detail={"returns": {t: round(v, 4) for t, v in ranked}},
    )


def _rank_annual_returns(args: dict[str, Any]) -> ToolResult:
    start, end, exact_dates = _window(args)
    tickers = universe(args)
    results: dict[str, float] = {}
    notes: list[str] = []
    for ticker in tickers:
        value, ticker_notes = _window_return(ticker, start, end, exact_dates)
        notes.extend(ticker_notes)
        if value is not None:
            results[ticker] = value
    if not results:
        return ToolResult.failure(TOOL, args, "no returns could be computed")

    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    label = _label(start, end)
    facts = [
        Fact("best", ranked[0][0], f"{ranked[0][0]} was best at {fmt_pct(ranked[0][1])}{label}"),
        Fact("worst", ranked[-1][0], f"{ranked[-1][0]} was worst at {fmt_pct(ranked[-1][1])}{label}"),
    ]
    # Positions are only stated when the question actually asked for a ranking,
    # so a "best and worst" answer is not padded with unrequested placings.
    if args.get("top_n"):
        for position, (ticker, value) in enumerate(ranked[: int(args["top_n"])], start=1):
            facts.append(Fact(f"rank_{position}", ticker, f"{position}) {ticker} {fmt_pct(value)}"))

    summary = ", ".join(f"{i}) {t} {fmt_pct(v)}" for i, (t, v) in enumerate(ranked[:5], start=1))
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"Ranked returns{label}: {summary}. Worst: {ranked[-1][0]} {fmt_pct(ranked[-1][1])}.",
        facts=facts,
        notes=notes + _tabcorp_note(results),
        detail={"ranking": [{"ticker": t, "return_pct": round(v, 4)} for t, v in ranked]},
    )


def _basket_return(args: dict[str, Any]) -> ToolResult:
    start, end, exact_dates = _window(args)
    tickers = universe(args)
    if not tickers:
        return ToolResult.failure(TOOL, args, "no matching tickers for the basket")

    per_ticker: dict[str, float] = {}
    notes: list[str] = []
    for ticker in tickers:
        value, ticker_notes = _window_return(ticker, start, end, exact_dates)
        notes.extend(ticker_notes)
        if value is not None:
            per_ticker[ticker] = value
    if not per_ticker:
        return ToolResult.failure(TOOL, args, "no constituent returns could be computed")

    basket = sum(per_ticker.values()) / len(per_ticker)
    label = _label(start, end)
    verb = "rose" if basket >= 0 else "fell"
    facts = [
        Fact(
            "basket_return",
            round(basket, 4),
            f"the {len(per_ticker)}-stock basket {verb} {fmt_pct(basket)}{label}",
        )
    ]
    for ticker in _resolve_many(args.get("report_tickers")):
        if ticker in per_ticker:
            facts.append(
                Fact(f"return_{ticker}", round(per_ticker[ticker], 4), f"{ticker} {fmt_pct(per_ticker[ticker])}")
            )

    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f"Basket of {len(per_ticker)} tickers {verb} {fmt_pct(basket)}{label} "
            f"(arithmetic mean of constituent close-to-close returns)."
        ),
        facts=facts,
        notes=notes + _tabcorp_note(per_ticker),
        detail={"constituents": {t: round(v, 4) for t, v in sorted(per_ticker.items())}},
    )


def _max_drawdown(args: dict[str, Any]) -> ToolResult:
    start, end, exact_dates = _window(args)
    tickers = universe(args)
    results: list[tuple[str, float, date, date]] = []
    for ticker in tickers:
        bars = _slice(ticker, start, end)
        if len(bars) < 2:
            continue
        peak, peak_day = bars[0].close, bars[0].day
        worst, worst_peak_day, worst_day = 0.0, bars[0].day, bars[0].day
        for bar in bars:
            if bar.close > peak:
                peak, peak_day = bar.close, bar.day
            drop = (bar.close / peak - 1.0) * 100.0
            if drop < worst:
                worst, worst_peak_day, worst_day = drop, peak_day, bar.day
        results.append((ticker, worst, worst_peak_day, worst_day))
    if not results:
        return ToolResult.failure(TOOL, args, "no drawdown could be computed")

    results.sort(key=lambda row: row[1])
    top_n = int(args.get("top_n", 3))
    facts = [
        Fact(
            f"rank_{position}",
            ticker,
            f"{position}) {ticker} {fmt_pct(value)}, {iso(peak_day)} to {iso(trough_day)}",
        )
        for position, (ticker, value, peak_day, trough_day) in enumerate(results[:top_n], start=1)
    ]
    summary = "; ".join(
        f"{i}) {t} {fmt_pct(v)} {iso(p)} to {iso(q)}" for i, (t, v, p, q) in enumerate(results[:5], start=1)
    )
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"Worst maximum drawdowns{_label(start, end)}: {summary}.",
        facts=facts,
        notes=_tabcorp_note(t for t, *_ in results),
        detail={
            "drawdowns": [
                {"ticker": t, "drawdown_pct": round(v, 4), "peak_date": iso(p), "trough_date": iso(q)}
                for t, v, p, q in results
            ]
        },
    )


def _avg_volume(args: dict[str, Any]) -> ToolResult:
    start, end, exact_dates = _window(args)
    tickers = universe(args)
    results: dict[str, float] = {}
    for ticker in tickers:
        bars = _slice(ticker, start, end)
        if bars:
            results[ticker] = sum(b.volume for b in bars) / len(bars)
    if not results:
        return ToolResult.failure(TOOL, args, "no volume data in the requested window")

    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    label = _label(start, end)
    facts = [
        Fact(
            "highest_avg_volume",
            ranked[0][0],
            f"{ranked[0][0]} has the highest average daily volume at {fmt_num(ranked[0][1], 2)} shares per trading day{label}",
        )
    ]
    if len(ranked) > 1:
        facts.append(
            Fact(
                "lowest_avg_volume",
                ranked[-1][0],
                f"{ranked[-1][0]} has the lowest average daily volume at {fmt_num(ranked[-1][1], 2)} shares per trading day",
            )
        )
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"Average daily volume{label}: " + "; ".join(f"{t} {fmt_num(v, 2)}" for t, v in ranked[:5]) + ".",
        facts=facts,
        notes=_tabcorp_note(results),
        detail={"average_volume": {t: round(v, 2) for t, v in ranked}},
    )


def _daily_returns(bars: Sequence[Bar]) -> list[float]:
    return [(b.close / a.close - 1.0) for a, b in zip(bars, bars[1:])]


def _volatility(args: dict[str, Any]) -> ToolResult:
    start, end, exact_dates = _window(args)
    annualise = bool(args.get("annualize", args.get("annualise", True)))
    tickers = universe(args)
    results: dict[str, float] = {}
    for ticker in tickers:
        rets = _daily_returns(_slice(ticker, start, end))
        if len(rets) < 2:
            continue
        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sigma = math.sqrt(variance) * 100.0
        results[ticker] = sigma * math.sqrt(252) if annualise else sigma
    if not results:
        return ToolResult.failure(TOOL, args, "not enough data to compute volatility")

    ranked = sorted(results.items(), key=lambda kv: -kv[1])
    kind = "annualised" if annualise else "daily"
    label = _label(start, end)
    facts = [
        Fact(f"volatility_{t}", round(v, 4), f"{t} {kind} volatility was {v:.2f}%{label}")
        for t, v in ranked[:5]
    ]
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"{kind.capitalize()} volatility{label}: " + "; ".join(f"{t} {v:.2f}%" for t, v in ranked[:6]) + ".",
        facts=facts,
        notes=_tabcorp_note(results),
        detail={"volatility_pct": {t: round(v, 4) for t, v in ranked}, "annualised": annualise},
    )


def _correlation(args: dict[str, Any]) -> ToolResult:
    start, end, exact_dates = _window(args)
    tickers = universe(args)
    if len(tickers) < 2:
        return ToolResult.failure(TOOL, args, "correlation needs two tickers")

    a, b = tickers[0], tickers[1]
    bars_a = {bar.day: bar.close for bar in _slice(a, start, end)}
    bars_b = {bar.day: bar.close for bar in _slice(b, start, end)}
    common = sorted(set(bars_a) & set(bars_b))
    if len(common) < 3:
        return ToolResult.failure(TOOL, args, "not enough overlapping sessions")

    ra = [bars_a[y] / bars_a[x] - 1 for x, y in zip(common, common[1:])]
    rb = [bars_b[y] / bars_b[x] - 1 for x, y in zip(common, common[1:])]
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = math.sqrt(sum((x - mean_a) ** 2 for x in ra))
    var_b = math.sqrt(sum((y - mean_b) ** 2 for y in rb))
    if var_a == 0 or var_b == 0:
        return ToolResult.failure(TOOL, args, "zero variance; correlation undefined")

    rho = cov / (var_a * var_b)
    label = _label(start, end)
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"Daily-return correlation between {a} and {b}{label} is {rho:.3f} over {len(ra)} sessions.",
        facts=[Fact("correlation", round(rho, 4), f"the correlation between {a} and {b} was {rho:.3f}{label}")],
        detail={"sessions": len(ra)},
    )


def _close_on(args: dict[str, Any]) -> ToolResult:
    when = parse_date(args.get("date") or args.get("date_from"))
    if when is None:
        return ToolResult.failure(TOOL, args, "close_on requires a 'date' argument")
    tickers = universe(args)
    if not tickers:
        return ToolResult.failure(TOOL, args, "no matching tickers")

    facts, notes, detail = [], [], {}
    for ticker in tickers[:8]:
        bar, exact = _bar_on(ticker, when)
        if bar is None:
            notes.append(f"{ticker} has no data on or before {iso(when)}")
            continue
        if not exact:
            notes.append(f"{ticker} did not trade on {iso(when)}; used {iso(bar.day)}")
        facts.append(Fact(f"close_{ticker}", bar.close, f"{ticker} closed at {bar.close:.4f} on {iso(bar.day)}"))
        detail[ticker] = {"date": iso(bar.day), "close": bar.close, "volume": bar.volume}
    return ToolResult(
        tool=TOOL,
        args=args,
        summary="; ".join(f.text for f in facts) or "no closing prices found",
        facts=facts,
        notes=notes,
        detail=detail,
    )


_METRICS = {
    "coverage": _coverage,
    "annual_return": lambda a: _return_metric(a, True),
    "window_return": lambda a: _return_metric(a, True),
    "full_sample_return": lambda a: _return_metric(a, False),
    "rank_annual_returns": _rank_annual_returns,
    "basket_return": _basket_return,
    "max_drawdown": _max_drawdown,
    "avg_volume": _avg_volume,
    "volatility": _volatility,
    "correlation": _correlation,
    "close_on": _close_on,
}

METRICS = tuple(_METRICS)


def run(metric: str, args: dict[str, Any]) -> ToolResult:
    handler = _METRICS.get(metric)
    if handler is None:
        return ToolResult.failure(
            TOOL, args, f"unknown asx metric {metric!r}; available: {', '.join(METRICS)}"
        )
    return handler(args)
