"""Tool registry: the only path from the models to the data.

Qwen requests calls, this module validates and executes them, and the structured
result goes back as compact text (brain) plus full facts (synthesis model).
Neither model ever touches a file.
"""

from __future__ import annotations

from typing import Any

from . import data_afr, data_asx, data_rba
from .schemas import Fact, ToolResult

QUERY_DATA = "query_data"
RETRIEVE = "retrieve"

DATASETS = ("rba", "asx", "afr", "cross")

#: Compact catalog injected into the brain prompt. Kept terse on purpose - the
#: brain has a 4096-token context on the supplied vLLM deployment.
METRIC_CATALOG = """\
rba  : coverage | count | count_changes | count_increases | count_decreases | extremes
       max_hold_streak | lookup_rate(date) | changes_in_period | cycle_summary(direction=hikes|cuts) | list
asx  : coverage | annual_return(year) | window_return(date_from,date_to) | full_sample_return
       rank_annual_returns(year) | basket_return(date_from,date_to,report_tickers)
       max_drawdown(top_n) | avg_volume | volatility | correlation(tickers=[a,b]) | close_on(date)
afr  : coverage | count(pattern,group_by=year|month) | count_by_month(pattern) | share(pattern)
cross: coverage  -> date ranges of all three datasets, for answering whether an analysis is supported"""


def _cross_coverage(args: dict[str, Any]) -> ToolResult:
    rba = data_rba.load()
    asx = data_asx.load()
    index = data_afr.afr_index.get_index()
    afr_dates = index.dates[index.dates > 0]
    afr_first, afr_last = int(afr_dates.min()), int(afr_dates.max())

    def _iso(compact: int) -> str:
        return f"{compact // 10000}-{compact // 100 % 100:02d}-{compact % 100:02d}"

    asx_first = min(bars[0].day for bars in asx.values())
    asx_last = max(bars[-1].day for bars in asx.values())

    facts = [
        Fact("rba_range", f"{rba[0].effective}..{rba[-1].effective}",
             f"the RBA data covers {rba[0].effective} to {rba[-1].effective}"),
        Fact("asx_range", f"{asx_first}..{asx_last}",
             f"the ASX data covers {asx_first} to {asx_last}"),
        Fact("afr_range", f"{_iso(afr_first)}..{_iso(afr_last)}",
             f"the AFR data covers {_iso(afr_first)} to {_iso(afr_last)}"),
        Fact("overlap_end", str(min(asx_last.isoformat(), _iso(afr_last))),
             f"AFR and ASX both end in {asx_last.year}, so any analysis after that year is unsupported"),
    ]
    return ToolResult(
        tool=QUERY_DATA,
        args=args,
        summary=(
            f"Coverage - RBA {rba[0].effective} to {rba[-1].effective} ({len(rba)} records); "
            f"ASX {asx_first} to {asx_last} ({len(asx)} tickers); "
            f"AFR {_iso(afr_first)} to {_iso(afr_last)} ({index.document_count:,} articles)."
        ),
        facts=facts,
        notes=[
            "AFR and ASX end in 2021 while RBA continues past it; cross-dataset "
            "questions covering later periods cannot be fully observed."
        ],
    )


def run_tool(name: str, args: dict[str, Any]) -> ToolResult:
    """Validate and execute one model-requested tool call."""
    args = dict(args or {})

    if name == RETRIEVE:
        return data_afr.run("find_article", args)

    if name != QUERY_DATA:
        return ToolResult.failure(name, args, f"unknown tool {name!r}; available: {QUERY_DATA}, {RETRIEVE}")

    dataset = str(args.pop("dataset", "") or "").strip().lower()
    metric = str(args.pop("metric", "") or "").strip().lower()
    if dataset not in DATASETS:
        return ToolResult.failure(
            QUERY_DATA, args, f"unknown dataset {dataset!r}; available: {', '.join(DATASETS)}"
        )
    if not metric:
        return ToolResult.failure(QUERY_DATA, args, "a 'metric' argument is required")

    # Nested {"args": {...}} is accepted as well as flat keyword arguments.
    nested = args.pop("args", None)
    if isinstance(nested, dict):
        args = {**nested, **args}
    args["dataset"], args["metric"] = dataset, metric

    try:
        if dataset == "rba":
            return data_rba.run(metric, args)
        if dataset == "asx":
            return data_asx.run(metric, args)
        if dataset == "afr":
            return data_afr.run(metric, args)
        return _cross_coverage(args)
    except Exception as exc:  # never let a tool bug become a 500
        return ToolResult.failure(QUERY_DATA, args, f"{type(exc).__name__}: {exc}")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": QUERY_DATA,
            "description": (
                "Exact structured query over the RBA cash-rate, ASX price, and AFR news "
                "datasets. Always use this for any number, date, count, ranking or return."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "enum": list(DATASETS)},
                    "metric": {"type": "string", "description": "Metric name from the catalog in the system prompt."},
                    "pattern": {"type": "string", "description": "AFR only. Python regex, use \\b anchors for whole words."},
                    "tickers": {"type": "array", "items": {"type": "string"}, "description": "ASX tickers, e.g. ['CBA.AX']. Omit for all 18."},
                    "exclude_tickers": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['TAH.AX'] to exclude Tabcorp."},
                    "report_tickers": {"type": "array", "items": {"type": "string"}, "description": "basket_return only: also report these constituents individually."},
                    "date": {"type": "string", "description": "YYYY-MM-DD, for lookup_rate and close_on."},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD window start."},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD window end."},
                    "year": {"type": "integer"},
                    "group_by": {"type": "string", "enum": ["year", "month", "year_and_month"]},
                    "direction": {"type": "string", "enum": ["hikes", "cuts", "all"]},
                    "top_n": {"type": "integer"},
                },
                "required": ["dataset", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": RETRIEVE,
            "description": (
                "Fetch one AFR article by headline (and optionally publication date). "
                "The full body text is passed to the synthesis model, not returned here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD publication date."},
                },
                "required": ["headline"],
            },
        },
    },
]
