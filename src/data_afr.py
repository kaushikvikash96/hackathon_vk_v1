"""AFR news corpus queries built on the inverted index.

Challenge rules enforced here:
  * patterns are matched across HEADLINE + SUBHEAD + INTRO + TEXT combined;
  * matching is case-insensitive and counted once per record;
  * whole-word searches must use ``\\b`` anchors - callers supply the regex and
    it is passed through untouched.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np

from . import afr_index
from .schemas import Fact, ToolResult
from .util import fmt_int, parse_date

TOOL = "query_data"
DATASET = "afr"

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


def _as_int_date(value: Any, *, end: bool = False) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}", text):
        return int(text + ("1231" if end else "0101"))
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = text.split("-")
        return int(f"{year}{month}{'31' if end else '01'}")
    parsed = parse_date(text)
    return int(parsed.strftime("%Y%m%d")) if parsed else None


def _window(args: dict[str, Any]) -> tuple[int | None, int | None]:
    start = _as_int_date(args.get("date_from"))
    end = _as_int_date(args.get("date_to"), end=True)
    year = args.get("year")
    if year and start is None and end is None:
        start, end = int(f"{int(year)}0101"), int(f"{int(year)}1231")
    return start, end


def _filter_by_date(doc_ids: np.ndarray, start: int | None, end: int | None) -> np.ndarray:
    if start is None and end is None:
        return doc_ids
    dates = afr_index.get_index().dates[doc_ids]
    mask = np.ones(doc_ids.shape, dtype=bool)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return doc_ids[mask]


def _window_label(args: dict[str, Any], start: int | None, end: int | None) -> str:
    if args.get("year"):
        return f" in {int(args['year'])}"
    if start and end:
        return f" between {start} and {end}"
    if start:
        return f" from {start}"
    if end:
        return f" up to {end}"
    return ""


def _require_pattern(args: dict[str, Any]) -> str | None:
    pattern = args.get("pattern") or args.get("query")
    return str(pattern) if pattern else None


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _boundary_note(pattern: str) -> list[str]:
    tokens = re.findall(r"\w+", pattern)
    if tokens and "\\b" not in pattern and any(len(t) <= 4 for t in tokens):
        return [
            "The pattern has no word-boundary anchors; short terms may match "
            "inside unrelated words and inflate the count."
        ]
    return []


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _coverage(args: dict[str, Any]) -> ToolResult:
    index = afr_index.get_index()
    dates = index.dates[index.dates > 0]
    first, last = int(dates.min()), int(dates.max())
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f"AFR coverage: {index.document_count:,} articles from "
            f"{first // 10000}-{first // 100 % 100:02d} to {last // 10000}-{last // 100 % 100:02d}."
        ),
        facts=[
            Fact("article_count", index.document_count, f"the AFR corpus holds {fmt_int(index.document_count)} articles"),
            Fact("first_date", first, f"the earliest publication date is {first}"),
            Fact("last_date", last, f"the latest publication date is {last}"),
        ],
    )


def _count(args: dict[str, Any]) -> ToolResult:
    pattern = _require_pattern(args)
    if not pattern:
        return ToolResult.failure(TOOL, args, "afr count requires a 'pattern' (Python regex) argument")
    try:
        matches = afr_index.get_index().search(pattern)
    except re.error as exc:
        return ToolResult.failure(TOOL, args, f"invalid regex {pattern!r}: {exc}")

    start, end = _window(args)
    matches = _filter_by_date(matches, start, end)
    label = _window_label(args, start, end)
    total = int(matches.size)

    facts = [
        Fact("match_count", total, f"there are {fmt_int(total)} AFR records matching {pattern}{label}"),
    ]
    detail: dict[str, Any] = {"pattern": pattern}

    group_by = str(args.get("group_by", "")).lower()
    if group_by in {"year", "month", "year_and_month", "both"} or args.get("group_by") is True:
        dates = afr_index.get_index().dates[matches]
        by_year = Counter(int(d) // 10000 for d in dates if d > 0)
        by_month = Counter(int(d) // 100 for d in dates if d > 0)
        detail["by_year"] = {str(k): v for k, v in sorted(by_year.items())}
        if group_by in {"month", "year_and_month", "both"} or args.get("group_by") is True:
            detail["by_month"] = {f"{k // 100}-{k % 100:02d}": v for k, v in sorted(by_month.items())}

        if by_year:
            top_year, top_year_count = by_year.most_common(1)[0]
            facts.append(
                Fact("peak_year", top_year, f"it peaked in {top_year} with {fmt_int(top_year_count)} matching records")
            )
        if by_month:
            top_month, top_month_count = by_month.most_common(1)[0]
            month_label = f"{_MONTH_NAMES[top_month % 100]} {top_month // 100}"
            facts.append(
                Fact("peak_month", f"{top_month // 100}-{top_month % 100:02d}",
                     f"{month_label} is the peak month with {fmt_int(top_month_count)}")
            )

    summary = f"{total:,} AFR records match {pattern}{label}."
    if "peak_year" in {f.label for f in facts}:
        summary += " " + " ".join(f.text.capitalize() + "." for f in facts[1:])
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=summary,
        facts=facts,
        notes=_boundary_note(pattern),
        detail=detail,
    )


def _count_by_month(args: dict[str, Any]) -> ToolResult:
    merged = dict(args)
    merged["group_by"] = "month"
    return _count(merged)


def _share(args: dict[str, Any]) -> ToolResult:
    pattern = _require_pattern(args)
    if not pattern:
        return ToolResult.failure(TOOL, args, "afr share requires a 'pattern' argument")

    index = afr_index.get_index()
    start, end = _window(args)
    try:
        matches = _filter_by_date(index.search(pattern), start, end)
    except re.error as exc:
        return ToolResult.failure(TOOL, args, f"invalid regex {pattern!r}: {exc}")

    universe = _filter_by_date(np.arange(index.document_count, dtype=np.uint32), start, end)
    total = int(universe.size)
    hits = int(matches.size)
    share = (hits / total * 100.0) if total else 0.0
    label = _window_label(args, start, end)
    return ToolResult(
        tool=TOOL,
        args=args,
        summary=f"{hits:,} of {total:,} AFR records{label} match {pattern} ({share:.2f}%).",
        facts=[
            Fact("match_count", hits, f"{fmt_int(hits)} AFR records match {pattern}{label}"),
            Fact("total_records", total, f"out of {fmt_int(total)} records{label}"),
            Fact("share_pct", round(share, 4), f"a share of {share:.2f}%"),
        ],
        notes=_boundary_note(pattern),
    )


def _find_article(args: dict[str, Any]) -> ToolResult:
    index = afr_index.get_index()
    headline = str(args.get("headline") or args.get("title") or "").strip()
    if not headline:
        return ToolResult.failure(TOOL, args, "find_article requires a 'headline' argument")

    target = _normalise(headline)
    when = _as_int_date(args.get("date") or args.get("publication_date"))

    exact, partial = [], []
    for doc_id, normalised in enumerate(index.normalised_headlines()):
        if normalised == target:
            exact.append(doc_id)
        elif target and target in normalised:
            partial.append(doc_id)

    doc_ids = exact or partial
    if when is not None:
        dated = [d for d in doc_ids if int(index.dates[d]) == when]
        doc_ids = dated or doc_ids
    if not doc_ids:
        return ToolResult.failure(TOOL, args, f"no AFR article found with headline {headline!r}")

    doc_id = doc_ids[0]
    record = index.source_record(doc_id)
    published = str(record.get("PUBLICATIONDATE") or "")
    iso_date = f"{published[:4]}-{published[4:6]}-{published[6:8]}" if len(published) == 8 else published
    body = " ".join(
        part for part in (record.get("INTRO") or "", record.get("TEXT") or "") if part
    ).strip()

    return ToolResult(
        tool=TOOL,
        args=args,
        summary=(
            f'Found AFR article "{record.get("HEADLINE")}" published {iso_date} '
            f"({len(body):,} chars of body text stored as evidence for synthesis)."
        ),
        facts=[
            Fact("headline", record.get("HEADLINE"), f'the article is "{record.get("HEADLINE")}"'),
            Fact("publication_date", iso_date, f"published {iso_date}"),
        ],
        notes=[] if len(doc_ids) == 1 else [f"{len(doc_ids)} articles matched; using the first."],
        detail={
            "headline": record.get("HEADLINE"),
            "subhead": record.get("SUBHEAD"),
            "publication_date": iso_date,
            "article_text": body[:12000],
        },
    )


_METRICS = {
    "coverage": _coverage,
    "count": _count,
    "count_by_month": _count_by_month,
    "share": _share,
    "find_article": _find_article,
}

METRICS = tuple(_METRICS)


def run(metric: str, args: dict[str, Any]) -> ToolResult:
    handler = _METRICS.get(metric)
    if handler is None:
        return ToolResult.failure(
            TOOL, args, f"unknown afr metric {metric!r}; available: {', '.join(METRICS)}"
        )
    return handler(args)
