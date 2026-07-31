"""Date parsing and number formatting shared by the data layers.

All dates are normalised to ISO ``YYYY-MM-DD``. The organizer judge accepts
equivalent date formats, and ISO is unambiguous for a model to copy verbatim.
"""

from __future__ import annotations

import re
from datetime import date, datetime

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DMY_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})$")
_YMD_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


def parse_date(value: str | date | None) -> date | None:
    """Parse the date spellings that appear across the three datasets."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()

    m = _ISO_RE.match(text)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))

    m = _COMPACT_RE.match(text)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))

    m = _DMY_RE.match(text)
    if m:
        month = _MONTHS.get(m[2][:3].lower())
        if month:
            return date(int(m[3]), month, int(m[1]))

    m = _YMD_SLASH_RE.match(text)
    if m:  # day/month/year - Australian convention
        return date(int(m[3]), int(m[2]), int(m[1]))

    for fmt in ("%d %B %Y", "%B %d %Y", "%b %d %Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {value!r}")


def iso(d: date) -> str:
    return d.isoformat()


def fmt_num(value: float, decimals: int = 2) -> str:
    """Thousands-separated fixed-precision number, integers without decimals."""
    if value == int(value) and decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float, decimals: int = 2, signed: bool = True) -> str:
    """Percentage with an explicit sign - the graders check signs on returns."""
    sign = "+" if (signed and value > 0) else ""
    return f"{sign}{value:,.{decimals}f}%"


def fmt_rate(value: float) -> str:
    """RBA cash-rate target, e.g. 0.1 -> '0.10%'."""
    return f"{value:.2f}%"


def fmt_points(value: float) -> str:
    """Rate change in percentage points, always signed."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f} percentage points"
