"""The AFR index must return exactly what a brute-force scan returns.

The index only proposes candidates; the real regex decides. These tests prove
that on a fixed slice of the corpus, across the pattern shapes the challenge
uses: anchored words, phrases, optional suffixes, alternations, unanchored
substrings, and patterns that cannot be decomposed at all.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import afr_index  # noqa: E402

#: Comparing against a full scan of 219k articles takes ~15s per pattern, so
#: equivalence is proven on a fixed prefix of the corpus instead.
SLICE = 25_000

PATTERNS = [
    r"\bunemployment\b",
    r"\bQBE\b",
    r"\bNAB\b",
    r"cash rate",
    r"interest rates?",
    r"rate cut|rate hike",
    r"interest rates?|cash rate|rate cut|rate hike|\bRBA\b",
    r"climate change",
    r"superannuation",          # unanchored substring
    r"\bhousing\b|\bproperty\b",
    r"budget \d{4}",            # character class inside a phrase
    r"\biron ore\b",
]


@pytest.fixture(scope="module")
def index():
    if not afr_index.is_built():
        pytest.skip("AFR index not built; run python scripts/build_afr_index.py")
    return afr_index.get_index()


def naive_scan(index, pattern: str, limit: int) -> set[int]:
    matcher = re.compile(pattern, re.IGNORECASE)
    return {d for d in range(limit) if matcher.search(index.doc_text(d))}


@pytest.mark.parametrize("pattern", PATTERNS)
def test_index_matches_brute_force(index, pattern):
    limit = min(SLICE, index.document_count)
    expected = naive_scan(index, pattern, limit)
    actual = {int(d) for d in index.search(pattern) if int(d) < limit}
    assert actual == expected, (
        f"{pattern!r}: index missed {sorted(expected - actual)[:5]}, "
        f"over-matched {sorted(actual - expected)[:5]}"
    )


def test_candidates_are_a_superset(index):
    """Correctness rests on candidates never omitting a true match."""
    limit = min(SLICE, index.document_count)
    for pattern in PATTERNS:
        split = index.candidates(pattern)
        if split is None:
            continue  # decomposition declined; a full scan is used instead
        confirmed, pending = split
        proposed = set(int(d) for d in confirmed) | set(int(d) for d in pending)
        missed = naive_scan(index, pattern, limit) - proposed
        assert not missed, f"{pattern!r}: candidates missed {sorted(missed)[:5]}"


def test_confirmed_docs_need_no_verification(index):
    r"""A \b-anchored single word is answered from postings alone."""
    split = index.candidates(r"\bunemployment\b")
    assert split is not None
    confirmed, pending = split
    assert confirmed.size > 0
    assert pending.size == 0


def test_literal_prefix_keeps_optional_suffixes_safe():
    assert afr_index.literal_prefix(r"interest rates?") == "interest rate"
    assert afr_index.literal_prefix(r"cash rate") == "cash rate"
    assert afr_index.literal_prefix(r"\bRBA\b") == "rba"
    assert afr_index.literal_prefix(r"\d{4}") is None


def test_alternation_splitting_respects_groups_and_classes():
    assert afr_index.split_alternatives("a|b|c") == ["a", "b", "c"]
    assert afr_index.split_alternatives(r"(a|b)|c") == ["(a|b)", "c"]
    assert afr_index.split_alternatives(r"[a|b]c") == ["[a|b]c"]
    assert afr_index.split_alternatives(r"\bRBA\b|cash rate") == [r"\bRBA\b", "cash rate"]


def test_search_is_cached(index):
    first = index.search(r"\bunemployment\b")
    second = index.search(r"\bunemployment\b")
    assert first is second


def test_counts_match_published_reference_values(index):
    """Values quoted in the public calibration questions."""
    assert len(index.search(r"\bunemployment\b")) == 5997
