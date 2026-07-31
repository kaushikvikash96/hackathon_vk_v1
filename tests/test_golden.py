"""Golden tests: every public calibration question, checked at the tool layer.

These assert the *numbers*, independent of any model. If a question regresses
here, no amount of prompting will recover it during evaluation.

Expected values come from Participant_Package/public_questions.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import run_tool  # noqa: E402

EXCLUDE_TABCORP = ["TAH.AX"]


def summary(**kwargs) -> str:
    result = run_tool(kwargs.pop("tool", "query_data"), kwargs)
    assert result.ok, result.error
    return result.summary


def facts(**kwargs) -> dict[str, object]:
    result = run_tool(kwargs.pop("tool", "query_data"), kwargs)
    assert result.ok, result.error
    return {f.label: f.value for f in result.facts}


# --------------------------------------------------------------------------
# MHQ001 - RBA changes, increases, decreases
# --------------------------------------------------------------------------
def test_mhq001_rba_change_counts():
    values = facts(dataset="rba", metric="count_changes")
    assert values["changes"] == 41
    assert values["increases"] == 20
    assert values["decreases"] == 21
    assert "175" in summary(dataset="rba", metric="count_changes")


# --------------------------------------------------------------------------
# MHQ035 - 2011-2013 easing period
# --------------------------------------------------------------------------
def test_mhq035_easing_cycle():
    values = facts(
        dataset="rba", metric="cycle_summary", direction="cuts",
        date_from="2011-01-01", date_to="2013-12-31",
    )
    assert values["count"] == 8
    assert values["cumulative_change"] == -2.25
    assert values["target_before"] == 4.75
    assert values["target_after"] == 2.50

    result = run_tool("query_data", {
        "dataset": "rba", "metric": "cycle_summary", "direction": "cuts",
        "date_from": "2011-01-01", "date_to": "2013-12-31",
    })
    assert result.detail["by_year"] == {"2011": 2, "2012": 4, "2013": 2}


# --------------------------------------------------------------------------
# MHQ040 - ASX dimensions
# --------------------------------------------------------------------------
def test_mhq040_asx_coverage():
    values = facts(dataset="asx", metric="coverage")
    assert values["ticker_count"] == 18
    assert values["rows_per_ticker"] == 1774
    assert values["first_date"] == "2015-01-02"
    assert values["last_date"] == "2021-12-30"


# --------------------------------------------------------------------------
# MHQ045 - best and worst 2018 return excluding Tabcorp
# --------------------------------------------------------------------------
def test_mhq045_2018_best_worst():
    result = run_tool("query_data", {
        "dataset": "asx", "metric": "rank_annual_returns",
        "year": 2018, "exclude_tickers": EXCLUDE_TABCORP,
    })
    returns = dict((row["ticker"], row["return_pct"]) for row in result.detail["ranking"])
    assert result.detail["ranking"][0]["ticker"] == "BHP.AX"
    assert result.detail["ranking"][-1]["ticker"] == "AMP.AX"
    assert returns["BHP.AX"] == pytest.approx(22.17, abs=0.02)
    assert returns["AMP.AX"] == pytest.approx(-50.04, abs=0.02)
    assert "TAH.AX" not in returns


# --------------------------------------------------------------------------
# MHQ049 - highest average daily volume excluding Tabcorp
# --------------------------------------------------------------------------
def test_mhq049_average_volume():
    result = run_tool("query_data", {
        "dataset": "asx", "metric": "avg_volume", "exclude_tickers": EXCLUDE_TABCORP,
    })
    assert result.detail["average_volume"]["AMP.AX"] == pytest.approx(11_635_671.71, abs=1.0)
    assert list(result.detail["average_volume"])[0] == "AMP.AX"


# --------------------------------------------------------------------------
# MHQ055 - three worst drawdowns with peak and trough dates
# --------------------------------------------------------------------------
def test_mhq055_worst_drawdowns():
    result = run_tool("query_data", {
        "dataset": "asx", "metric": "max_drawdown",
        "exclude_tickers": EXCLUDE_TABCORP, "top_n": 3,
    })
    top3 = result.detail["drawdowns"][:3]
    assert [row["ticker"] for row in top3] == ["AMP.AX", "AGL.AX", "QAN.AX"]
    assert top3[0]["drawdown_pct"] == pytest.approx(-82.45, abs=0.02)
    assert top3[0]["peak_date"] == "2015-03-20"
    assert top3[0]["trough_date"] == "2021-12-17"
    assert top3[1]["drawdown_pct"] == pytest.approx(-76.24, abs=0.02)
    assert (top3[1]["peak_date"], top3[1]["trough_date"]) == ("2017-04-10", "2021-11-16")
    assert top3[2]["drawdown_pct"] == pytest.approx(-71.08, abs=0.02)
    assert (top3[2]["peak_date"], top3[2]["trough_date"]) == ("2019-12-19", "2020-03-19")


# --------------------------------------------------------------------------
# MHQ058 / MHQ067 / MHQ080 - article retrieval plus the rate in force
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "headline,published,expected_rate",
    [
        ("Travel stocks take off on vaccine rollout", "2021-02-23", 0.10),
        ("Why investors don't believe the RBA on interest rates", "2021-11-25", 0.10),
        ("Energy stocks shine as vaccines fuel oil rally", "2020-11-28", 0.10),
    ],
)
def test_article_retrieval_and_rate(headline, published, expected_rate):
    article = run_tool("retrieve", {"headline": headline, "date": published})
    assert article.ok, article.error
    assert article.detail["publication_date"] == published
    assert len(article.detail["article_text"]) > 100

    rate = facts(dataset="rba", metric="lookup_rate", date=published)
    assert rate["target"] == pytest.approx(expected_rate)


def test_rate_lookup_uses_on_or_before():
    """The decision in force, never the next future decision."""
    values = facts(dataset="rba", metric="lookup_rate", date="2021-02-23")
    assert values["effective_date"] == "2021-02-03"


# --------------------------------------------------------------------------
# MHQ061 - whole-word unemployment peaks
# --------------------------------------------------------------------------
def test_mhq061_unemployment_peaks():
    values = facts(
        dataset="afr", metric="count", pattern=r"\bunemployment\b", group_by="year_and_month"
    )
    assert values["peak_year"] == 2020
    assert values["peak_month"] == "2020-05"
    result = run_tool("query_data", {
        "dataset": "afr", "metric": "count",
        "pattern": r"\bunemployment\b", "group_by": "year_and_month",
    })
    assert result.detail["by_year"]["2020"] == 1452
    assert result.detail["by_month"]["2020-05"] == 218


# --------------------------------------------------------------------------
# MHQ072 - post-cut basket and named constituents
# --------------------------------------------------------------------------
def test_mhq072_june_2019_basket():
    result = run_tool("query_data", {
        "dataset": "asx", "metric": "basket_return",
        "date_from": "2019-06-05", "date_to": "2019-06-12",
        "exclude_tickers": EXCLUDE_TABCORP,
        "report_tickers": ["CBA.AX", "NAB.AX", "ANZ.AX", "BHP.AX", "RIO.AX"],
    })
    values = {f.label: f.value for f in result.facts}
    assert values["basket_return"] == pytest.approx(2.88, abs=0.02)
    assert values["return_CBA.AX"] == pytest.approx(0.60, abs=0.02)
    assert values["return_NAB.AX"] == pytest.approx(1.39, abs=0.02)
    assert values["return_ANZ.AX"] == pytest.approx(0.89, abs=0.02)
    assert values["return_BHP.AX"] == pytest.approx(5.89, abs=0.02)
    assert values["return_RIO.AX"] == pytest.approx(2.91, abs=0.02)
    assert len(result.detail["constituents"]) == 17

    rate = facts(dataset="rba", metric="lookup_rate", date="2019-06-05")
    assert rate["target"] == pytest.approx(1.25)


# --------------------------------------------------------------------------
# MHQ074 - one-week basket return after each 2019 cut
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "start,end,expected",
    [("2019-06-05", "2019-06-12", 2.88), ("2019-07-03", "2019-07-10", 0.24), ("2019-10-02", "2019-10-09", -2.17)],
)
def test_mhq074_weekly_baskets(start, end, expected):
    values = facts(
        dataset="asx", metric="basket_return",
        date_from=start, date_to=end, exclude_tickers=EXCLUDE_TABCORP,
    )
    assert values["basket_return"] == pytest.approx(expected, abs=0.02)


def test_mhq074_2019_cut_targets():
    result = run_tool("query_data", {
        "dataset": "rba", "metric": "changes_in_period",
        "date_from": "2019-01-01", "date_to": "2019-12-31",
    })
    targets = [row["target"] for row in result.detail["changes"]]
    assert targets == [1.25, 1.00, 0.75]


# --------------------------------------------------------------------------
# MHQ076 - QBE AFR count and 2021 return rank
# --------------------------------------------------------------------------
def test_mhq076_qbe_2021():
    afr = facts(dataset="afr", metric="count", pattern=r"\bQBE\b", year=2021)
    assert afr["match_count"] == 369

    ranking = run_tool("query_data", {
        "dataset": "asx", "metric": "rank_annual_returns",
        "year": 2021, "exclude_tickers": EXCLUDE_TABCORP,
    })
    best = ranking.detail["ranking"][0]
    assert best["ticker"] == "QBE.AX"
    assert best["return_pct"] == pytest.approx(35.57, abs=0.02)


# --------------------------------------------------------------------------
# MHQ080 - five-session basket window after publication
# --------------------------------------------------------------------------
def test_mhq080_post_publication_window():
    values = facts(
        dataset="asx", metric="basket_return",
        date_from="2020-11-30", date_to="2020-12-07", exclude_tickers=EXCLUDE_TABCORP,
    )
    assert values["basket_return"] == pytest.approx(2.37, abs=0.02)


# --------------------------------------------------------------------------
# MHQ084 - 2019 across all three datasets
# --------------------------------------------------------------------------
def test_mhq084_2019_three_datasets():
    cuts = facts(
        dataset="rba", metric="cycle_summary", direction="cuts",
        date_from="2019-01-01", date_to="2019-12-31",
    )
    assert cuts["count"] == 3
    assert cuts["cumulative_change"] == -0.75
    assert cuts["target_after"] == 0.75

    afr = facts(
        dataset="afr", metric="count",
        pattern=r"interest rates?|cash rate|rate cut|rate hike|\bRBA\b", year=2019,
    )
    assert afr["match_count"] == 3181

    basket = facts(
        dataset="asx", metric="basket_return", year=2019, exclude_tickers=EXCLUDE_TABCORP
    )
    assert basket["basket_return"] == pytest.approx(20.11, abs=0.02)


# --------------------------------------------------------------------------
# MHQ090 - unsupported cross-dataset analysis
# --------------------------------------------------------------------------
def test_mhq090_coverage_mismatch():
    result = run_tool("query_data", {"dataset": "cross", "metric": "coverage"})
    assert result.ok
    assert "2021" in result.summary
    assert any("unsupported" in note or "cannot be fully observed" in note for note in result.notes)

    hikes = facts(
        dataset="rba", metric="cycle_summary", direction="hikes",
        date_from="2022-01-01", date_to="2023-12-31",
    )
    assert hikes["count"] == 13
    assert hikes["cumulative_change"] == 4.25
    assert hikes["target_before"] == 0.10
    assert hikes["target_after"] == 4.35


# --------------------------------------------------------------------------
# Documented tool contracts
# --------------------------------------------------------------------------
def test_unknown_metric_is_a_helpful_error_not_a_crash():
    result = run_tool("query_data", {"dataset": "rba", "metric": "does_not_exist"})
    assert not result.ok
    assert "count_changes" in result.error


def test_unknown_dataset_is_a_helpful_error():
    result = run_tool("query_data", {"dataset": "nasdaq", "metric": "count"})
    assert not result.ok
    assert "rba" in result.error


def test_afr_count_requires_a_pattern():
    result = run_tool("query_data", {"dataset": "afr", "metric": "count"})
    assert not result.ok
    assert "pattern" in result.error


def test_brain_view_never_leaks_article_text():
    """Article bodies must reach synthesis only, never the 4k brain context."""
    result = run_tool("retrieve", {"headline": "Travel stocks take off on vaccine rollout"})
    assert len(result.brain_view()) <= 700
    assert result.detail["article_text"] not in result.brain_view()
