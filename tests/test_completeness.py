"""Dataset-level completeness summaries (stablebound.completeness).

The per-row columns are tested in test_stats.py. These cover the collapse to
one row per group, which is what a user actually reads — and which broke twice
in development on dtype handling that the per-row tests could not catch.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound.completeness import format_report, summarize

STABLE_COLS = [
    "year", "season", "variable", "stable_id", "value", "n_constituents",
    "n_in_group", "completeness", "complete", "missing_unit_ids",
    "constituent_ids", "late_reporting",
]


def _stable(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=STABLE_COLS)


def _row(year, var, sid, n_con, n_grp, comp, complete, missing, late=False):
    return (year, "Annual", var, sid, 1.0, n_con, n_grp, comp, complete,
            missing, "", late)


def test_summarize_groups_by_variable_and_year():
    df = _stable([
        _row(2015, "area_ha", "A", 2, 2, 1.0, True, ""),
        _row(2015, "area_ha", "B", 1, 2, 0.5, False, "B2"),
        _row(2016, "area_ha", "A", 2, 2, 1.0, True, ""),
    ])
    out = summarize(df)
    assert list(out.columns[:2]) == ["variable", "year"]
    y2015 = out[out["year"] == 2015].iloc[0]
    assert y2015["n_cells"] == 2
    assert y2015["n_complete"] == 1
    assert y2015["pct_complete"] == 0.5
    assert y2015["mean_completeness"] == 0.75
    assert y2015["min_completeness"] == 0.5


def test_summarize_handles_object_dtype_columns():
    """completeness/complete arrive as object dtype (floats mixed with pd.NA).

    pandas' numeric methods reject object columns outright, so the summary
    has to coerce. This broke on real data while the per-row tests passed.
    """
    df = _stable([
        _row(2015, "area_ha", "A", 2, 2, 1.0, True, ""),
        _row(2015, "yield", "A", pd.NA, pd.NA, pd.NA, pd.NA, ""),
    ])
    assert df["completeness"].dtype == object
    out = summarize(df)
    assert len(out) == 2
    # The all-NA group reports NA rather than crashing or inventing a number.
    y = out[out["variable"] == "yield"].iloc[0]
    assert pd.isna(y["mean_completeness"])


def test_late_reporting_rows_are_excluded_but_counted():
    df = _stable([
        _row(2015, "area_ha", "A", 2, 2, 1.0, True, ""),
        _row(2015, "area_ha", "X", pd.NA, pd.NA, pd.NA, pd.NA, "", late=True),
    ])
    out = summarize(df)
    row = out.iloc[0]
    # The late row must not drag the average down — it has no group to be
    # complete against — but it must still be visible.
    assert row["n_cells"] == 1
    assert row["pct_complete"] == 1.0
    assert row["n_late_reporting_rows"] == 1


def test_summarize_by_stable_id_finds_chronic_offenders():
    df = _stable([
        _row(2015, "area_ha", "GOOD", 2, 2, 1.0, True, ""),
        _row(2016, "area_ha", "GOOD", 2, 2, 1.0, True, ""),
        _row(2015, "area_ha", "BAD", 1, 4, 0.25, False, "b,c,d"),
        _row(2016, "area_ha", "BAD", 1, 4, 0.25, False, "b,c,d"),
    ])
    out = summarize(df, by="stable_id")
    bad = out[out["stable_id"] == "BAD"].iloc[0]
    assert bad["pct_complete"] == 0.0
    assert bad["mean_completeness"] == 0.25


def test_worst_ids_names_the_least_complete_units():
    df = _stable([
        _row(2015, "area_ha", "A", 4, 4, 1.0, True, ""),
        _row(2015, "area_ha", "B", 1, 4, 0.25, False, "x,y,z"),
        _row(2015, "area_ha", "C", 2, 4, 0.5, False, "x,y"),
    ])
    out = summarize(df, worst_n=2)
    assert out.iloc[0]["worst_ids"] == "B,C"


def test_modern_frame_gets_a_provenance_summary_not_a_ratio():
    # A modern value is a redistribution, so there is no expected-members
    # denominator; reporting a completeness ratio would be inventing one.
    df = pd.DataFrame({
        "year": [2015, 2015], "season": ["Annual", "Annual"],
        "variable": ["area_ha", "area_ha"], "modern_id": ["M1", "M2"],
        "value": [1.0, 2.0], "sources": ["a,b", "c"],
        "n_sources": [2, 1], "min_n_common_observations": [3, pd.NA],
        "has_nan_fraction": [False, True], "fraction_method": ["seasonal", "area"],
    })
    out = summarize(df)
    row = out.iloc[0]
    assert "pct_complete" not in out.columns
    assert row["mean_n_sources"] == 1.5
    assert row["min_n_sources"] == 1.0
    assert row["min_n_common_observations"] == 3.0
    assert row["n_cells_nan_fraction"] == 1
    # "area" is a worse tier than "seasonal", so it is what gets reported.
    assert row["worst_fraction_method"] == "area"


def test_empty_frame_returns_empty_not_an_error():
    assert summarize(pd.DataFrame()).empty


def test_unrecognised_frame_raises_with_a_useful_message():
    with pytest.raises(ValueError, match="neither 'completeness'"):
        summarize(pd.DataFrame({"variable": ["a"], "year": [2015]}))


def test_format_report_is_printable():
    df = _stable([
        _row(2015, "area_ha", "A", 2, 2, 1.0, True, ""),
        _row(2015, "area_ha", "B", 1, 2, 0.5, False, "B2"),
    ])
    text = format_report(summarize(df), "Stable completeness")
    assert "Stable completeness" in text
    assert "Least complete groups" in text
    assert "50.0%" in text
