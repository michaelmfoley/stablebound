"""Tests for the BEAST-based breakpoint analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("Rbeast")

from stablebound import analyze_breakpoints, BreakpointResult  # noqa: E402

RNG = np.random.default_rng(20260514)


def _step_series(unit: str, crop: str, step_year: int, levels=(5.0, 10.0),
                 years=range(2000, 2025), noise=0.3):
    """Build a long-form synthetic Total Year series with a step change."""
    rows = []
    for y in years:
        v = levels[0] if y < step_year else levels[1]
        v += RNG.normal(0, noise)
        rows.append(
            {"unit_id": unit, "crop": crop, "season": "Total Year",
             "year": y, "value": v}
        )
    return pd.DataFrame(rows)


def test_single_series_detects_step():
    df = _step_series("A", "Rice", step_year=2012)
    res = analyze_breakpoints(df, value_col="value")
    assert isinstance(res, BreakpointResult)
    assert len(res.per_series) == 25  # one row per year
    # The step year (or its near neighbors) should have the highest posterior.
    series = res.per_series.set_index("year")["posterior"]
    peak_year = int(series.idxmax())
    assert abs(peak_year - 2012) <= 1, f"peak at {peak_year}, expected ~2012"
    assert series.loc[peak_year] > 0.5


def test_multicrop_product_aligned_steps():
    """Two crops co-stepping at the same year → multicrop sharper at that year."""
    df = pd.concat(
        [
            _step_series("A", "Rice", step_year=2012),
            _step_series("A", "Wheat", step_year=2012),
        ],
        ignore_index=True,
    )
    res = analyze_breakpoints(df, value_col="value")
    mc = res.multicrop(reduce_cols=("crop",))
    assert set(mc.columns) >= {"unit_id", "year", "multicrop_posterior", "n_series"}
    series = mc.set_index("year")["multicrop_posterior"]
    peak_year = int(series.idxmax())
    assert abs(peak_year - 2012) <= 1
    assert (mc["n_series"] == 2).all()


def test_multicrop_product_disjoint_steps_near_zero():
    """Two crops stepping in different years → product near zero everywhere."""
    df = pd.concat(
        [
            _step_series("A", "Rice", step_year=2008),
            _step_series("A", "Wheat", step_year=2018),
        ],
        ignore_index=True,
    )
    res = analyze_breakpoints(df, value_col="value")
    mc = res.multicrop(reduce_cols=("crop",))
    # No single year should have both crops co-flagging strongly.
    assert mc["multicrop_posterior"].max() < 0.3


def test_single_crop_multicrop_equals_per_series():
    """Single-crop district: multicrop == per-crop (modulo column rename)."""
    df = _step_series("A", "Rice", step_year=2010)
    res = analyze_breakpoints(df, value_col="value")
    mc = res.multicrop(reduce_cols=("crop",))
    merged = res.per_series.merge(
        mc, on=["unit_id", "year"], suffixes=("_pc", "_mc")
    )
    np.testing.assert_allclose(
        merged["posterior"], merged["multicrop_posterior"], atol=1e-9
    )
    assert (merged["n_series"] == 1).all()


def test_min_years_filter():
    """Series shorter than min_years is excluded from per_series."""
    short = _step_series("A", "Rice", step_year=2003, years=range(2000, 2006))
    long_ = _step_series("B", "Rice", step_year=2012, years=range(2000, 2020))
    res = analyze_breakpoints(
        pd.concat([short, long_], ignore_index=True),
        value_col="value",
        min_years=8,
    )
    assert set(res.per_series["unit_id"]) == {"B"}


def test_flagged_threshold():
    df = _step_series("A", "Rice", step_year=2010)
    res = analyze_breakpoints(df, value_col="value")
    flagged = res.flagged(threshold=0.5)
    assert (flagged["posterior"] > 0.5).all()
    assert len(flagged) >= 1


def test_missing_required_column_raises():
    df = pd.DataFrame({"unit_id": ["A"], "year": [2000], "value": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        analyze_breakpoints(df, value_col="production")  # 'crop' default in group_cols


def test_season_filter_default():
    """Rows not matching the season filter are dropped before analysis."""
    df = pd.concat(
        [
            _step_series("A", "Rice", step_year=2010),  # season="Total Year"
            _step_series("A", "Rice", step_year=2010).assign(season="Kharif"),
        ],
        ignore_index=True,
    )
    res = analyze_breakpoints(df, value_col="value")
    # 25 years of Total Year rows survive; we should still have 25 result rows
    # (not 50 — Kharif was filtered out).
    assert len(res.per_series) == 25


def test_no_season_filter_sums_all_rows():
    """With season_filter=None, repeated (group, year) rows are summed."""
    df = pd.concat(
        [
            _step_series("A", "Rice", step_year=2010).assign(season="Kharif"),
            _step_series("A", "Rice", step_year=2010).assign(season="Rabi"),
        ],
        ignore_index=True,
    )
    res = analyze_breakpoints(df, value_col="value", season_filter=None)
    # 25 unique years remain (summed across the two seasons).
    assert len(res.per_series) == 25


def test_same_seed_reproduces_posteriors():
    """BEAST is an MCMC sampler, so a published breakpoint count has to name a seed.

    Left unseeded, two runs over the same crop series disagree in the third
    decimal, and any count thresholded off those posteriors drifts between
    runs. Every number in the paper's relationship-table audit is generated
    this way, so the seeding has to hold.
    """
    stats = pd.concat([_step_series("A", "rice", 2012), _step_series("B", "wheat", 2015)])
    first = analyze_breakpoints(stats, value_col="value", mcmc_seed=7)
    second = analyze_breakpoints(stats, value_col="value", mcmc_seed=7)
    pd.testing.assert_frame_equal(first.per_series, second.per_series)


def test_different_seeds_move_posteriors():
    """The guard above is only worth having if the seed is doing something.

    If two seeds returned identical posteriors, the sampler would not be
    sampling and the reproducibility test would pass for the wrong reason.
    """
    stats = pd.concat([_step_series("A", "rice", 2012), _step_series("B", "wheat", 2015)])
    a = analyze_breakpoints(stats, value_col="value", mcmc_seed=1).per_series
    b = analyze_breakpoints(stats, value_col="value", mcmc_seed=99).per_series
    assert not a.posterior.equals(b.posterior)
