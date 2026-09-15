"""Bayesian breakpoint detection on per-(unit, crop) timeseries.

Wraps the ``Rbeast`` PyPI package to flag candidate boundary-shift
artifacts (or other discontinuities) in long-form stats. The function
is intentionally generic: it accepts any DataFrame with grouping
columns + year + value, so users can run it on:

  - pre-stable raw stats (highlights issues in the upstream data)
  - post-aggregation stable boundary stats (shows which issues the
    stable-boundary product fixed)
  - modern boundary stats (shows residual / induced breakpoints)
  - any other long-form timeseries

Per-series posteriors can be reduced across an axis (e.g. crops) via
:meth:`BreakpointResult.multicrop` to produce a multicrop posterior
per (unit, year) under an independence assumption — useful when several
crops in a district co-break at the same year, suggesting a boundary
artifact rather than crop-specific noise.

Cross-checking flagged years against the lineage file is **manual** in
this iteration. The output is designed to make that cross-check easy.

Dependency note: ``Rbeast`` is an optional dependency. Install with
``pip install stablebound[analysis]`` (or ``pip install Rbeast``
directly). The module imports it lazily so the rest of the package
works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "analyze_breakpoints",
    "BreakpointResult",
]


@dataclass
class BreakpointResult:
    """Per-series breakpoint posteriors plus reducers.

    Attributes
    ----------
    per_series : pd.DataFrame
        Long-form: columns ``(*group_cols, year_col, "posterior")``.
        One row per (group, year) with the BEAST changepoint-occurrence
        posterior for that year.
    group_cols : tuple of str
        The grouping column names used in ``per_series``.
    year_col : str
        Year column name.
    value_col : str
        Name of the value column that was analyzed (e.g. ``"area"``).
    """

    per_series: pd.DataFrame
    group_cols: tuple
    year_col: str
    value_col: str

    def multicrop(
        self,
        reduce_cols: Sequence[str] = ("crop",),
        method: str = "product",
    ) -> pd.DataFrame:
        """Collapse posteriors across one or more grouping dims.

        Parameters
        ----------
        reduce_cols : sequence of str
            Group columns to reduce over (e.g. ``("crop",)`` to combine
            across crops within each unit-year).
        method : {"product"}
            Currently only the independence product is implemented:
            ``P_multi(year) = ∏ over series-present P_i(year)``.
            Series that don't appear in a given (unit, year) drop out
            of the product naturally — they neither penalize nor
            inflate the result.

        Returns
        -------
        pd.DataFrame
            Columns ``(*keep_cols, year_col, "multicrop_posterior",
            "n_series")`` where ``keep_cols = group_cols - reduce_cols``.
        """
        if method != "product":
            raise ValueError(
                f"method={method!r} not implemented; use 'product'."
            )
        reduce_cols = tuple(reduce_cols)
        keep_cols = tuple(c for c in self.group_cols if c not in reduce_cols)
        if not keep_cols:
            raise ValueError(
                "reduce_cols cannot include every group column; "
                "you must leave at least one to group by."
            )

        df = self.per_series
        grp = df.groupby([*keep_cols, self.year_col], dropna=False)["posterior"]
        out = grp.agg(
            multicrop_posterior=lambda s: float(np.prod(s.values)),
            n_series="count",
        ).reset_index()
        return out

    def flagged(
        self,
        threshold: float = 0.5,
        posterior_col: str = "posterior",
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Filter to rows where the posterior exceeds ``threshold``.

        If ``df`` is omitted, filters ``self.per_series``. Pass a
        multicrop result to filter that instead.
        """
        target = df if df is not None else self.per_series
        return target[target[posterior_col] > threshold].reset_index(drop=True)


# BEAST estimates changepoint posteriors by MCMC. Left to itself it seeds from
# the clock, so two runs over the same crop series disagree in the third decimal
# and any count thresholded off those posteriors moves between runs. Every
# published breakpoint number therefore has to name a seed. This is the default
# so that results reproduce unless a caller opts out.
DEFAULT_MCMC_SEED = 1


#: How a gap in a series reaches BEAST. ``"interpolate"`` (the default, and what
#: every published number was computed with) fills missing years linearly on the
#: regular annual grid, extending the ends; ``"native"`` hands the grid to BEAST
#: with the gaps as NaN and lets its sampler treat them as missing, which Rbeast
#: supports. The grid itself is a choice, not a constraint: BEAST accepts missing
#: values, and the interpolated default exists so that every series has the same
#: time base. Use ``"native"`` to measure what that choice costs.
MISSING_MODES = ("interpolate", "native")


def _beast_one_series(
    years: np.ndarray,
    values: np.ndarray,
    mcmc_seed: int = DEFAULT_MCMC_SEED,
    missing: str = "interpolate",
) -> np.ndarray:
    """Run BEAST on one series. Returns per-year posterior of being a changepoint.

    The input series is placed on a regular annual grid spanning
    ``years.min()..years.max()``. With ``missing="interpolate"`` the gaps are
    filled linearly (ends extended); with ``missing="native"`` they stay NaN
    and BEAST handles them as missing values. Either way the returned posterior
    is restricted back to the input years.
    """
    import Rbeast as rb  # lazy

    if missing not in MISSING_MODES:
        raise ValueError(f"missing must be one of {MISSING_MODES}, got {missing!r}")
    y_min, y_max = int(years.min()), int(years.max())
    full_years = np.arange(y_min, y_max + 1)
    series = pd.Series(values, index=years).reindex(full_years)
    if missing == "interpolate":
        series = series.interpolate(limit_direction="both")
    out = rb.beast(
        series.values,
        season="none",
        quiet=True,
        print_param=False,
        print_progress=False,
        mcmc_seed=mcmc_seed,
    )
    posteriors = np.asarray(out.trend.cpOccPr, dtype=float)
    return pd.Series(posteriors, index=full_years).reindex(years).values


def analyze_breakpoints(
    stats: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Sequence[str] = ("unit_id", "crop"),
    year_col: str = "year",
    season_col: str | None = "season",
    season_filter: str | None = "Total Year",
    min_years: int = 8,
    mcmc_seed: int = DEFAULT_MCMC_SEED,
    missing: str = "interpolate",
) -> BreakpointResult:
    """Run BEAST changepoint detection on each (group_cols) timeseries.

    Parameters
    ----------
    stats : pd.DataFrame
        Long-form input with at least ``[*group_cols, year_col, value_col]``.
        If ``season_col`` is present and ``season_filter`` is set, rows
        are filtered to ``stats[season_col] == season_filter`` first.
        Within each group, multiple rows for the same year are summed
        (so a single value per year is fed to BEAST).
    value_col : str
        The variable to analyze (e.g. ``"area"`` or ``"production"``).
    group_cols : sequence of str, default ``("unit_id", "crop")``
        Grouping columns. One BEAST run per unique combination.
    year_col : str, default ``"year"``
    season_col : str or None, default ``"season"``
        Column name to filter on. If ``None`` or absent from ``stats``,
        no season filtering is done.
    season_filter : str or None, default ``"Total Year"``
        Value to keep in ``season_col``. Set ``None`` to skip filtering.
    min_years : int, default 8
        Skip any group with fewer than this many distinct years of data.
        BEAST needs a few points to be informative; very short series
        yield uninformative posteriors and slow the pipeline.
    mcmc_seed : int, default ``DEFAULT_MCMC_SEED``
        Seed for BEAST's sampler. BEAST estimates its posteriors by MCMC,
        so an unseeded run returns slightly different numbers every time
        and no result built on it reproduces. Pass ``0`` for BEAST's own
        random seeding; vary it deliberately to measure how much of a
        reported breakpoint count is sampling noise.
    missing : {"interpolate", "native"}, default ``"interpolate"``
        How a missing year inside a series reaches BEAST: filled linearly on
        the regular annual grid (the default, which every published number
        used), or handed to BEAST as NaN for its own missing-value handling.
        See ``MISSING_MODES``.

    Returns
    -------
    BreakpointResult
    """
    group_cols = tuple(group_cols)
    if missing not in MISSING_MODES:
        raise ValueError(f"missing must be one of {MISSING_MODES}, got {missing!r}")
    absent = [c for c in (*group_cols, year_col, value_col) if c not in stats.columns]
    if absent:
        raise ValueError(
            f"stats is missing required columns {absent}. "
            f"Have: {list(stats.columns)}."
        )

    df = stats
    if season_col and season_col in df.columns and season_filter is not None:
        df = df[df[season_col] == season_filter]
    df = df.dropna(subset=[value_col])
    if df.empty:
        return BreakpointResult(
            per_series=pd.DataFrame(columns=[*group_cols, year_col, "posterior"]),
            group_cols=group_cols,
            year_col=year_col,
            value_col=value_col,
        )

    # Sum multiple rows per (group, year) — handles repeated seasons even
    # when not filtered, and is a no-op for clean Total-Year inputs.
    df = (
        df.groupby([*group_cols, year_col], dropna=False)[value_col]
        .sum()
        .reset_index()
    )

    rows = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        sub = sub.sort_values(year_col)
        years = sub[year_col].to_numpy()
        values = sub[value_col].to_numpy(dtype=float)
        if len(years) < min_years:
            continue
        try:
            posteriors = _beast_one_series(years, values, mcmc_seed=mcmc_seed, missing=missing)
        except Exception:
            # Any individual BEAST failure is logged via skipping; we
            # don't want one bad series to abort the whole sweep.
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        for year, post in zip(years, posteriors):
            row = dict(zip(group_cols, keys))
            row[year_col] = year
            row["posterior"] = float(post) if np.isfinite(post) else 0.0
            rows.append(row)

    per_series = pd.DataFrame(rows, columns=[*group_cols, year_col, "posterior"])
    return BreakpointResult(
        per_series=per_series,
        group_cols=group_cols,
        year_col=year_col,
        value_col=value_col,
    )
