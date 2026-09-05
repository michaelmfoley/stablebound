"""Reconciliation diagnostics — paper Algorithm 4.

.. warning::

    **2026-06-04: this module is currently disabled at the StableBoundary
    public API.** Empirical evaluation on India 1997-2022 found that the
    window-averaged drop and sum-jump tests cannot distinguish the
    canonical post-event reporting handoff (parent reports the event year,
    children begin reporting the next year) from sustained double-counting,
    and flag it at a ~99% rate. The active ``merge`` / ``subtract`` modes
    delete legitimate post-event child rows in response, causing
    substantial data loss. ``StableBoundary.aggregate_stats`` and
    ``StableBoundary.reconcile_stats`` raise ``NotImplementedError`` for
    any mode other than ``"off"``. The underlying ``reconcile()`` function
    in this module is left callable so the unit tests can continue to
    exercise the test math, and so that future fixes can be implemented
    incrementally. The diagnostics that surfaced this are in the authors'
    diagnostic notebook (not part of this repository).

    **Treat this module as experimental.** It is not imported by either
    product's happy path and nothing in ``docs/`` describes it as usable.
    A rework needs to replace the window-averaged comparison with a
    per-year overlap test: the failure mode is that averaging over
    ``window`` years smears the one-year parent/child handoff across the
    whole window, so a clean handoff and a genuine multi-year double-count
    produce the same statistic. Any fix must also be evaluated against the
    handoff case explicitly — a lower false-positive rate on India is the
    minimum bar for re-enabling it.

Given a long-form stats frame and a relationship graph, the algorithm
identifies events (Split or Merge) where the reported data behaves as if
the boundary change weren't yet reflected — that is, where the parent
unit's row appears to "double-count" its successors' territory, or vice
versa for merges.

Two diagnostics, computed over a rolling pre/post window of length
``window`` (default 3 years):

**Drop test (eq. 3 in the paper).** For a split event affecting parent ``i``
in year ``s`` with successor(s) ``j ∈ C(i)``:

    r_t = (Σ_j Y_{j,t}) / Y_{i,t-1}    expected fractional drop
    d_t = 1 - Y_{i,t} / Y_{i,t-1}      observed fractional drop
    flag if  d_t < r_t - tau_drop

The parent's observed drop falls short of the expected territorial loss by
more than the tolerance.

**Sum-jump test (eq. 4 in the paper).**

    flag if  (Y_{i,t} + Σ_j Y_{j,t}) / Y_{i,t-1}  >  1 + tau_sum

Post-event sum of parent + successors exceeds the pre-event baseline by
more than the tolerance — suggests the parent is still reporting the full
pre-event territory.

For merge events, the same diagnostics apply with the roles of parent and
successor reversed: the persisting-parent row plays the parent role and
the new merged-unit row plays the successor role.

**Four modes** (`off`, `flag`, `merge`, `subtract`).

- ``off``: skip the diagnostic entirely. Returns the inputs unchanged
  with an empty flags DataFrame. Useful for exploratory work or when
  the user has decided the diagnostic isn't yet mature enough to
  consume — both reconciliation and breakpoint detection are
  intentionally toggleable, since neither will ever be perfectly
  solved and the package shouldn't force unfinished diagnostics on
  users who'd rather skip them.
- ``flag``: run diagnostics; return inputs unchanged plus a populated
  flags DataFrame. Package default. The underlying data is never
  modified.
- ``merge``: union flagged stable polygons; drop redundant rows.
  Modifies the remap.
- ``subtract``: subtract double-counted rows from parent/merged-unit
  values per (year, season, variable). Modifies the stats. Falls
  back to merge when subtract is undefined.

See module docstring of ``boundary.py`` and ``docs/methodology.md``
for the per-mode definitions.

Redistribute events do not admit this style of diagnostic (the territory
transferred depends on land use that cannot be inferred from reported
totals alone — paper §3.3 final paragraph). They are surfaced in the flags
frame as ``mode_applied="redistribute_unreconcilable"`` for transparency
but not auto-modified.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

from .lineage import LineageGraph

ReconcileMode = Literal["off", "flag", "merge", "subtract"]
_VALID_MODES = ("off", "flag", "merge", "subtract")

# Empty-shape flags DataFrame returned by mode="off" (and by other code
# paths that need an empty-but-typed frame). Kept as a module-level
# template so the column list stays consistent across producers.
_EMPTY_FLAGS_COLUMNS = (
    "event_year", "event_type", "unit_id", "variable", "season",
    "r_t", "d_t", "sum_jump_ratio",
    "drop_flag", "sum_flag", "flagged", "mode_applied", "note",
)


def _empty_flags_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_EMPTY_FLAGS_COLUMNS))


# --- Public API ----------------------------------------------------------


def reconcile(
    stats_df: pd.DataFrame,
    graph: LineageGraph,
    remap: dict[str, str],
    *,
    mode: ReconcileMode = "flag",
    tau_drop: float = 0.15,
    tau_sum: float = 0.15,
    window: int = 3,
    test_variables: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Run drop + sum-jump diagnostics; optionally modify stats/remap per ``mode``.

    Returns:
        ``(reconciled_stats, modified_remap, flags_df)``. In ``flag`` mode
        ``reconciled_stats`` and ``modified_remap`` are unmodified copies of
        the inputs.

    The ``flags_df`` columns:
        ``event_year, event_type, unit_id, variable, season, r_t, d_t,
         sum_jump_ratio, drop_flag, sum_flag, flagged, mode_applied,
         note``

    where ``unit_id`` is the parent for splits and the persisting parent
    (or merged child) for merges; ``season`` is always ``"All"`` because
    the diagnostic sums values across seasons (see ``_build_window_lookup``);
    ``flagged = drop_flag or sum_flag``; ``mode_applied`` is the mode
    actually applied to this event (may differ from ``mode`` when a
    subtract→merge fallback occurs).
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}.")

    # "off" mode: skip diagnostics entirely. Useful when the user wants
    # aggregation without the reconciliation step (e.g., exploratory
    # work, or when the diagnostic's noise/maturity makes its output
    # not worth surfacing). Both reconciliation and breakpoint
    # detection are intentionally toggleable — neither will ever be
    # perfectly solved, and the package shouldn't force unfinished
    # diagnostics on users who'd rather skip them.
    if mode == "off":
        return stats_df.copy(), dict(remap), _empty_flags_df()

    # Group raw rows into structured event objects. The RT encodes a
    # 1-parent-N-children Split as N rows; we want one event per
    # (year, parent_id) with a tuple of children. Symmetrically for Merges.
    splits = _group_splits(graph)
    merges = _group_merges(graph)

    # Default to running diagnostics on every variable in the stats frame.
    # Callers can pass a subset if they want to focus on (e.g.) just area
    # variables, which are usually the cleanest signal of territorial change.
    if test_variables is None:
        test_variables = sorted(stats_df["variable"].unique().tolist())
    test_variables = list(test_variables)

    # Pre-pivot stats into ``{(unit_id, variable): {year: value}}``. The
    # alternative — filtering the full DataFrame inside the (event ×
    # variable) loop — is O(N_events × N_variables × N_stats_rows). For
    # India that's ~100,000 × ~370K = 37 billion comparisons, which took
    # ~17 minutes. The pre-pivot is one groupby pass plus O(window)
    # dict lookups inside the loop, taking aggregate_stats from 17 min →
    # ~2 sec on the same data. Algorithm unchanged.
    lookup = _build_window_lookup(stats_df)

    # Run drop + sum-jump diagnostics for every (event, variable) pair.
    # _evaluate_* returns a row dict regardless of whether anything fired,
    # so the flags_df has uniform shape regardless of input.
    flag_rows: list[dict] = []
    for ev in splits:
        for var in test_variables:
            flag_rows.append(_evaluate_split(ev, var, lookup, tau_drop, tau_sum, window))
    for ev in merges:
        for var in test_variables:
            flag_rows.append(_evaluate_merge(ev, var, lookup, tau_drop, tau_sum, window))

    flags_df = pd.DataFrame(flag_rows)

    # Redistribute events get audit rows but no flag — the paper §3.3
    # final paragraph notes redistributes can't be reconciled from totals
    # alone. Surfacing them as ``mode_applied="redistribute_unreconcilable"``
    # is just so users know the events were considered.
    redistribute_rows = _redistribute_audit_rows(graph)
    if redistribute_rows:
        flags_df = pd.concat([flags_df, pd.DataFrame(redistribute_rows)], ignore_index=True)

    if flags_df.empty:
        flags_df = _empty_flags_df()

    if mode == "flag":
        flags_df.loc[flags_df["flagged"] == True, "mode_applied"] = "flag"  # noqa: E712
        return stats_df.copy(), dict(remap), flags_df

    if mode == "merge":
        new_stats, new_remap, flags_df = _apply_merge_mode(
            stats_df, remap, flags_df, splits, merges
        )
        return new_stats, new_remap, flags_df

    # mode == "subtract"
    new_stats, new_remap, flags_df = _apply_subtract_mode(
        stats_df, remap, flags_df, splits, merges
    )
    return new_stats, new_remap, flags_df


# --- Event grouping -------------------------------------------------------


@dataclass(frozen=True)
class _SplitEvent:
    year: int
    parent_id: str
    children: tuple[str, ...]


@dataclass(frozen=True)
class _MergeEvent:
    year: int
    parents: tuple[str, ...]
    child_id: str


def _group_splits(graph: LineageGraph) -> list[_SplitEvent]:
    """All Split events, one entry per (year, parent_id) with its children."""
    ev = graph.events
    splits = ev[ev["event_type"] == "Split"]
    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    for _, row in splits.iterrows():
        grouped[(int(row["event_year"]), row["parent_id"])].append(row["child_id"])
    return [
        _SplitEvent(year=year, parent_id=parent, children=tuple(sorted(set(kids))))
        for (year, parent), kids in sorted(grouped.items())
    ]


def _group_merges(graph: LineageGraph) -> list[_MergeEvent]:
    """All Merge events, one entry per (year, child_id) with its parents."""
    ev = graph.events
    merges = ev[ev["event_type"] == "Merge"]
    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    for _, row in merges.iterrows():
        grouped[(int(row["event_year"]), row["child_id"])].append(row["parent_id"])
    return [
        _MergeEvent(year=year, parents=tuple(sorted(set(parents))), child_id=child)
        for (year, child), parents in sorted(grouped.items())
    ]


# --- Diagnostics ---------------------------------------------------------


def _build_window_lookup(stats_df: pd.DataFrame) -> dict[tuple[str, str], dict[int, float]]:
    """Pre-pivot stats into ``{(unit_id, variable): {year: summed_value}}``.

    Performance optimization: this single groupby pass replaces what would
    otherwise be O(N_events × N_variables) repeated filterings of the full
    DataFrame inside the diagnostic loop. Values are summed across seasons
    so that a single-variable diagnostic captures the unit's full reporting.
    """
    if stats_df.empty:
        return {}
    grouped = stats_df.groupby(["unit_id", "variable", "year"], sort=False)["value"].sum()
    out: dict[tuple[str, str], dict[int, float]] = {}
    for (uid, var, year), val in grouped.items():
        out.setdefault((str(uid), str(var)), {})[int(year)] = float(val)
    return out


def _window_mean(
    lookup: dict[tuple[str, str], dict[int, float]],
    unit_id: str,
    variable: str,
    years: range,
) -> float | None:
    """Mean of pre-pivoted values for ``(unit_id, variable)`` across ``years``.

    ``None`` if no year in the window has a value.
    """
    series = lookup.get((unit_id, variable))
    if not series:
        return None
    vals = [series[y] for y in years if y in series]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _evaluate_split(
    event: _SplitEvent, variable: str,
    lookup: dict[tuple[str, str], dict[int, float]],
    tau_drop: float, tau_sum: float, window: int,
) -> dict:
    """Drop and sum-jump tests for a Split event on a single variable.

    Paper equations 3 and 4. Computed over rolling windows of length
    ``window`` on each side of the event year — single-year tests are
    too sensitive to drought / bumper-year shocks.
    """
    # The pre-window is [year - window, year). The post-window is
    # [year, year + window). Note the half-open intervals: the event year
    # itself is in "post" — the paper convention is that boundary events
    # take effect at the START of the named year, so reporting in that
    # year already reflects the new geography.
    pre_years = range(event.year - window, event.year)
    post_years = range(event.year, event.year + window)

    # Parent's mean reported value before and after the split.
    parent_pre = _window_mean(lookup, event.parent_id, variable, pre_years)
    parent_post = _window_mean(lookup, event.parent_id, variable, post_years)

    # Sum of all successor (child) means in the post window. We use ANY
    # successor with data — partial reporting still informs the test.
    successor_post_total = 0.0
    have_any_successor = False
    for child in event.children:
        sm = _window_mean(lookup, child, variable, post_years)
        if sm is not None:
            successor_post_total += sm
            have_any_successor = True

    # Default flag row — every field starts neutral. We fill in computed
    # values as we go so the flags_df has uniform shape even when a test
    # couldn't run (e.g., no pre-split data).
    base = {
        "event_year": event.year,
        "event_type": "Split",
        "unit_id": event.parent_id,
        "variable": variable,
        # Diagnostic sums values across all seasons (see _build_window_lookup),
        # so the season column is "All". Keeping the column makes downstream
        # code that joins on (event_year, variable, season) keys work uniformly.
        "season": "All",
        "r_t": float("nan"),
        "d_t": float("nan"),
        "sum_jump_ratio": float("nan"),
        "drop_flag": False,
        "sum_flag": False,
        "flagged": False,
        "mode_applied": "none",
        "note": "",
    }

    # Bail-out conditions: no pre-split baseline (can't compute drop), or
    # no post-split data at all (can't compute either test). Surface a
    # note for the user; the row carries no flags.
    if parent_pre is None or parent_pre == 0:
        base["note"] = "no pre-split parent data"
        return base
    if parent_post is None and not have_any_successor:
        base["note"] = "no post-split data for parent or successors"
        return base

    # r_t: expected fractional drop in the parent (= the share of its
    # pre-event reporting now attributable to the successors). Under
    # clean reporting, the parent's d_t should ≈ r_t.
    if have_any_successor:
        base["r_t"] = successor_post_total / parent_pre

    # d_t: observed fractional drop. d_t = 1 - Y_{i, post} / Y_{i, pre}.
    # If the parent didn't drop (still reporting full pre-event total),
    # d_t ≈ 0; if the parent dropped to zero, d_t ≈ 1.
    if parent_post is not None:
        base["d_t"] = 1.0 - parent_post / parent_pre

    # Sum-jump test (eq. 4): is the post-event sum (parent + successors)
    # bigger than the pre-event parent baseline by more than tau_sum?
    # Under clean reporting, the sum should be ≈ pre-event value.
    if parent_post is not None and have_any_successor:
        base["sum_jump_ratio"] = (parent_post + successor_post_total) / parent_pre
        base["sum_flag"] = base["sum_jump_ratio"] > 1.0 + tau_sum

    # Drop test (eq. 3): is the observed drop short of the expected drop
    # by more than tau_drop? d_t < r_t - tau_drop ⇔ parent under-dropped.
    if (
        parent_post is not None
        and have_any_successor
        and base["d_t"] < base["r_t"] - tau_drop
    ):
        base["drop_flag"] = True

    # Either test firing flags the event. The two diagnostics are
    # somewhat correlated but not redundant — an old-row that under-drops
    # also typically inflates the sum, but tau values can place one or
    # the other on the threshold.
    base["flagged"] = base["drop_flag"] or base["sum_flag"]
    return base


def _evaluate_merge(
    event: _MergeEvent, variable: str,
    lookup: dict[tuple[str, str], dict[int, float]],
    tau_drop: float, tau_sum: float, window: int,
) -> dict:
    """Drop and sum-jump tests for a Merge event on a single variable.

    Symmetric to the split case with parent/successor roles reversed:
    the persisting parents play the role of the "stable row" (they're the
    ones that should drop after the merge); the new merged-unit child
    plays the role of the "successor" (it appears for the first time in
    the post window).

    A common failure mode this catches: a merge where one of the
    pre-merge parent rows continues to report at its pre-merge level
    after the merge — double-counting against the merged-unit row.
    """
    pre_years = range(event.year - window, event.year)
    post_years = range(event.year, event.year + window)

    # Sum of all parents' pre-merge means. This is the "baseline" that
    # post-merge reporting should approximately reproduce.
    parent_pre_total = 0.0
    have_any_parent_pre = False
    for parent in event.parents:
        pm = _window_mean(lookup, parent, variable, pre_years)
        if pm is not None:
            parent_pre_total += pm
            have_any_parent_pre = True

    # We pick a "persisting parent" — the parent that's still reporting
    # post-merge (which it shouldn't be, in clean data). Heuristic: pick
    # the parent with the largest post-merge mean. This handles the
    # multi-parent merge case where some parents fully ceased and others
    # continued; we want to flag the strongest persistence.
    persisting_parent_post = None
    persisting_id: str | None = None
    for parent in event.parents:
        pm = _window_mean(lookup, parent, variable, post_years)
        if pm is not None and (persisting_parent_post is None or pm > persisting_parent_post):
            persisting_parent_post = pm
            persisting_id = parent

    # Merged-unit child's post-merge mean. This is the "new combined"
    # reporting that should reflect all of the merged territory.
    child_post = _window_mean(lookup, event.child_id, variable, post_years)

    base = {
        "event_year": event.year,
        "event_type": "Merge",
        "unit_id": persisting_id if persisting_id is not None else event.child_id,
        "variable": variable,
        "season": "All",  # diagnostic sums across seasons
        "r_t": float("nan"),
        "d_t": float("nan"),
        "sum_jump_ratio": float("nan"),
        "drop_flag": False,
        "sum_flag": False,
        "flagged": False,
        "mode_applied": "none",
        "note": "",
    }

    if not have_any_parent_pre:
        base["note"] = "no pre-merge parent data"
        return base
    if child_post is None and persisting_parent_post is None:
        base["note"] = "no post-merge data"
        return base

    # Drop test for merges: in clean data the persisting parents should
    # cease reporting after the merge. Expected drop = 1 (full
    # cessation). If observed drop < 1 - tau_drop, the parent is still
    # over-reporting — flag.
    if persisting_parent_post is not None and parent_pre_total > 0:
        base["d_t"] = 1.0 - persisting_parent_post / parent_pre_total
        if base["d_t"] < 1.0 - tau_drop:
            base["drop_flag"] = True

    # Sum-jump test for merges: persisting-parent + merged-child summed
    # should ≈ parent_pre_total. If > 1 + tau_sum, the parents and
    # merged unit are double-counting.
    if (
        child_post is not None
        and persisting_parent_post is not None
        and parent_pre_total > 0
    ):
        base["sum_jump_ratio"] = (persisting_parent_post + child_post) / parent_pre_total
        base["sum_flag"] = base["sum_jump_ratio"] > 1.0 + tau_sum

    base["flagged"] = base["drop_flag"] or base["sum_flag"]
    return base


def _redistribute_audit_rows(graph: LineageGraph) -> list[dict]:
    """Audit rows for Redistribute events (not auto-reconciled — paper §3.3)."""
    ev = graph.events
    redists = ev[ev["event_type"] == "Redistribute"]
    rows = []
    for (year, parent), _g in redists.groupby(["event_year", "parent_id"]):
        rows.append(
            {
                "event_year": int(year),
                "event_type": "Redistribute",
                "unit_id": parent,
                "variable": "",
                "season": "All",
                "r_t": float("nan"),
                "d_t": float("nan"),
                "sum_jump_ratio": float("nan"),
                "drop_flag": False,
                "sum_flag": False,
                "flagged": False,
                "mode_applied": "redistribute_unreconcilable",
                "note": "redistribute events are not auto-reconciled (paper §3.3)",
            }
        )
    return rows


# --- Mode application -----------------------------------------------------


def _apply_merge_mode(
    stats_df: pd.DataFrame,
    remap: dict[str, str],
    flags_df: pd.DataFrame,
    splits: list[_SplitEvent],
    merges: list[_MergeEvent],
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Merge mode: union flagged stable polygons; drop redundant child/parent rows.

    For each split event flagged on any variable: ensure parent and all
    successor children share the same stable_id (rewriting remap), and drop
    successor rows from the stats frame in post-event years.

    For each merge event flagged on any variable: ensure all parents and
    the merged child share the same stable_id; drop the persisting-parent
    rows in post-event years.
    """
    new_remap = dict(remap)
    rows_to_drop = pd.Series(False, index=stats_df.index)

    flagged = flags_df[flags_df["flagged"] == True]  # noqa: E712

    for ev in splits:
        ev_flags = flagged[
            (flagged["event_year"] == ev.year)
            & (flagged["event_type"] == "Split")
            & (flagged["unit_id"] == ev.parent_id)
        ]
        if ev_flags.empty:
            continue
        # Union the parent and all children into one stable group.
        members = (ev.parent_id, *ev.children)
        new_remap = _union_into_smallest(new_remap, members)
        # Drop successor rows from post-event years (they double-count the parent).
        for child in ev.children:
            mask = (stats_df["unit_id"] == child) & (stats_df["year"] >= ev.year)
            rows_to_drop |= mask
        flags_df.loc[ev_flags.index, "mode_applied"] = "merge"

    for ev in merges:
        ev_flags = flagged[
            (flagged["event_year"] == ev.year) & (flagged["event_type"] == "Merge")
        ]
        ev_flags = ev_flags[
            ev_flags["unit_id"].isin(ev.parents) | (ev_flags["unit_id"] == ev.child_id)
        ]
        if ev_flags.empty:
            continue
        members = (*ev.parents, ev.child_id)
        new_remap = _union_into_smallest(new_remap, members)
        # Drop persisting-parent rows from post-event years.
        for parent in ev.parents:
            mask = (stats_df["unit_id"] == parent) & (stats_df["year"] >= ev.year)
            rows_to_drop |= mask
        flags_df.loc[ev_flags.index, "mode_applied"] = "merge"

    new_stats = stats_df[~rows_to_drop].reset_index(drop=True)
    return new_stats, new_remap, flags_df


def _apply_subtract_mode(
    stats_df: pd.DataFrame,
    remap: dict[str, str],
    flags_df: pd.DataFrame,
    splits: list[_SplitEvent],
    merges: list[_MergeEvent],
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Subtract mode: subtract double-counted rows from parent/merged-unit rows.

    For splits: subtract each successor's value from the parent's value in
    post-event years (per (year, season, variable)).

    For merges: subtract each persisting-parent's value from the merged
    child's value in post-event years. **Fallback**: if the parent and its
    co-parents map to the same stable polygon (i.e., the base year postdates
    the merge, so they're already in one group), subtract is undefined for
    that conflict; merge mode is applied instead.
    """
    df = stats_df.copy()
    df["__row__"] = range(len(df))
    new_remap = dict(remap)

    flagged = flags_df[flags_df["flagged"] == True]  # noqa: E712

    # Splits: subtract successor rows from parent rows in post-event years.
    for ev in splits:
        ev_flags = flagged[
            (flagged["event_year"] == ev.year)
            & (flagged["event_type"] == "Split")
            & (flagged["unit_id"] == ev.parent_id)
        ]
        if ev_flags.empty:
            continue
        for var in ev_flags["variable"].unique():
            df = _subtract_split(df, ev, var)
        flags_df.loc[ev_flags.index, "mode_applied"] = "subtract"

    # Merges: subtract persisting-parent rows from merged-child rows.
    for ev in merges:
        ev_flags = flagged[
            (flagged["event_year"] == ev.year) & (flagged["event_type"] == "Merge")
        ]
        if ev_flags.empty:
            continue
        # Fallback condition: if all parents + child are already in one
        # stable group, subtract is undefined.
        all_members = (*ev.parents, ev.child_id)
        stable_ids = {new_remap.get(m, m) for m in all_members}
        if len(stable_ids) == 1:
            # All in one polygon; subtract undefined → fall back to merge.
            new_remap = _union_into_smallest(new_remap, all_members)
            for parent in ev.parents:
                mask = (df["unit_id"] == parent) & (df["year"] >= ev.year)
                df = df[~mask].copy()
            flags_df.loc[ev_flags.index, "mode_applied"] = "subtract→merge_fallback"
        else:
            for var in ev_flags["variable"].unique():
                df = _subtract_merge(df, ev, var)
            flags_df.loc[ev_flags.index, "mode_applied"] = "subtract"

    df = df.drop(columns=["__row__"]).reset_index(drop=True)
    return df, new_remap, flags_df


def _subtract_split(df: pd.DataFrame, ev: _SplitEvent, variable: str) -> pd.DataFrame:
    """For each post-event (year, season), subtract Σ child values from parent value."""
    # The situation this addresses: after a district splits, some sources
    # keep publishing a figure for the old district that already includes its
    # offspring, while also publishing the offspring separately. Adding those
    # together double-counts, so the offspring's share is taken back out of
    # the parent's figure.
    #
    # Whether this is the right reading of such data is exactly what has not
    # been established -- see the warning at the top of this module.
    for year in sorted(df.loc[df["year"] >= ev.year, "year"].unique()):
        year_df = df[(df["year"] == year) & (df["variable"] == variable)]
        for season in year_df["season"].unique():
            sub = year_df[year_df["season"].astype(str) == str(season)]
            parent_rows = sub[sub["unit_id"] == ev.parent_id]
            children_rows = sub[sub["unit_id"].isin(ev.children)]
            # Only act where both sides are actually present. One without the
            # other is not double-counting, it is just a normal report.
            if parent_rows.empty or children_rows.empty:
                continue
            child_total = float(children_rows["value"].sum())
            for idx in parent_rows.index:
                df.at[idx, "value"] = float(df.at[idx, "value"]) - child_total
    return df


def _subtract_merge(df: pd.DataFrame, ev: _MergeEvent, variable: str) -> pd.DataFrame:
    """For each post-merge (year, season), subtract Σ persisting-parent values from child value."""
    # The mirror image for merges: a district absorbs another, and the source
    # publishes a combined figure for the enlarged district while still
    # publishing the absorbed one separately. Same remedy, opposite direction.
    for year in sorted(df.loc[df["year"] >= ev.year, "year"].unique()):
        year_df = df[(df["year"] == year) & (df["variable"] == variable)]
        for season in year_df["season"].unique():
            sub = year_df[year_df["season"].astype(str) == str(season)]
            child_rows = sub[sub["unit_id"] == ev.child_id]
            parent_rows = sub[sub["unit_id"].isin(ev.parents)]
            if child_rows.empty or parent_rows.empty:
                continue
            parent_total = float(parent_rows["value"].sum())
            for idx in child_rows.index:
                df.at[idx, "value"] = float(df.at[idx, "value"]) - parent_total
    return df


def _union_into_smallest(remap: dict[str, str], members: Iterable[str]) -> dict[str, str]:
    """Union all ``members`` and any units already mapped to their stable_ids
    into a single group. The new stable_id is the lex-smallest among them.
    """
    members = list(members)
    # Collect every unit currently sharing a stable_id with any member.
    target_stables = {remap.get(m, m) for m in members}
    affected = [u for u, s in remap.items() if s in target_stables]
    affected.extend(m for m in members if m not in remap)
    candidates = list(target_stables) + members
    canonical = min(candidates)
    new = dict(remap)
    for u in set(affected) | set(members):
        new[u] = canonical
    return new
