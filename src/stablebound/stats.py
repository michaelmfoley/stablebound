"""Long-form stats aggregation (paper Algorithm 5) + intensive recomputation.

The aggregation routes each (unit_id, year, season, variable) row to its
stable polygon via the remap and sums values. Three routing cases:

1. **Standard case**: ``unit_id`` is in the remap AND alive in the
   snapshot at the row's year → route to ``remap[unit_id]``.

2. **Early reporting**: a unit reports before its official creation
   year. ``unit_id`` may be in the remap (via the modern shapefile or
   baseline) but is NOT in ``snapshot(year)``. Flagged
   ``late_reporting=True`` and surfaced under ``stable_id=unit_id``
   rather than silently rerouted.

3. **Late reporting** (paper §3.4): a unit
   continues to report after it was officially dissolved. Same
   handling as early reporting — ``late_reporting=True``,
   ``stable_id=unit_id``, surfaced rather than re-routed. The
   ``late_reporting`` flag is **snapshot-aware**: it's True whenever
   the unit isn't in the year's snapshot, regardless of whether the
   remap happens to know about the unit (e.g., via the singleton
   fallback).

A ``UserWarning`` is emitted at the end of ``aggregate`` if any
late-reporting rows are present, with a count and a small sample, so
the diagnostic doesn't get lost in a notebook scroll.

**Completeness columns.** Each aggregated row carries:

  - ``n_constituents``: distinct ``unit_id`` values that contributed
  - ``n_in_group``: count of remap members alive in ``snapshot(year)``
    (NA for late-reporting rows — the concept doesn't apply)
  - ``complete``: True iff every alive-at-year group member reported
  - ``missing_unit_ids``: comma-joined list of alive-at-year members
    that DIDN'T report
  - ``constituent_ids``: audit trail of the reporters

Note the two different year windows in play, which answer different
questions. ``late_reporting`` asks "did this unit report outside its
lifespan?" and allows a one-year grace (``snapshot(y) | snapshot(y+1)``)
so the normal parent→child reporting handoff at an event year isn't
flagged. The completeness columns ask "who should have reported this
year?" and use the strict ``snapshot(y)`` — a parent ceasing during year
y and its children first existing in y+1 are not simultaneously members
of year y.

**Intensive variables.** ``aggregate(..., intensive={...})`` emits one derived
row per cell for each declared ratio (yield = production / area). The ratio is
computed from a separate aggregation of only those units that reported BOTH
the numerator and the denominator in that (unit, year, season), so a partial
report (area without production, or the reverse) never biases it. The derived
row's completeness columns name exactly the units used. ``derive_intensive``
remains available for callers holding an already-aggregated frame (the modern
product), where it divides the full sums.
"""

from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

from .lineage import LineageGraph
from .schemas import validate_stats
from .snapshot import build_snapshot

# Row-count threshold above which aggregate() prints a progress
# heads-up. Tuned so synthetic test data (typically <100 rows) stays
# silent but real-world stats (India: 750K) get a "this may take a few
# seconds" line.
_PROGRESS_THRESHOLD_ROWS = 10_000


def aggregate(
    stats_df: pd.DataFrame,
    remap: dict[str, str],
    graph: LineageGraph,
    base_year: int,
    max_year: int,
    *,
    intensive: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Long-form group-by ``(year, season, variable, stable_id)``.

    Args:
        stats_df: Long-form stats with the canonical schema.
        remap: ``unit_id → stable_id`` mapping (from ``build_stable_groups``).
        graph: The lineage graph. Used to compute per-year snapshots
            for snapshot-aware late-reporting detection.
        base_year: Stats rows with ``year < base_year`` are rejected.
        max_year: Used to mark rows beyond the analysis window. Late-reporting
            rows (units no longer in the snapshot at their reporting year)
            are flagged.
        intensive: Optional ``{name: (numerator_var, denominator_var)}``.
            For each pair, a derived row is emitted per cell whose value is
            ``Σ numerator / Σ denominator`` **summed only over the constituent
            units that reported both variables** in that (unit, year, season).
            A unit that reported area but not production contributes to the
            area row and not to the yield row, so the yield is never biased
            by a partial report. The derived row's completeness columns
            describe that set: ``constituent_ids`` are the units used,
            ``missing_unit_ids`` the alive members not used. Any input rows
            whose ``variable`` equals an intensive name are dropped first —
            an intensive is recomputed, never summed.

    Returns:
        DataFrame with columns:
            ``year, season, variable, stable_id, value,
             n_constituents, n_in_group, completeness, complete,
             missing_unit_ids, constituent_ids, late_reporting``
        ``late_reporting`` is True for rows whose ``unit_id`` is not in
        the snapshot at the row's year — typically a unit reporting
        after dissolution or before creation. ``complete`` / ``n_in_group``
        / ``missing_unit_ids`` are NA / "" for late-reporting rows
        because the "members of this stable group" concept doesn't
        apply to a unit reporting outside its lifespan.
    """
    # Schema check (rejects pre-base-year rows, missing columns, NaNs
    # in required fields incl. season). Fail-fast at the data boundary.
    validate_stats(stats_df, base_year=base_year)
    if max_year < base_year:
        raise ValueError(f"max_year ({max_year}) must be >= base_year ({base_year}).")

    df = stats_df.copy()

    # An intensive is recomputed from its inputs, never summed. If the caller's
    # stats already carry rows under an intensive name (a source file that
    # ships yield alongside area and production), drop them here so they can
    # never be added across constituents. derive_intensive used to do this
    # after the fact; doing it first keeps the extensive pass honest too.
    if intensive:
        df = df[~df["variable"].isin(set(intensive.keys()))]

    # Empty-frame guard. Callers (e.g., StableBoundary.aggregate_stats)
    # may pass an empty frame after a variable filter wiped everything
    # out — they already warn about that. Skip the per-year snapshot
    # work and return an empty result with the canonical columns.
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    # Progress heads-up for large datasets. Prints once to stderr at
    # the start of the run so notebook users see something is
    # happening; gated by row count so synthetic test data stays
    # silent. The matching "done" line at the end prints elapsed time.
    _verbose = len(df) >= _PROGRESS_THRESHOLD_ROWS
    _t_start = time.perf_counter()
    if _verbose:
        print(
            f"[stablebound] aggregating {len(df):,} stats rows "
            f"({df['year'].nunique()} years × {df['variable'].nunique()} "
            "variables)...",
            file=sys.stderr,
            flush=True,
        )

    # Work out which districts existed in each year, once per year rather
    # than once per row — a national statistics file has hundreds of
    # thousands of rows but only a few dozen distinct years.
    #
    # The following year is worked out too, because of what happens around a
    # change. A district that split during 2011 may file its 2011 figures
    # either as its old self or as its new halves, and both are legitimate:
    # the change happened partway through the year. So a district is treated
    # as reporting normally if it existed in either that year or the next,
    # and only flagged when it reports well outside its lifetime. Without
    # that leeway every district would be flagged in the year it changed,
    # burying the genuine cases.
    unique_years = sorted(int(y) for y in df["year"].dropna().unique())
    additional = set(remap.keys())
    snapshots_by_year: dict[int, set[str]] = {}
    for y in unique_years:
        snapshots_by_year.setdefault(
            y, set(build_snapshot(graph, y, additional_units=additional))
        )
        snapshots_by_year.setdefault(
            y + 1, set(build_snapshot(graph, y + 1, additional_units=additional))
        )

    # Invert the remap for "group members of stable_id" lookups.
    members_by_stable: dict[str, set[str]] = {}
    for uid, sid in remap.items():
        members_by_stable.setdefault(sid, set()).add(uid)

    # Snapshot-aware late_reporting: True when unit_id isn't in
    # snapshot(year) AND isn't in snapshot(year+1). Catches both
    # late-after-dissolution and early-before-creation while granting
    # a one-event-year grace (so a child reporting in its creation
    # year, or a parent reporting in its dissolution year, isn't
    # flagged). The narrower "unit_id not in remap" check (old
    # behavior) missed cases where a singleton-fallback entry made the
    # remap know about a unit it shouldn't have.
    #
    # Vectorized: build a flat set of (year, unit_id) pairs that are
    # considered alive (including the grace year), then run one list
    # comprehension over zipped columns. df.apply(axis=1) is a Python
    # loop with per-call pandas overhead — on India (~750K rows) it
    # was ~50s; the comprehension is ~0.5s.
    alive_with_grace: dict[int, set[str]] = {}
    for y in snapshots_by_year:
        alive_with_grace[y] = snapshots_by_year.get(y, set()) | snapshots_by_year.get(y + 1, set())

    alive_pairs: set[tuple[int, str]] = set()
    for y, units in alive_with_grace.items():
        for u in units:
            alive_pairs.add((y, u))

    # The extensive pass: every reported row, routed and summed.
    grouped = _route_and_group(df, remap, snapshots_by_year, alive_pairs, members_by_stable)

    # The intensive pass. Each ratio is built from its OWN aggregation of
    # only the rows whose unit reported both inputs in that cell, so the
    # numerator and denominator always describe the same set of units. The
    # routing, grouping and completeness logic is the same helper as above;
    # only the input rows differ.
    if intensive:
        derived_frames = []
        for name, (num_var, den_var) in intensive.items():
            paired = _rows_reporting_both(df, num_var, den_var)
            if paired.empty:
                # Neither input present anywhere — nothing to derive.
                continue
            paired_agg = _route_and_group(
                paired, remap, snapshots_by_year, alive_pairs, members_by_stable
            )
            derived = derive_intensive(paired_agg, {name: (num_var, den_var)})
            derived_frames.append(derived[derived["variable"] == name])
        if derived_frames:
            grouped = pd.concat([grouped, *derived_frames], ignore_index=True)
            grouped = grouped[_OUTPUT_COLUMNS].sort_values(
                ["variable", "year", "season", "stable_id"], na_position="last"
            ).reset_index(drop=True)

    # Say something now about districts reporting outside their lifetime,
    # rather than leaving it as a column in a file nobody opens. It is not
    # an error — the figures are kept, filed under the district that
    # reported them — but it usually points at something wrong upstream.
    # Counted on the extensive rows only, so a derived yield row does not
    # double-count the unit that produced it.
    extensive_rows = grouped[~grouped["variable"].isin(set((intensive or {}).keys()))]
    late_mask = extensive_rows["late_reporting"] == True  # noqa: E712
    n_late = int(late_mask.sum())
    if n_late > 0:
        sample = extensive_rows[late_mask].head(5)
        lines = [
            f"{n_late} late-reporting row(s) detected: units reporting "
            "outside their lifespan (per snapshot at row's year). These "
            "rows are kept in the output with late_reporting=True and "
            "stable_id=unit_id (no auto-rerouting). Sample:",
        ]
        for _, r in sample.iterrows():
            lines.append(
                f"  unit_id={r['stable_id']}  year={int(r['year'])}  "
                f"variable={r['variable']}  value={r['value']}"
            )
        if n_late > len(sample):
            lines.append(f"  ... ({n_late} total)")
        lines.append(
            "To act: filter `late_reporting=True` rows out, or investigate "
            "the lineage end-dates for these units. Future versions may add "
            "an opt-in rerouting-to-parent mode if this becomes a common "
            "data-quality issue."
        )
        warnings.warn("\n".join(lines), UserWarning, stacklevel=2)

    if _verbose:
        print(
            f"[stablebound] aggregation done in "
            f"{time.perf_counter() - _t_start:.1f}s "
            f"({len(grouped):,} output rows).",
            file=sys.stderr,
            flush=True,
        )
    return grouped


#: The canonical output columns of :func:`aggregate`, in order.
_OUTPUT_COLUMNS = [
    "year", "season", "variable", "stable_id", "value",
    "n_constituents", "n_in_group", "completeness", "complete",
    "missing_unit_ids", "constituent_ids", "late_reporting",
]


def _rows_reporting_both(df: pd.DataFrame, num_var: str, den_var: str) -> pd.DataFrame:
    """The numerator and denominator rows of units that reported BOTH.

    A cell is keyed by ``(unit_id, year, season)``. A unit that reported only
    one of the two variables in a cell — or reported it as NaN — is left out
    of both sides, so the ratio built downstream compares like with like.
    Returns an empty frame (same columns) when no unit reported both.
    """
    keys = ["unit_id", "year", "season"]
    num = df[(df["variable"] == num_var) & df["value"].notna()]
    den = df[(df["variable"] == den_var) & df["value"].notna()]
    if num.empty or den.empty:
        return df.iloc[0:0]
    both = num[keys].drop_duplicates().merge(den[keys].drop_duplicates(), on=keys)
    if both.empty:
        return df.iloc[0:0]
    return pd.concat(
        [num.merge(both, on=keys), den.merge(both, on=keys)], ignore_index=True
    )


def _route_and_group(
    stats_rows: pd.DataFrame,
    remap: dict[str, str],
    snapshots_by_year: dict[int, set[str]],
    alive_pairs: set[tuple[int, str]],
    members_by_stable: dict[str, set[str]],
) -> pd.DataFrame:
    """Route each row to its stable polygon, sum, and attach completeness.

    Shared by the extensive pass and each intensive pass of :func:`aggregate`
    so that both apply exactly the same routing and completeness rules. The
    caller supplies the per-year snapshots and the grace-window alive set,
    which are computed once for the whole run.
    """
    df = stats_rows.copy()

    df["late_reporting"] = [
        (int(y), uid) not in alive_pairs
        for y, uid in zip(df["year"], df["unit_id"])
    ]

    # Route: in-snapshot rows go to remap[unit_id]; out-of-snapshot
    # (late_reporting) rows stay under their own ID so they're
    # surfaced rather than silently rerouted.
    df["stable_id"] = df["unit_id"].map(remap)
    df.loc[df["stable_id"].isna(), "stable_id"] = df.loc[df["stable_id"].isna(), "unit_id"]
    df.loc[df["late_reporting"], "stable_id"] = df.loc[df["late_reporting"], "unit_id"]

    # Aggregation. ``dropna=False`` is no longer needed because
    # validate_stats now rejects NaN seasons — but kept for safety
    # on any other nullable group keys.
    # ``min_count=1`` ensures all-NaN groups yield NaN, not 0
    # (preserves "not reported" semantics for null values).
    grouped = (
        df.groupby(
            ["year", "season", "variable", "stable_id", "late_reporting"],
            dropna=False,
        )
        .agg(
            value=("value", lambda s: s.sum(min_count=1)),
            n_constituents=("unit_id", "nunique"),
            constituent_ids=("unit_id", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )

    # Now work out how complete each figure is: of the districts that made
    # up this group in this year, how many actually reported? Answer it once
    # per group per year and look it up, rather than recomputing per row.
    #
    # Note this uses the strict year, NOT the leeway applied above.
    # Completeness uses the STRICT snapshot, not the grace window that
    # `late_reporting` uses above. The grace window is right for "did this
    # unit report outside its lifespan?" — it suppresses the one-year
    # parent/child handoff. It is wrong for "who should have reported this
    # year?", because it counts a ceasing parent AND its not-yet-existing
    # children as members of the same year. Measured on India 1997-2025 that
    # inflated the denominator for 9,418 of 556,316 non-late cells and
    # flipped `complete` on 4,589 of them — e.g. Gujarat 1997, where both
    # parents of the Patan split reported and the row was still marked
    # incomplete, naming three districts that did not exist until 1998.
    alive_members_by_year_sid: dict[tuple[int, str], set[str]] = {}
    for sid, members in members_by_stable.items():
        for y in snapshots_by_year:
            alive_members_by_year_sid[(y, sid)] = members & snapshots_by_year[y]

    def _row_completeness(year: int, sid: str, late: bool, constituent_ids: str
                          ) -> tuple[object, object, str]:
        if late:
            return pd.NA, pd.NA, ""
        alive = alive_members_by_year_sid.get((year, sid), set())
        reporters = set(constituent_ids.split(",")) if constituent_ids else set()
        missing = sorted(alive - reporters)
        # Subset, not cardinality. `len(reporters) == len(alive)` could read
        # True while `missing` was non-empty whenever a reporter fell outside
        # the alive set (an early-reporting unit), which is exactly the case
        # the strict window above makes more common. `complete` and
        # `missing_unit_ids` must never disagree.
        return len(alive), (not missing), ",".join(missing)

    triples = [
        _row_completeness(int(y), sid, bool(late), cids)
        for y, sid, late, cids in zip(
            grouped["year"], grouped["stable_id"],
            grouped["late_reporting"], grouped["constituent_ids"],
        )
    ]
    grouped["n_in_group"] = [t[0] for t in triples]
    grouped["complete"] = [t[1] for t in triples]
    grouped["missing_unit_ids"] = [t[2] for t in triples]
    # Fraction of the units that SHOULD have reported that actually did.
    #
    # Deliberately not n_constituents / n_in_group: n_constituents counts
    # every reporter, including an early reporter that is not yet in the
    # year's snapshot, which pushed the ratio above 1.0. Measuring coverage
    # of the expected set keeps it in [0, 1] and makes it agree with
    # `complete` by construction (1.0 exactly when nothing is missing).
    grouped["completeness"] = [
        # notna first: `pd.NA and x` raises rather than short-circuiting.
        ((d - m) / d) if (pd.notna(d) and d) else pd.NA
        for d, m in zip(
            grouped["n_in_group"],
            [len(x.split(",")) if x else 0 for x in grouped["missing_unit_ids"]],
        )
    ]

    # Reorder columns and sort for cross-run determinism.
    return grouped[_OUTPUT_COLUMNS].sort_values(
        ["variable", "year", "season", "stable_id"], na_position="last"
    ).reset_index(drop=True)


def derive_intensive(
    agg: pd.DataFrame,
    intensive_pairs: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Recompute intensive variables from already-aggregated extensives.

    For each ``(intensive_name → (numerator_var, denominator_var))`` pair,
    join the numerator and denominator rows on
    ``(year, season, stable_id)`` and emit a new row with
    ``variable = intensive_name``, ``value = numerator / denominator``.
    Where the denominator is zero or missing, the intensive value is NaN.

    Returns the input frame plus the new derived rows. Existing rows with
    ``variable == intensive_name`` are dropped first (they would have been
    summed across constituents, which is wrong for intensives).
    """
    df = agg.copy()
    if not intensive_pairs:
        return df

    # Defensive: drop any pre-existing rows for the intensive variable
    # names. If the stats input erroneously included yield (or any other
    # intensive) as a row, ``aggregate`` would have summed it across
    # constituents — which is the exact wrong thing for an intensive.
    # Drop those rows here so the caller's bad input doesn't survive.
    df = df[~df["variable"].isin(intensive_pairs.keys())].copy()

    new_rows = []
    # Join keys: an intensive value is defined per (year, season,
    # stable_id, late_reporting) tuple. We merge num and den frames on
    # this key. The late_reporting flag is included so num and den match
    # only when both share the same flag value.
    keys = ["year", "season", "stable_id", "late_reporting"]
    # Completeness columns are carried through the join, not blanked. A yield
    # is only as complete as the scarcer of its two inputs, restricted to the
    # units that reported BOTH — see _intensive_completeness below.
    carry = ["n_in_group", "constituent_ids", "missing_unit_ids"]
    # The modern product's frame doesn't carry completeness columns yet, and
    # it round-trips through this function (modern.py::_derive_intensive_for_modern).
    # Degrade to NA rather than raising: an absent input can't produce a
    # completeness figure, and inventing one would be worse than saying so.
    has_completeness = all(c in df.columns for c in carry)
    if not has_completeness:
        carry = []
    for intensive_name, (num_var, den_var) in intensive_pairs.items():
        num = df[df["variable"] == num_var][keys + ["value"] + carry].rename(
            columns={"value": "_num", **{c: f"{c}_num" for c in carry}}
        )
        den = df[df["variable"] == den_var][keys + ["value"] + carry].rename(
            columns={"value": "_den", **{c: f"{c}_den" for c in carry}}
        )
        if num.empty or den.empty:
            # Either component absent — can't compute the intensive.
            # Silently skip rather than emit NaN rows.
            continue
        # Inner join: emit a row only when both numerator and denominator
        # are present for the same (year, season, stable_id) cell.
        joined = num.merge(den, on=keys, how="inner")
        # Avoid divide-by-zero by masking zero denominators to NaN. The
        # standard pandas behavior of dividing by 0 → inf would mislead
        # downstream consumers.
        joined["value"] = joined["_num"] / joined["_den"].where(joined["_den"] != 0)
        joined["variable"] = intensive_name
        # A yield is only as trustworthy as the two figures it came from, so
        # carry a completeness for it too rather than leaving it blank.
        if has_completeness:
            _intensive_completeness(joined)
        else:
            # Nothing to carry when the inputs never had it.
            for col, blank in (
                ("n_constituents", pd.NA), ("n_in_group", pd.NA),
                ("completeness", pd.NA), ("complete", pd.NA),
                ("missing_unit_ids", ""), ("constituent_ids", ""),
            ):
                joined[col] = blank
        new_rows.append(joined[
            ["year", "season", "variable", "stable_id", "value",
             "n_constituents", "n_in_group", "completeness", "complete",
             "missing_unit_ids", "constituent_ids", "late_reporting"]
        ])

    if new_rows:
        df = pd.concat([df, *new_rows], ignore_index=True)
    return df.sort_values(
        ["variable", "year", "season", "stable_id"], na_position="last"
    ).reset_index(drop=True)


def _split_ids(value: object) -> set[str]:
    """Parse a comma-joined id column back into a set (empty for NA/"")."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value)
    return set(text.split(",")) if text else set()


def _intensive_completeness(joined: pd.DataFrame) -> None:
    """Fill an intensive row's completeness columns in place.

    A ratio is only trustworthy where *both* of its inputs are, so:

    - ``constituent_ids`` — the **intersection**: units that reported the
      numerator AND the denominator. A unit that reported only production
      contributed to neither side of the ratio for its area.
    - ``missing_unit_ids`` — the **union** of what each side was missing.
    - ``n_in_group`` — the same group size as the inputs; the numerator and
      denominator rows share a ``(year, season, stable_id)``, so their group
      membership is identical by construction.
    - ``complete`` — true only when nothing is missing on either side.

    Previously all of these were hardcoded to NA/"" , which left yield — usually
    the headline variable — with no completeness information at all.
    """
    contributors = [
        _split_ids(a) & _split_ids(b)
        for a, b in zip(joined["constituent_ids_num"], joined["constituent_ids_den"])
    ]
    missing = [
        _split_ids(a) | _split_ids(b)
        for a, b in zip(joined["missing_unit_ids_num"], joined["missing_unit_ids_den"])
    ]
    joined["constituent_ids"] = [",".join(sorted(s)) for s in contributors]
    joined["missing_unit_ids"] = [",".join(sorted(s)) for s in missing]
    joined["n_constituents"] = [len(s) for s in contributors]
    # Both sides share the group, so either column works; prefer the
    # numerator's and fall back for safety.
    joined["n_in_group"] = joined["n_in_group_num"].where(
        joined["n_in_group_num"].notna(), joined["n_in_group_den"]
    )
    joined["complete"] = [not s for s in missing]
    joined["completeness"] = [
        # Same definition as the extensive path: coverage of the expected
        # set, so it stays in [0, 1] and equals 1.0 exactly when complete.
        ((d - len(m)) / d) if (pd.notna(d) and d) else pd.NA
        for d, m in zip(joined["n_in_group"], missing)
    ]
