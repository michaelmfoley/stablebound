"""Modern boundary algorithm — fraction computation and ledger composition.

Covers the cases enumerated in docs/PLAN.md Phase M1: pure Split, pure
Merge, Redistribute pool, cascade depth, NaN propagation, insufficient
window data, NameChange / Coarse passthrough, per-variable windows,
late-reporting, and the no-events trivial case.
"""

from __future__ import annotations

import math

import pandas as pd

from stablebound.lineage import LineageGraph
from stablebound.modern_algorithm import build_modern_ledger


def _rt(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name",
            "child_id", "child_name"]
    return pd.DataFrame(rows, columns=cols)


def _stats(rows):
    """Build a long-form stats frame from (unit_id, year, season, variable, value) tuples."""
    return pd.DataFrame(
        rows,
        columns=["unit_id", "year", "season", "variable", "value"],
    )


def _modern_value(stats_modern, modern_id, year, variable, season="annual"):
    """Look up a single (modern_id, year, variable, season) cell's value.

    Returns the float value, or raises if the cell is missing.
    """
    sel = stats_modern[
        (stats_modern["modern_id"] == modern_id)
        & (stats_modern["year"] == year)
        & (stats_modern["variable"] == variable)
        & (stats_modern["season"] == season)
    ]
    assert len(sel) == 1, f"expected 1 row, got {len(sel)} for {modern_id} {year} {variable}"
    return float(sel.iloc[0]["value"])


# --- Pure Split -------------------------------------------------------------

def test_pure_split_distributes_pre_event_data_by_post_event_fractions():
    # Base year 2000. Parent A reports 1000 area in 2001, 2002, 2003.
    # In 2004 A splits into A1 and A2. Post-event window 2005-2009: A1
    # reports 600/yr, A2 reports 400/yr. Fractions: 0.6 and 0.4.
    # Pre-2004 A's data should land 60% on A1 and 40% on A2.
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", y, "annual", "area", 1000.0) for y in (2001, 2002, 2003)]
        + [("A1", y, "annual", "area", 600.0) for y in range(2005, 2010)]
        + [("A2", y, "annual", "area", 400.0) for y in range(2005, 2010)]
    )

    sm, fr, late = build_modern_ledger(
        g, stats, modern_unit_ids={"A1", "A2"},
        base_year=2000, max_year=2010, default_window=5,
    )

    # Pre-event allocations
    assert _modern_value(sm, "A1", 2001, "area") == 600.0
    assert _modern_value(sm, "A1", 2002, "area") == 600.0
    assert _modern_value(sm, "A2", 2003, "area") == 400.0

    # Post-event reports survive untouched
    assert _modern_value(sm, "A1", 2005, "area") == 600.0
    assert _modern_value(sm, "A2", 2009, "area") == 400.0

    # No late-reporting rows
    assert late.empty

    # Audit rows: 2 (one per child), variable="area", fractions sum to 1
    fa = fr[fr["variable"] == "area"]
    assert len(fa) == 2
    assert math.isclose(fa["fraction"].sum(), 1.0)


def test_pure_split_fraction_uses_only_post_event_window():
    # Base year 2000. Same shape as above but window is 3 years —
    # 2005-2007. We give A1 a non-representative pre-event-window
    # spike in 2008 to verify the window cutoff is enforced.
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)
    stats = _stats(
        [("A", 2003, "annual", "area", 100.0)]
        # 2005-2007 (in window): A1=70, A2=30 → fraction 0.7/0.3
        + [("A1", y, "annual", "area", 70.0) for y in (2005, 2006, 2007)]
        + [("A2", y, "annual", "area", 30.0) for y in (2005, 2006, 2007)]
        # 2008 (out of window): A1 spikes to 1e6 — should be ignored
        + [("A1", 2008, "annual", "area", 1_000_000.0)]
        + [("A2", 2008, "annual", "area", 30.0)]
    )

    sm, _, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A1", "A2"},
        base_year=2000, max_year=2010, default_window=3,
    )

    # Pre-event A=100 → 70 on A1, 30 on A2 — confirming the window=3 truncation
    assert _modern_value(sm, "A1", 2003, "area") == 70.0
    assert _modern_value(sm, "A2", 2003, "area") == 30.0


# --- Pure Merge -------------------------------------------------------------

def test_pure_merge_pools_parents_and_pushes_full_value_to_child():
    # Base year 2000. A and B exist independently and report area in
    # 2001-2003. In 2004 they merge into C.
    # Result on C: pre-2004 area = sum of A and B; post-event = C's reports.
    df = _rt([
        (2004, "Merge", "A", "A", "C", "C"),
        (2004, "Merge", "B", "B", "C", "C"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", y, "annual", "area", 100.0) for y in (2001, 2002, 2003)]
        + [("B", y, "annual", "area", 50.0) for y in (2001, 2002, 2003)]
        + [("C", y, "annual", "area", 150.0) for y in range(2005, 2010)]
    )

    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"C"},
        base_year=2000, max_year=2010, default_window=5,
    )

    for y in (2001, 2002, 2003):
        assert _modern_value(sm, "C", y, "area") == 150.0
    for y in range(2005, 2010):
        assert _modern_value(sm, "C", y, "area") == 150.0

    # One audit row per parent edge; the single child gets fraction = 1.0
    fa = fr[fr["variable"] == "area"]
    assert len(fa) == 2
    assert set(fa["parent_ids"]) == {"A", "B"}
    assert all(math.isclose(float(f), 1.0) for f in fa["fraction"])


# --- Redistribute (multi-parent + multi-child, pooled) ----------------------

def test_redistribute_pools_parents_then_distributes_by_child_fractions():
    # Base year 2000. A and B both report area in 2001-2003.
    # In 2004 a redistribute: parents {A, B} → children {X, Y}.
    # Post-event X reports 80/yr, Y reports 20/yr → fractions 0.8/0.2.
    # Pool of A+B in 2003 = 100 + 50 = 150. X gets 120, Y gets 30.
    df = _rt([
        (2004, "Redistribute", "A", "A", "X", "X"),
        (2004, "Redistribute", "A", "A", "Y", "Y"),
        (2004, "Redistribute", "B", "B", "X", "X"),
        (2004, "Redistribute", "B", "B", "Y", "Y"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", y, "annual", "area", 100.0) for y in (2001, 2002, 2003)]
        + [("B", y, "annual", "area", 50.0) for y in (2001, 2002, 2003)]
        + [("X", y, "annual", "area", 80.0) for y in range(2005, 2010)]
        + [("Y", y, "annual", "area", 20.0) for y in range(2005, 2010)]
    )

    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"X", "Y"},
        base_year=2000, max_year=2010, default_window=5,
    )

    # 2003 pool = 150 → X 120, Y 30
    assert math.isclose(_modern_value(sm, "X", 2003, "area"), 120.0)
    assert math.isclose(_modern_value(sm, "Y", 2003, "area"), 30.0)

    # Audit: one row per parent edge (per-parent distribution since the
    # 2026-06-05 E1-inflation fix); each parent has 2 children → "Split"
    fa = fr[(fr["variable"] == "area") & (fr["child_id"] == "X")]
    assert len(fa) == 2
    assert set(fa["parent_ids"]) == {"A", "B"}
    assert set(fa["event_type"]) == {"Split"}


# --- Cascading lineage (sequential composition emerges) ---------------------

def test_cascade_depth_2_produces_sequential_fraction_product():
    # Base year 1990. P1 reports 1000 in 1991-1994.
    # 1995: P1 → {P2, P2_sib}. P2 reports 700 in 1995-1999, P2_sib 300 → f_1995 = 0.7.
    # 2010: P2 → {M, M_sib}. M reports 350 in 2010-2014, M_sib 350 → f_2010 = 0.5.
    # M's pre-1995 share of P1 = 0.5 × 0.7 × 1000 = 350 per year.
    # M's 1995-2009 share of P2 = 0.5 × P2 = 0.5 × 700 = 350 per year.
    df = _rt([
        (1995, "Split", "P1", "P1", "P2", "P2"),
        (1995, "Split", "P1", "P1", "P2_sib", "P2_sib"),
        (2010, "Split", "P2", "P2", "M", "M"),
        (2010, "Split", "P2", "P2", "M_sib", "M_sib"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        # P1 pre-event reports
        [("P1", y, "annual", "area", 1000.0) for y in (1991, 1992, 1993, 1994)]
        # P2 + P2_sib post-1995, used for f_1995 and as P2's data through 2009
        + [("P2", y, "annual", "area", 700.0) for y in range(1995, 2010)]
        + [("P2_sib", y, "annual", "area", 300.0) for y in range(1995, 2010)]
        # M + M_sib post-2010, used for f_2010
        + [("M", y, "annual", "area", 350.0) for y in range(2010, 2015)]
        + [("M_sib", y, "annual", "area", 350.0) for y in range(2010, 2015)]
    )

    sm, _, _ = build_modern_ledger(
        g, stats,
        modern_unit_ids={"M", "M_sib", "P2_sib"},
        base_year=1990, max_year=2014, default_window=5,
    )

    # M's pre-1995 share: 0.5 × 0.7 × 1000 = 350
    assert math.isclose(_modern_value(sm, "M", 1991, "area"), 350.0)
    assert math.isclose(_modern_value(sm, "M", 1994, "area"), 350.0)
    # M's 1995-2009 share: 0.5 × 700 = 350
    assert math.isclose(_modern_value(sm, "M", 2000, "area"), 350.0)
    # M's own 2010+ reports
    assert math.isclose(_modern_value(sm, "M", 2010, "area"), 350.0)

    # P2_sib's pre-1995 share: 0.3 × 1000 = 300
    assert math.isclose(_modern_value(sm, "P2_sib", 1991, "area"), 300.0)


# --- NaN handling (data gaps) -----------------------------------------------

def test_child_with_zero_post_event_reports_falls_through_to_area_tier():
    # Same as the basic split, but A2 has no reports post-event.
    # Under common-years semantics: A1 ∩ A2 = ∅ → seasonal fails.
    # Total-year also fails (A2 has zero cells for this var across any
    # season). Area tier kicks in if modern_areas is supplied, and
    # splits A's data by geographic share rather than letting A1
    # absorb all of it (the old behavior, which silently assigned
    # 100% of an unknowable share to the only reporter).
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", y, "annual", "area", 1000.0) for y in (2001, 2002, 2003)]
        + [("A1", y, "annual", "area", 600.0) for y in range(2005, 2010)]
        # A2 reports nothing in the window
    )

    # Pass modern_areas: A1 = 30 units, A2 = 70 units. The area tier
    # will split A's pre-event data 30/70 by geography.
    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A1", "A2"},
        base_year=2000, max_year=2010, default_window=5,
        modern_areas={"A1": 30.0, "A2": 70.0},
    )

    # Area-tier allocation: A's 1000 splits 300/700 by area.
    assert math.isclose(_modern_value(sm, "A1", 2003, "area"), 300.0)
    a2_2003 = sm[
        (sm["modern_id"] == "A2") & (sm["year"] == 2003)
        & (sm["variable"] == "area")
    ].iloc[0]
    assert math.isclose(float(a2_2003["value"]), 700.0)
    # Cell carries the area-tier mark.
    assert a2_2003["fraction_method"] == "area"

    # Audit shows fraction_method='area' for both children.
    a1_frac = fr[(fr["child_id"] == "A1") & (fr["variable"] == "area")].iloc[0]
    a2_frac = fr[(fr["child_id"] == "A2") & (fr["variable"] == "area")].iloc[0]
    assert a1_frac["fraction_method"] == "area"
    assert a2_frac["fraction_method"] == "area"
    assert math.isclose(float(a1_frac["fraction"]), 0.3)
    assert math.isclose(float(a2_frac["fraction"]), 0.7)

    # Conservation: sum across A1+A2 == A's pre-event total.
    a_total_2003 = sum(
        float(sm[
            (sm["modern_id"] == c) & (sm["year"] == 2003)
            & (sm["variable"] == "area") & (sm["season"] == "annual")
        ]["value"].iloc[0])
        for c in ("A1", "A2")
    )
    assert math.isclose(a_total_2003, 1000.0)


def test_common_years_intersection_partial_overlap():
    """B reports 5 years, C reports 4 years (missing 2005). Fraction is
    computed over the 4 common years (2006-2009), not biased by B's
    extra 2005 datum. The audit shows n_common=4 with n_individual
    differing per child to make the imbalance visible.
    """
    df = _rt([
        (2004, "Split", "A", "A", "B", "B"),
        (2004, "Split", "A", "A", "C", "C"),
    ])
    g = LineageGraph.from_dataframe(df)

    # B reports a steady 600. C reports 400, but missed 2005.
    # If 2005 were anomalous for B (e.g., 9000), the OLD per-child
    # mean would inflate B's fraction; the new intersection eats it.
    stats = _stats(
        [("A", y, "annual", "area", 1000.0) for y in (2001, 2002, 2003)]
        + [("B", 2005, "annual", "area", 9000.0)]            # anomalous year, B-only
        + [("B", y, "annual", "area", 600.0) for y in (2006, 2007, 2008, 2009)]
        + [("C", y, "annual", "area", 400.0) for y in (2006, 2007, 2008, 2009)]
    )

    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"B", "C"},
        base_year=2000, max_year=2010, default_window=5,
    )

    # Fraction should be 600 / (600 + 400) = 0.6 / 0.4, NOT biased by
    # the 9000 outlier in B's 2005 (which is outside the common set).
    fb = fr[(fr["child_id"] == "B") & (fr["variable"] == "area")].iloc[0]
    fc = fr[(fr["child_id"] == "C") & (fr["variable"] == "area")].iloc[0]
    assert fb["fraction_method"] == "seasonal"
    assert math.isclose(float(fb["fraction"]), 0.6)
    assert math.isclose(float(fc["fraction"]), 0.4)
    # Audit reflects the asymmetry: n_common=4, n_individual differs.
    assert int(fb["n_common_observations"]) == 4
    assert int(fc["n_common_observations"]) == 4
    assert int(fb["n_individual_observations"]) == 5  # B had 5 (incl. anomalous)
    assert int(fc["n_individual_observations"]) == 4

    # Pre-event distribution uses the apples-to-apples fraction.
    assert math.isclose(_modern_value(sm, "B", 2003, "area"), 600.0)
    assert math.isclose(_modern_value(sm, "C", 2003, "area"), 400.0)


def test_child_with_zero_post_event_reports_no_areas_is_undefined():
    # Same scenario as above but WITHOUT modern_areas. The cascade has
    # no last-resort fallback → fractions are NaN, A's data is lost.
    # This is the documented honest outcome when an algorithm-direct
    # caller skips areas (StableBoundary.aggregate_stats always
    # passes them).
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", y, "annual", "area", 1000.0) for y in (2001, 2002, 2003)]
        + [("A1", y, "annual", "area", 600.0) for y in range(2005, 2010)]
        # A2 reports nothing
    )

    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A1", "A2"},
        base_year=2000, max_year=2010, default_window=5,
        # No modern_areas passed
    )

    a1_frac = fr[(fr["child_id"] == "A1") & (fr["variable"] == "area")].iloc[0]
    assert a1_frac["fraction_method"] == "undefined"
    assert math.isnan(float(a1_frac["fraction"]))


# --- Insufficient window (warns but still computes) ------------------------

def test_partial_window_still_computes_fraction(caplog):
    # Window length 5, but data only exists 2 years post-event.
    # Should still compute fractions from those 2 years and emit a warning.
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)
    stats = _stats(
        [("A", 2003, "annual", "area", 1000.0)]
        + [("A1", y, "annual", "area", 600.0) for y in (2005, 2006)]
        + [("A2", y, "annual", "area", 400.0) for y in (2005, 2006)]
    )

    import logging
    with caplog.at_level(logging.WARNING):
        sm, fr, _ = build_modern_ledger(
            g, stats, modern_unit_ids={"A1", "A2"},
            base_year=2000, max_year=2010, default_window=5,
        )

    # Fractions still come out 0.6 / 0.4
    assert math.isclose(_modern_value(sm, "A1", 2003, "area"), 600.0)
    assert math.isclose(_modern_value(sm, "A2", 2003, "area"), 400.0)
    # Audit reports n_common_observations=2 for both (both children
    # reported the same 2 years — common-years intersection = 2)
    fa = fr[fr["variable"] == "area"]
    assert set(fa["n_common_observations"]) == {2}
    assert set(fa["n_individual_observations"]) == {2}
    # Warning logged
    assert any("post-event years" in rec.message for rec in caplog.records)


# --- NameChange / Coarse passthrough ----------------------------------------

def test_namechange_does_not_disturb_ledger():
    # A NameChange (parent_id == child_id) at year 2005 should leave the
    # ledger untouched — territorial events filter excludes it. The unit's
    # data flows straight to the modern output.
    df = _rt([
        (2005, "NameChange", "A", "OldName", "A", "NewName"),
    ])
    g = LineageGraph.from_dataframe(df)
    stats = _stats(
        [("A", y, "annual", "area", 100.0) for y in range(2001, 2010)]
    )

    sm, fr, late = build_modern_ledger(
        g, stats, modern_unit_ids={"A"},
        base_year=2000, max_year=2010, default_window=5,
    )

    # A's full 9 years of data flow through identically
    for y in range(2001, 2010):
        assert _modern_value(sm, "A", y, "area") == 100.0
    # No fractions audit rows (no territorial event)
    assert fr.empty
    # No late reporting
    assert late.empty


def test_coarse_event_does_not_disturb_ledger():
    # Coarse events (parent_id == child_id, parent admin level reassigned)
    # are also non-territorial. Same passthrough as NameChange.
    df = _rt([
        (2005, "Coarse", "A", "A", "A", "A"),
    ])
    g = LineageGraph.from_dataframe(df)
    stats = _stats(
        [("A", y, "annual", "area", 100.0) for y in range(2001, 2010)]
    )

    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A"},
        base_year=2000, max_year=2010, default_window=5,
    )

    for y in range(2001, 2010):
        assert _modern_value(sm, "A", y, "area") == 100.0
    assert fr.empty


# --- Per-variable window ----------------------------------------------------

def test_per_variable_window_overrides_default():
    # area uses window=5; production uses window=2.
    # Set up data so each variable would yield different fractions if the
    # other window were used.
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", 2003, "annual", "area", 1000.0)]
        + [("A", 2003, "annual", "production", 1000.0)]
        # area: 5-year mean — A1 90/yr, A2 10/yr in 2005-2009 → 0.9 / 0.1
        + [("A1", y, "annual", "area", 90.0) for y in range(2005, 2010)]
        + [("A2", y, "annual", "area", 10.0) for y in range(2005, 2010)]
        # production: A1 70/yr in 2005-2006, A2 30/yr → 0.7 / 0.3
        # Then in 2007-2009 the ratio swings to 0.1 / 0.9 — but window=2
        # should ignore this.
        + [("A1", y, "annual", "production", 70.0) for y in (2005, 2006)]
        + [("A2", y, "annual", "production", 30.0) for y in (2005, 2006)]
        + [("A1", y, "annual", "production", 10.0) for y in (2007, 2008, 2009)]
        + [("A2", y, "annual", "production", 90.0) for y in (2007, 2008, 2009)]
    )

    sm, _, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A1", "A2"},
        base_year=2000, max_year=2010,
        window_per_var={"production": 2}, default_window=5,
    )

    # area: 0.9 / 0.1 of 1000 = 900 / 100
    assert math.isclose(_modern_value(sm, "A1", 2003, "area"), 900.0)
    assert math.isclose(_modern_value(sm, "A2", 2003, "area"), 100.0)
    # production: 0.7 / 0.3 of 1000 = 700 / 300 (using only 2-year window)
    assert math.isclose(_modern_value(sm, "A1", 2003, "production"), 700.0)
    assert math.isclose(_modern_value(sm, "A2", 2003, "production"), 300.0)


# --- Late-reporting ---------------------------------------------------------

def test_late_reporting_unit_emits_into_late_csv():
    # A reports in 2002-2003 (pre-event), splits at 2004 into A1 only,
    # then continues to (incorrectly) report under "A" in 2005-2006.
    # Pre-event data should land on A1; post-event reports under A go
    # into the late_reporting frame.
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", 2002, "annual", "area", 100.0)]
        + [("A", 2003, "annual", "area", 100.0)]
        # A1 reports normally
        + [("A1", y, "annual", "area", 100.0) for y in range(2005, 2010)]
        # A also reports incorrectly post-event — late reporting
        + [("A", y, "annual", "area", 50.0) for y in (2005, 2006)]
    )

    sm, _, late = build_modern_ledger(
        g, stats, modern_unit_ids={"A1"},
        base_year=2000, max_year=2010, default_window=5,
    )

    # A1 has 2002-2009 data
    assert _modern_value(sm, "A1", 2002, "area") == 100.0
    # Late frame has the post-event A reports
    assert not late.empty
    a_late = late[(late["unit_id"] == "A") & (late["variable"] == "area")]
    assert set(a_late["year"]) == {2005, 2006}
    assert all(a_late["value"] == 50.0)


# --- No-events trivial case -------------------------------------------------

def test_unit_with_no_events_passes_through_identically():
    # No territorial events at all. Every unit's stats flow straight to
    # modern output unmodified.
    df = pd.DataFrame(
        columns=["event_year", "event_type", "parent_id", "parent_name",
                 "child_id", "child_name"]
    )
    g = LineageGraph.from_dataframe(df, validate=False)

    stats = _stats(
        [("A", y, "annual", "area", 100.0) for y in range(2001, 2010)]
        + [("B", y, "annual", "area", 50.0) for y in range(2001, 2010)]
    )
    sm, fr, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A", "B"},
        base_year=2000, max_year=2010, default_window=5,
    )

    assert fr.empty
    for y in range(2001, 2010):
        assert _modern_value(sm, "A", y, "area") == 100.0
        assert _modern_value(sm, "B", y, "area") == 50.0


# --- Multiple seasons -------------------------------------------------------

def test_seasons_compute_separate_fractions():
    # Same units, two seasons. A1 wins kharif but A2 wins rabi.
    df = _rt([
        (2004, "Split", "A", "A", "A1", "A1"),
        (2004, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(df)

    stats = _stats(
        [("A", 2003, "kharif", "area", 100.0)]
        + [("A", 2003, "rabi", "area", 100.0)]
        # kharif: A1 80, A2 20
        + [("A1", y, "kharif", "area", 80.0) for y in range(2005, 2010)]
        + [("A2", y, "kharif", "area", 20.0) for y in range(2005, 2010)]
        # rabi: A1 30, A2 70
        + [("A1", y, "rabi", "area", 30.0) for y in range(2005, 2010)]
        + [("A2", y, "rabi", "area", 70.0) for y in range(2005, 2010)]
    )

    sm, _, _ = build_modern_ledger(
        g, stats, modern_unit_ids={"A1", "A2"},
        base_year=2000, max_year=2010, default_window=5,
    )

    assert math.isclose(_modern_value(sm, "A1", 2003, "area", season="kharif"), 80.0)
    assert math.isclose(_modern_value(sm, "A2", 2003, "area", season="kharif"), 20.0)
    assert math.isclose(_modern_value(sm, "A1", 2003, "area", season="rabi"), 30.0)
    assert math.isclose(_modern_value(sm, "A2", 2003, "area", season="rabi"), 70.0)
