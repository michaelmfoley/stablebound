"""Snapshot construction + shapefile-year inference."""

from __future__ import annotations

import pandas as pd

from stablebound.lineage import LineageGraph
from stablebound.snapshot import build_snapshot, infer_year


def _rt(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name"]
    return pd.DataFrame(rows, columns=cols)


def test_snapshot_before_any_event_is_initial_units():
    # Convention: event_year=T means event happens DURING T, so the
    # year-T snapshot is pre-event (start of year T). At year=2002 (one
    # year before the 2003 event), the snapshot is just the initial unit.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    assert build_snapshot(g, year=2002) == {"A"}


def test_snapshot_at_event_year_is_still_pre_event():
    # Under the canonical convention, snapshot at year==event_year
    # reflects the START of that year — pre-event. The split happens
    # during 2003, so snapshot[2003] = {A}. snapshot[2004] = {A1, A2}.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    assert build_snapshot(g, year=2003) == {"A"}
    assert build_snapshot(g, year=2004) == {"A1", "A2"}
    assert build_snapshot(g, year=2010) == {"A1", "A2"}


def test_snapshot_after_merge_consolidates():
    # Merge happens during 2010. snapshot[2010] = pre-event = {A, B}.
    # snapshot[2011] = post-event = {C}.
    df = _rt(
        [
            (2010, "Merge", "A", "A", "C", "C"),
            (2010, "Merge", "B", "B", "C", "C"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    assert build_snapshot(g, year=2009) == {"A", "B"}
    assert build_snapshot(g, year=2010) == {"A", "B"}
    assert build_snapshot(g, year=2011) == {"C"}


def test_snapshot_namechange_does_not_alter_active_set():
    # NameChange events are non-territorial; build_snapshot ignores them
    # entirely. The 2003 Split happens during 2003, so by year=2010 the
    # active set is {A1, A2}.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
            (2010, "NameChange", "A1", "A", "A1", "A_renamed"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    assert build_snapshot(g, year=2010) == {"A1", "A2"}


def test_infer_year_picks_smallest_mismatch():
    # Splits at 2003 and 2010 happen during those years. Snapshot {A2, X}
    # (A1 was split, X exists) only appears starting at year=2011 under
    # the canonical convention.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
            (2010, "Split", "A1", "A", "X", "X"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    shapefile = {"A2", "X"}
    best, mismatches = infer_year(shapefile, g, range(2002, 2013))
    assert best == 2011
    assert mismatches[2011] == 0
    # 2010 is pre-event of the 2010 split → still has A1, no X.
    assert mismatches[2010] == 2


def test_infer_year_breaks_ties_by_earlier_year():
    # The 2003 Split takes effect in the 2004 snapshot. {A1, A2} matches
    # snapshot at year=2004 onward; tie-break favors the earliest year.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    best, _ = infer_year({"A1", "A2"}, g, [2010, 2004, 2005])
    assert best == 2004


def test_infer_year_seeds_additional_units():
    """A baseline unit no event ever mentions must not count as a mismatch.

    build_snapshot starts from the units territorial events reach, so a unit
    that simply exists and never changes is in NO candidate snapshot unless
    additional_units is seeded. Unseeded, it is counted as missing in every
    year at once.

    That does not usually move the argmin -- it is a constant added to every
    candidate -- which is exactly why it survived: the answer stayed right
    while the evidence for it became unreadable. On India's shipped shapefile
    the chosen year reported 240 discrepancies unseeded against 12 seeded.
    """
    df = pd.DataFrame(
        [
            {"event_year": 2005, "event_type": "Split", "parent_id": "A",
             "parent_name": "A", "child_id": "A1", "child_name": "A1"},
            {"event_year": 2005, "event_type": "Split", "parent_id": "A",
             "parent_name": "A", "child_id": "A2", "child_name": "A2"},
        ]
    )
    g = LineageGraph.from_dataframe(df)
    # Z never appears in any event; it is only in the baseline.
    shapefile = {"A1", "A2", "Z"}

    unseeded_year, unseeded = infer_year(shapefile, g, [2006, 2007])
    seeded_year, seeded = infer_year(shapefile, g, [2006, 2007],
                                     additional_units={"A", "Z"})

    # The chosen year is unchanged -- the bug never moved the answer.
    assert unseeded_year == seeded_year == 2006
    # But unseeded, Z is reported missing in every year; seeded, nothing is.
    assert unseeded[2006] == 1, "Z should be counted as a mismatch when unseeded"
    assert seeded[2006] == 0, "seeding additional_units should clear the mismatch"
