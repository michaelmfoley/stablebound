"""Reconciliation diagnostics + flag/merge/subtract modes."""

from __future__ import annotations

import pandas as pd

from stablebound.lineage import LineageGraph
from stablebound.reconcile import reconcile


def _rt(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name"]
    return pd.DataFrame(rows, columns=cols)


def _stats(rows):
    cols = ["unit_id", "year", "season", "variable", "value"]
    return pd.DataFrame(rows, columns=cols)


def _split_a_into_a_and_b():
    """Standard split fixture: A → A1, A2 in 2010."""
    return LineageGraph.from_dataframe(
        _rt(
            [
                (2010, "Split", "A", "A", "A1", "A"),
                (2010, "Split", "A", "A", "A2", "B"),
            ]
        )
    )


def test_clean_reporting_does_not_flag():
    # Pre-2010: A reports area=100. Post-2010: A1 reports 60, A2 reports 40.
    # Total preserved, parent properly drops to 0 (no row for A post-split → handled gracefully).
    graph = _split_a_into_a_and_b()
    stats = _stats(
        [
            ("A", 2007, "Annual", "area_ha", 100.0),
            ("A", 2008, "Annual", "area_ha", 100.0),
            ("A", 2009, "Annual", "area_ha", 100.0),
            ("A1", 2010, "Annual", "area_ha", 60.0),
            ("A1", 2011, "Annual", "area_ha", 60.0),
            ("A1", 2012, "Annual", "area_ha", 60.0),
            ("A2", 2010, "Annual", "area_ha", 40.0),
            ("A2", 2011, "Annual", "area_ha", 40.0),
            ("A2", 2012, "Annual", "area_ha", 40.0),
        ]
    )
    _, _, flags = reconcile(stats, graph, remap={"A": "A", "A1": "A", "A2": "A"}, mode="flag")
    split_flags = flags[flags["event_type"] == "Split"]
    # Parent has no post-split rows so d_t is NaN; sum_jump_ratio also NaN.
    # No flag should fire.
    assert (split_flags["flagged"] == False).all()  # noqa: E712


def test_drop_test_flags_when_parent_continues_reporting_full():
    # A continues reporting at 100 post-2010 — the canonical "old district keeping
    # the parent's name and not dropping". A2 also reports 40. Sum jumps.
    graph = _split_a_into_a_and_b()
    stats = _stats(
        [
            ("A", 2007, "Annual", "area_ha", 100.0),
            ("A", 2008, "Annual", "area_ha", 100.0),
            ("A", 2009, "Annual", "area_ha", 100.0),
            ("A", 2010, "Annual", "area_ha", 100.0),  # didn't drop
            ("A", 2011, "Annual", "area_ha", 100.0),
            ("A", 2012, "Annual", "area_ha", 100.0),
            ("A2", 2010, "Annual", "area_ha", 40.0),
            ("A2", 2011, "Annual", "area_ha", 40.0),
            ("A2", 2012, "Annual", "area_ha", 40.0),
        ]
    )
    _, _, flags = reconcile(
        stats,
        graph,
        remap={"A": "A", "A1": "A", "A2": "A"},
        mode="flag",
    )
    split_flags = flags[(flags["event_type"] == "Split") & (flags["unit_id"] == "A")]
    assert split_flags["flagged"].any()
    row = split_flags.iloc[0]
    # Expected r_t = 40/100 = 0.4; observed d_t = 0; 0 < 0.4 - 0.15 = 0.25 → drop_flag
    assert row["drop_flag"] == True  # noqa: E712
    # Sum jump: (100 + 40) / 100 = 1.4 > 1 + 0.15 → sum_flag
    assert row["sum_flag"] == True  # noqa: E712


def test_flag_mode_does_not_modify_stats_or_remap():
    graph = _split_a_into_a_and_b()
    stats = _stats(
        [
            ("A", 2008, "Annual", "area_ha", 100.0),
            ("A", 2009, "Annual", "area_ha", 100.0),
            ("A", 2010, "Annual", "area_ha", 100.0),
            ("A", 2011, "Annual", "area_ha", 100.0),
            ("A2", 2010, "Annual", "area_ha", 40.0),
            ("A2", 2011, "Annual", "area_ha", 40.0),
        ]
    )
    remap = {"A": "A", "A1": "A", "A2": "A"}
    new_stats, new_remap, flags = reconcile(stats, graph, remap=remap, mode="flag")
    pd.testing.assert_frame_equal(new_stats, stats)
    assert new_remap == remap
    # Flags carry mode_applied=='flag' for flagged rows.
    flagged = flags[flags["flagged"] == True]  # noqa: E712
    assert (flagged["mode_applied"] == "flag").all()


def test_merge_mode_drops_successor_rows():
    graph = _split_a_into_a_and_b()
    stats = _stats(
        [
            ("A", 2008, "Annual", "area_ha", 100.0),
            ("A", 2009, "Annual", "area_ha", 100.0),
            ("A", 2010, "Annual", "area_ha", 100.0),
            ("A", 2011, "Annual", "area_ha", 100.0),
            ("A", 2012, "Annual", "area_ha", 100.0),
            ("A2", 2010, "Annual", "area_ha", 40.0),
            ("A2", 2011, "Annual", "area_ha", 40.0),
            ("A2", 2012, "Annual", "area_ha", 40.0),
        ]
    )
    new_stats, new_remap, flags = reconcile(
        stats,
        graph,
        remap={"A": "A", "A1": "A", "A2": "A"},
        mode="merge",
    )
    # Successor (A1, A2) post-2010 rows should be dropped.
    assert ((new_stats["unit_id"] == "A2") & (new_stats["year"] >= 2010)).sum() == 0
    # Parent A rows survive.
    assert ((new_stats["unit_id"] == "A") & (new_stats["year"] >= 2010)).sum() == 3
    flagged = flags[flags["flagged"] == True]  # noqa: E712
    assert (flagged["mode_applied"] == "merge").all()


def test_subtract_mode_subtracts_successor_value_from_parent():
    graph = _split_a_into_a_and_b()
    stats = _stats(
        [
            ("A", 2008, "Annual", "area_ha", 100.0),
            ("A", 2009, "Annual", "area_ha", 100.0),
            ("A", 2010, "Annual", "area_ha", 100.0),
            ("A", 2011, "Annual", "area_ha", 100.0),
            ("A2", 2010, "Annual", "area_ha", 40.0),
            ("A2", 2011, "Annual", "area_ha", 40.0),
        ]
    )
    new_stats, new_remap, flags = reconcile(
        stats,
        graph,
        remap={"A": "A", "A1": "A", "A2": "A"},
        mode="subtract",
    )
    # Parent A's 2010 and 2011 values should be reduced by 40.
    a2010 = new_stats[(new_stats["unit_id"] == "A") & (new_stats["year"] == 2010)]
    a2011 = new_stats[(new_stats["unit_id"] == "A") & (new_stats["year"] == 2011)]
    assert float(a2010["value"].iloc[0]) == 60.0
    assert float(a2011["value"].iloc[0]) == 60.0
    # Successor row preserved.
    a2_2010 = new_stats[(new_stats["unit_id"] == "A2") & (new_stats["year"] == 2010)]
    assert float(a2_2010["value"].iloc[0]) == 40.0
    flagged = flags[flags["flagged"] == True]  # noqa: E712
    assert (flagged["mode_applied"] == "subtract").all()


def test_subtract_mode_falls_back_to_merge_when_base_postdates_merge():
    # Merge fixture: F + G → H in 2005. base_year = 2010 (postdates merge)
    # → all already in one stable group → subtract undefined → merge fallback.
    graph = LineageGraph.from_dataframe(
        _rt(
            [
                (2005, "Merge", "F", "F", "H", "H"),
                (2005, "Merge", "G", "G", "H", "H"),
            ]
        )
    )
    stats = _stats(
        [
            ("F", 2002, "Annual", "area_ha", 50.0),
            ("F", 2003, "Annual", "area_ha", 50.0),
            ("F", 2004, "Annual", "area_ha", 50.0),
            ("F", 2005, "Annual", "area_ha", 50.0),  # F still reporting after merge
            ("F", 2006, "Annual", "area_ha", 50.0),
            ("G", 2002, "Annual", "area_ha", 50.0),
            ("G", 2003, "Annual", "area_ha", 50.0),
            ("G", 2004, "Annual", "area_ha", 50.0),
            ("H", 2005, "Annual", "area_ha", 100.0),
            ("H", 2006, "Annual", "area_ha", 100.0),
        ]
    )
    # Base year 2010 → all map to one stable_id.
    remap = {"F": "F", "G": "F", "H": "F"}
    new_stats, new_remap, flags = reconcile(
        stats,
        graph,
        remap=remap,
        mode="subtract",
    )
    flagged = flags[flags["flagged"] == True]  # noqa: E712
    assert (flagged["mode_applied"] == "subtract→merge_fallback").all()
    # F's post-merge rows should be dropped (the merge-mode behavior).
    assert ((new_stats["unit_id"] == "F") & (new_stats["year"] >= 2005)).sum() == 0


def test_off_mode_returns_empty_flags_and_unmodified_inputs():
    """mode='off' short-circuits: no diagnostics, no mutations."""
    graph = _split_a_into_a_and_b()
    stats = _stats([
        # Construct a stats frame that WOULD flag under mode='flag':
        # A keeps reporting at full pre-split level after the split.
        ("A", 2007, "Annual", "area_ha", 100.0),
        ("A", 2008, "Annual", "area_ha", 100.0),
        ("A", 2009, "Annual", "area_ha", 100.0),
        ("A", 2010, "Annual", "area_ha", 100.0),
        ("A", 2011, "Annual", "area_ha", 100.0),
        ("A1", 2010, "Annual", "area_ha", 60.0),
        ("A2", 2010, "Annual", "area_ha", 40.0),
    ])
    remap = {"A": "A", "A1": "A", "A2": "A"}

    # Sanity: with flag mode, this scenario DOES flag.
    _, _, flag_flags = reconcile(stats, graph, remap, mode="flag")
    assert (flag_flags["flagged"] == True).any()  # noqa: E712

    # With off mode: empty flags, unmodified stats + remap.
    reconciled, new_remap, flags = reconcile(stats, graph, remap, mode="off")
    assert len(flags) == 0
    assert list(flags.columns) == [
        "event_year", "event_type", "unit_id", "variable", "season",
        "r_t", "d_t", "sum_jump_ratio",
        "drop_flag", "sum_flag", "flagged", "mode_applied", "note",
    ]
    pd.testing.assert_frame_equal(reconciled, stats)
    assert new_remap == remap
    # And it must still be a copy — mutating it shouldn't affect the input.
    new_remap["A"] = "ZZZ"
    assert remap["A"] == "A"


def test_invalid_mode_lists_off_as_an_option():
    """The validator's error message should mention 'off' so users see the switch exists."""
    import pytest
    graph = _split_a_into_a_and_b()
    with pytest.raises(ValueError, match=r"off.*flag.*merge.*subtract|.*off"):
        reconcile(_stats([]), graph, remap={}, mode="nonsense")


def test_redistribute_audit_row_is_emitted_but_not_flagged_or_modified():
    graph = LineageGraph.from_dataframe(
        _rt(
            [
                (2010, "Redistribute", "A", "A", "A_new", "A"),
                (2010, "Redistribute", "B", "B", "A_new", "A"),
                (2010, "Redistribute", "A", "A", "B_new", "B"),
                (2010, "Redistribute", "B", "B", "B_new", "B"),
            ]
        )
    )
    stats = _stats([("A", 2009, "Annual", "area_ha", 50.0)])
    _, _, flags = reconcile(stats, graph, remap={"A": "A", "B": "B"}, mode="merge")
    redist = flags[flags["event_type"] == "Redistribute"]
    assert len(redist) >= 1
    assert (redist["mode_applied"] == "redistribute_unreconcilable").all()
    assert (redist["flagged"] == False).all()  # noqa: E712
