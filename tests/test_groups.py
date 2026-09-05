"""Stable-group construction (paper Algorithm 3) and the redistribute-safe rule."""

from __future__ import annotations

import pandas as pd

from stablebound.lineage import LineageGraph
from stablebound.groups import build_stable_groups


def _rt(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name"]
    return pd.DataFrame(rows, columns=cols)


def test_clean_split_unions_descendants_with_parent():
    # Base year 2002. After 2003, A splits into A1 and A2.
    # All three are in A's stable group; stable_id = "A" (smallest lex ID).
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    remap = build_stable_groups(g, base_year=2002)
    assert remap["A"] == "A"
    assert remap["A1"] == "A"
    assert remap["A2"] == "A"


def test_clean_merge_unions_pre_merge_parents_into_one_group():
    # Base year 2009. A and B exist independently, then merge into C in 2010.
    # Under paper Algorithm 3, the merge child C unions A's and B's groups
    # into a single stable group containing {A, B, C}. The 2009 stable
    # polygon for that region is the union of A's and B's territories —
    # resolution lost, but data sums cleanly across years (no allocation
    # assumption needed for C's reports).
    df = _rt(
        [
            (2010, "Merge", "A", "A", "C", "C"),
            (2010, "Merge", "B", "B", "C", "C"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    remap = build_stable_groups(g, base_year=2009)
    assert remap["A"] == remap["B"] == remap["C"]


def test_redistribute_unions_all_ancestor_groups():
    # Base year 2009. A and B both base-year units. In 2010 a redistribute
    # creates A_new and B_new, each with parents {A, B}. Under paper
    # Algorithm 3, the multi-parent children union A's and B's groups into
    # one stable group containing {A, B, A_new, B_new}. The 2010
    # reorganization can't be cleanly decomposed without inventing a map
    # of how territory actually flowed — merging is the only assumption-
    # free choice.
    df = _rt(
        [
            (2010, "Redistribute", "A", "A", "A_new", "A"),
            (2010, "Redistribute", "B", "B", "A_new", "A"),
            (2010, "Redistribute", "A", "A", "B_new", "B"),
            (2010, "Redistribute", "B", "B", "B_new", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    remap = build_stable_groups(g, base_year=2009)
    canonical = remap["A"]
    assert remap["B"] == canonical
    assert remap["A_new"] == canonical
    assert remap["B_new"] == canonical


def test_redistribute_within_same_base_group_still_unions():
    # When all parents of a redistribute child trace back to the SAME base
    # unit, the child IS unioned in. Setup:
    # Base year 2002. A splits in 2003 into A1, A2. In 2010 A1 and A2 do
    # an internal redistribute creating A1', A2' (each with parents {A1, A2}).
    # All five units should land in A's group.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
            (2010, "Redistribute", "A1", "A", "A1p", "A"),
            (2010, "Redistribute", "A2", "B", "A1p", "A"),
            (2010, "Redistribute", "A1", "A", "A2p", "B"),
            (2010, "Redistribute", "A2", "B", "A2p", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    remap = build_stable_groups(g, base_year=2002)
    assert remap["A"] == remap["A1"] == remap["A2"] == "A"
    assert remap["A1p"] == "A"
    assert remap["A2p"] == "A"


def test_stable_id_is_lexicographically_smallest():
    # Determinism guarantee: stable_id of a group is the smallest unit_id.
    df = _rt(
        [
            (2003, "Split", "ZZZZZ", "Z", "AAAAA", "Z1"),
            (2003, "Split", "ZZZZZ", "Z", "MMMMM", "Z2"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    remap = build_stable_groups(g, base_year=2002)
    # All three are unioned, smallest is "AAAAA".
    assert set(remap.values()) == {"AAAAA"}


def test_namechange_does_not_alter_groups():
    # NameChange events are not territorial and should not affect the remap.
    df_with_nc = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
            (2015, "NameChange", "A1", "A", "A1", "A_renamed"),
        ]
    )
    df_no_nc = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
        ]
    )
    g_nc = LineageGraph.from_dataframe(df_with_nc)
    g_no = LineageGraph.from_dataframe(df_no_nc)
    assert build_stable_groups(g_nc, base_year=2002) == build_stable_groups(
        g_no, base_year=2002
    )
