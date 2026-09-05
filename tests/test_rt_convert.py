"""Tests for the old-format → canonical RT converter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stablebound import (
    LineageGraph,
    convert_relationship_table_to_lineage,
    read_legacy_relationship_table,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rt_convert"


# ----------------------------------------------------------------------
# Golden test: the hand-authored legacy fixture against its canonical twin
# ----------------------------------------------------------------------

SYNTHETIC_TWIN = Path(__file__).parent / "fixtures" / "synthetic" / "legacy_admin1"


def test_golden_match_against_the_hand_authored_twin():
    """Converter output equals the canonical fixture row for row.

    ``relationshiptable_XX.csv`` and ``synthetic/legacy_admin1/`` describe the
    same country in two dialects, and neither is generated from the other (see
    the README beside the CSV), so this compares two independent statements of
    one lineage. Ids are part of the comparison: the fixture was authored with
    the ids ``_assign_ids`` allocates, 00007 included-then-collapsed.
    """
    old = read_legacy_relationship_table(FIXTURE_DIR / "relationshiptable_XX.csv")
    lineage, baseline = convert_relationship_table_to_lineage(
        old, country="XX", admin_level=1
    )

    key = ["event_year", "parent_id", "child_name"]
    expected_lineage = (
        pd.read_csv(SYNTHETIC_TWIN / "lineage.csv").sort_values(key).reset_index(drop=True)
    )
    got = lineage.sort_values(key).reset_index(drop=True)[expected_lineage.columns]
    pd.testing.assert_frame_equal(got, expected_lineage, check_dtype=False)

    cols = ["unit_id", "name", "year"]
    expected_baseline = (
        pd.read_csv(SYNTHETIC_TWIN / "baseline.csv").sort_values("unit_id").reset_index(drop=True)
    )
    got_baseline = baseline.sort_values("unit_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(got_baseline[cols], expected_baseline[cols], check_dtype=False)


def test_output_consumable_by_lineagegraph():
    """The lineage output passes LineageGraph validation without modification."""
    old = read_legacy_relationship_table(FIXTURE_DIR / "relationshiptable_XX.csv")
    lineage, _ = convert_relationship_table_to_lineage(
        old, country="XX", admin_level=1
    )
    graph = LineageGraph.from_dataframe(lineage)
    assert graph.min_event_year == 2015
    assert graph.max_event_year == 2025
    assert all(eid.startswith("XX.ADM1.") for eid in graph.all_unit_ids())


# ----------------------------------------------------------------------
# Synthetic micro-cases
# ----------------------------------------------------------------------

def _make_old_rt(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal old-format RT for unit tests.

    Each `row` dict must supply: category, relationship_type,
    from_unit, to_unit, from_unit_name, to_unit_name, fnid_from_unit,
    fnid_to_unit. (Hierarchical rows: relationship_type is irrelevant
    but the column must be present.)
    """
    df = pd.DataFrame(rows)
    df["label"] = ""
    df["can_aggregate_to_from"] = False
    df["can_aggregate_from_to"] = True
    # Apply the derived columns the converter expects:
    from stablebound.rt_convert import _extract_name, _extract_year_from_fnid
    df["from_name"] = df["from_unit_name"].apply(_extract_name)
    df["to_name"] = df["to_unit_name"].apply(_extract_name)
    df["from_year"] = df["fnid_from_unit"].apply(_extract_year_from_fnid)
    df["to_year"] = df["fnid_to_unit"].apply(_extract_year_from_fnid)
    return df


def test_split_one_parent_two_children():
    """Single Split: 1 parent → 2 children. Children get new IDs."""
    rows = [
        # 2000 snapshot — one unit
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="Alpha", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        # 2005 snapshot — two units (the split children)
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="Alpha-North", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="12", to_unit="0",
             from_unit_name="Alpha-South", to_unit_name="Country",
             fnid_from_unit="XX2005A102", fnid_to_unit="XX2005A0"),
        # Temporal: Alpha (2000) → Alpha-North + Alpha-South (2005)
        dict(category="temporal", relationship_type="split",
             from_unit="10", to_unit="11",
             from_unit_name="Alpha", to_unit_name="Alpha-North",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
        dict(category="temporal", relationship_type="split",
             from_unit="10", to_unit="12",
             from_unit_name="Alpha", to_unit_name="Alpha-South",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A102"),
    ]
    lineage, baseline = convert_relationship_table_to_lineage(
        _make_old_rt(rows), country="XX"
    )
    assert len(lineage) == 2
    assert (lineage["event_type"] == "Split").all()
    assert (lineage["event_year"] == 2005).all()
    assert (lineage["parent_id"] == "XX.ADM1.00001").all()
    assert set(lineage["child_id"]) == {"XX.ADM1.00002", "XX.ADM1.00003"}
    # Baseline has the single 2000 unit.
    assert len(baseline) == 1
    assert baseline.iloc[0]["name"] == "Alpha"
    assert baseline.iloc[0]["unit_id"] == "XX.ADM1.00001"


def test_merge_two_parents_one_child():
    """Merge: two parents → one child. Child gets a new ID."""
    rows = [
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="B", to_unit_name="Country",
             fnid_from_unit="XX2000A102", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="12", to_unit="0",
             from_unit_name="AB", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="merge",
             from_unit="10", to_unit="12",
             from_unit_name="A", to_unit_name="AB",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
        dict(category="temporal", relationship_type="merge",
             from_unit="11", to_unit="12",
             from_unit_name="B", to_unit_name="AB",
             fnid_from_unit="XX2000A102", fnid_to_unit="XX2005A101"),
    ]
    lineage, _ = convert_relationship_table_to_lineage(_make_old_rt(rows), country="XX")
    assert len(lineage) == 2
    assert (lineage["event_type"] == "Merge").all()
    assert set(lineage["parent_name"]) == {"A", "B"}
    assert (lineage["child_name"] == "AB").all()


def test_redistribute_multi_to_multi():
    """Multi-parent multi-child → Redistribute."""
    rows = [
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="B", to_unit_name="Country",
             fnid_from_unit="XX2000A102", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="20", to_unit="0",
             from_unit_name="C", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="21", to_unit="0",
             from_unit_name="D", to_unit_name="Country",
             fnid_from_unit="XX2005A102", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="split",
             from_unit="10", to_unit="20",
             from_unit_name="A", to_unit_name="C",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
        dict(category="temporal", relationship_type="split",
             from_unit="10", to_unit="21",
             from_unit_name="A", to_unit_name="D",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A102"),
        dict(category="temporal", relationship_type="split",
             from_unit="11", to_unit="20",
             from_unit_name="B", to_unit_name="C",
             fnid_from_unit="XX2000A102", fnid_to_unit="XX2005A101"),
        dict(category="temporal", relationship_type="split",
             from_unit="11", to_unit="21",
             from_unit_name="B", to_unit_name="D",
             fnid_from_unit="XX2000A102", fnid_to_unit="XX2005A102"),
    ]
    lineage, _ = convert_relationship_table_to_lineage(_make_old_rt(rows), country="XX")
    assert (lineage["event_type"] == "Redistribute").all()
    assert len(lineage) == 4


def test_pure_successor_emits_no_event():
    """1→1 successor: no lineage row; child inherits parent's ID."""
    rows = [
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="successor",
             from_unit="10", to_unit="11",
             from_unit_name="A", to_unit_name="A",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
    ]
    lineage, baseline = convert_relationship_table_to_lineage(
        _make_old_rt(rows), country="XX"
    )
    assert lineage.empty
    assert len(baseline) == 1
    assert baseline.iloc[0]["unit_id"] == "XX.ADM1.00001"


def test_invalid_admin_level_raises():
    with pytest.raises(ValueError, match="admin_level must be a positive int"):
        convert_relationship_table_to_lineage(
            pd.DataFrame({"category": ["hierarchical"]}),
            country="XX",
            admin_level=0,
        )


def test_admin_level_3_accepted_with_correct_prefix():
    """ADM3 (and deeper) levels build IDs with the matching ADMn prefix."""
    rows = [
        dict(category="hierarchical", relationship_type="admin3_2",
             from_unit="100", to_unit="10",
             from_unit_name="Tehsil1", to_unit_name="Dist1",
             fnid_from_unit="XX2000A301", fnid_to_unit="XX2000A201"),
    ]
    _, baseline = convert_relationship_table_to_lineage(
        _make_old_rt(rows), country="XX", admin_level=3
    )
    assert baseline.iloc[0]["unit_id"] == "XX.ADM3.00001"


def test_no_hierarchical_rows_raises():
    """Need at least one snapshot to assign baseline IDs."""
    df = pd.DataFrame(
        {
            "category": ["temporal"],
            "relationship_type": ["successor"],
            "from_unit": ["10"],
            "to_unit": ["11"],
            "from_unit_name": ["A"],
            "to_unit_name": ["A"],
            "fnid_from_unit": ["XX2000A101"],
            "fnid_to_unit": ["XX2005A101"],
            "from_name": ["A"],
            "to_name": ["A"],
            "from_year": [2000],
            "to_year": [2005],
        }
    )
    with pytest.raises(ValueError, match="No hierarchical rows"):
        convert_relationship_table_to_lineage(df, country="XX")


# ----------------------------------------------------------------------
# NameChange detection (both FEWS encodings)
# ----------------------------------------------------------------------

def test_explicit_name_change_emits_NameChange_event():
    """A `relationship_type == "name change"` row → NameChange event."""
    rows = [
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="OldName", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="NewName", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="name change",
             from_unit="10", to_unit="11",
             from_unit_name="OldName", to_unit_name="NewName",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
    ]
    lineage, _ = convert_relationship_table_to_lineage(_make_old_rt(rows), country="XX")
    assert len(lineage) == 1
    assert lineage.iloc[0]["event_type"] == "NameChange"
    assert lineage.iloc[0]["parent_name"] == "OldName"
    assert lineage.iloc[0]["child_name"] == "NewName"
    # NameChange invariant: same canonical ID on both sides.
    assert lineage.iloc[0]["parent_id"] == lineage.iloc[0]["child_id"]


def test_successor_with_differing_names_emits_NameChange():
    """1-to-1 successor with differing names → NameChange (not dropped)."""
    rows = [
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="Bombay", to_unit_name="Country",
             fnid_from_unit="XX1990A101", fnid_to_unit="XX1990A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="Mumbai", to_unit_name="Country",
             fnid_from_unit="XX1995A101", fnid_to_unit="XX1995A0"),
        dict(category="temporal", relationship_type="successor",
             from_unit="10", to_unit="11",
             from_unit_name="Bombay", to_unit_name="Mumbai",
             fnid_from_unit="XX1990A101", fnid_to_unit="XX1995A101"),
    ]
    lineage, _ = convert_relationship_table_to_lineage(_make_old_rt(rows), country="XX")
    assert len(lineage) == 1
    row = lineage.iloc[0]
    assert row["event_type"] == "NameChange"
    assert row["parent_id"] == row["child_id"] == "XX.ADM1.00001"
    assert row["parent_name"] == "Bombay"
    assert row["child_name"] == "Mumbai"


def test_namechange_output_validates_with_LineageGraph():
    """NameChange rows pass through validate_relationship_table."""
    rows = [
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="A-renamed", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="successor",
             from_unit="10", to_unit="11",
             from_unit_name="A", to_unit_name="A-renamed",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
    ]
    lineage, _ = convert_relationship_table_to_lineage(_make_old_rt(rows), country="XX")
    graph = LineageGraph.from_dataframe(lineage)
    assert graph is not None


# ----------------------------------------------------------------------
# Admin level filtering (mixed-level files)
# ----------------------------------------------------------------------

def _make_mixed_level_rt() -> pd.DataFrame:
    """Synthetic file with both ADM1 and ADM2 rows, plus an ADM2 rename."""
    return _make_old_rt([
        # ADM1 snapshot 2000 + 2005 + a Split
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="State1", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="State1-N", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="12", to_unit="0",
             from_unit_name="State1-S", to_unit_name="Country",
             fnid_from_unit="XX2005A102", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="split",
             from_unit="10", to_unit="11",
             from_unit_name="State1", to_unit_name="State1-N",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
        dict(category="temporal", relationship_type="split",
             from_unit="10", to_unit="12",
             from_unit_name="State1", to_unit_name="State1-S",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A102"),
        # ADM2 snapshot 2000 + 2005, with a rename
        dict(category="hierarchical", relationship_type="admin2_1",
             from_unit="100", to_unit="10",
             from_unit_name="Dist1", to_unit_name="State1",
             fnid_from_unit="XX2000A20101", fnid_to_unit="XX2000A101"),
        dict(category="hierarchical", relationship_type="admin2_1",
             from_unit="101", to_unit="11",
             from_unit_name="Dist1-new", to_unit_name="State1-N",
             fnid_from_unit="XX2005A20101", fnid_to_unit="XX2005A101"),
        dict(category="temporal", relationship_type="successor",
             from_unit="100", to_unit="101",
             from_unit_name="Dist1", to_unit_name="Dist1-new",
             fnid_from_unit="XX2000A20101", fnid_to_unit="XX2005A20101"),
    ])


def test_admin_level_1_skips_admin2_rows():
    lineage, baseline = convert_relationship_table_to_lineage(
        _make_mixed_level_rt(), country="XX", admin_level=1
    )
    # Only the ADM1 Split shows up (2 rows for 2 children).
    assert (lineage["event_type"] == "Split").all()
    assert len(lineage) == 2
    assert all(eid.startswith("XX.ADM1.") for eid in lineage["parent_id"])
    assert all(eid.startswith("XX.ADM1.") for eid in baseline["unit_id"])


def test_admin_level_2_skips_admin1_rows():
    lineage, baseline = convert_relationship_table_to_lineage(
        _make_mixed_level_rt(), country="XX", admin_level=2
    )
    # Only the ADM2 NameChange shows up.
    assert len(lineage) == 1
    assert lineage.iloc[0]["event_type"] == "NameChange"
    assert all(eid.startswith("XX.ADM2.") for eid in baseline["unit_id"])


def test_no_hierarchical_at_admin_level_raises():
    """Requesting ADM2 from an ADM1-only file fails with a level-aware message."""
    rt = _make_old_rt([
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="10", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
    ])
    with pytest.raises(ValueError, match="admin_level=2"):
        convert_relationship_table_to_lineage(rt, country="XX", admin_level=2)


def test_relationship_type_case_normalized():
    """Mixed-case rel_type values still parse (FEWS files are inconsistent)."""
    rows = [
        dict(category="hierarchical", relationship_type="ADMIN1_0",
             from_unit="10", to_unit="0",
             from_unit_name="A", to_unit_name="Country",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2000A0"),
        dict(category="hierarchical", relationship_type="admin1_0",
             from_unit="11", to_unit="0",
             from_unit_name="A2", to_unit_name="Country",
             fnid_from_unit="XX2005A101", fnid_to_unit="XX2005A0"),
        dict(category="temporal", relationship_type="Successor",
             from_unit="10", to_unit="11",
             from_unit_name="A", to_unit_name="A2",
             fnid_from_unit="XX2000A101", fnid_to_unit="XX2005A101"),
    ]
    lineage, _ = convert_relationship_table_to_lineage(_make_old_rt(rows), country="XX")
    # Uppercase "Successor" should still be recognized → NameChange (names differ).
    assert len(lineage) == 1
    assert lineage.iloc[0]["event_type"] == "NameChange"
