"""LineageGraph parsing + adjacency."""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound import SchemaError
from stablebound.lineage import LineageGraph, _coarse_name_of_unit_in_year


def _rt(rows):
    """Tiny RT-builder. Each row: (year, type, p_id, p_name, c_id, c_name)."""
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name"]
    return pd.DataFrame(rows, columns=cols)


def test_from_dataframe_validates_and_keeps_canonical_columns():
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    assert list(g.events.columns)[:6] == [
        "event_year",
        "event_type",
        "parent_id",
        "parent_name",
        "child_id",
        "child_name",
    ]
    assert g.events["event_year"].dtype.kind == "i"


def test_from_dataframe_rejects_unknown_event_type():
    df = _rt([(2003, "Bogus", "A", "A", "A1", "A")])
    with pytest.raises(SchemaError):
        LineageGraph.from_dataframe(df)


def test_initial_units_excludes_territorial_children_and_namechange_only():
    # A splits into A1, A2 (territorial). B has only a NameChange row
    # (non-territorial). Initial units = {A} only — B doesn't qualify
    # via territorial events. NameChange-only units must be supplied via
    # baseline/additional_units to appear in snapshots.
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
            (2010, "NameChange", "B", "old_B", "B", "new_B"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    assert g.initial_units() == {"A"}


def test_adjacency_after_filters_by_year():
    df = _rt(
        [
            (2003, "Split", "A", "A", "A1", "A"),
            (2003, "Split", "A", "A", "A2", "B"),
            (2010, "Split", "A1", "A", "X", "X"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    p2c, c2p = g.adjacency_after(year=2005)
    assert p2c == {"A1": {"X"}}
    assert c2p == {"X": {"A1"}}


def test_adjacency_excludes_name_changes():
    df = _rt(
        [
            (2010, "NameChange", "A", "old", "A", "new"),
            (2011, "Split", "A", "A", "A1", "A"),
        ]
    )
    g = LineageGraph.from_dataframe(df)
    p2c, c2p = g.adjacency_after(year=2000)
    # Only the Split should appear; NameChange has same parent_id and child_id
    # but is excluded because it isn't territorial.
    assert p2c == {"A": {"A1"}}
    assert c2p == {"A1": {"A"}}


COARSE_COLS = [
    "event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name",
    "parent_coarse_id", "parent_coarse_name", "child_coarse_id", "child_coarse_name",
]


def test_coarse_name_survives_a_rename_of_the_unit_itself():
    """A district's own rename must not erase the state it is in.

    merge_name_changes injects NameChange rows carrying no coarse columns.
    Reading only the single latest past event means that when a district's most
    recent event is its own rename, both coarse columns are null and the unit
    loses its state from that year onward. On India this made every district
    renamed in 2018 report state None in 2019 and after.
    """
    df = pd.DataFrame(
        [
            [2000, "Split", "P", "P", "D", "Dtown", "S1", "Statia", "S1", "Statia"],
            # The unit's own rename: no coarse columns, as merge_name_changes emits.
            [2010, "NameChange", "D", "Dtown", "D", "Dcity", None, None, None, None],
        ],
        columns=COARSE_COLS,
    )
    g = LineageGraph.from_dataframe(df)
    assert _coarse_name_of_unit_in_year(g, "D", 2005) == "Statia"
    assert _coarse_name_of_unit_in_year(g, "D", 2015) == "Statia", (
        "the rename wiped the state instead of falling through to the split"
    )


def test_coarse_name_picks_up_an_upper_admin_rename():
    """A state's rename must reach districts whose own events predate it.

    The relationship table records the state name as of each event, which is
    correct but frozen. Without applying later upper-admin renames, a district
    whose last territorial event was in 2000 reports the state's year-2000 name
    forever. India's Odisha districts reported "Orissa" in 2024, thirteen years
    after the 2011 rename.
    """
    df = pd.DataFrame(
        [
            [2000, "Split", "P", "P", "D", "D", "S1", "Oldland", "S1", "Oldland"],
            # The STATE's rename. parent_id == child_id == the coarse id, and it
            # matches no district -- which is why it used to do nothing.
            [2011, "NameChange", "S1", "Oldland", "S1", "Newland", None, None, None, None],
        ],
        columns=COARSE_COLS,
    )
    g = LineageGraph.from_dataframe(df)
    assert _coarse_name_of_unit_in_year(g, "D", 2005) == "Oldland", "pre-rename must keep the old name"
    assert _coarse_name_of_unit_in_year(g, "D", 2015) == "Newland", (
        "the state rename never reached the district"
    )


def test_coarse_name_applies_a_rename_on_the_forward_fallback():
    """A district with no past events must still hear about its state's rename.

    A district that has simply existed, untouched, reaches a different branch
    than one with a territorial history: with nothing behind it, the lookup
    reads the earliest event ahead of it and takes the state name recorded
    there. That name is whatever the table's author wrote, and India's table
    was written with historical vocabulary -- so Odisha's untouched districts
    still read "Orissa" at 2024 even after the past-events branch had been
    fixed. Three of the fix's three paths matter; this is the second.
    """
    df = pd.DataFrame(
        [
            # The state's rename, thirteen years before the question is asked.
            [2011, "NameChange", "S1", "Oldland", "S1", "Newland", None, None, None, None],
            # D's only appearance is ahead of the query year, and it records
            # the pre-rename state name.
            [2030, "Split", "D", "D", "D2", "D2", "S1", "Oldland", "S1", "Oldland"],
        ],
        columns=COARSE_COLS,
    )
    g = LineageGraph.from_dataframe(df)
    assert _coarse_name_of_unit_in_year(g, "D", 2005) == "Oldland", "pre-rename must keep the old name"
    assert _coarse_name_of_unit_in_year(g, "D", 2015) == "Newland", (
        "the forward fallback ignored the state rename"
    )
