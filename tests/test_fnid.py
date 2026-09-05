"""FNID code-map and assignment tests."""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound.fnid import (
    CODE_ALPHABET,
    CODE_CAPACITY,
    FNIDOverflowError,
    assign_fnids,
    build_admin1_code_map,
    build_admin2_code_map,
    build_fnid,
    encode_code,
)
from stablebound.lineage import LineageGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rt(rows):
    cols = [
        "event_year", "event_type",
        "parent_id", "parent_name", "child_id", "child_name",
        "parent_coarse_id", "parent_coarse_name",
        "child_coarse_id", "child_coarse_name",
    ]
    return pd.DataFrame(rows, columns=cols)


def _baseline(rows):
    cols = ["year", "unit_id", "name", "coarse_id", "coarse_name"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def test_alphabet_size_and_first_few():
    assert CODE_CAPACITY == 359
    assert CODE_ALPHABET[0] == "01"
    assert CODE_ALPHABET[98] == "99"
    assert CODE_ALPHABET[99] == "A0"
    assert CODE_ALPHABET[-1] == "Z9"


def test_encode_code_round_trip():
    for i in (0, 1, 50, 99, 100, 358):
        assert CODE_ALPHABET.index(encode_code(i)) == i


def test_encode_code_overflow():
    with pytest.raises(FNIDOverflowError):
        encode_code(CODE_CAPACITY)
    with pytest.raises(FNIDOverflowError):
        encode_code(-1)


def test_build_fnid_format_admin2():
    assert build_fnid("IN", 2003, 2, "01", "07") == "IN2003A20107"


def test_build_fnid_format_admin1():
    assert build_fnid("VN", 1991, 1, "01") == "VN1991A101"


# ---------------------------------------------------------------------------
# Admin1 code map
# ---------------------------------------------------------------------------

def test_admin1_code_map_baseline_only():
    """All admin1s present in baseline get codes 01, 02, ... in sort order."""
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "District Alpha", "S1", "State One"),
        (1991, "U2", "District Beta",  "S2", "State Two"),
        (1991, "U3", "District Gamma", "S2", "State Two"),
    ])
    g = LineageGraph.from_dataframe(rt) if len(rt) else _empty_graph()
    cm = build_admin1_code_map(g, baseline, iso="XX")
    assert list(cm["ADMIN1_ID"]) == ["S1", "S2"]
    assert list(cm["SS"]) == ["01", "02"]


def test_admin1_code_map_new_state_appended_after_baseline():
    """A state introduced via a Coarse event after baseline gets a fresh later code."""
    rt = _rt([
        # Coarse event in 2000: U2 is reassigned from S1 to a new state S3
        (2000, "Coarse", "U2", "U2_name", "U2", "U2_name",
         "S1", "State One", "S3", "State Three"),
    ])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
        (1991, "U3", "Gamma", "S2", "State Two"),
    ])
    g = LineageGraph.from_dataframe(rt)
    cm = build_admin1_code_map(g, baseline, iso="XX")
    # S1 and S2 (founders) come first; S3 (post-baseline) comes last.
    assert list(cm["ADMIN1_ID"]) == ["S1", "S2", "S3"]
    assert list(cm["SS"]) == ["01", "02", "03"]


# ---------------------------------------------------------------------------
# Admin2 code map
# ---------------------------------------------------------------------------

def test_admin2_code_map_basic():
    """Two states, three districts → SS by state, DD by district within state."""
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha",   "S1", "State One"),
        (1991, "U2", "Beta",    "S1", "State One"),
        (1991, "U3", "Gamma",   "S2", "State Two"),
    ])
    g = _empty_graph()
    cm = build_admin2_code_map(g, baseline, iso="XX")
    rows = {r["ADMIN2_ID"]: (r["SS"], r["DD"]) for _, r in cm.iterrows()}
    assert rows["U1"] == ("01", "01")
    assert rows["U2"] == ("01", "02")
    assert rows["U3"] == ("02", "01")


def test_admin2_origin_state_persists_through_coarse():
    """A district reassigned via Coarse keeps its origin state's SS."""
    rt = _rt([
        (2000, "Coarse", "U2", "Beta", "U2", "Beta",
         "S1", "State One", "S3", "State Three"),
    ])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
    ])
    g = LineageGraph.from_dataframe(rt)
    cm = build_admin2_code_map(g, baseline, iso="XX")
    rows = {r["ADMIN2_ID"]: (r["SS"], r["ORIGIN_ADMIN1_ID"]) for _, r in cm.iterrows()}
    # U2's origin is still S1, so it carries S1's SS even after the Coarse.
    assert rows["U2"][1] == "S1"
    assert rows["U2"][0] == rows["U1"][0]


def test_admin2_split_creates_new_dd_with_origin_ss():
    """A district carved off via Split inherits the parent's origin SS but a new DD."""
    rt = _rt([
        (2003, "Split", "U2", "Beta",  "U4", "Beta_keep",
         "S1", "State One", "S1", "State One"),
        (2003, "Split", "U2", "Beta",  "U5", "Beta_new",
         "S1", "State One", "S1", "State One"),
    ])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
    ])
    g = LineageGraph.from_dataframe(rt)
    cm = build_admin2_code_map(g, baseline, iso="XX")
    rows = {r["ADMIN2_ID"]: (r["SS"], r["DD"]) for _, r in cm.iterrows()}
    # All three districts have S1's SS. DDs are 01..03, with ordering by ID.
    assert rows["U1"][0] == rows["U2"][0] == rows["U4"][0] == rows["U5"][0]
    dds = sorted(v[1] for v in rows.values())
    assert dds == ["01", "02", "03", "04"]


# ---------------------------------------------------------------------------
# FNID assignment to stats
# ---------------------------------------------------------------------------

def test_assign_fnids_admin2():
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
    ])
    g = _empty_graph()
    cm = build_admin2_code_map(g, baseline, iso="XX")

    stats = pd.DataFrame({
        "ADM2_ID": ["U1", "U2", "U1"],
        "Year":    [2003, 2003, 2010],
        "Crop":    ["x",  "y",  "x"],
    })
    out = assign_fnids(
        stats, cm, iso="XX", level=2,
        year_col="Year", unit_id_col="ADM2_ID",
    )
    assert list(out["FNID"]) == [
        "XX2003A20101",  # U1 (S1=01, U1=01)
        "XX2003A20102",  # U2 (S1=01, U2=02)
        "XX2010A20101",  # U1 again, 2010
    ]


def test_assign_fnids_admin1():
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S2", "State Two"),
    ])
    g = _empty_graph()
    cm = build_admin1_code_map(g, baseline, iso="XX")

    stats = pd.DataFrame({
        "ADM1_ID": ["S1", "S2", "S1"],
        "Year":    [2003, 2003, 2010],
    })
    out = assign_fnids(
        stats, cm, iso="XX", level=1,
        year_col="Year", unit_id_col="ADM1_ID",
    )
    assert list(out["FNID"]) == [
        "XX2003A101",  # S1=01
        "XX2003A102",  # S2=02
        "XX2010A101",
    ]


def test_assign_fnids_missing_unit_yields_empty():
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
    ])
    g = _empty_graph()
    cm = build_admin2_code_map(g, baseline, iso="XX")
    stats = pd.DataFrame({"ADM2_ID": ["U1", "GHOST"], "Year": [2003, 2003]})
    out = assign_fnids(stats, cm, iso="XX", level=2, year_col="Year", unit_id_col="ADM2_ID")
    assert list(out["FNID"]) == ["XX2003A20101", ""]


# ---------------------------------------------------------------------------
# Helpers for empty graph
# ---------------------------------------------------------------------------

def _empty_graph() -> LineageGraph:
    """A graph with no events (used for baseline-only tests)."""
    df = pd.DataFrame({
        "event_year": pd.Series([], dtype=int),
        "event_type": pd.Series([], dtype=object),
        "parent_id":  pd.Series([], dtype=object),
        "parent_name": pd.Series([], dtype=object),
        "child_id":   pd.Series([], dtype=object),
        "child_name": pd.Series([], dtype=object),
    })
    return LineageGraph.from_dataframe(df, validate=False)
