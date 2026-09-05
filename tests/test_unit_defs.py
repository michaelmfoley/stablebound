"""Unit-defs table + bulk write tests."""

from __future__ import annotations

import pandas as pd

from stablebound.fnid import build_admin1_code_map, build_admin2_code_map
from stablebound.lineage import LineageGraph
from stablebound.unit_defs import (
    build_admin1_attribution_table,
    build_unit_defs_table,
    write_unit_defs_files,
)


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


def _empty_graph() -> LineageGraph:
    df = pd.DataFrame({
        "event_year": pd.Series([], dtype=int),
        "event_type": pd.Series([], dtype=object),
        "parent_id":  pd.Series([], dtype=object),
        "parent_name": pd.Series([], dtype=object),
        "child_id":   pd.Series([], dtype=object),
        "child_name": pd.Series([], dtype=object),
    })
    return LineageGraph.from_dataframe(df, validate=False)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def test_attribution_at_baseline_year_matches_baseline():
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S2", "State Two"),
    ])
    g = _empty_graph()
    attr = build_admin1_attribution_table(g, baseline)
    rows_1991 = attr[attr["year"] == 1991]
    pairs = {(r["unit_id"], r["admin1_id"]) for _, r in rows_1991.iterrows()}
    assert pairs == {("U1", "S1"), ("U2", "S2")}


def test_attribution_after_coarse_event_reassigns_admin1():
    """A Coarse event in 2000 reassigns U2 from S1 to S3 starting in 2001."""
    rt = _rt([
        (2000, "Coarse", "U2", "Beta", "U2", "Beta",
         "S1", "State One", "S3", "State Three"),
    ])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
    ])
    g = LineageGraph.from_dataframe(rt)
    attr = build_admin1_attribution_table(g, baseline)

    # Pre-event year: still S1.
    rel_1999 = attr[(attr["year"] <= 1999) & (attr["unit_id"] == "U2")]
    assert rel_1999.sort_values("year").iloc[-1]["admin1_id"] == "S1"

    # Post-event year (event happens during 2000, takes effect in 2001).
    rel_2001 = attr[(attr["year"] <= 2001) & (attr["unit_id"] == "U2")]
    assert rel_2001.sort_values("year").iloc[-1]["admin1_id"] == "S3"


# ---------------------------------------------------------------------------
# Unit-defs table builder
# ---------------------------------------------------------------------------

def test_unit_defs_admin1_rows_match_active_admin1s():
    """At baseline year, level-1 file has one row per distinct admin1."""
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
        (1991, "U3", "Gamma", "S2", "State Two"),
    ])
    g = _empty_graph()
    cm1 = build_admin1_code_map(g, baseline, iso="XX")

    df = build_unit_defs_table(
        g, baseline, cm1,
        iso="XX", admin0="Examplestan",
        level=1, year=1991,
    )
    assert list(df.columns) == ["FNID", "EFF_YEAR", "COUNTRY", "admin0", "admin1"]
    assert sorted(df["admin1"]) == ["State One", "State Two"]
    assert set(df["FNID"]) == {"XX1991A101", "XX1991A102"}
    assert (df["EFF_YEAR"] == 1991).all()
    assert (df["COUNTRY"] == "XX").all()


def test_unit_defs_admin2_rows_match_active_admin2s():
    """At baseline year, level-2 file has one row per active admin2."""
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S1", "State One"),
        (1991, "U3", "Gamma", "S2", "State Two"),
    ])
    g = _empty_graph()
    cm2 = build_admin2_code_map(g, baseline, iso="XX")

    df = build_unit_defs_table(
        g, baseline, cm2,
        iso="XX", admin0="Examplestan",
        level=2, year=1991,
    )
    assert list(df.columns) == [
        "FNID", "EFF_YEAR", "COUNTRY", "admin0", "admin1", "admin2",
    ]
    assert len(df) == 3
    assert set(df["admin2"]) == {"Alpha", "Beta", "Gamma"}


def test_unit_defs_admin2_split_year_adds_new_district():
    """After a Split happens during 2003, the 2004 file shows the new child."""
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
    cm2 = build_admin2_code_map(g, baseline, iso="XX")

    pre = build_unit_defs_table(g, baseline, cm2, iso="XX", admin0="X", level=2, year=2003)
    post = build_unit_defs_table(g, baseline, cm2, iso="XX", admin0="X", level=2, year=2004)
    assert set(pre["admin2"]) == {"Alpha", "Beta"}
    assert set(post["admin2"]) == {"Alpha", "Beta_keep", "Beta_new"}


# ---------------------------------------------------------------------------
# Bulk write
# ---------------------------------------------------------------------------

def test_write_unit_defs_files_creates_expected_paths(tmp_path):
    rt = _rt([])
    baseline = _baseline([
        (1991, "U1", "Alpha", "S1", "State One"),
        (1991, "U2", "Beta",  "S2", "State Two"),
    ])
    g = _empty_graph()
    cm1 = build_admin1_code_map(g, baseline, iso="XX")
    cm2 = build_admin2_code_map(g, baseline, iso="XX")

    paths = write_unit_defs_files(
        g, baseline,
        admin1_code_map=cm1, admin2_code_map=cm2,
        iso="XX", admin0="Examplestan",
        years=[1991, 1992],
        levels=(1, 2),
        out_dir=tmp_path,
    )

    expected = {
        tmp_path / "admin1" / "XX_Admin1_1991.csv",
        tmp_path / "admin1" / "XX_Admin1_1992.csv",
        tmp_path / "admin2" / "XX_Admin2_1991.csv",
        tmp_path / "admin2" / "XX_Admin2_1992.csv",
    }
    assert set(paths) == expected
    for p in expected:
        assert p.exists()
        df = pd.read_csv(p)
        assert "FNID" in df.columns
        assert "EFF_YEAR" in df.columns
