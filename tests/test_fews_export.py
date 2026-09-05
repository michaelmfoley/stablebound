"""Tests for the country-agnostic FEWS deliverable builder (fews_export.py).

Uses a small synthetic two-level lineage (mirrors tests/test_unit_defs.py
fixtures). The byte-for-byte reproduction of India's frozen deliverable bundle
was a developer harness that is not part of this repository; these tests pin
the format/shape and the relationship-type coverage on a tiny, fully-controlled
example.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound.lineage import LineageGraph
from stablebound.fews_export import (
    DeliverableValidationError,
    attach_stats_fnids,
    build_deliverables,
    validate_no_within_admin1_duplicate,
)
from stablebound.fnid import build_admin2_code_map

RT_COLS = [
    "event_year", "event_type",
    "parent_id", "parent_name", "child_id", "child_name",
    "parent_coarse_id", "parent_coarse_name",
    "child_coarse_id", "child_coarse_name",
]
BASE_COLS = ["year", "unit_id", "name", "coarse_id", "coarse_name"]
YEARS = range(1997, 2007)


def _empty_graph() -> LineageGraph:
    return LineageGraph.from_dataframe(
        pd.DataFrame({c: pd.Series([], dtype=object) for c in
                      ["event_year", "event_type", "parent_id",
                       "parent_name", "child_id", "child_name"]}).astype(
            {"event_year": int}),
        validate=False,
    )


def _fixture():
    """Synthetic India-like lineage exercising all five relationship types."""
    rt = pd.DataFrame([
        # split: U1 Alpha carves off U4 Delta (2000)
        (2000, "Split", "U1", "Alpha", "U4", "Delta", "S1", "One", "S1", "One"),
        # merge: U3 Gamma merges into U4 (2002)
        (2002, "Merge", "U3", "Gamma", "U4", "Delta", "S2", "Two", "S1", "One"),
        # redistribute: some of U2 Beta goes to U4 (2003)
        (2003, "Redistribute", "U2", "Beta", "U4", "Delta", "S1", "One", "S1", "One"),
        # name change: U4 Delta -> Delta2 (2005)
        (2005, "NameChange", "U4", "Delta", "U4", "Delta2", "S1", "One", "S1", "One"),
    ], columns=RT_COLS)
    baseline = pd.DataFrame([
        (1991, "U1", "Alpha", "S1", "One"),
        (1991, "U2", "Beta",  "S1", "One"),
        (1991, "U3", "Gamma", "S2", "Two"),
    ], columns=BASE_COLS)
    return LineageGraph.from_dataframe(rt), baseline


def test_admin_definitions_format(tmp_path):
    graph, baseline = _fixture()
    res = build_deliverables(
        iso="XX", admin0="Examplestan", years=YEARS, out_dir=tmp_path,
        adm2_graph=graph, admin1_graph=_empty_graph(), baseline=baseline,
    )
    assert len(res["admin_definitions"]) == len(list(YEARS))
    xl = pd.ExcelFile(tmp_path / "XX_Admin_Definitions_1997.xlsx")
    assert xl.sheet_names == ["XX_admin1_1997", "XX_admin2_1997"]
    a1 = pd.read_excel(xl, "XX_admin1_1997")
    a2 = pd.read_excel(xl, "XX_admin2_1997")
    assert list(a1.columns) == ["FNID", "EFF_YEAR", "COUNTRY_CODE", "admin0", "admin1"]
    assert list(a2.columns) == ["FNID", "EFF_YEAR", "COUNTRY_CODE", "admin0", "admin1", "admin2"]
    assert list(a2["FNID"]) == sorted(a2["FNID"])          # sorted ascending by FNID
    assert (a2["COUNTRY_CODE"] == "XX").all()
    assert set(a2["admin2"]) == {"Alpha", "Beta", "Gamma"}  # 1997: before the split


def test_relationship_all_five_types(tmp_path):
    graph, baseline = _fixture()
    build_deliverables(
        iso="XX", admin0="Examplestan", years=YEARS, out_dir=tmp_path,
        adm2_graph=graph, admin1_graph=_empty_graph(), baseline=baseline,
    )
    rel = pd.read_csv(tmp_path / "XX_GeographicUnitRelationship.csv")
    types = set(rel["relationship_type"])
    assert types <= {"successor", "split", "merge", "redistribute", "name change"}
    for t in ("successor", "split", "merge", "redistribute", "name change"):
        assert t in types, f"missing relationship_type {t!r}"
    assert list(rel.columns) == [
        "relationship_type", "from_unit_name", "to_unit_name",
        "fnid_from_unit", "fnid_to_unit",
    ]
    # the split row: Alpha -> Delta across 2000->2001
    split = rel[rel["relationship_type"] == "split"]
    assert (("Alpha", "Delta") == (split.iloc[0]["from_unit_name"],
                                   split.iloc[0]["to_unit_name"]))
    assert split.iloc[0]["fnid_from_unit"].startswith("XX2000A2")
    assert split.iloc[0]["fnid_to_unit"].startswith("XX2001A2")


def test_name_style_with_country(tmp_path):
    graph, baseline = _fixture()
    build_deliverables(
        iso="XX", admin0="Examplestan", years=YEARS, out_dir=tmp_path,
        adm2_graph=graph, admin1_graph=_empty_graph(), baseline=baseline,
        name_style="with_country",
    )
    rel = pd.read_csv(tmp_path / "XX_GeographicUnitRelationship.csv")
    assert rel["from_unit_name"].str.endswith(", Examplestan").all()


def test_attach_stats_fnids(tmp_path):
    graph, baseline = _fixture()
    cm2 = build_admin2_code_map(graph, baseline, iso="XX")
    stats = pd.DataFrame({
        "ADM2_ID": ["U1", "U2", "NOPE"],
        "Year": [1998, 1998, 1998],
        "value": [1.0, 2.0, 3.0],
    })
    out = attach_stats_fnids(stats, cm2, iso="XX", level=2,
                             unit_id_col="ADM2_ID", year_col="Year")
    assert out.loc[0, "FNID"].startswith("XX1998A2")
    assert out.loc[1, "FNID"].startswith("XX1998A2")
    assert out.loc[2, "FNID"] == ""            # unit not in code map -> empty
    assert list(stats.columns) == ["ADM2_ID", "Year", "value"]  # input not mutated


def test_validator_raises_on_duplicate():
    a1 = pd.DataFrame({"FNID": ["XX1997A101", "XX1997A102"],
                       "admin1": ["One", "One"]})  # duplicate admin1 name
    with pytest.raises(DeliverableValidationError, match="duplicate admin1"):
        validate_no_within_admin1_duplicate(a1, None, 1997)

    a1_ok = pd.DataFrame({"FNID": ["XX1997A101"], "admin1": ["One"]})
    a2 = pd.DataFrame({"FNID": ["XX1997A20101", "XX1997A20102"],
                       "admin1": ["One", "One"], "admin2": ["Dup", "Dup"]})
    with pytest.raises(DeliverableValidationError, match="duplicate admin2"):
        validate_no_within_admin1_duplicate(a1_ok, a2, 1997)


# --- Admin1-only (single-level) countries -------------------------------

A1_RT_COLS = ["event_year", "event_type", "parent_id", "parent_name",
              "child_id", "child_name"]
A1_BASE_COLS = ["year", "unit_id", "name"]


def _admin1_only_fixture():
    """Single-level lineage (no coarse layer) with a rename and a split.

    FEWS-ordinal ids so SS is the id-stable ordinal (XX.ADM1.0000N -> 'N').
    """
    rt = pd.DataFrame([
        (2003, "NameChange", "XX.ADM1.00002", "Beta", "XX.ADM1.00002", "Beta Renamed"),
        (2007, "Split", "XX.ADM1.00001", "Alpha", "XX.ADM1.00003", "Gamma"),
        (2007, "Split", "XX.ADM1.00001", "Alpha", "XX.ADM1.00004", "Delta"),
    ], columns=A1_RT_COLS)
    baseline = pd.DataFrame([
        (2000, "XX.ADM1.00001", "Alpha"),
        (2000, "XX.ADM1.00002", "Beta"),
    ], columns=A1_BASE_COLS)
    return LineageGraph.from_dataframe(rt), baseline


def test_admin1_only_deliverables(tmp_path):
    graph, baseline = _admin1_only_fixture()
    years = range(2000, 2010)
    res = build_deliverables(
        iso="XX", admin0="Examplestan", years=years, out_dir=tmp_path,
        adm2_graph=None, admin1_graph=graph, baseline=baseline,
    )
    assert len(res["admin_definitions"]) == len(list(years))

    # 2004 workbook: single admin1 tab (no admin2), 10-char admin1 FNIDs.
    xl = pd.ExcelFile(tmp_path / "XX_Admin_Definitions_2004.xlsx")
    assert xl.sheet_names == ["XX_admin1_2004"]
    a1 = pd.read_excel(xl, "XX_admin1_2004")
    assert list(a1.columns) == ["FNID", "EFF_YEAR", "COUNTRY_CODE", "admin0", "admin1"]
    assert (a1["FNID"].str.len() == 10).all()
    assert a1["FNID"].str.startswith("XX2004A1").all()
    names = dict(zip(a1["FNID"], a1["admin1"]))
    assert names["XX2004A101"] == "Alpha"            # id-stable SS = ordinal
    assert names["XX2004A102"] == "Beta Renamed"     # year-accurate (renamed 2003)

    # 2001 (pre-rename) still shows the old name — names are per-year.
    a1_2001 = pd.read_excel(tmp_path / "XX_Admin_Definitions_2001.xlsx",
                            sheet_name="XX_admin1_2001")
    assert dict(zip(a1_2001["FNID"], a1_2001["admin1"]))["XX2001A102"] == "Beta"

    # 2008 (post-split): Alpha gone, Gamma + Delta present.
    a1_2008 = pd.read_excel(tmp_path / "XX_Admin_Definitions_2008.xlsx",
                            sheet_name="XX_admin1_2008")
    ids_2008 = set(a1_2008["FNID"])
    assert "XX2008A101" not in ids_2008                 # Alpha ended in 2007 split
    assert {"XX2008A103", "XX2008A104"} <= ids_2008     # Gamma, Delta

    # relationship table: split + name change + successor, all 10-char FNIDs.
    rel = pd.read_csv(tmp_path / "XX_GeographicUnitRelationship.csv")
    assert {"split", "name change", "successor"} <= set(rel["relationship_type"])
    assert (rel["fnid_from_unit"].str.len() == 10).all()
    assert (rel["fnid_to_unit"].str.len() == 10).all()
