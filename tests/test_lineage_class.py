"""Tests for the user-facing :class:`Lineage` class."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from stablebound import Lineage

EXAMPLELAND = Path(__file__).resolve().parents[1] / "examples" / "exampleland"


# --- Helpers -------------------------------------------------------------


def _exampleland_lineage() -> Lineage:
    """Custom-country Lineage pointing at the Exampleland fixtures.

    Exampleland is not bundled, so this exercises the custom-country
    branch of the constructor. The baseline is needed for units like
    Bravo that never appear in the RT.
    """
    return Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )


def _make_shapefile_gdf(names: list[str]) -> gpd.GeoDataFrame:
    geom = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i, _ in enumerate(names)]
    return gpd.GeoDataFrame({"name": names, "geometry": geom}, crs="EPSG:4326")


# --- Bundled-country construction ---------------------------------------


def test_bundled_india_construction():
    ln = Lineage("IN")
    assert ln.country_code == "IN"
    assert ln.validity_start_year == 1991
    assert "1991" in ln.notes


def test_bundled_country_without_a_name_change_log(bundle_synthetic):
    """The bundled branch with ``name_change_log_path=None``; India always has one."""
    bundle_synthetic("legacy_admin1", code="ZY", coverage_end_year=2030)
    ln = Lineage("ZY")
    assert ln.country_code == "ZY"
    assert ln.validity_start_year == 2010
    assert ln.name_change_log is None


def test_country_code_is_uppercased():
    ln = Lineage("in")
    assert ln.country_code == "IN"


def test_unknown_country_requires_rt_path():
    with pytest.raises(ValueError, match="not bundled"):
        Lineage("ZZ")


def test_unknown_country_with_rt_path_works():
    ln = _exampleland_lineage()
    assert ln.country_code == "ZZ"
    # No bundled validity year for a custom country.
    assert ln.validity_start_year is None
    # Exampleland's baseline carries year=2010, so min_year comes from
    # there (not from the lineage graph's first event year, 2014).
    assert ln.min_year == 2010


def test_min_year_falls_back_to_event_year_when_no_baseline():
    # Without a baseline, min_year falls back to the lineage graph's
    # earliest event year — the only signal left.
    ln = Lineage("ZZ", relationship_table_path=EXAMPLELAND / "relationship_table.csv")
    assert ln.baseline is None
    assert ln.min_year == ln.lineage.min_event_year


def test_missing_rt_path_raises():
    with pytest.raises(FileNotFoundError, match="relationship_table_path"):
        Lineage("ZZ", relationship_table_path="/nonexistent/rt.csv")


# --- Lazy loading -------------------------------------------------------


def test_lineage_property_loads_graph(monkeypatch):
    ln = Lineage("IN")
    assert ln._lineage is None       # not loaded yet
    g = ln.lineage
    assert ln._lineage is not None   # cached after first access
    # Same instance on subsequent access.
    assert ln.lineage is g


def test_relationship_table_merges_name_changes():
    # India bundle includes a name change log; the loaded RT should have
    # those rows folded in as NameChange events. Derived from the NCL rather
    # than hardcoded — a magic count goes stale the moment a bad row is
    # removed, which is exactly what happened to the previous `>= 50`.
    ln = Lineage("IN")
    n_ncl = len(ln.name_change_log)
    rt = ln.relationship_table
    nc_rows = rt[rt["event_type"] == "NameChange"]
    assert n_ncl > 0
    assert len(nc_rows) >= n_ncl


# --- min_year / years ---------------------------------------------------


def test_india_years_start_at_validity():
    ln = Lineage("IN")
    assert ln.min_year == 1991
    assert ln.years.start == 1991
    # Upper bound is max_event_year + 1 by default.
    assert ln.years.stop - 1 == ln.lineage.max_event_year + 1


def test_max_year_caps_years_range():
    ln = Lineage("IN", max_year=2000)
    assert max(ln.years) == 2000


# --- Snapshot -----------------------------------------------------------


def test_snapshot_default_year_is_latest():
    ln = Lineage("IN")
    snap = ln.snapshot()
    # Latest year defaults to max(years).
    assert (snap["year"] == max(ln.years)).all()
    # Hundreds of districts in modern India.
    assert len(snap) > 500


def test_snapshot_specific_year_returns_baseline_units():
    ln = Lineage("IN")
    snap_1991 = ln.snapshot(year=1991)
    # 1991 baseline has 467 districts (see project memory).
    assert len(snap_1991) > 400
    assert (snap_1991["year"] == 1991).all()
    assert {"unit_id", "unit_name", "year"} <= set(snap_1991.columns)


def test_snapshot_year_changes_unit_set():
    ln = Lineage("IN")
    snap_1991 = ln.snapshot(year=1991)
    snap_2020 = ln.snapshot(year=2020)
    # India added many districts between 1991 and 2020 via Split events.
    assert len(snap_2020) > len(snap_1991)


# --- Shapefile attachment workflow --------------------------------------


def test_propose_then_attach_round_trip(tmp_path):
    # Use Exampleland as a small custom-country test bed.
    ln = _exampleland_lineage()
    gdf = _make_shapefile_gdf(
        ["Alpha North", "Alpha South", "Bravo", "Charlie Renamed", "DeltaEcho"]
    )

    proposal = ln.propose_shapefile_mapping(gdf, name_column="name", year=2019)
    assert (proposal.proposals["method"] == "exact").all()

    mapping_csv = tmp_path / "mapping.csv"
    proposal.to_csv(mapping_csv)
    attached = ln.attach_shapefile(gdf, mapping=mapping_csv, name_column="name")
    assert "unit_id" in attached.columns
    assert set(attached["unit_id"]) == {"E.011", "E.012", "E.002", "E.003", "E.013"}

    # After attach, shapefile + shapefile_year are accessible.
    assert ln.shapefile is attached
    # Exampleland's "current-day" snapshot matches the modern shapefile
    # exactly at any year >= 2019 (post-merge). infer_year picks the
    # earliest year that matches, so we expect something near 2019.
    assert ln.shapefile_year >= 2019


def test_attach_shapefile_without_mapping_requires_id_column():
    ln = _exampleland_lineage()
    # A shapefile with unit_id already attached works directly.
    gdf = _make_shapefile_gdf(["Alpha North"])
    gdf["unit_id"] = ["E.011"]
    attached = ln.attach_shapefile(gdf)
    assert attached["unit_id"].tolist() == ["E.011"]


def test_attach_shapefile_without_mapping_or_id_column_errors():
    ln = _exampleland_lineage()
    gdf = _make_shapefile_gdf(["Alpha"])
    with pytest.raises(ValueError, match="No mapping provided"):
        ln.attach_shapefile(gdf)


def test_shapefile_accessors_raise_before_attach():
    ln = Lineage("IN")
    with pytest.raises(RuntimeError, match="No shapefile attached"):
        _ = ln.shapefile
    with pytest.raises(RuntimeError, match="No shapefile attached"):
        _ = ln.shapefile_year


# --- Stats mapping helper -----------------------------------------------


def test_propose_stats_mapping_via_lineage_method(tmp_path):
    ln = _exampleland_lineage()
    stats = pd.DataFrame(
        {
            "district": ["Alpha", "Bravo", "Charlie Renamed"],
            "year": [2010, 2011, 2017],
            "value": [1.0, 2.0, 3.0],
        }
    )
    proposal = ln.propose_stats_mapping(stats, name_column="district", year_column="year")
    assert len(proposal.proposals) == 3
    by_name = {r["source_name"]: r for _, r in proposal.proposals.iterrows()}
    assert by_name["Alpha"]["proposed_unit_id"] == "E.001"
    assert by_name["Charlie Renamed"]["proposed_unit_id"] == "E.003"


# --- Repr ----------------------------------------------------------------


def test_repr_indicates_attachment_state():
    ln = Lineage("IN")
    assert "no shapefile" in repr(ln)
    assert "country_code='IN'" in repr(ln)


def test_repr_progressive_disclosure():
    # Fresh Lineage: only identity + shapefile state. Cheap to print
    # (no lazy load triggered).
    ln = Lineage("IN")
    fresh_repr = repr(ln)
    assert "events=" not in fresh_repr
    assert "years=" not in fresh_repr
    # After accessing .lineage, the repr starts surfacing years/events.
    _ = ln.lineage
    full_repr = repr(ln)
    assert "events=" in full_repr
    assert "years=" in full_repr


# --- Legacy-RT constructor -----------------------------------------------

XX_FEWS_FIXTURE = (
    Path(__file__).parent / "fixtures" / "rt_convert" / "relationshiptable_XX.csv"
)


def test_from_legacy_rt_synthetic_fixture_end_to_end():
    """Build a Lineage straight from the hand-authored FEWS-format fixture."""
    ln = Lineage.from_legacy_rt(XX_FEWS_FIXTURE, country="XX", admin_level=1)
    assert ln.country_code == "XX"
    # Baseline derived from the 2010 hierarchical rows.
    assert ln.validity_start_year == 2010
    # The graph parses without errors and covers the expected year span.
    graph = ln.lineage
    assert graph.min_event_year >= 2015
    assert graph.max_event_year >= 2020
    assert all(eid.startswith("XX.ADM1.") for eid in graph.all_unit_ids())
    # No NCL was passed → name_change_log stays None.
    assert ln.name_change_log is None


def test_from_legacy_rt_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="relationship_table_path"):
        Lineage.from_legacy_rt(tmp_path / "missing.csv", country="XX")


def test_from_legacy_rt_missing_ncl_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="name_change_log_path"):
        Lineage.from_legacy_rt(
            XX_FEWS_FIXTURE,
            country="XX",
            name_change_log_path=tmp_path / "missing_ncl.csv",
        )


def test_from_legacy_rt_baseline_served_from_cache():
    """Derived baseline is reachable even though _baseline_path is unset."""
    ln = Lineage.from_legacy_rt(XX_FEWS_FIXTURE, country="XX", admin_level=1)
    base = ln.baseline
    assert base is not None
    assert len(base) > 0
    assert (base["year"] == 2010).all()
    assert all(uid.startswith("XX.ADM1.") for uid in base["unit_id"])


def test_from_legacy_rt_with_sidecar_ncl_merges(tmp_path):
    """A sidecar NCL is loaded and its rows show up in the merged RT."""
    ncl_path = tmp_path / "ncl.csv"
    pd.DataFrame(
        {
            "event_year": [2012],
            "unit_id": ["XX.ADM1.00001"],
            "old_name": ["OldAlpha"],
            "new_name": ["NewAlpha"],
        }
    ).to_csv(ncl_path, index=False)
    ln = Lineage.from_legacy_rt(
        XX_FEWS_FIXTURE,
        country="XX",
        admin_level=1,
        name_change_log_path=ncl_path,
    )
    assert ln.name_change_log is not None
    assert len(ln.name_change_log) == 1
    # The merged RT should now include a NameChange row for XX.ADM1.00001.
    nc_rows = ln.relationship_table[
        ln.relationship_table["event_type"] == "NameChange"
    ]
    assert ((nc_rows["parent_id"] == "XX.ADM1.00001")
            & (nc_rows["parent_name"] == "OldAlpha")
            & (nc_rows["child_name"] == "NewAlpha")).any()


# --- Attach-time shapefile checks ----------------------------------------


def _attach_ids(ln, ids):
    gdf = _make_shapefile_gdf([f"f{i}" for i in range(len(ids))])
    gdf["unit_id"] = ids
    return gdf


def test_attach_shapefile_reports_findings_and_stays_quiet_when_clean():
    """Exampleland's 2019 map, correct and then with one id swapped.

    E.011 is Alpha North, created by the 2015 split of Alpha (E.001). Putting
    the retired parent's id on that polygon is the defect India shipped, and
    it is invisible to the existing attach checks because the id is neither
    null nor malformed.
    """
    ln = _exampleland_lineage()
    good = ln.attach_shapefile(_attach_ids(ln, ["E.002", "E.003", "E.011", "E.012", "E.013"]))
    assert len(good) == 5
    assert ln.shapefile_issues == []
    assert "no findings" in ln.shapefile_report()

    ln2 = _exampleland_lineage()
    with pytest.warns(UserWarning, match="finding"):
        ln2.attach_shapefile(_attach_ids(ln2, ["E.002", "E.003", "E.001", "E.012", "E.013"]))
    cats = {i.category for i in ln2.shapefile_issues}
    assert "shapefile_retired_id" in cats
    assert "shapefile_unit_without_polygon" in cats
    assert "E.001" in ln2.shapefile_report()


def test_attach_shapefile_validation_can_be_switched_off():
    ln = _exampleland_lineage()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        ln.attach_shapefile(
            _attach_ids(ln, ["E.002", "E.003", "E.001", "E.012", "E.013"]),
            validate=False,
        )
    assert ln.shapefile_issues == []


def test_attach_shapefile_reports_a_duplicate_id():
    """The check that the first wiring silently disabled.

    attach_shapefile deduplicates ids into a set for infer_year; handing that
    set to the validator makes a duplicate unfindable. This asserts the
    duplicate survives the trip through attach, not just through the
    validator called directly.
    """
    ln = _exampleland_lineage()
    with pytest.warns(UserWarning, match="more than one polygon"):
        ln.attach_shapefile(_attach_ids(ln, ["E.002", "E.003", "E.011", "E.011", "E.013"]))
    dup = [i for i in ln.shapefile_issues if i.category == "shapefile_duplicate_id"]
    assert dup and dup[0].ids == ["E.011"]
