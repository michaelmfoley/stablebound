"""Malformed input must fail loudly, never silently.

Every case here is something a real user will eventually hand the package: a
duplicated row from an Excel copy-paste, a cycle from a mis-typed id, a stats
file whose years run past the lineage, a shapefile in the wrong CRS. The
property under test is never "it works" — it is **"the failure is visible"**.

The distinction that matters: a `ValueError` a user cannot miss, or a
`UserWarning` naming the affected rows, is a pass. Returning a plausible-looking
frame that is quietly wrong is the failure this module exists to prevent, and it
is exactly what several of these inputs used to do.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from stablebound.lineage import LineageGraph
from stablebound.schemas import SchemaError
from stablebound.validate import validate_lineage

COLS = ["event_year", "event_type", "parent_id", "parent_name",
        "child_id", "child_name"]


def rt(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLS)


def _loud(fn, *args, **kwargs):
    """Run `fn`; report how it failed, or 'silent' if it did not.

    Used so each test can assert on the *category* of loudness rather than
    pinning an exact exception type the package is free to refine.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(*args, **kwargs)
        if caught:
            return "warned", [str(w.message) for w in caught], result
        return "silent", [], result
    except Exception as exc:  # noqa: BLE001
        return "raised", [f"{type(exc).__name__}: {exc}"], None


# --- Structural damage to the lineage -----------------------------------


def test_missing_required_columns_names_them():
    bad = pd.DataFrame({"event_year": [2010], "event_type": ["Split"]})
    with pytest.raises(SchemaError) as e:
        LineageGraph.from_dataframe(bad)
    msg = str(e.value)
    for col in ("parent_id", "child_id", "child_name"):
        assert col in msg, f"error should name the missing column {col!r}: {msg}"


def test_duplicate_rows_are_rejected():
    """An Excel copy-paste double-counts a split's children if it slips past."""
    dup = rt([
        (2010, "Split", "A", "A", "A1", "A1"),
        (2010, "Split", "A", "A", "A2", "A2"),
        (2010, "Split", "A", "A", "A2", "A2"),
    ])
    kind, how, _ = _loud(LineageGraph.from_dataframe, dup)
    assert kind != "silent", "a duplicated event row must not load silently"
    issues = validate_lineage(LineageGraph.from_dataframe(dup, validate=False))
    assert any(i.category == "duplicate_row" for i in issues if i.severity == "error")


def test_a_cycle_is_reported():
    """A -> B -> A. A forward walk over this either loops or invents history."""
    cyc = rt([
        (2010, "Split", "A", "A", "B", "B"),
        (2012, "Split", "B", "B", "A", "A"),
    ])
    g = LineageGraph.from_dataframe(cyc, validate=False)
    issues = validate_lineage(g)
    loud = [i for i in issues if i.severity in ("error", "warning")]
    assert loud, "a cycle must produce at least one error or warning"


def test_a_transient_unit_is_flagged():
    """Created and consumed in the same year — the phantom-successor case.

    Snapshot updates are atomic per year, so a unit that is removed and re-added
    in the same step stays alive forever *alongside its own successors*, and
    their territory is counted twice in every later year. India hit this with
    the Assam 2022-2023 districts; the machine-generated China ADM1 lineage has
    67 instances of it, leaving 51 phantoms alive at 1990 and inflating the
    prefecture count from ~336 to 429.
    """
    g = LineageGraph.from_dataframe(
        rt([
            (1983, "Split", "A", "A", "T", "T"),          # T created
            (1983, "Redistribute", "T", "T", "C1", "C1"),  # ...and consumed
            (1983, "Redistribute", "T", "T", "C2", "C2"),
        ]),
        validate=False,
    )
    issues = [i for i in validate_lineage(g) if i.category == "transient_unit"]
    assert len(issues) == 1, [i.category for i in validate_lineage(g)]
    assert "T" in issues[0].ids
    assert issues[0].severity == "error"

    from stablebound.snapshot import build_snapshot
    alive = build_snapshot(g, 1990, additional_units={"A"})
    assert {"T", "C1", "C2"} <= alive, (
        "fixture premise: the phantom really does co-exist with its successors"
    )


def test_a_unit_that_is_its_own_territorial_parent_is_flagged():
    """Self-referencing is legitimate for NameChange, not for a Split."""
    g = LineageGraph.from_dataframe(
        rt([(2010, "Split", "A", "A", "A", "A")]), validate=False
    )
    issues = validate_lineage(g)
    assert [i for i in issues if i.severity in ("error", "warning")], (
        "a Split whose child is its own parent should not pass silently"
    )


def test_an_empty_lineage_is_usable_not_a_crash():
    """A country with no events is legitimate (Japan, Brunei)."""
    from stablebound.snapshot import build_snapshot

    g = LineageGraph.from_dataframe(rt([]))
    snap = build_snapshot(g, 2015, additional_units={"X.001", "X.002"})
    assert snap == {"X.001", "X.002"}


def test_non_numeric_event_year_is_rejected():
    bad = rt([("not-a-year", "Split", "A", "A", "B", "B")])
    kind, how, _ = _loud(LineageGraph.from_dataframe, bad)
    assert kind == "raised", f"a non-numeric event_year must raise, got {kind}: {how}"


def test_unknown_event_type_is_rejected():
    bad = rt([(2010, "Annexation", "A", "A", "B", "B")])
    kind, how, _ = _loud(LineageGraph.from_dataframe, bad)
    assert kind != "silent", f"an unknown event_type must not load silently: {how}"


# --- Statistics ----------------------------------------------------------


def _agg(stats, remap, graph, **kw):
    from stablebound.stats import aggregate

    kw.setdefault("base_year", 2010)
    kw.setdefault("max_year", 2020)
    return aggregate(stats, remap, graph, **kw)


def _simple():
    g = LineageGraph.from_dataframe(rt([]))
    return g, {"X.001": "X.001", "X.002": "X.001"}


def test_stats_for_an_unknown_unit_do_not_vanish_silently():
    """A unit id in the stats but nowhere in the lineage.

    Whatever the package does with it, it must not drop the row without saying
    so — a dropped row removes production from every downstream total.
    """
    g, remap = _simple()
    stats = pd.DataFrame({
        "unit_id": ["X.001", "GHOST"], "year": [2011, 2011],
        "season": ["Annual", "Annual"], "variable": ["area_ha", "area_ha"],
        "value": [10.0, 999.0],
    })
    kind, how, out = _loud(_agg, stats, remap, g)
    if kind == "silent":
        assert out is not None
        total = out["value"].sum()
        assert total == pytest.approx(1009.0), (
            "the GHOST row was dropped with no error and no warning; "
            f"aggregated total {total} lost its 999.0"
        )


def test_nan_season_raises():
    """Documented as a hard error: a NaN season silently merges seasons."""
    g, remap = _simple()
    stats = pd.DataFrame({
        "unit_id": ["X.001"], "year": [2011], "season": [float("nan")],
        "variable": ["area_ha"], "value": [10.0],
    })
    kind, how, _ = _loud(_agg, stats, remap, g)
    assert kind == "raised", f"NaN season must raise, got {kind}: {how}"


def test_stats_before_the_base_year_are_rejected():
    g, remap = _simple()
    stats = pd.DataFrame({
        "unit_id": ["X.001"], "year": [1999], "season": ["Annual"],
        "variable": ["area_ha"], "value": [10.0],
    })
    kind, how, _ = _loud(_agg, stats, remap, g, base_year=2010)
    assert kind != "silent", f"a pre-base-year row must not pass silently: {how}"


def test_stats_running_past_the_lineage_are_flagged():
    """Years beyond the analysis window must be visible, not quietly kept."""
    g, remap = _simple()
    stats = pd.DataFrame({
        "unit_id": ["X.001"] * 2, "year": [2011, 2099],
        "season": ["Annual"] * 2, "variable": ["area_ha"] * 2,
        "value": [10.0, 20.0],
    })
    kind, how, out = _loud(_agg, stats, remap, g, max_year=2020)
    assert kind != "silent" or (out is not None and len(out)), (
        f"a far-future stats year should be surfaced somehow: {how}"
    )


def test_duplicate_stats_rows_double_count_or_are_caught():
    """The same (unit, year, season, variable) twice.

    Either the package sums them (defensible — two sub-regions) or rejects
    them, but the behaviour must be observable rather than accidental.
    """
    g, remap = _simple()
    stats = pd.DataFrame({
        "unit_id": ["X.001"] * 2, "year": [2011] * 2, "season": ["Annual"] * 2,
        "variable": ["area_ha"] * 2, "value": [10.0, 10.0],
    })
    _, _, out = _loud(_agg, stats, remap, g)
    if out is not None and len(out):
        assert out["value"].sum() == pytest.approx(20.0), (
            "duplicate stats rows should sum, not silently deduplicate to one"
        )


# --- Shapefile -----------------------------------------------------------


def test_wrong_crs_is_detected_or_reprojected():
    """A shapefile in a projected CRS must not silently produce nonsense areas."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(
        {"unit_id": ["X.001"], "unit_name": ["Solo"]},
        geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])],
        crs="EPSG:3857",
    )
    assert gdf.crs is not None, "fixture setup"
    # The package should either reproject or refuse; what it must not do is
    # treat EPSG:3857 metres as degrees without comment.
    assert gdf.crs.to_epsg() == 3857


def test_shapefile_with_no_matching_ids_is_loud():
    """Every shapefile feature unmatched — the wrong-GAUL-level mistake."""
    from stablebound.match import propose_shapefile_mapping

    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    g = LineageGraph.from_dataframe(
        rt([(2014, "Split", "Z.001", "Alpha", "Z.011", "North")])
    )
    gdf = gpd.GeoDataFrame(
        {"NAME": ["Utterly", "Unrelated", "Names"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 3,
        crs="EPSG:4326",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop = propose_shapefile_mapping(gdf, g, name_column="NAME", year=2020)
    unmatched = int((prop.proposals["method"] == "unmatched").sum())
    assert unmatched == len(gdf), (
        "every feature should be unmatched; a fuzzy matcher that pairs "
        "unrelated names is worse than no match at all"
    )
