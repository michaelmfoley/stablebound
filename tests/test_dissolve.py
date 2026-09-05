"""Geometry dissolution tests using simple synthetic polygons."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon, box

from stablebound.dissolve import dissolve


def _square(x_min, y_min, side=1.0):
    return box(x_min, y_min, x_min + side, y_min + side)


def _gdf(rows):
    """rows: list of (unit_id, polygon)."""
    return gpd.GeoDataFrame(
        {"unit_id": [r[0] for r in rows], "geometry": [r[1] for r in rows]},
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_dissolve_unions_polygons_sharing_stable_id():
    # Three adjacent unit squares, all in the same stable group → one polygon.
    gdf = _gdf(
        [
            ("U1", _square(0, 0)),
            ("U2", _square(1, 0)),
            ("U3", _square(2, 0)),
        ]
    )
    remap = {"U1": "S", "U2": "S", "U3": "S"}
    out = dissolve(gdf, remap, id_column="unit_id")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["stable_id"] == "S"
    assert row["n_modern"] == 3
    assert row["source_ids"] == "U1,U2,U3"
    # Three side-by-side unit squares should give a 3x1 rectangle of area 3.
    assert abs(row["geometry"].area - 3.0) < 1e-9


def test_dissolve_keeps_separate_groups_separate():
    gdf = _gdf(
        [
            ("U1", _square(0, 0)),
            ("U2", _square(1, 0)),
            ("U3", _square(0, 5)),  # geographically distant
        ]
    )
    remap = {"U1": "A", "U2": "A", "U3": "B"}
    out = dissolve(gdf, remap, id_column="unit_id")
    assert len(out) == 2
    by_id = {row["stable_id"]: row for _, row in out.iterrows()}
    assert by_id["A"]["n_modern"] == 2
    assert by_id["B"]["n_modern"] == 1


def test_dissolve_drops_units_missing_from_remap():
    gdf = _gdf(
        [
            ("U1", _square(0, 0)),
            ("U2", _square(1, 0)),
            ("ORPHAN", _square(10, 10)),
        ]
    )
    remap = {"U1": "A", "U2": "A"}
    out = dissolve(gdf, remap, id_column="unit_id")
    assert len(out) == 1
    assert out.iloc[0]["stable_id"] == "A"
    assert out.iloc[0]["n_modern"] == 2


def test_dissolve_repairs_invalid_geometry_before_unioning():
    # Bowtie polygon — invalid until make_valid()'d.
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not bowtie.is_valid
    gdf = _gdf([("U1", bowtie), ("U2", _square(2, 0))])
    remap = {"U1": "S", "U2": "S"}
    out = dissolve(gdf, remap, id_column="unit_id")
    # Should not raise; should produce one row.
    assert len(out) == 1


def test_dissolve_preserves_crs():
    gdf = _gdf([("U1", _square(0, 0)), ("U2", _square(1, 0))])
    remap = {"U1": "S", "U2": "S"}
    out = dissolve(gdf, remap, id_column="unit_id")
    assert out.crs is not None
    assert out.crs.to_string() == "EPSG:4326"
