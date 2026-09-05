"""Geometry dissolution: modern shapefile + stable-group remap → stable shapefile.

Given the modern shapefile and a remap (``unit_id → stable_id``), produce a
GeoDataFrame with one row per stable group whose geometry is the union of
its constituent modern polygons. Invalid geometries are repaired with
``shapely.make_valid`` before unioning.

For per-year stable shapefiles, the orchestrator calls
``build_stable_groups(graph, base_year=y)`` for each ``y`` in
``[config.base_year, config.max_year]`` to obtain a year-specific remap,
then calls ``dissolve`` with that remap.
"""

from __future__ import annotations

import geopandas as gpd
from shapely import make_valid
from shapely.ops import unary_union


def dissolve(
    modern_gdf: gpd.GeoDataFrame,
    remap: dict[str, str],
    id_column: str,
) -> gpd.GeoDataFrame:
    """Dissolve modern polygons by stable group.

    Args:
        modern_gdf: GeoDataFrame of the modern shapefile. Must contain
            ``id_column`` whose values match keys in ``remap``.
        remap: ``unit_id → stable_id`` mapping (from
            :func:`stablebound.groups.build_stable_groups`).
        id_column: name of the column in ``modern_gdf`` that holds unit IDs.

    Returns:
        GeoDataFrame with one row per stable group, columns:
        ``stable_id``, ``n_modern``, ``source_ids`` (comma-joined), ``geometry``.
        Modern units missing from ``remap`` are dropped; their absence is
        not a fatal error — callers can audit beforehand if needed.
    """
    # Fail fast on a missing ID column rather than silently produce
    # all-NaN stable_ids.
    if id_column not in modern_gdf.columns:
        raise KeyError(
            f"id_column {id_column!r} not in modern_gdf columns: {list(modern_gdf.columns)}"
        )

    # Project to just (id, geometry) to keep the working frame small;
    # any other shapefile columns aren't relevant to the dissolution.
    df = modern_gdf[[id_column, "geometry"]].copy()
    df["stable_id"] = df[id_column].map(remap)
    # Drop features whose unit_id isn't in the remap. With the singleton
    # fallback in ``build_stable_groups``, this is rare — any feature
    # unioned in via ``additional_units`` gets a stable_id. But a
    # researcher who passes ``additional_units=None`` (skipping the
    # fallback) may legitimately have unmapped features.
    df = df.dropna(subset=["stable_id"])

    # Geometry hygiene before unary_union: shapely's union routines fail
    # silently or produce inconsistent results on invalid geometries
    # (bowtie polygons, self-intersecting rings, etc.). make_valid()
    # repairs them in a deterministic way.
    df["geometry"] = df["geometry"].apply(_repair)

    # Group features by stable_id and union their geometries. unary_union
    # handles arbitrary-cardinality unions in one call (faster + more
    # numerically stable than an iterative pairwise union).
    rows = []
    for stable_id, group in df.groupby("stable_id"):
        unioned = unary_union(list(group["geometry"]))
        rows.append(
            {
                "stable_id": stable_id,
                "n_modern": len(group),
                # Audit trail: which modern features dissolved into this
                # stable polygon? Sorted + comma-joined for stable display.
                "source_ids": ",".join(sorted(group[id_column].astype(str))),
                "geometry": unioned,
            }
        )

    # CRS preserved from the input. Sorting by stable_id makes diffs
    # against the legacy regression outputs deterministic.
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=modern_gdf.crs)
    return out.sort_values("stable_id").reset_index(drop=True)


def _repair(geom):
    """Validate and repair a geometry before unioning.

    Returns the input unchanged if it's already valid (the common case);
    only invokes make_valid for invalid geometries since make_valid is
    relatively expensive.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    return make_valid(geom)
