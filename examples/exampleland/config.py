"""Exampleland — a synthetic dataset bundled with the stablebound package.

Five base-year (2010) units, with a split in 2014, a name change in
2016, and a merge in 2018, producing five modern (2020) units that
aggregate into four stable polygons.

Usage:

    from examples.exampleland.config import lineage, INTENSIVE, STATS_PATH
    from stablebound import StableBoundary

    sb = StableBoundary(lineage, target_year=2010, max_year=2020)
    sb.build_boundaries()
    sb.aggregate_stats(stats=STATS_PATH, intensive=INTENSIVE)

The module-level ``lineage`` is constructed once and shared across
tests. Tests pass their own ``output_dir`` to the products rather
than mutating the lineage.

Expected behavior on a clean run: the stats deliberately include rows
that exercise the late-reporting diagnostics, so ``UserWarning`` lines
about late reports and partial post-event windows will print. These are
not failures — they're the package surfacing its diagnostics on the
example data. Inspect ``modern/late_reporting.csv`` in the output
directory to see which rows are flagged.
"""

from __future__ import annotations

from pathlib import Path

from stablebound import Lineage

_HERE = Path(__file__).resolve().parent

# Exampleland is a custom country (no ISO code "EX" in the bundled
# registry), so we pass the file paths explicitly. The baseline lists
# the five 2010 units (Alpha, Bravo, Charlie, Delta, Echo). The RT
# adds the 2014 split, 2016 name change, and 2018 merge.
lineage = Lineage(
    "EX",
    relationship_table_path=_HERE / "relationship_table.csv",
    baseline_path=_HERE / "baseline.csv",
)
# The modern shapefile ships with canonical ``unit_id``s already
# attached — no name-matching step required. attach_shapefile passes
# through unmodified.
lineage.attach_shapefile(_HERE / "modern.geojson")

# Constants used by the test suite. Not part of the public API.
STATS_PATH = _HERE / "stats.csv"
INTENSIVE = {"yield_mt_ha": ("rice_production_mt", "rice_area_ha")}
TARGET_YEAR = 2010
MAX_YEAR = 2020
