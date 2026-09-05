"""End-to-end smoke test on the bundled Exampleland dataset."""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path


from stablebound import StableBoundary

EXAMPLELAND = Path(__file__).resolve().parents[1] / "examples" / "exampleland"


@contextlib.contextmanager
def _exampleland_fixture():
    """Yield a (lineage, output_dir, stats_path, intensive) tuple.

    Uses the shared Exampleland Lineage but a per-test output_dir so
    runs don't collide.
    """
    sys.path.insert(0, str(EXAMPLELAND.parent.parent))
    from examples.exampleland.config import (
        INTENSIVE,
        MAX_YEAR,
        STATS_PATH,
        TARGET_YEAR,
        lineage,
    )

    with tempfile.TemporaryDirectory() as td:
        yield {
            "lineage": lineage,
            "output_dir": Path(td) / "out",
            "stats_path": STATS_PATH,
            "intensive": INTENSIVE,
            "target_year": TARGET_YEAR,
            "max_year": MAX_YEAR,
        }


def test_exampleland_full_pipeline():
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=f["max_year"],
            output_dir=f["output_dir"],
        )
        sb.build_boundaries()

        # Per-year shapefiles exist for every year in [2010, 2020].
        out = f["output_dir"]
        for year in range(2010, 2021):
            assert (out / f"stable_{year}.geojson").exists(), f"missing stable_{year}.geojson"

        # Canonical files are present.
        assert (out / "remap.json").exists()
        assert (out / "snapshots.json").exists()
        assert (out / "name_history.csv").exists()
        assert (out / "summary.json").exists()
        assert (out / "shapefile_vintage_report.txt").exists()

        # Summary sanity.
        s = sb.summary()
        assert s["country_code"] == "EX"
        assert s["n_modern_units"] == 5
        # 4 stable groups under paper Algorithm 3:
        #   {E.001, E.011, E.012} (split lineage)
        #   {E.002}                (unchanged)
        #   {E.003}                (NameChange only)
        #   {E.004, E.005, E.013}  (merge)
        assert s["n_stable_groups_at_base"] == 4
        assert s["year_range"] == [2010, 2020]

        # 4 polygons at target year (groups consolidated). 5 polygons
        # at the modern year — no events at or after 2020 to consolidate.
        base_gdf = sb.get_boundary(2010)
        assert len(base_gdf) == 4
        modern_gdf = sb.get_boundary(2020)
        assert len(modern_gdf) == 5

        sb.aggregate_stats(
            stats=f["stats_path"],
            intensive=f["intensive"],
        )
        agg = sb.get_stats()
        # Must be a non-empty long-form frame.
        assert {"year", "season", "variable", "stable_id", "value"}.issubset(agg.columns)
        assert len(agg) > 0

        # Yield must be derived (not summed) — we recompute from production / area.
        yield_rows = agg[agg["variable"] == "yield_mt_ha"]
        assert len(yield_rows) > 0
        # Spot-check 2018 stable group E.001: production 132+88=220, area 60+40=100, yield 2.2.
        e001_2018 = yield_rows[(yield_rows["stable_id"] == "E.001") & (yield_rows["year"] == 2018)]
        assert len(e001_2018) == 1
        assert abs(float(e001_2018.iloc[0]["value"]) - 2.2) < 1e-9

        # 2026-06-04: reconcile is disabled by default (mode="off"). The
        # diagnostic was found to flag clean post-event reporting handoffs
        # as suspected double-counting at high rates on real data. The in-memory
        # flags object should therefore be an empty (but typed) frame.
        assert len(sb._reconcile_flags) == 0
        assert not (f["output_dir"] / "reconciliation_flags.csv").exists()


def test_exampleland_aggregation_routes_to_correct_stable_groups():
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=f["max_year"],
            output_dir=f["output_dir"],
        )
        sb.build_boundaries()
        sb.aggregate_stats(stats=f["stats_path"], intensive=f["intensive"])

        # E.001's stable group contains {E.001, E.011, E.012}.
        # In 2018 the constituents are E.011 (60) + E.012 (40) for area_ha = 100.
        rice_area = sb.get_stats(variable="rice_area_ha", year=2018)
        e001_2018 = rice_area[rice_area["stable_id"] == "E.001"]
        assert len(e001_2018) == 1
        assert float(e001_2018.iloc[0]["value"]) == 100.0
        assert e001_2018.iloc[0]["n_constituents"] == 2

        # In 2010, only E.001 reports (E.011 and E.012 don't exist yet) — area_ha = 100.
        rice_area_2010 = sb.get_stats(variable="rice_area_ha", year=2010)
        e001_2010 = rice_area_2010[rice_area_2010["stable_id"] == "E.001"]
        assert len(e001_2010) == 1
        assert float(e001_2010.iloc[0]["value"]) == 100.0
        assert e001_2010.iloc[0]["n_constituents"] == 1

        # E.013 (DeltaEcho) is the merge child of E.004 and E.005. The
        # stable group is {E.004, E.005, E.013} with stable_id="E.004".
        # At 2018 the only reporting member is E.013 (parents have ceased).
        e004_2018 = rice_area[(rice_area["stable_id"] == "E.004") & (rice_area["year"] == 2018)]
        assert len(e004_2018) == 1
        assert e004_2018.iloc[0]["late_reporting"] == False  # noqa: E712
        assert float(e004_2018.iloc[0]["value"]) == 80.0


def test_aggregate_stats_off_mode_skips_reconcile_file():
    """reconcile_mode='off' produces stats but no reconciliation_flags.csv."""
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=f["max_year"],
            output_dir=f["output_dir"],
        )
        sb.build_boundaries()
        sb.aggregate_stats(
            stats=f["stats_path"],
            intensive=f["intensive"],
            reconcile_mode="off",
        )
        # Aggregated stats present...
        agg = sb.get_stats()
        assert len(agg) > 0
        # ...but no flags file.
        assert not (f["output_dir"] / "reconciliation_flags.csv").exists()
        # In-memory flags object is an empty (but typed) frame.
        assert len(sb._reconcile_flags) == 0


def test_aggregate_stats_non_off_mode_raises():
    """2026-06-04: reconcile_mode != 'off' is disabled pending diagnostic rework."""
    import pytest
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=f["max_year"],
            output_dir=f["output_dir"],
        )
        sb.build_boundaries()
        for mode in ("flag", "merge", "subtract"):
            with pytest.raises(NotImplementedError):
                sb.aggregate_stats(
                    stats=f["stats_path"],
                    intensive=f["intensive"],
                    reconcile_mode=mode,
                )


def test_reconcile_stats_standalone_off_mode_returns_empty_flags():
    """2026-06-04: only mode='off' is supported. Returns an empty flags frame."""
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=f["max_year"],
            output_dir=f["output_dir"],
        )
        # No build_boundaries — reconcile_stats should auto-call it.
        flags = sb.reconcile_stats(stats=f["stats_path"])
        # Default mode is now "off"; returns an empty (but typed) frame.
        assert len(flags) == 0
        # No flags file written.
        assert not (f["output_dir"] / "reconciliation_flags.csv").exists()
        # No stats_aggregated.csv either — this method never aggregates.
        assert not (f["output_dir"] / "stats_aggregated.csv").exists()


def test_reconcile_stats_off_mode_returns_empty():
    """reconcile_stats(mode='off') is a no-op diagnostic — still loads + validates stats."""
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=f["max_year"],
            output_dir=f["output_dir"],
        )
        flags = sb.reconcile_stats(stats=f["stats_path"], mode="off")
        assert len(flags) == 0
        # No file written; if a stale one existed, it'd be removed.
        assert not (f["output_dir"] / "reconciliation_flags.csv").exists()
