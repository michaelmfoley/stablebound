"""End-to-end test of ModernBoundary on the bundled Exampleland dataset.

Verifies the orchestrator wiring (load → algorithm → derive_intensive
→ disk write → in-memory cache + reload). The arithmetic correctness
of the algorithm itself is covered by ``test_modern_algorithm.py``;
this test focuses on integration.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

EXAMPLELAND = Path(__file__).resolve().parents[1] / "examples" / "exampleland"


@pytest.fixture
def exampleland_fixture(tmp_path):
    """Yield (lineage, output_dir, stats_path, intensive) for a single test."""
    sys.path.insert(0, str(EXAMPLELAND.parent.parent))
    from examples.exampleland.config import (
        INTENSIVE,
        STATS_PATH,
        TARGET_YEAR,
        lineage,
    )

    return {
        "lineage": lineage,
        "output_dir": tmp_path / "exampleland_out",
        "stats_path": STATS_PATH,
        "intensive": INTENSIVE,
        "target_year": TARGET_YEAR,
    }


def _build_modern(fix):
    from stablebound import ModernBoundary

    mb = ModernBoundary(
        fix["lineage"],
        target_year=fix["target_year"],
        output_dir=fix["output_dir"],
    )
    mb.aggregate_stats(stats=fix["stats_path"], intensive=fix["intensive"])
    return mb


def test_build_modern_writes_expected_files(exampleland_fixture):
    mb = _build_modern(exampleland_fixture)
    modern_dir = exampleland_fixture["output_dir"] / "modern"
    assert (modern_dir / "stats_modern.csv").exists()
    assert (modern_dir / "event_fractions.csv").exists()
    assert (modern_dir / "late_reporting.csv").exists()
    assert (modern_dir / "summary.json").exists()


def test_modern_stats_schema(exampleland_fixture):
    mb = _build_modern(exampleland_fixture)
    sm = mb.get_modern_stats()
    expected = {
        "year", "season", "variable", "modern_id", "value",
        "sources", "lineage_depth", "has_nan_fraction",
    }
    assert expected.issubset(set(sm.columns))


def test_split_distributes_e001_pre_event_to_e011_and_e012(exampleland_fixture):
    # E.001 (Alpha) splits at 2014 into E.011 and E.012.
    # Post-event window 2015-2019: E.011 reports area=60/yr, E.012=40/yr.
    # E.001's pre-event (2010-2013) area=100/yr should split to 60/40.
    mb = _build_modern(exampleland_fixture)
    sm = mb.get_modern_stats()

    e011_2010 = sm[
        (sm["modern_id"] == "E.011")
        & (sm["year"] == 2010)
        & (sm["variable"] == "rice_area_ha")
    ].iloc[0]
    e012_2010 = sm[
        (sm["modern_id"] == "E.012")
        & (sm["year"] == 2010)
        & (sm["variable"] == "rice_area_ha")
    ].iloc[0]
    assert math.isclose(float(e011_2010["value"]), 60.0)
    assert math.isclose(float(e012_2010["value"]), 40.0)
    assert "E.001" in str(e011_2010["sources"])
    assert int(e011_2010["lineage_depth"]) == 1


def test_merge_pools_e004_and_e005_to_e013(exampleland_fixture):
    # E.004 + E.005 merge into E.013 at 2018. E.013 should inherit the
    # sum of both parents' pre-2018 data.
    mb = _build_modern(exampleland_fixture)
    sm = mb.get_modern_stats()
    e013_2010 = sm[
        (sm["modern_id"] == "E.013")
        & (sm["year"] == 2010)
        & (sm["variable"] == "rice_area_ha")
    ].iloc[0]
    assert math.isclose(float(e013_2010["value"]), 80.0)
    sources = str(e013_2010["sources"])
    assert "E.004" in sources and "E.005" in sources


def test_unchanged_unit_e002_passes_through(exampleland_fixture):
    # E.002 (Bravo) has no events; values should pass through unmodified.
    mb = _build_modern(exampleland_fixture)
    sm = mb.get_modern_stats()
    e002_2010 = sm[
        (sm["modern_id"] == "E.002")
        & (sm["year"] == 2010)
        & (sm["variable"] == "rice_area_ha")
    ].iloc[0]
    assert math.isclose(float(e002_2010["value"]), 80.0)
    assert int(e002_2010["lineage_depth"]) == 0


def test_namechange_unit_e003_passes_through(exampleland_fixture):
    # E.003 (Charlie) has a NameChange at 2016. Non-territorial — data
    # flows through unmodified.
    mb = _build_modern(exampleland_fixture)
    sm = mb.get_modern_stats()
    e003_2010 = sm[
        (sm["modern_id"] == "E.003")
        & (sm["year"] == 2010)
        & (sm["variable"] == "rice_area_ha")
    ].iloc[0]
    e003_2018 = sm[
        (sm["modern_id"] == "E.003")
        & (sm["year"] == 2018)
        & (sm["variable"] == "rice_area_ha")
    ].iloc[0]
    assert math.isclose(float(e003_2010["value"]), 90.0)
    assert math.isclose(float(e003_2018["value"]), 92.0)
    assert int(e003_2010["lineage_depth"]) == 0


def test_intensive_yield_recomputed_post_aggregation(exampleland_fixture):
    # intensive declares yield_mt_ha = production / area.
    # For E.013 in 2010: production=160 (100+60), area=80 (50+30) → yield=2.0
    mb = _build_modern(exampleland_fixture)
    sm = mb.get_modern_stats()
    yields = sm[sm["variable"] == "yield_mt_ha"]
    assert not yields.empty
    e013_2010_yield = yields[
        (yields["modern_id"] == "E.013") & (yields["year"] == 2010)
    ].iloc[0]
    assert math.isclose(float(e013_2010_yield["value"]), 2.0)


def test_late_reporting_includes_e001_post_event_rows(exampleland_fixture):
    # E.001 reports for 2014 and 2015. Event_year=2014 → 2014 is
    # pre-event under the canonical convention; only 2015 is late-reporting.
    mb = _build_modern(exampleland_fixture)
    late = mb.get_late_reporting()
    e001_late = late[late["unit_id"] == "E.001"]
    assert not e001_late.empty
    assert set(e001_late["year"]) == {2015}


def test_event_fractions_audit_table(exampleland_fixture):
    mb = _build_modern(exampleland_fixture)
    fr = mb.get_event_fractions()
    # 2014 split has 2 children × 2 variables = 4 audit rows
    fr_2014 = fr[fr["event_year"] == 2014]
    assert len(fr_2014) == 4
    assert set(fr_2014["child_id"]) == {"E.011", "E.012"}
    # 2018 merge has 2 parents × 1 child × 2 variables = 4 rows (one per
    # parent edge), fraction = 1.0 each
    fr_2018 = fr[fr["event_year"] == 2018]
    assert len(fr_2018) == 4
    assert all(math.isclose(f, 1.0) for f in fr_2018["fraction"])
    assert "computed_at" in fr.columns
    assert all(fr["computed_at"].notna())


def test_summary_counts(exampleland_fixture):
    mb = _build_modern(exampleland_fixture)
    s = mb.summary()
    assert s["n_events_processed"] == 4  # 2 splits + 2 merges; NameChange excluded
    assert s["n_modern_units"] == 5
    assert s["default_window"] == 5


def test_disk_cache_short_circuits_second_call(exampleland_fixture):
    # First aggregate writes outputs; a fresh ModernBoundary instance
    # over the same output_dir should rehydrate from disk.
    mb1 = _build_modern(exampleland_fixture)
    from stablebound import ModernBoundary
    mb2 = ModernBoundary(
        exampleland_fixture["lineage"],
        target_year=exampleland_fixture["target_year"],
        output_dir=exampleland_fixture["output_dir"],
    )
    # No stats arg needed here — short-circuit hits before stats logic
    # would matter. We still must pass `stats=` because the kwarg is
    # required, but it will go unread when the cache short-circuits.
    mb2.aggregate_stats(
        stats=exampleland_fixture["stats_path"],
        intensive=exampleland_fixture["intensive"],
    )

    sm1 = mb1.get_modern_stats()
    sm2 = mb2.get_modern_stats()
    assert len(sm1) == len(sm2)
    assert mb1.summary()["n_modern_units"] == mb2.summary()["n_modern_units"]
