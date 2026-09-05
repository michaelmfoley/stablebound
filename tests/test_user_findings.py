"""Tests for fixes from the post-v0.1.0 user-testing review.

Each test corresponds to one finding from the two user-testing review
rounds (internal notes, kept outside the repo). Grouped here so the
regression surface for those specific fixes stays visible.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stablebound.lineage import LineageGraph


def _aggregate_empty_graph() -> LineageGraph:
    """Empty graph for aggregate tests that don't exercise snapshot semantics."""
    return LineageGraph.from_dataframe(
        pd.DataFrame(columns=[
            "event_year", "event_type",
            "parent_id", "parent_name", "child_id", "child_name",
        ])
    )

EXAMPLELAND = Path(__file__).resolve().parents[1] / "examples" / "exampleland"


@contextlib.contextmanager
def _exampleland_fixture():
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


# --- Finding #5: silent-empty-filter warnings --------------------------


def test_aggregate_stats_warns_when_extensive_misses_all_variables():
    """A typo'd variable name used to silently produce 0 rows; now it warns."""
    from stablebound import StableBoundary

    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"], target_year=f["target_year"],
            max_year=f["max_year"], output_dir=f["output_dir"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sb.aggregate_stats(
                stats=f["stats_path"],
                extensive=["nonexistent_variable_ha"],
            )
            msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("not present in stats" in m for m in msgs), (
            f"expected a UserWarning naming the missing variable; got: {msgs}"
        )


def test_aggregate_stats_warns_when_filter_empties_frame():
    """Empty post-filter frame triggers an explicit 'nothing to aggregate' warning."""
    from stablebound import StableBoundary

    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"], target_year=f["target_year"],
            max_year=f["max_year"], output_dir=f["output_dir"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sb.aggregate_stats(
                stats=f["stats_path"],
                extensive=["nonexistent_variable_ha"],
            )
            msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("empty frame" in m for m in msgs), (
            f"expected an 'empty frame' UserWarning; got: {msgs}"
        )


def test_aggregate_stats_no_warning_when_filter_matches():
    """No spurious warning when extensive correctly matches stats variables."""
    from stablebound import StableBoundary

    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"], target_year=f["target_year"],
            max_year=f["max_year"], output_dir=f["output_dir"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sb.aggregate_stats(
                stats=f["stats_path"],
                extensive=["rice_area_ha", "rice_production_mt"],
                intensive=f["intensive"],
            )
            user_warns = [
                str(w.message) for w in caught
                if issubclass(w.category, UserWarning)
                and ("not present in stats" in str(w.message) or "empty frame" in str(w.message))
            ]
        assert not user_warns, f"unexpected warnings: {user_warns}"


# --- Finding #6: schema-versioned summary -------------------------------


def test_stable_summary_carries_schema_field():
    from stablebound import StableBoundary

    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"], target_year=f["target_year"],
            max_year=f["max_year"], output_dir=f["output_dir"],
        )
        sb.build_boundaries()
        summary = json.loads((f["output_dir"] / "summary.json").read_text())
        assert summary.get("_schema") == 3


def test_stable_cache_rejected_when_schema_mismatches():
    """A stale on-disk summary (different/missing _schema) should NOT
    short-circuit the build."""
    from stablebound import StableBoundary

    with _exampleland_fixture() as f:
        out = f["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        # Pre-seed the cache with an old-shape summary (no _schema) +
        # a remap.json so the existence check passes.
        (out / "summary.json").write_text(json.dumps({
            "country_code": "EX",
            "country_name": "Exampleland",   # old field that's been dropped
            "n_modern_units": 999,
            "year_range": [2010, 2020],
        }))
        (out / "remap.json").write_text("{}")

        sb = StableBoundary(
            f["lineage"], target_year=f["target_year"],
            max_year=f["max_year"], output_dir=out,
        )
        sb.build_boundaries()
        # The cache was rebuilt; the new summary has _schema and no
        # country_name.
        summary = json.loads((out / "summary.json").read_text())
        assert summary.get("_schema") == 3
        assert "country_name" not in summary
        assert summary["n_modern_units"] == 5


def test_modern_summary_carries_schema_field():
    from stablebound import ModernBoundary

    with _exampleland_fixture() as f:
        mb = ModernBoundary(
            f["lineage"], target_year=f["target_year"],
            output_dir=f["output_dir"],
        )
        mb.aggregate_stats(stats=f["stats_path"], intensive=f["intensive"])
        summary = json.loads((f["output_dir"] / "modern" / "summary.json").read_text())
        assert summary.get("_schema") == 3


# --- Finding #4: validation report API ----------------------------------


def test_lineage_validation_report_returns_text():
    from stablebound import Lineage
    ln = Lineage("IN")
    text = ln.validation_report()
    # India has dozens of non-error findings; the report should
    # acknowledge them, not be empty.
    assert isinstance(text, str)
    assert "Lineage(IN)" in text
    assert "finding" in text


def test_lineage_validation_issues_is_cached():
    from stablebound import Lineage
    ln = Lineage("IN")
    a = ln.validation_issues
    b = ln.validation_issues
    # Same list object — proves caching, also confirms we don't re-run
    # validate_lineage on every property access.
    assert a is b


def test_stable_boundary_validation_report_three_sections():
    from stablebound import StableBoundary
    with _exampleland_fixture() as f:
        sb = StableBoundary(
            f["lineage"], target_year=f["target_year"],
            max_year=f["max_year"], output_dir=f["output_dir"],
        )
        sb.build_boundaries()
        sb.aggregate_stats(stats=f["stats_path"], intensive=f["intensive"])
        report = sb.validation_report()
        # Three section headers expected.
        assert "Lineage" in report
        assert "Stable-group contiguity" in report
        assert "Stats × lineage" in report


def test_modern_boundary_validation_report_two_sections():
    from stablebound import ModernBoundary
    with _exampleland_fixture() as f:
        mb = ModernBoundary(
            f["lineage"], target_year=f["target_year"],
            output_dir=f["output_dir"],
        )
        mb.aggregate_stats(stats=f["stats_path"], intensive=f["intensive"])
        report = mb.validation_report()
        assert "Lineage" in report
        assert "Stats × lineage" in report
        # ModernBoundary doesn't run the stable contiguity check.
        assert "Stable-group contiguity" not in report


def test_auto_log_demoted_to_info(caplog):
    """The auto-emitted validation log line is INFO-level now, not WARNING.

    A first-time researcher running with default logging shouldn't be
    shouted at; they get the line at INFO if they opt into it.
    """
    import logging
    from stablebound import Lineage
    with caplog.at_level(logging.INFO, logger="stablebound.lineage"):
        ln = Lineage("IN")
        _ = ln.lineage   # triggers the log
    # Confirm: at INFO level, the line is captured. (At WARNING, it'd
    # also be captured, but the *level* would differ.)
    info_lines = [r for r in caplog.records if r.levelno == logging.INFO]
    warning_lines = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "validation" in r.getMessage().lower()
    ]
    assert info_lines, "expected INFO-level validation log from lineage load"
    assert not warning_lines, f"expected no WARNING-level validation logs, got: {warning_lines}"


# --- Phase 1: data-correctness fixes (handoff #1, #4) -------------------


def test_modern_default_max_year_includes_post_event_stats():
    """ModernBoundary's default max_year used to be graph.max_event_year,
    which silently dropped stats rows beyond the last event year.
    """
    from stablebound import ModernBoundary

    with _exampleland_fixture() as f:
        # Exampleland's last lineage event is 2018; stats run through 2020.
        # Default max_year should respect the stats.
        mb = ModernBoundary(
            f["lineage"],
            target_year=f["target_year"],
            output_dir=f["output_dir"],
            # no max_year — exercise the default
        )
        mb.aggregate_stats(stats=f["stats_path"], intensive=f["intensive"])
        sm = mb.get_modern_stats()
        years_present = set(sm["year"].unique())
        assert 2019 in years_present, f"2019 missing; years = {sorted(years_present)}"
        assert 2020 in years_present, f"2020 missing; years = {sorted(years_present)}"
        assert mb.summary()["max_year"] >= 2020


def test_modern_explicit_max_year_still_caps():
    """Explicit max_year is honored even when stats extend further."""
    from stablebound import ModernBoundary

    with _exampleland_fixture() as f:
        mb = ModernBoundary(
            f["lineage"],
            target_year=f["target_year"],
            max_year=2017,
            output_dir=f["output_dir"],
        )
        mb.aggregate_stats(stats=f["stats_path"], intensive=f["intensive"])
        sm = mb.get_modern_stats()
        assert sm["year"].max() <= 2017


def test_aggregate_preserves_all_nan_groups_as_nan():
    """All-NaN groups used to yield 0.0; null value = 'not reported',
    so they should yield NaN.
    """
    from stablebound.stats import aggregate

    df = pd.DataFrame({
        "unit_id": ["U1", "U1"],
        "year": [2010, 2010],
        "season": ["Annual", "Annual"],
        "variable": ["rice", "rice"],
        "value": [np.nan, np.nan],
    })
    out = aggregate(df, remap={"U1": "U1"}, graph=_aggregate_empty_graph(),
                    base_year=2010, max_year=2010)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["value"])


def test_aggregate_preserves_mixed_nan_as_real_sum():
    """A group with one NaN and one real value still sums to the real
    value — pandas skipna=True default behavior preserved.
    """
    from stablebound.stats import aggregate

    df = pd.DataFrame({
        "unit_id": ["U1", "U1"],
        "year": [2010, 2010],
        "season": ["Annual", "Annual"],
        "variable": ["rice", "rice"],
        "value": [np.nan, 5.0],
    })
    out = aggregate(df, remap={"U1": "U1"}, graph=_aggregate_empty_graph(),
                    base_year=2010, max_year=2010)
    assert len(out) == 1
    assert float(out.iloc[0]["value"]) == 5.0


# --- Phase 2: lineage semantics (handoff #6, #7) ------------------------


def test_india_1991_snapshot_has_only_adm2_ids():
    """The bundled India name_change_log carries 4 ADM1-level renames.
    Pre-fix those polluted the 1991 ADM2 snapshot (471 rows). Post-fix:
    467 rows, none with ADM1 IDs.
    """
    from stablebound import Lineage

    ln = Lineage("IN")
    snap = ln.snapshot(year=1991)
    assert len(snap) == 467
    assert not snap["unit_id"].astype(str).str.startswith("IN.ADM1.").any(), (
        f"ADM1 IDs leaked into snapshot: "
        f"{[u for u in snap['unit_id'] if str(u).startswith('IN.ADM1.')]}"
    )


def test_namechange_only_unit_in_baseline_appears_in_snapshot():
    """A unit with only a NameChange row drops out of initial_units, but
    a baseline entry should still bring it back via additional_units.
    Exampleland's Charlie (E.003) has a NameChange event AND is in
    baseline — it must remain in the 2010 snapshot.
    """
    from stablebound import Lineage

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    snap = ln.snapshot(year=2010)
    assert "E.003" in set(snap["unit_id"])


def test_empty_rt_min_max_event_year_return_none():
    """LineageGraph on an empty events DataFrame returns None for the
    year bounds instead of choking on int(NaN).
    """
    from stablebound.lineage import LineageGraph

    empty = pd.DataFrame(columns=[
        "event_year", "event_type", "parent_id", "parent_name",
        "child_id", "child_name",
    ])
    g = LineageGraph(events=empty)
    assert g.min_event_year is None
    assert g.max_event_year is None


def test_lineage_with_empty_rt_and_baseline_year_works(tmp_path):
    """A 'static country' (empty RT but a baseline with year=2010) can
    answer min_year / years / snapshot without crashing.
    """
    from stablebound import Lineage

    # Synthesize an empty RT and a small baseline.
    rt = tmp_path / "empty_rt.csv"
    rt.write_text(
        "event_year,event_type,parent_id,parent_name,child_id,child_name\n"
    )
    baseline = tmp_path / "baseline.csv"
    baseline.write_text(
        "unit_id,name,year\n"
        "U1,One,2010\n"
        "U2,Two,2010\n"
    )

    ln = Lineage("ZZ", relationship_table_path=rt, baseline_path=baseline)
    assert ln.min_year == 2010
    assert list(ln.years) == [2010]
    snap = ln.snapshot(year=2010)
    assert set(snap["unit_id"]) == {"U1", "U2"}


def test_lineage_with_no_signals_raises_clear_error(tmp_path):
    """A custom country with no validity year, no baseline, and an empty
    RT has no min_year to infer. Raise rather than return garbage.
    """
    from stablebound import Lineage

    rt = tmp_path / "empty_rt.csv"
    rt.write_text(
        "event_year,event_type,parent_id,parent_name,child_id,child_name\n"
    )
    ln = Lineage("ZZ", relationship_table_path=rt)
    with pytest.raises(ValueError, match="Cannot infer min_year"):
        _ = ln.min_year


# --- Phase 3: unmatched-features transparency (handoff #2) --------------


def _exampleland_with_extra_unmatched_shapefile(tmp_path):
    """Build a fixture: Exampleland lineage + a modern shapefile with
    5 matched + 2 mystery features. Returns (lineage, shapefile gdf).
    """
    import geopandas as gpd
    from shapely.geometry import Polygon
    from stablebound import Lineage

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    base_gdf = gpd.read_file(EXAMPLELAND / "modern.geojson")
    # Synthesize 2 extra features named "Mystery*" with no canonical IDs.
    extra = gpd.GeoDataFrame({
        "unit_name": ["MysteryA", "MysteryB"],
        "unit_id": [None, None],  # blank, simulates unmatched mapping
        "geometry": [
            Polygon([(5, 0), (6, 0), (6, 1), (5, 1)]),
            Polygon([(6, 0), (7, 0), (7, 1), (6, 1)]),
        ],
    }, crs=base_gdf.crs)
    augmented = pd.concat([base_gdf, extra], ignore_index=True)
    return ln, gpd.GeoDataFrame(augmented, crs=base_gdf.crs)


def test_attach_shapefile_keep_default_creates_sentinels(tmp_path):
    """Default on_unmatched='keep' replaces NaN unit_ids with sentinel
    strings so the features survive into the products.
    """
    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf)  # default on_unmatched="keep"
    ids = set(ln.shapefile["unit_id"].astype(str))
    sentinels = {x for x in ids if x.startswith("UNMATCHED_")}
    assert len(sentinels) == 2, f"expected 2 sentinels, got {sentinels}"


def test_attach_shapefile_drop_removes_rows(tmp_path):
    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf, on_unmatched="drop")
    # Dropped from .shapefile.
    assert len(ln.shapefile) == 5
    # Still recorded for transparency.
    assert len(ln.unmatched_features) == 2


def test_attach_shapefile_error_raises(tmp_path):
    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    with pytest.raises(ValueError, match="2 feature.* have no unit_id"):
        ln.attach_shapefile(gdf, on_unmatched="error")


def test_attach_shapefile_invalid_on_unmatched_raises(tmp_path):
    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    with pytest.raises(ValueError, match="on_unmatched must be"):
        ln.attach_shapefile(gdf, on_unmatched="bogus")


def test_unmatched_features_property_empty_when_all_matched():
    from stablebound import Lineage
    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    ln.attach_shapefile(EXAMPLELAND / "modern.geojson")
    assert ln.unmatched_features.empty


def test_stable_output_has_is_matched_column(tmp_path):
    """Sentinel singletons flow through build_boundaries; stable_<year>.
    geojson gains an is_matched=False column for them.
    """
    import geopandas as gpd
    from stablebound import StableBoundary

    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf)
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path / "out")
    sb.build_boundaries()

    stable_2020 = gpd.read_file(tmp_path / "out" / "stable_2020.geojson")
    assert "is_matched" in stable_2020.columns
    # 5 matched + 2 sentinel singletons → 7 rows at 2020 (no further
    # consolidation past 2018).
    assert len(stable_2020) == 7
    assert int((~stable_2020["is_matched"]).sum()) == 2


def test_stable_summary_carries_n_unmatched(tmp_path):
    from stablebound import StableBoundary

    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf)
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path / "out")
    sb.build_boundaries()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["n_unmatched_features"] == 2


def test_unmatched_features_geojson_written(tmp_path):
    import geopandas as gpd
    from stablebound import StableBoundary

    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf)
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path / "out")
    sb.build_boundaries()
    path = tmp_path / "out" / "unmatched_features.geojson"
    assert path.exists()
    sidecar = gpd.read_file(path)
    assert len(sidecar) == 2


def test_modern_output_includes_unmatched_sidecar(tmp_path):
    """ModernBoundary writes its own unmatched_features.geojson sidecar
    under the modern subdir.
    """
    from stablebound import ModernBoundary

    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf)
    mb = ModernBoundary(ln, target_year=2010, output_dir=tmp_path / "out")
    mb.aggregate_stats(
        stats=EXAMPLELAND / "stats.csv",
        intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
    )
    sidecar = tmp_path / "out" / "modern" / "unmatched_features.geojson"
    assert sidecar.exists()
    summary = json.loads((tmp_path / "out" / "modern" / "summary.json").read_text())
    assert summary["n_unmatched_features"] == 2


def test_aggregate_stats_drops_unmatched_mapping_rows_with_warning(tmp_path):
    """A stats mapping with blank proposed_unit_id used to fail schema
    validation outright. Now: warn + drop the affected rows.
    """
    from stablebound import Lineage, StableBoundary

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    ln.attach_shapefile(EXAMPLELAND / "modern.geojson")

    stats = pd.DataFrame({
        "district_name": ["Alpha"] * 2 + ["Mystery"] * 2,
        "year": [2010, 2011, 2010, 2011],
        "season": ["Annual"] * 4,
        "variable": ["rice_area_ha"] * 4,
        "value": [100.0, 110.0, 99.0, 88.0],
    })
    mapping = pd.DataFrame({
        "source_name": ["Alpha", "Mystery"],
        "proposed_unit_id": ["E.001", ""],   # Mystery deliberately blank
    })

    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path / "out")
    sb.build_boundaries()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sb.aggregate_stats(
            stats=stats, mapping=mapping, name_column="district_name",
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("Dropped" in m and "Mystery" in m for m in msgs), msgs
    # Aggregated output contains Alpha rows but not Mystery.
    agg = sb.get_stats()
    assert "Mystery" not in set(agg.get("constituent_ids", pd.Series()).astype(str))


def test_validation_report_includes_unmatched_section(tmp_path):
    from stablebound import StableBoundary

    ln, gdf = _exampleland_with_extra_unmatched_shapefile(tmp_path)
    ln.attach_shapefile(gdf)
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path / "out")
    sb.build_boundaries()
    report = sb.validation_report()
    assert "Unmatched features" in report
    assert "2 feature" in report


# --- Phase 4: cache invalidation (handoff #3, pragmatic) ----------------


def test_stable_cache_rejected_on_target_year_mismatch(tmp_path):
    """Building once with target_year=1997, then constructing a new
    StableBoundary with target_year=1991 on the same output_dir,
    must NOT return the 1997 cache.
    """
    from stablebound import Lineage, StableBoundary

    ln = Lineage("IN")
    PREPARED = Path(__file__).resolve().parents[1] / "tools" / "india" / "_prepared"
    shp = PREPARED / "India_modern_with_ids.geojson"
    if not shp.exists():
        pytest.skip("requires tools/india/_prepared (generated locally by prepare_shapefile.py)")
    ln.attach_shapefile(shp)

    sb1 = StableBoundary(ln, target_year=1997, output_dir=tmp_path)
    sb1.build_boundaries()
    assert sb1.summary()["year_range"][0] == 1997

    # Same output_dir, different target_year. Must reject the cache and rebuild.
    sb2 = StableBoundary(ln, target_year=1991, output_dir=tmp_path)
    sb2.build_boundaries()
    assert sb2.summary()["year_range"][0] == 1991


def test_stable_cache_rejected_on_country_code_mismatch(tmp_path):
    """If the on-disk summary was written for a different country
    (improbable in practice; programmer error), rebuild rather than
    hydrate stale outputs.
    """
    import json
    from stablebound import Lineage, StableBoundary

    ln = Lineage("IN")
    PREPARED = Path(__file__).resolve().parents[1] / "tools" / "india" / "_prepared"
    shp = PREPARED / "India_modern_with_ids.geojson"
    if not shp.exists():
        pytest.skip("requires tools/india/_prepared (generated locally by prepare_shapefile.py)")
    ln.attach_shapefile(shp)

    sb1 = StableBoundary(ln, target_year=1997, output_dir=tmp_path)
    sb1.build_boundaries()

    # Tamper with the summary.json to flip the country_code.
    summary_p = tmp_path / "summary.json"
    summary = json.loads(summary_p.read_text())
    summary["country_code"] = "ZZ"
    summary_p.write_text(json.dumps(summary))

    sb2 = StableBoundary(ln, target_year=1997, output_dir=tmp_path)
    sb2.build_boundaries()
    # Rebuild restored the correct country_code.
    assert sb2.summary()["country_code"] == "IN"


def test_stable_aggregate_stats_reruns_with_different_stats(tmp_path):
    """Calling aggregate_stats twice on the same instance with
    different stats inputs must produce different outputs.
    Pre-fix it short-circuited and returned the first call's
    aggregated frame.
    """
    from stablebound import Lineage, StableBoundary

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    ln.attach_shapefile(EXAMPLELAND / "modern.geojson")
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path)
    sb.build_boundaries()

    # First call: real stats file.
    sb.aggregate_stats(
        stats=EXAMPLELAND / "stats.csv",
        intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
    )
    first_rows = len(sb.get_stats())
    assert first_rows > 0

    # Second call: synthetic 1-row stats. Pre-fix this short-circuited.
    tiny = pd.DataFrame({
        "unit_id": ["E.001"],
        "year": [2010],
        "season": ["Annual"],
        "variable": ["rice_area_ha"],
        "value": [42.0],
    })
    sb.aggregate_stats(stats=tiny)
    second_rows = len(sb.get_stats())
    assert second_rows < first_rows, (
        f"second aggregate_stats appears to have returned the first call's "
        f"cached result (rows: {first_rows} vs {second_rows})"
    )


def test_modern_aggregate_stats_reruns_with_different_stats(tmp_path):
    from stablebound import Lineage, ModernBoundary

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    ln.attach_shapefile(EXAMPLELAND / "modern.geojson")
    mb = ModernBoundary(ln, target_year=2010, output_dir=tmp_path)

    mb.aggregate_stats(
        stats=EXAMPLELAND / "stats.csv",
        intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
    )
    first_rows = len(mb.get_modern_stats())

    tiny = pd.DataFrame({
        "unit_id": ["E.002"],
        "year": [2010],
        "season": ["Annual"],
        "variable": ["rice_area_ha"],
        "value": [99.0],
    })
    mb.aggregate_stats(stats=tiny)
    second_rows = len(mb.get_modern_stats())
    assert second_rows < first_rows


# --- Phase 5: API consistency + polish (handoff #5, #8, #9) -------------


def test_attach_shapefile_renames_custom_id_column(tmp_path):
    """A user passing `id_column='my_id'` used to leave the shapefile
    with 'my_id' which downstream products couldn't find. attach now
    renames internally to canonical 'unit_id'.
    """
    import geopandas as gpd
    from stablebound import Lineage, StableBoundary

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    gdf = gpd.read_file(EXAMPLELAND / "modern.geojson")
    # Rename the canonical unit_id column to something user-y.
    gdf = gdf.rename(columns={"unit_id": "my_id"})
    ln.attach_shapefile(gdf, id_column="my_id")
    # Internal storage is always under 'unit_id'.
    assert "unit_id" in ln.shapefile.columns
    # Products run end-to-end.
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path)
    sb.build_boundaries()
    assert sb.summary()["n_modern_units"] == 5


def test_get_stats_raises_on_unknown_filter(tmp_path):
    from stablebound import Lineage, StableBoundary

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    ln.attach_shapefile(EXAMPLELAND / "modern.geojson")
    sb = StableBoundary(ln, target_year=2010, max_year=2020, output_dir=tmp_path)
    sb.build_boundaries()
    sb.aggregate_stats(stats=EXAMPLELAND / "stats.csv")
    with pytest.raises(ValueError, match="unknown filter"):
        sb.get_stats(varible="rice")   # deliberate typo


def test_get_modern_stats_raises_on_unknown_filter(tmp_path):
    from stablebound import Lineage, ModernBoundary

    ln = Lineage(
        "ZZ",
        relationship_table_path=EXAMPLELAND / "relationship_table.csv",
        baseline_path=EXAMPLELAND / "baseline.csv",
    )
    ln.attach_shapefile(EXAMPLELAND / "modern.geojson")
    mb = ModernBoundary(ln, target_year=2010, output_dir=tmp_path)
    mb.aggregate_stats(stats=EXAMPLELAND / "stats.csv")
    with pytest.raises(ValueError, match="unknown filter"):
        mb.get_modern_stats(yr=2018)   # deliberate typo


def test_py_typed_marker_present():
    """pyproject.toml declares py.typed in package-data; the file
    must actually exist or type-checkers can't pick the package up.
    """
    import importlib.resources
    files = importlib.resources.files("stablebound")
    py_typed = files / "py.typed"
    assert py_typed.is_file(), "py.typed marker missing from package data"


def test_docs_no_build_modern_references():
    """Sweep public docs for stale build_modern() references."""
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    readme = Path(__file__).resolve().parents[1] / "README.md"
    pattern_files = list(docs_dir.glob("*.md")) + [readme]
    offenders = []
    for f in pattern_files:
        text = f.read_text()
        for line_no, line in enumerate(text.splitlines(), 1):
            # build_modern_ledger is a real internal function; allow it.
            if "build_modern_ledger" in line:
                continue
            if "build_modern" in line or ".build()" in line:
                offenders.append(f"{f.name}:{line_no}: {line.strip()}")
    assert not offenders, "stale build_modern references:\n" + "\n".join(offenders)
