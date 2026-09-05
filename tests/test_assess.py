"""The new-country readiness report.

`assess_country` is the first thing someone onboarding a country runs, so its
failure modes matter more than most: a section that grades PASS on broken input
is worse than no report at all, and one that grades FAIL on good input trains
people to ignore it.

Both directions are tested here — every section is exercised against input that
should pass and input that should fail.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stablebound import assess_country, detect_dialect
from stablebound.assess import ReadinessReport, Section

REPO = Path(__file__).resolve().parents[1]
SYN = REPO / "tests/fixtures/synthetic"
LEGACY_RT = REPO / "tests/fixtures/rt_convert/relationshiptable_XX.csv"

COLS = ["event_year", "event_type", "parent_id", "parent_name",
        "child_id", "child_name"]


def _rt(rows):
    return pd.DataFrame(rows, columns=COLS)


def _baseline(ids, year=2010):
    return pd.DataFrame({"unit_id": ids, "name": ids, "year": year})


# --- grading mechanics ---------------------------------------------------


def test_overall_grade_is_the_worst_section():
    r = ReadinessReport("X", [
        Section("a", "PASS", ""), Section("b", "WARN", ""), Section("c", "FAIL", ""),
    ])
    assert r.grade == "FAIL"
    assert [s.name for s in r.failures] == ["c"]


def test_skip_sections_do_not_drag_the_grade_down():
    """A country assessed with only a lineage must not read as failing."""
    r = ReadinessReport("X", [Section("a", "PASS", ""), Section("b", "SKIP", "")])
    assert r.grade == "PASS"


def test_all_skip_is_skip_not_pass():
    r = ReadinessReport("X", [Section("a", "SKIP", ""), Section("b", "SKIP", "")])
    assert r.grade == "SKIP"


def test_an_invalid_grade_is_rejected_at_construction():
    with pytest.raises(ValueError, match="grade must be one of"):
        Section("a", "GREAT", "")


# --- dialect detection ---------------------------------------------------


def test_detects_the_canonical_dialect():
    assert detect_dialect(_rt([(2010, "Split", "A", "A", "B", "B")])) == "canonical"


def test_detects_the_legacy_fews_dialect():
    legacy = pd.read_csv(LEGACY_RT)
    assert detect_dialect(legacy) == "legacy"


def test_an_unrecognisable_table_fails_rather_than_guessing():
    junk = pd.DataFrame({"foo": [1], "bar": [2]})
    assert detect_dialect(junk) == "unknown"
    report = assess_country(relationship_table=junk, country="Junk")
    assert report.grade == "FAIL"
    assert "could not identify" in report.sections[0].summary


def test_a_user_need_not_know_which_dialect_they_have():
    """The whole point: same call, either format."""
    # A path, not a DataFrame: the legacy branch derives its year/name columns
    # in the reader, and assess_country hands an in-memory frame straight to
    # the converter.
    report = assess_country(relationship_table=LEGACY_RT, country="XX", admin_level=1)
    assert report.sections[0].data["dialect"] == "legacy"
    assert report.grade != "FAIL", str(report)


def test_uppercase_headers_are_flagged_but_still_usable():
    """Generated lineages often ship UPPERCASE; the loader is case-sensitive."""
    upper = _rt([(2010, "Split", "A", "A", "B", "B")])
    upper.columns = [c.upper() for c in upper.columns]
    report = assess_country(relationship_table=upper,
                            baseline=_baseline(["A"]), country="Upper")
    dialect = report.sections[0]
    assert dialect.grade == "WARN"
    assert any("lowercase" in d for d in dialect.detail)
    # ...and it still built the graph rather than bailing.
    assert report.sections[1].name == "Lineage health"


# --- lineage health ------------------------------------------------------


def test_a_transient_unit_fails_the_health_section():
    """The China pathology: created and consumed in one year."""
    report = assess_country(
        relationship_table=_rt([
            (1983, "Split", "A", "A", "T", "T"),
            (1983, "Redistribute", "T", "T", "C1", "C1"),
            (1983, "Redistribute", "T", "T", "C2", "C2"),
        ]),
        baseline=_baseline(["A"], year=1980),
        country="Transient",
    )
    health = next(s for s in report.sections if s.name == "Lineage health")
    assert health.grade == "FAIL"
    assert "transient_unit" in health.data["categories"]


def test_events_predating_the_baseline_fail_coverage():
    report = assess_country(
        relationship_table=_rt([(1995, "Split", "A", "A", "B", "B")]),
        baseline=_baseline(["A"], year=2010),
        country="Early",
    )
    cov = next(s for s in report.sections if s.name == "Baseline coverage")
    assert cov.grade == "FAIL"
    assert any("predate" in d for d in cov.detail)


def test_an_orphan_parent_fails_coverage():
    """A parent that exists in neither the baseline nor an earlier event."""
    report = assess_country(
        relationship_table=_rt([(2015, "Split", "GHOST", "Ghost", "B", "B")]),
        baseline=_baseline(["A"]),
        country="Orphan",
    )
    cov = next(s for s in report.sections if s.name == "Baseline coverage")
    assert cov.grade == "FAIL"
    assert cov.data["n_orphans"] == 1


def test_no_baseline_warns_about_invisible_units():
    report = assess_country(
        relationship_table=_rt([(2015, "Split", "A", "A", "B", "B")]),
        country="NoBase",
    )
    cov = next(s for s in report.sections if s.name == "Baseline coverage")
    assert cov.grade == "WARN"


# --- shapefile -----------------------------------------------------------


def test_a_missing_name_column_says_what_is_available():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame({"WRONG": ["a"]},
                           geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326")
    report = assess_country(
        relationship_table=_rt([(2015, "Split", "A", "A", "B", "B")]),
        baseline=_baseline(["A"]), shapefile=gdf, shapefile_name_column="NAME",
        country="BadCol",
    )
    shp = next(s for s in report.sections if s.name == "Shapefile match")
    assert shp.grade == "FAIL"
    assert "WRONG" in " ".join(shp.detail)


def test_a_large_unit_count_gap_is_called_out_as_a_level_mistake():
    """C14: GAUL L1 has 17 Philippine regions where the RT has ~80 provinces.

    Reported as a probable wrong-admin-level error rather than a matching
    failure, because the fix is 'use the other file', not 'add aliases'.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"NAME": [f"Region {i}" for i in range(4)]},
                           geometry=[poly] * 4, crs="EPSG:4326")
    report = assess_country(
        relationship_table=_rt([]),
        baseline=_baseline([f"U.{i:03d}" for i in range(40)]),
        shapefile=gdf, shapefile_name_column="NAME", country="LevelGap",
    )
    shp = next(s for s in report.sections if s.name == "Shapefile match")
    assert shp.grade == "FAIL"
    assert shp.data.get("probable_wrong_admin_level") is True
    assert any("admin level" in d for d in shp.detail)


# --- statistics ----------------------------------------------------------


def test_fnid_bearing_stats_skip_name_matching():
    """FEWS-sourced statistics carry an FNID; that is an exact join."""
    stats = pd.DataFrame({"FNID": ["TH1980A101"], "Year": [2010], "value": [1.0]})
    report = assess_country(
        relationship_table=_rt([(2015, "Split", "A", "A", "B", "B")]),
        baseline=_baseline(["A"]), stats=stats, country="Fnid",
    )
    st = next(s for s in report.sections if s.name == "Statistics match")
    assert st.grade == "PASS"
    assert st.data["fnid_column"] == "FNID"
    assert "attach_stats_by_fnid" in " ".join(st.detail)


def test_completeness_forecast_reports_the_worst_year():
    """Answers 'how complete will this be' from the inputs, before a full run."""
    stats = pd.DataFrame({
        "unit_id": ["A", "B", "A"], "year": [2010, 2010, 2011],
        "value": [1.0, 2.0, 3.0],
    })
    report = assess_country(
        relationship_table=_rt([]), baseline=_baseline(["A", "B"]),
        stats=stats, stats_year_column="year", country="Forecast",
    )
    fc = next(s for s in report.sections if s.name == "Completeness forecast")
    assert fc.data["worst_year"] == 2011, str(report)
    assert fc.data["mean"] == pytest.approx(0.75)


# --- FEWS exportability --------------------------------------------------


def test_admin2_without_coarse_attribution_is_blocked():
    report = assess_country(
        relationship_table=_rt([(2015, "Split", "A", "A", "B", "B")]),
        baseline=_baseline(["A"]), admin_level=2, country="NoCoarse",
    )
    fews = next(s for s in report.sections if s.name == "FEWS exportability")
    assert fews.grade == "FAIL"
    assert any("coarse_id" in d for d in fews.detail)


def test_level_three_is_refused_with_the_reason():
    report = assess_country(
        relationship_table=_rt([(2015, "Split", "A", "A", "B", "B")]),
        baseline=_baseline(["A"]), admin_level=3, country="Deep",
    )
    fews = next(s for s in report.sections if s.name == "FEWS exportability")
    assert fews.grade == "FAIL"
    assert "code width" in " ".join(fews.detail).lower()


def test_code_space_overflow_is_predicted_not_discovered_at_write():
    """China ADM1 has 1094 units against a 2-character (capacity 100) space."""
    many = [f"U.{i:04d}" for i in range(150)]
    report = assess_country(
        relationship_table=_rt([]), baseline=_baseline(many),
        admin_level=1, country="Overflow",
    )
    fews = next(s for s in report.sections if s.name == "FEWS exportability")
    assert fews.grade == "FAIL"
    assert any("exceed" in d for d in fews.detail)


# --- the rendered report -------------------------------------------------


def test_str_renders_every_section_and_the_overall_grade():
    report = assess_country(
        relationship_table=SYN / "admin2_coarse/lineage.csv",
        baseline=SYN / "admin2_coarse/baseline.csv",
        country="Coarseland",
    )
    text = str(report)
    for name in ("Relationship table dialect", "Lineage health",
                 "Baseline coverage", "FEWS exportability"):
        assert name in text
    assert "OVERALL:" in text
    assert report.to_dict()["country"] == "Coarseland"
