"""End-to-end onboarding, driven exactly as a new user would drive it.

Every other test exercises one function. This one answers the question the
release actually rests on: *can somebody take four files and reach a finished
product without editing library source?*

It uses Exampleland, the five-unit synthetic country bundled with the
examples, so a genuine end-to-end run costs no external data. The path is the
one the documentation describes:

    assess_country  ->  propose_shapefile_mapping  ->  review  ->
    attach_shapefile  ->  build_boundaries  ->  validation_report  ->  export_fews

Exampleland's names match its lineage exactly, so what these tests can assert
about the *matcher* is structural (the report has the right sections, the
review CSV has the right columns, the counts add up). Behaviour on imperfect
real-world names is covered by ``test_match.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples/exampleland"
EX_SHP = EX / "modern.geojson"
NAME_COL = "unit_name"


@pytest.fixture(scope="module")
def modern():
    gpd = pytest.importorskip("geopandas")
    return gpd.read_file(EX_SHP)


def test_step1_assess_before_committing(exampleland, modern):
    """The first thing a user runs. Every section must render and grade."""
    from stablebound import assess_country

    report = assess_country(
        relationship_table=exampleland._rt_path,
        baseline=exampleland._baseline_path,
        shapefile=modern,
        shapefile_name_column=NAME_COL,
        country="Exampleland",
        admin_level=1,
    )
    assert report.grade in ("PASS", "WARN"), str(report)
    shp = next(s for s in report.sections if s.name == "Shapefile match")
    # Exampleland's names match its lineage exactly; a lower rate here means
    # the matcher or the geojson changed.
    assert shp.data["rate"] == 1.0, str(report)
    assert shp.data["unmatched"] == 0


def test_step2_the_review_artifact_is_actionable(exampleland, modern, tmp_path):
    """A user reviews a CSV, not a data structure. It has to be usable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proposal = exampleland.propose_shapefile_mapping(modern, name_column=NAME_COL)

    out = tmp_path / "mapping.csv"
    proposal.to_csv(out)
    written = pd.read_csv(out)
    assert len(written) == len(modern)
    for col in ("source_name", "proposed_unit_id", "method", "score"):
        assert col in written.columns, f"review CSV needs {col!r}"

    text = proposal.summary()
    assert "Total source rows" in text
    # The counts must sum to the total, or a reviewer silently loses rows.
    base = proposal.proposals["method"].astype(str).str.split("+").str[0]
    assert base.value_counts().sum() == len(proposal.proposals)


def test_step3_full_run_to_stable_boundaries(exampleland, modern, tmp_path):
    """Attach, build, and check the products actually landed on disk."""
    from stablebound import StableBoundary

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proposal = exampleland.propose_shapefile_mapping(modern, name_column=NAME_COL)
        exampleland.attach_shapefile(modern, mapping=proposal.proposals, name_column=NAME_COL)
        sb = StableBoundary(exampleland, output_dir=tmp_path / "out")
        sb.build_boundaries()

    summary = sb.summary()
    assert summary.get("n_stable_groups_at_base", 0) > 0, summary
    written = list((tmp_path / "out").rglob("*"))
    assert any(p.suffix == ".geojson" for p in written), "no geometry written"
    assert any(p.name == "remap.json" for p in written), "no remap written"


def test_step5_fews_export_admin1_only(exampleland, tmp_path):
    """The admin1-only branch of the FEWS export, on a country with no upper level."""
    res = exampleland.export_fews(
        tmp_path / "fews",
        years=range(exampleland.min_year, exampleland.min_year + 4),
        admin0="Exampleland",
    )
    rel = res["relationship"][0]
    assert rel.exists() and rel.name == "EX_GeographicUnitRelationship.csv"
    assert len(res["admin_definitions"]) == 4

    df = pd.read_csv(rel)
    assert len(df) > 0
    for p in res["admin_definitions"]:
        sheets = pd.ExcelFile(p).sheet_names
        assert any("admin1" in s for s in sheets)
        assert not any("admin2" in s for s in sheets), (
            "an admin1-only country must not emit an admin2 tab"
        )


def test_the_whole_path_needs_no_library_edits(exampleland):
    """The acceptance bar, stated as a test.

    Everything above runs through the public API only. This asserts that
    explicitly: the names used here are the names a user is told about.
    """
    import stablebound

    public = set(stablebound.__all__)
    for name in ("Lineage", "StableBoundary", "assess_country"):
        assert name in public, f"{name} must be part of the documented surface"
    for method in ("propose_shapefile_mapping", "attach_shapefile", "export_fews"):
        assert hasattr(exampleland, method), f"Lineage.{method} missing"
