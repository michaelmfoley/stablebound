"""A country's facts can come from the caller: ``Lineage(...)`` keywords.

India is built from its published files plus the keywords (admin level,
coarse lineage, validity and coverage years, notes, normalizer, column
mapping), by hand, and must equal ``Lineage.from_folder`` of the same folder
on everything a user can read off it, including the files ``export_fews``
writes. Both routes are then checked against India's B0 anchors — numbers
recorded before 0.2.0 from the then-bundled copy, not from this code. Each
planted fact change must be reported: a comparison that cannot fail proves
nothing.
"""

from __future__ import annotations

import pytest

from stablebound import Lineage, normalize_name, normalizer_from_rules, read_country_description
from tests.conftest import INDIA_FIXTURE
from tests.india_reference import B0_ANCHORS, INDIA_COLUMNS, INDIA_NAME_RULES, anchors_of, india_names
from tests.lineage_equivalence import lineage_differences

YEARS = range(1991, 2026)
EXPORT_YEARS = [2000, 2001, 2014, 2015]  # state splits (2000) and Telangana (2014)
DESC = read_country_description(INDIA_FIXTURE)


@pytest.fixture(scope="module")
def india_from_folder() -> Lineage:
    return Lineage.from_folder(INDIA_FIXTURE)


@pytest.fixture(scope="module")
def names() -> list[str]:
    return india_names()


def _keywords(**override) -> dict:
    """India's files and facts, written out by hand (not read from the JSON)."""
    kw = dict(
        relationship_table_path=DESC.files["relationship_table"],
        baseline_path=DESC.files["baseline"],
        name_change_log_path=DESC.files["name_change_log"],
        coarse_lineage_path=DESC.files["coarse_lineage"],
        admin_level=2,
        validity_start_year=1991,
        coverage_end_year=2025,
        notes=DESC.notes,
        normalizer=normalizer_from_rules(INDIA_NAME_RULES),
        columns=INDIA_COLUMNS,
    )
    kw.update(override)
    return kw


def test_keywords_reproduce_the_folder_route(tmp_path, india_from_folder, names):
    by_hand = Lineage("IN", **_keywords())
    diffs = lineage_differences(
        india_from_folder, by_hand, years=YEARS, names=names,
        export_dir=tmp_path, export_years=EXPORT_YEARS, admin0="India",
    )
    assert diffs == []


def test_both_routes_give_the_b0_anchors(india_from_folder):
    assert anchors_of(india_from_folder) == B0_ANCHORS
    assert anchors_of(Lineage("IN", **_keywords())) == B0_ANCHORS


@pytest.mark.parametrize(
    "override, expect",
    [
        (dict(admin_level=1, coarse_lineage_path=None), "admin_level"),
        (dict(coverage_end_year=2024), "coverage_end_year"),
        (dict(notes="something else"), "notes"),
        (dict(normalizer=normalize_name), "default_normalizer"),
    ],
    ids=["admin_level", "coverage_end_year", "notes", "normalizer"],
)
def test_a_planted_fact_is_reported(india_from_folder, names, override, expect):
    """Positive control for the comparison above."""
    ln = Lineage("IN", **_keywords(**override))
    diffs = lineage_differences(india_from_folder, ln, years=(1991, 2024), names=names)
    assert any(d.startswith(expect) for d in diffs), diffs


def test_without_the_facts_a_district_lineage_reads_as_state_level():
    """What the keywords fix: paths alone carry none of the country's facts."""
    bare = Lineage(
        "IN",
        relationship_table_path=DESC.files["relationship_table"],
        baseline_path=DESC.files["baseline"],
        name_change_log_path=DESC.files["name_change_log"],
        columns=INDIA_COLUMNS,
    )
    assert bare.admin_level == 1
    assert bare.coverage_end_year is None
    assert bare.default_normalizer("Villupuram (TN)") == "villupuram tn"


def test_coarse_lineage_needs_admin_level_two():
    with pytest.raises(ValueError, match="admin_level >= 2"):
        Lineage("IN", relationship_table_path=DESC.files["relationship_table"],
                coarse_lineage_path=DESC.files["coarse_lineage"])


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_admin_level_must_be_a_whole_number(bad):
    with pytest.raises(ValueError, match="admin_level"):
        Lineage("IN", relationship_table_path=DESC.files["relationship_table"], admin_level=bad)


def test_unknown_columns_table_is_refused():
    with pytest.raises(ValueError, match="unknown table"):
        Lineage("IN", relationship_table_path=DESC.files["relationship_table"],
                columns={"baselines": {"ADM2_ID": "unit_id"}})


def test_missing_coarse_lineage_file_is_reported_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match="coarse_lineage_path"):
        Lineage("IN", relationship_table_path=DESC.files["relationship_table"],
                admin_level=2, coarse_lineage_path=tmp_path / "nope.xlsx")
