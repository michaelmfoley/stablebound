"""A country's facts can come from the caller instead of the built-in list.

India is built from its files plus the new keywords (admin level, coarse
lineage, validity and coverage years, notes, normalizer) with "IN" removed
from the built-in list, so every fact must come from the keywords. It must
equal ``Lineage("IN")`` on everything a user can read off it, including the
files ``export_fews`` writes. Each planted fact change must be reported — a
comparison that cannot fail proves nothing.
"""

from __future__ import annotations

import pytest

import stablebound.data as data_mod
from stablebound import BUNDLED_COUNTRIES, Lineage, normalize_name
from tests.india_reference import india_names
from tests.lineage_equivalence import lineage_differences

YEARS = range(1991, 2026)
EXPORT_YEARS = [2000, 2001, 2014, 2015]  # state splits (2000) and Telangana (2014)


@pytest.fixture(scope="module")
def india_bundled() -> Lineage:
    return Lineage("IN")


@pytest.fixture(scope="module")
def names() -> list[str]:
    return india_names()


def _keywords(**override) -> dict:
    b = BUNDLED_COUNTRIES["IN"]
    kw = dict(
        relationship_table_path=b.lineage_path,
        baseline_path=b.baseline_path,
        name_change_log_path=b.name_change_log_path,
        coarse_lineage_path=b.coarse_lineage_path,
        admin_level=b.admin_level,
        validity_start_year=b.validity_start_year,
        coverage_end_year=b.coverage_end_year,
        notes=b.notes,
        normalizer=b.normalizer,
    )
    kw.update(override)
    return kw


def _unregistered(monkeypatch, **override) -> Lineage:
    kw = _keywords(**override)
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    return Lineage("IN", **kw)


def test_keywords_alone_reproduce_the_bundled_lineage(
    monkeypatch, tmp_path, india_bundled, names
):
    ln = _unregistered(monkeypatch)
    assert "IN" not in BUNDLED_COUNTRIES  # nothing can leak in from the list
    diffs = lineage_differences(
        india_bundled, ln, years=YEARS, names=names,
        export_dir=tmp_path, export_years=EXPORT_YEARS, admin0="India",
    )
    assert diffs == []


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
def test_a_planted_fact_is_reported(monkeypatch, india_bundled, names, override, expect):
    """Positive control for the comparison above."""
    ln = _unregistered(monkeypatch, **override)
    diffs = lineage_differences(india_bundled, ln, years=(1991, 2024), names=names)
    assert any(d.startswith(expect) for d in diffs), diffs


def test_without_the_facts_a_district_lineage_reads_as_state_level(monkeypatch):
    """What the keywords fix: paths alone lose everything the list knew."""
    b = BUNDLED_COUNTRIES["IN"]
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    bare = Lineage(
        "IN",
        relationship_table_path=b.lineage_path,
        baseline_path=b.baseline_path,
        name_change_log_path=b.name_change_log_path,
    )
    assert bare.admin_level == 1
    assert bare.coverage_end_year is None
    assert bare.default_normalizer("Villupuram (TN)") == "villupuram tn"


def test_keywords_override_the_built_in_list(india_bundled):
    ln = Lineage("IN", coverage_end_year=2020, notes="tried a correction")
    assert (ln.coverage_end_year, ln.notes) == (2020, "tried a correction")
    assert ln.admin_level == india_bundled.admin_level  # unstated facts still come from the list


def test_coarse_lineage_needs_admin_level_two(monkeypatch):
    b = BUNDLED_COUNTRIES["IN"]
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    with pytest.raises(ValueError, match="admin_level >= 2"):
        Lineage("IN", relationship_table_path=b.lineage_path,
                coarse_lineage_path=b.coarse_lineage_path)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_admin_level_must_be_a_whole_number(bad):
    with pytest.raises(ValueError, match="admin_level"):
        Lineage("IN", admin_level=bad)


def test_unknown_columns_table_is_refused():
    with pytest.raises(ValueError, match="unknown table"):
        Lineage("IN", columns={"baselines": {"ADM2_ID": "unit_id"}})


def test_missing_coarse_lineage_file_is_reported_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError, match="coarse_lineage_path"):
        Lineage("IN", coarse_lineage_path=tmp_path / "nope.xlsx")


def test_the_bundled_object_keeps_its_normalizer_if_the_list_changes(monkeypatch):
    """Built once, a Lineage does not change when the registry is edited later."""
    ln = Lineage("IN")
    before = ln.default_normalizer("Villupuram (TN)")
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    assert ln.default_normalizer("Villupuram (TN)") == before == "villupuram"
