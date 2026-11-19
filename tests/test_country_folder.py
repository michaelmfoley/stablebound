"""Loading a relationship-table folder as it is shipped (``stablebound.json``).

The acceptance test: India's four tables as the StableBound-India deposit
publishes them (``ADM2_ID``, ``CHANGE_YEAR`` ... ; the frozen fixture in
tests/fixtures/india_as_shipped) load with ``Lineage.from_folder`` and give
India's B0 anchors — numbers recorded on 18 Nov 2026 from the copy then
bundled in the package, before any of this code existed. Planted faults (a
wrong mapping, a wrong fact, a missing rule) must each be caught, and a
broken description must fail at once with a message naming the problem.
"""

from __future__ import annotations

import json
import shutil

import pytest

from stablebound import (
    CountryFolderError,
    Lineage,
    read_country_description,
    write_country_description,
)
from stablebound.schemas import SchemaError
from tests.conftest import EXAMPLELAND, INDIA_FIXTURE
from tests.india_reference import (
    B0_ANCHORS,
    INDIA_COLUMNS,
    INDIA_NAME_RULES,
    anchors_of,
    copy_india_folder,
)
from tests.lineage_equivalence import lineage_differences


@pytest.fixture(scope="module")
def india():
    return Lineage.from_folder(INDIA_FIXTURE)


# --- the acceptance test ----------------------------------------------------


def test_india_as_published_loads_and_gives_the_b0_anchors(india):
    assert anchors_of(india) == B0_ANCHORS


def test_the_fixture_describes_itself_as_india_is_published():
    """The description carries India's mapping and rules, and nothing invented."""
    raw = json.loads((INDIA_FIXTURE / "stablebound.json").read_text(encoding="utf-8"))
    columns = {t: {k: v for k, v in m.items() if not k.startswith("_")}
               for t, m in raw["columns"].items() if not t.startswith("_")}
    rules = {k: v for k, v in raw["name_rules"].items() if not k.startswith("_")}
    assert columns == INDIA_COLUMNS
    assert rules == INDIA_NAME_RULES
    assert raw["files"] == {
        "relationship_table": "lineage_adm2.xlsx",
        "coarse_lineage": "lineage_adm1.xlsx",
        "baseline": "baseline_1991.csv",
        "name_change_log": "name_changes.xlsx",
    }


def test_the_published_headers_really_are_not_canonical():
    """Guard: the fixture must exercise the mapping, not bypass it."""
    head = (INDIA_FIXTURE / "baseline_1991.csv").read_text().splitlines()[0]
    assert head.startswith("YEAR,ADM1_ID,ADM1_NAME,ADM2_ID,ADM2_NAME")


# --- planted faults ----------------------------------------------------------


def test_a_wrong_mapping_is_refused(tmp_path):
    bad_cols = {
        "baseline": {"ADM2_IDX": "unit_id", "ADM2_NAME": "name"},
    }
    bad = copy_india_folder(tmp_path / "bad", columns=bad_cols)
    with pytest.raises(SchemaError, match="ADM2_IDX|adm2_idx"):
        _ = Lineage.from_folder(bad).baseline


def test_a_missing_mapping_is_refused(tmp_path):
    bad = copy_india_folder(tmp_path / "nomap", columns=None)
    with pytest.raises(SchemaError):
        _ = Lineage.from_folder(bad).baseline


def test_a_wrong_fact_is_reported(tmp_path, india):
    desc = json.loads((INDIA_FIXTURE / "stablebound.json").read_text())
    files = {k: v for k, v in desc["files"].items() if k != "coarse_lineage"}
    bad = copy_india_folder(tmp_path / "lvl1", admin_level=1, files=files)
    diffs = lineage_differences(india, Lineage.from_folder(bad), years=(1991,))
    assert any(d.startswith("admin_level") for d in diffs), diffs


def test_a_missing_name_rule_is_reported(tmp_path, india):
    rules = {k: v for k, v in INDIA_NAME_RULES.items() if k != "strip_before_lowercase"}
    bad = copy_india_folder(tmp_path / "rules", name_rules=rules)
    diffs = lineage_differences(india, Lineage.from_folder(bad), names=["Villupuram (TN)"])
    assert any(d.startswith("default_normalizer") for d in diffs), diffs


# --- the description file itself --------------------------------------------


@pytest.mark.parametrize(
    "changes, message",
    [
        (dict(format="stablebound-country/2"), "format"),
        (dict(admin_levle=2), "unknown key"),
        (dict(country_code=""), "country_code"),
        (dict(admin_level="2"), "whole number"),
        (dict(files={"baseline": "baseline_1991.csv"}), "relationship_table is required"),
        (dict(files={"relationship_table": "nope.xlsx"}), "does not exist"),
        (dict(files={"relationship_table": "lineage_adm2.xlsx", "lineages": "x"}),
         "unknown table"),
        (dict(columns={"coarse": {"A": "b"}}), "unknown table"),
        (dict(name_rules={"remove_patterns": ["(unclosed"]}), "not a valid pattern"),
        (dict(name_rules={"remove_patterns": "\\bsri\\s+"}), "list of strings"),
    ],
    ids=["format", "typo-key", "code", "level-type", "no-rt", "missing-file",
         "bad-table", "bad-columns-table", "bad-regex", "rules-not-list"],
)
def test_a_broken_description_fails_at_once(tmp_path, changes, message):
    bad = copy_india_folder(tmp_path / "broken", **changes)
    with pytest.raises(CountryFolderError, match=message):
        read_country_description(bad)


def test_a_coarse_lineage_needs_admin_level_two(tmp_path):
    bad = copy_india_folder(tmp_path / "coarse", admin_level=1)
    with pytest.raises(CountryFolderError, match="coarse_lineage"):
        read_country_description(bad)


def test_columns_for_a_table_that_is_not_there(tmp_path):
    bad = copy_india_folder(tmp_path / "orphan",
                            files={"relationship_table": "lineage_adm2.xlsx"},
                            admin_level=1)
    with pytest.raises(CountryFolderError, match="files.baseline is not"):
        read_country_description(bad)


def test_no_description_file(tmp_path):
    with pytest.raises(CountryFolderError, match="no stablebound.json"):
        read_country_description(tmp_path)


def test_comment_keys_are_ignored(tmp_path):
    folder = copy_india_folder(tmp_path / "comments", _note="anything at all")
    assert read_country_description(folder).country_code == "IN"


def test_the_path_of_the_json_file_itself_also_works():
    assert read_country_description(INDIA_FIXTURE / "stablebound.json") == \
        read_country_description(INDIA_FIXTURE)


def test_the_format_round_trips(tmp_path):
    folder = tmp_path / "rt"
    shutil.copytree(INDIA_FIXTURE, folder)
    desc = read_country_description(folder)
    out = write_country_description(desc, folder / "copy.json")
    assert read_country_description(out) == desc


# --- Exampleland, the documented example -----------------------------------


def test_exampleland_loads_from_its_folder_as_before(exampleland):
    names = list(exampleland.baseline["name"]) + ["Alpha North", "Alpha South"]
    diffs = lineage_differences(exampleland, Lineage.from_folder(EXAMPLELAND),
                                years=range(2010, 2021), names=names)
    assert diffs == []
