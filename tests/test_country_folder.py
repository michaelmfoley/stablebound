"""Loading a relationship-table folder as it is shipped (``stablebound.json``).

The acceptance test: India's four tables, under the headers India publishes
them with (``ADM2_ID``, ``CHANGE_YEAR`` ...), plus a ``stablebound.json``,
load with ``Lineage.from_folder`` — "IN" removed from the built-in list, so
the folder alone supplies every fact — and equal ``Lineage("IN")`` on
everything a user can read off it. Planted faults (a wrong mapping, a wrong
fact, a missing rule) must each be caught.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import stablebound.data as data_mod
from stablebound import (
    CountryFolderError,
    Lineage,
    read_country_description,
    write_country_description,
)
from stablebound.schemas import SchemaError
from tests.conftest import EXAMPLELAND
from tests.india_reference import INDIA_NAME_RULES, india_names, write_india_folder
from tests.lineage_equivalence import lineage_differences

YEARS = range(1991, 2026)


@pytest.fixture(scope="module")
def india_bundled() -> Lineage:
    return Lineage("IN")


@pytest.fixture(scope="module")
def names() -> list[str]:
    return india_names()


@pytest.fixture(scope="module")
def india_folder(tmp_path_factory) -> Path:
    return write_india_folder(tmp_path_factory.mktemp("india_as_published"))


def _edit(folder: Path, dest: Path, **changes) -> Path:
    """Copy ``folder`` to ``dest`` with top-level JSON keys replaced (None deletes)."""
    shutil.copytree(folder, dest)
    p = dest / "stablebound.json"
    data = json.loads(p.read_text())
    for k, v in changes.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    p.write_text(json.dumps(data))
    return dest


# --- the acceptance test ----------------------------------------------------


def test_india_as_published_loads_equal_to_the_bundle(
    monkeypatch, tmp_path, india_bundled, india_folder, names
):
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    ln = Lineage.from_folder(india_folder)
    diffs = lineage_differences(
        india_bundled, ln, years=YEARS, names=names,
        export_dir=tmp_path, export_years=[2000, 2001, 2014, 2015], admin0="India",
    )
    assert diffs == []


def test_the_published_headers_really_are_not_canonical(india_folder):
    """Guard: the folder must exercise the mapping, not bypass it."""
    head = (india_folder / "baseline_1991.csv").read_text().splitlines()[0]
    assert head.startswith("YEAR,ADM1_ID,ADM1_NAME,ADM2_ID,ADM2_NAME")


# --- planted faults ----------------------------------------------------------


def test_a_wrong_mapping_is_refused(monkeypatch, tmp_path, india_folder):
    bad_cols = {
        "baseline": {"YEAR": "year", "ADM2_IDX": "unit_id", "ADM2_NAME": "name"},
    }
    bad = _edit(india_folder, tmp_path / "bad", columns=bad_cols)
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    with pytest.raises(SchemaError, match="ADM2_IDX|adm2_idx"):
        _ = Lineage.from_folder(bad).baseline


def test_a_missing_mapping_is_refused(monkeypatch, tmp_path, india_folder):
    bad = _edit(india_folder, tmp_path / "nomap", columns=None)
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    with pytest.raises(SchemaError):
        _ = Lineage.from_folder(bad).baseline


def test_a_wrong_fact_is_reported(monkeypatch, tmp_path, india_bundled, india_folder, names):
    desc = json.loads((india_folder / "stablebound.json").read_text())
    files = {k: v for k, v in desc["files"].items() if k != "coarse_lineage"}
    bad = _edit(india_folder, tmp_path / "lvl1", admin_level=1, files=files)
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    diffs = lineage_differences(india_bundled, Lineage.from_folder(bad), years=(1991,),
                                names=names)
    assert any(d.startswith("admin_level") for d in diffs), diffs


def test_a_missing_name_rule_is_reported(monkeypatch, tmp_path, india_bundled, india_folder,
                                         names):
    rules = {k: v for k, v in INDIA_NAME_RULES.items() if k != "strip_before_lowercase"}
    bad = _edit(india_folder, tmp_path / "rules", name_rules=rules)
    monkeypatch.delitem(data_mod.BUNDLED_COUNTRIES, "IN")
    diffs = lineage_differences(india_bundled, Lineage.from_folder(bad), names=names)
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
def test_a_broken_description_fails_at_once(tmp_path, india_folder, changes, message):
    bad = _edit(india_folder, tmp_path / "broken", **changes)
    with pytest.raises(CountryFolderError, match=message):
        read_country_description(bad)


def test_a_coarse_lineage_needs_admin_level_two(tmp_path, india_folder):
    bad = _edit(india_folder, tmp_path / "coarse", admin_level=1)
    with pytest.raises(CountryFolderError, match="coarse_lineage"):
        read_country_description(bad)


def test_columns_for_a_table_that_is_not_there(tmp_path, india_folder):
    bad = _edit(india_folder, tmp_path / "orphan",
                files={"relationship_table": "lineage_adm2.xlsx"},
                admin_level=1)
    with pytest.raises(CountryFolderError, match="files.baseline is not"):
        read_country_description(bad)


def test_no_description_file(tmp_path):
    with pytest.raises(CountryFolderError, match="no stablebound.json"):
        read_country_description(tmp_path)


def test_comment_keys_are_ignored(tmp_path, india_folder):
    folder = _edit(india_folder, tmp_path / "comments", _note="anything at all")
    assert read_country_description(folder).country_code == "IN"


def test_the_path_of_the_json_file_itself_also_works(india_folder):
    assert read_country_description(india_folder / "stablebound.json") == \
        read_country_description(india_folder)


def test_the_format_round_trips(tmp_path, india_folder):
    folder = tmp_path / "rt"
    shutil.copytree(india_folder, folder)
    desc = read_country_description(folder)
    out = write_country_description(desc, folder / "copy.json")
    assert read_country_description(out) == desc


# --- Exampleland, the documented example -----------------------------------


def test_exampleland_loads_from_its_folder_as_before(exampleland):
    names = list(exampleland.baseline["name"]) + ["Alpha North", "Alpha South"]
    diffs = lineage_differences(exampleland, Lineage.from_folder(EXAMPLELAND),
                                years=range(2010, 2021), names=names)
    assert diffs == []
