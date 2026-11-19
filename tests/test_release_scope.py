"""The release scope: the package ships no country data (0.2.0).

A country's relationship tables are published by whoever maintains them and
loaded with ``Lineage.from_folder``; the package is the engine only. These
tests pin that: no data directory or country-specific module in the source
tree, no package-data glob that could pick a table up, no registry left
behind in the public API — and the one real country the tests use still
loads, as shipped, without warnings.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import stablebound
from tests.conftest import load_india  # India from tests/fixtures/india_as_shipped

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "stablebound"
TABLE_SUFFIXES = {".xlsx", ".xls", ".csv", ".geojson", ".shp", ".gpkg", ".parquet"}


def test_no_country_data_directory_or_module_in_the_source_tree():
    """The one assertion that catches a half-done removal or a re-added bundle."""
    assert not (SRC / "data").exists(), "src/stablebound/data/ is back"
    assert not (SRC / "india").exists(), "src/stablebound/india/ is back"
    tables = [p for p in SRC.rglob("*") if p.suffix.lower() in TABLE_SUFFIXES]
    assert tables == [], f"data files inside the package source: {tables}"


def test_package_data_ships_only_the_type_marker():
    text = (REPO / "pyproject.toml").read_text()
    block = re.search(r"\[tool\.setuptools\.package-data\]\s*stablebound = \[(.*?)\]", text, re.S)
    assert block, "package-data block not found"
    entries = re.findall(r'"([^"]+)"', block.group(1))
    assert entries == ["py.typed"], entries


def test_no_builtin_country_registry_in_the_public_api():
    for name in ("BUNDLED_COUNTRIES", "BundledCountry"):
        assert not hasattr(stablebound, name), f"stablebound.{name} is back"
        assert name not in stablebound.__all__


def test_loading_a_country_as_shipped_emits_no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_india()
    assert caught == [], [str(w.message) for w in caught]
