"""The release scope: India is the only bundled country.

These tests pin the three properties that make a single-country release safe to
ship: the registry holds exactly India, every path it points at is inside the
package data directory and exists, and the wheel's package-data globs cannot
pick up anything else.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from stablebound import BUNDLED_COUNTRIES, Lineage

REPO = Path(__file__).resolve().parents[1]
PACKAGE_DATA = REPO / "src" / "stablebound" / "data"


def test_india_is_the_only_bundled_country():
    assert sorted(BUNDLED_COUNTRIES) == ["IN"]


def test_every_registry_path_is_inside_the_package_data_directory():
    for cc, entry in BUNDLED_COUNTRIES.items():
        for label in ("lineage_path", "baseline_path", "name_change_log_path",
                      "coarse_lineage_path"):
            p = getattr(entry, label)
            if p is None:
                continue
            assert PACKAGE_DATA in p.parents, f"{cc}.{label} resolved outside package data: {p}"
            assert p.exists(), f"{cc}.{label} missing: {p}"


def test_no_stray_country_directories_survive_in_package_data():
    """The one assertion that catches a half-done removal or a half-done addition."""
    dirs = sorted(p.name for p in PACKAGE_DATA.iterdir()
                  if p.is_dir() and not p.name.startswith("__"))
    assert dirs == sorted(BUNDLED_COUNTRIES), dirs


def test_package_data_globs_ship_india_only():
    text = (REPO / "pyproject.toml").read_text()
    globs = re.findall(r'"(data/[^"]+)"', text)
    assert globs, "no package-data globs found"
    assert all(g.startswith("data/IN/") for g in globs), globs


def test_loading_a_bundled_country_emits_no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Lineage("IN")
    assert caught == [], [str(w.message) for w in caught]
