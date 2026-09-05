"""Shared fixtures.

Two jobs. First, expose the synthetic fixture registry under
``tests/fixtures/synthetic/`` so one invariant battery can be parameterized
across every tiny country in it (see ``test_qualification.py``). Second, hold
the Exampleland and graph helpers that were previously copy-pasted across
``test_conservation.py``, ``test_exampleland.py``, ``test_groups.py`` and
``test_modern_boundary.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

# Google Drive Desktop syncs this checkout and resolves conflicts by writing a
# " 2" copy of a file next to the original. Those copies are importable Python,
# so pytest collects them as real test modules and silently runs a stale second
# edition of whole files — the suite went 577 -> 767 that way, and a duplicated
# conftest would have shadowed fixtures too. Refuse to collect them, so a sync
# artifact can never be mistaken for a passing test run.
collect_ignore_glob = ["* 2.py", "*/* 2.py"]

TESTS = Path(__file__).resolve().parent
SYNTHETIC = TESTS / "fixtures" / "synthetic"
EXAMPLELAND = TESTS.parent / "examples" / "exampleland"


@dataclass(frozen=True)
class SyntheticCountry:
    """One synthetic fixture: the four user-supplied files plus expectations."""

    name: str
    path: Path

    @property
    def lineage_path(self) -> Path:
        return self.path / "lineage.csv"

    @property
    def baseline_path(self) -> Path:
        return self.path / "baseline.csv"

    @property
    def shapefile_path(self) -> Path:
        return self.path / "modern.geojson"

    @property
    def stats_path(self) -> Path:
        return self.path / "stats.csv"

    @property
    def expected(self) -> dict:
        return json.loads((self.path / "expected.json").read_text(encoding="utf-8"))

    @property
    def doc(self) -> str:
        return self.expected.get("_doc", "")

    def lineage_df(self) -> pd.DataFrame:
        return pd.read_csv(self.lineage_path)

    def baseline_df(self) -> pd.DataFrame:
        return pd.read_csv(self.baseline_path)

    def stats_df(self) -> pd.DataFrame:
        return pd.read_csv(self.stats_path)

    def graph(self):
        from stablebound.lineage import LineageGraph

        return LineageGraph.from_dataframe(self.lineage_df(), validate=False)

    def lineage(self):
        """A `Lineage` wired to this fixture's files.

        The country code is synthetic and unregistered, which is deliberate —
        it exercises the same path a new Asian country takes before it is ever
        bundled.
        """
        from stablebound import Lineage

        return Lineage(
            self.iso,
            relationship_table_path=self.lineage_path,
            baseline_path=self.baseline_path,
        )

    @property
    def iso(self) -> str:
        """Two-letter code derived from the unit ids this fixture uses."""
        return str(self.baseline_df()["unit_id"].iloc[0]).split(".")[0][:2].upper()

    def __str__(self) -> str:  # nicer pytest ids
        return self.name


def _all_synthetic() -> list[SyntheticCountry]:
    if not SYNTHETIC.exists():
        return []
    return [
        SyntheticCountry(d.name, d)
        for d in sorted(SYNTHETIC.iterdir())
        if d.is_dir() and (d / "expected.json").exists()
    ]


SYNTHETIC_COUNTRIES = _all_synthetic()


@pytest.fixture(params=SYNTHETIC_COUNTRIES, ids=lambda c: c.name)
def synthetic(request) -> SyntheticCountry:
    """Every synthetic fixture, one per test invocation."""
    return request.param


@pytest.fixture(scope="session")
def synthetic_registry() -> list[SyntheticCountry]:
    """The whole registry at once, for tests that compare across fixtures."""
    return SYNTHETIC_COUNTRIES


def get_synthetic(name: str) -> SyntheticCountry:
    """One fixture by name, for tests that target a specific hazard."""
    for c in SYNTHETIC_COUNTRIES:
        if c.name == name:
            return c
    raise KeyError(f"no synthetic fixture {name!r}; have {[c.name for c in SYNTHETIC_COUNTRIES]}")



@pytest.fixture
def bundle_synthetic(monkeypatch):
    """Register a synthetic fixture in ``BUNDLED_COUNTRIES`` for one test.

    The bundled branch of ``Lineage.__init__`` is what sets ``validity_start_year``,
    ``coverage_end_year``, ``admin_level`` and ``notes`` from the registry; none of
    it is reachable from a custom-country ``Lineage``, and India is the only real
    entry. India cannot express ``coverage_end_year > max_event_year`` (both are
    2025), so the coverage-window behaviour has no bundled subject without this.

    ``monkeypatch.setitem`` on the registry dict is picked up by ``Lineage``
    (which imports the same dict object) and reverted after the test.
    """
    import stablebound.data as data_mod
    from stablebound.data import BundledCountry

    def _register(name: str, *, code: str, coverage_end_year: int | None = None,
                  admin_level: int = 1, notes: str = "") -> BundledCountry:
        c = get_synthetic(name)
        base_year = int(c.baseline_df()["year"].min())
        entry = BundledCountry(
            country_code=code,
            lineage_path=c.lineage_path,
            baseline_path=c.baseline_path,
            name_change_log_path=None,
            validity_start_year=base_year,
            coverage_end_year=coverage_end_year,
            admin_level=admin_level,
            notes=notes or f"synthetic fixture {name!r} registered for one test",
        )
        monkeypatch.setitem(data_mod.BUNDLED_COUNTRIES, code, entry)
        return entry

    return _register


# --- Exampleland ---------------------------------------------------------


@pytest.fixture(scope="session")
def exampleland_paths() -> dict:
    return {
        "relationship_table": EXAMPLELAND / "relationship_table.csv",
        "baseline": EXAMPLELAND / "baseline.csv",
        "shapefile": EXAMPLELAND / "modern.geojson",
        "stats": EXAMPLELAND / "stats.csv",
    }


@pytest.fixture
def exampleland(exampleland_paths):
    """A fresh Exampleland `Lineage` — fresh so tests can't leak state."""
    from stablebound import Lineage

    return Lineage(
        "EX",
        relationship_table_path=exampleland_paths["relationship_table"],
        baseline_path=exampleland_paths["baseline"],
    )


# --- small helpers previously duplicated across test modules -------------


def rt(rows) -> pd.DataFrame:
    """Canonical lineage frame from tuples, for inline one-off graphs."""
    return pd.DataFrame(
        rows,
        columns=["event_year", "event_type", "parent_id", "parent_name",
                 "child_id", "child_name"],
    )


def graph_from(rows, *, validate: bool = True):
    from stablebound.lineage import LineageGraph

    return LineageGraph.from_dataframe(rt(rows), validate=validate)


@pytest.fixture
def make_graph():
    """Factory so a test can build a throwaway graph without importing helpers."""
    return graph_from
