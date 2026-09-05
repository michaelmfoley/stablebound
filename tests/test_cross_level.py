"""Cross-level (admin2 ↔ admin1) validation.

A two-level country carries its upper-level attribution on the lower level:
``coarse_id`` on the baseline, ``parent_coarse_id`` / ``child_coarse_id`` on
events. The FEWS deliverable keys every admin2 code on the admin1 the unit
originated in, so a disagreement between the levels ships as a district filed
under the wrong state.

The hard part of this check is **not firing on correct data**. An upper-level
lineage records events, not membership, so most of a country's upper units are
absent from it by construction — see the two regression tests below, each
pinning a false-positive class that a naive version of this check produced on
India (22 errors, then 84).
"""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound import BUNDLED_COUNTRIES, Lineage
from stablebound.lineage import LineageGraph
from stablebound.validate import coarse_universe, validate_coarse_references


@pytest.fixture(scope="module")
def india():
    return Lineage("IN")


# --- Regressions: the two false-positive classes ------------------------


def test_state_that_never_had_an_event_is_not_dangling(india):
    """A state absent from the upper lineage is normal, not an error.

    India's ADM1 lineage has 16 rows naming 21 states; India has 32. Kerala,
    Punjab and Gujarat never split, so they appear in it nowhere and are known
    only from the districts that carry them as ``coarse_id``. An earlier
    version of this check compared references against
    ``coarse_graph.all_unit_ids()`` and reported all 22 as dangling.
    """
    adm1 = india.graph(level=1)
    known = set(adm1.all_unit_ids())
    referenced = set(india.baseline["coarse_id"].dropna().astype(str))
    absent = referenced - known
    assert len(absent) > 15, "fixture drift: India should have many static states"

    issues = validate_coarse_references(india.lineage, adm1, baseline=india.baseline)
    assert [i for i in issues if i.severity == "error"] == []


def test_namechange_only_state_is_not_reported_dead(india):
    """Orissa appears in the upper lineage solely as a 2011 NameChange.

    ``build_snapshot`` seeds from ``initial_units()``, which is territorial-only,
    so a metadata-only unit is in *no* bare snapshot — India's ADM1 graph yields
    7 states for 1991 rather than 32. Passing the coarse universe as
    ``additional_units`` is what makes the temporal check meaningful; without it
    every district in Orissa was reported as filed under a non-existent state
    (84 errors).
    """
    from stablebound.snapshot import build_snapshot

    adm1 = india.graph(level=1)
    orissa = "IN.ADM1.00025"
    assert orissa not in set(build_snapshot(adm1, 1991)), "fixture drift"

    universe = coarse_universe(india.lineage, adm1, baseline=india.baseline)
    seeded = set(build_snapshot(adm1, 1991, additional_units=universe))
    assert orissa in seeded
    assert len(seeded) == 32, "1991 India should have 32 states/UTs"


def test_bundled_countries_are_clean():
    """No bundled country may ship with a cross-level error."""
    for cc in BUNDLED_COUNTRIES:
        errors = [i for i in Lineage(cc).validate_levels() if i.severity == "error"]
        assert errors == [], f"{cc}: {[i.message for i in errors]}"


# --- The check must actually bite --------------------------------------


def _india_events_with(india, index_filter, column, value):
    ev = india.lineage.events.copy()
    ev.loc[index_filter(ev), column] = value
    return LineageGraph(ev)


def test_district_filed_under_a_state_that_has_ceased(india):
    """Bihar ceased in 2000; a 2006 event may not attribute a child to it."""
    graph = _india_events_with(
        india,
        lambda ev: ev[ev["event_year"] > 2005].index[0],
        "child_coarse_id",
        "IN.ADM1.00005",
    )
    issues = validate_coarse_references(
        graph, india.graph(level=1), baseline=india.baseline
    )
    hits = [i for i in issues if i.category == "coarse_reference_not_alive"]
    assert len(hits) == 1
    assert "IN.ADM1.00005" in hits[0].ids


def test_district_filed_under_a_state_that_does_not_exist_yet(india):
    """Telangana was created in 2014; a 1991 event may not reference it."""
    graph = _india_events_with(
        india,
        lambda ev: ev[ev["event_year"] < 2000].index[0],
        "child_coarse_id",
        "IN.ADM1.00039",
    )
    issues = validate_coarse_references(
        graph, india.graph(level=1), baseline=india.baseline
    )
    hits = [i for i in issues if i.category == "coarse_reference_not_alive"]
    assert len(hits) == 1
    assert "IN.ADM1.00039" in hits[0].ids


def test_baseline_unit_with_no_coarse_id_is_an_error(india):
    baseline = india.baseline.copy()
    baseline.loc[baseline.index[0], "coarse_id"] = None
    issues = validate_coarse_references(
        india.lineage, india.graph(level=1), baseline=baseline
    )
    hits = [i for i in issues if i.category == "baseline_unit_without_coarse_id"]
    assert len(hits) == 1
    assert hits[0].severity == "error"


def test_baseline_without_a_coarse_column_at_all_is_an_error(india):
    baseline = india.baseline.drop(columns=["coarse_id"])
    issues = validate_coarse_references(
        india.lineage, india.graph(level=1), baseline=baseline
    )
    assert any(i.category == "baseline_has_no_coarse_attribution" for i in issues)


def test_typo_is_only_dangling_when_a_coarse_baseline_makes_it_provable(india):
    """Without an upper baseline there is no independent evidence of a typo.

    The lower level's own references are the only other source for upper units,
    so checking one against the other is circular. Supplying a real coarse
    baseline is what turns this into a sound check — so the same typo is
    reported only in the second case.
    """
    adm1 = india.graph(level=1)
    graph = _india_events_with(
        india, lambda ev: ev.index[0], "child_coarse_id", "IN.ADM1.99999"
    )

    lenient = validate_coarse_references(graph, adm1, baseline=india.baseline)
    assert not any(i.category == "dangling_coarse_reference" for i in lenient)

    coarse_baseline = pd.DataFrame(
        {"unit_id": sorted(coarse_universe(india.lineage, adm1, baseline=india.baseline)
                           - {"IN.ADM1.99999"})}
    )
    strict = validate_coarse_references(
        graph, adm1, baseline=india.baseline, coarse_baseline=coarse_baseline
    )
    hits = [i for i in strict if i.category == "dangling_coarse_reference"]
    assert len(hits) == 1
    assert hits[0].ids == ["IN.ADM1.99999"]


def test_no_coarse_attribution_anywhere_warns(india):
    """A lineage with no coarse columns can't be cross-checked; say so."""
    ev = india.lineage.events.drop(
        columns=["parent_coarse_id", "child_coarse_id"], errors="ignore"
    )
    issues = validate_coarse_references(LineageGraph(ev), india.graph(level=1))
    assert any(i.category == "no_coarse_references" for i in issues)


# --- The universe helper ------------------------------------------------


def test_coarse_universe_unions_both_levels(india):
    adm1 = india.graph(level=1)
    universe = coarse_universe(india.lineage, adm1, baseline=india.baseline)
    assert set(adm1.all_unit_ids()) <= universe
    assert set(india.baseline["coarse_id"].dropna().astype(str)) <= universe


def test_export_fews_refuses_on_a_cross_level_error(india, tmp_path, monkeypatch):
    """A misfiled district must block the deliverable, not ship silently."""
    from stablebound.validate import LineageDataError, LineageIssue

    monkeypatch.setattr(
        type(india),
        "validate_levels",
        lambda self: [
            LineageIssue(
                severity="error",
                category="coarse_reference_not_alive",
                message="synthetic",
                detail="synthetic",
            )
        ],
    )
    with pytest.raises(LineageDataError, match="cross-level"):
        india.export_fews(tmp_path, years=range(2015, 2017), admin0="India")
