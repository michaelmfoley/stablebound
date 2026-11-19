"""Regression tests for India's snapshot counts (the frozen India fixture).

These tests pin down the expected admin unit counts for India — loaded as
shipped from tests/fixtures/india_as_shipped — at its baseline year and at
the shipped shapefile's vintage. They catch cross-admin-level pollution
(NameChange-only ADM1 ids leaking into an ADM2 snapshot) and baseline/RT
misinteractions that the integration tests might miss.

The file name is historical: until 0.2.0 these tables were bundled inside the
package. The fixture is frozen (see its PROVENANCE.md), so the counts here
change only if the fixture is deliberately replaced.
"""

from __future__ import annotations

from tests.conftest import load_india  # India from tests/fixtures/india_as_shipped, as shipped


# --- India ------------------------------------------------------------


def test_india_1991_snapshot_count_and_level():
    """India's 1991 baseline lists 467 ADM2 districts. The 1991
    snapshot should match that count exactly — no ADM1 pollution from
    state-level NameChange events in the name-change log.
    """
    ln = load_india()
    snap = ln.snapshot(year=1991)
    assert len(snap) == 467, f"expected 467 units; got {len(snap)}"
    # Every ID must be ADM2.
    bad = [u for u in snap["unit_id"] if not str(u).startswith("IN.ADM2.")]
    assert not bad, f"non-ADM2 IDs leaked into 1991 snapshot: {bad[:5]}"


def test_india_2024_snapshot_is_modern_district_count():
    """India's 2024 (post-event-2024) snapshot is 785 modern districts
    (recorded 18 Nov 2026 from the then-bundled copy; the fixture is the
    same tables). Kept as a window so the message stays readable."""
    ln = load_india()
    snap = ln.snapshot(year=2024)
    assert len(snap) == 785, f"unexpected 2024 snapshot size: {len(snap)}"
    bad = [u for u in snap["unit_id"] if not str(u).startswith("IN.ADM2.")]
    assert not bad, f"non-ADM2 IDs in 2024 snapshot: {bad[:5]}"


# --- Invariants on a real country ---------------------------------------


def test_no_sentinel_ids_in_a_real_snapshot():
    """A clean snapshot must not contain any UNMATCHED_* sentinel — those are
    only produced when a user attaches a shapefile with unmatched features,
    and none is attached here."""
    snap = load_india().snapshot()
    offenders = [u for u in snap["unit_id"] if str(u).startswith("UNMATCHED_")]
    assert not offenders, f"sentinel IDs leaked: {offenders[:3]}"


def test_validity_start_year_consistent_with_baseline():
    """The stated validity_start_year (India's stablebound.json) equals the
    minimum year in the baseline. Catches drift if the baseline is replaced
    but the description isn't."""
    ln = load_india()
    yrs = ln.baseline["year"].dropna()
    assert int(yrs.min()) == ln.validity_start_year == 1991
