"""Regression tests for the bundled India snapshot counts.

These tests pin down the expected admin unit counts for `Lineage("IN")` at
its baseline year and at the shipped shapefile's vintage. They catch
cross-admin-level pollution (NameChange-only ADM1 ids leaking into an ADM2
snapshot) and baseline/RT misinteractions that the integration tests might
miss. The cross-cutting checks at the end loop over every registry entry, so
they need no edit when a country is added.

If the bundled lineage genuinely changes (e.g., a few districts are added to
India's 1991 baseline), update the expected counts here with a brief note.
"""

from __future__ import annotations



# --- India ------------------------------------------------------------


def test_india_1991_snapshot_count_and_level():
    """India's 1991 baseline lists 467 ADM2 districts. The 1991
    snapshot should match that count exactly — no ADM1 pollution from
    state-level NameChange events in the bundled NCL.
    """
    from stablebound import Lineage

    ln = Lineage("IN")
    snap = ln.snapshot(year=1991)
    assert len(snap) == 467, f"expected 467 units; got {len(snap)}"
    # Every ID must be ADM2.
    bad = [u for u in snap["unit_id"] if not str(u).startswith("IN.ADM2.")]
    assert not bad, f"non-ADM2 IDs leaked into 1991 snapshot: {bad[:5]}"


def test_india_2024_snapshot_is_modern_district_count():
    """India's 2024 (post-event-2024) snapshot should be ~786 modern
    districts. Pinned with a tolerance to accommodate small bundled-RT
    updates that don't materially change the structure.
    """
    from stablebound import Lineage

    ln = Lineage("IN")
    snap = ln.snapshot(year=2024)
    # Memory pin: ~786 modern districts. Allow a small drift window so
    # bundled-RT updates don't break this test for minor edits.
    assert 770 <= len(snap) <= 800, f"unexpected 2024 snapshot size: {len(snap)}"
    bad = [u for u in snap["unit_id"] if not str(u).startswith("IN.ADM2.")]
    assert not bad, f"non-ADM2 IDs in 2024 snapshot: {bad[:5]}"


# --- Cross-cutting invariants ---------------------------------------


def test_no_sentinel_ids_in_bundled_snapshots():
    """A clean bundled-country snapshot must not contain any
    UNMATCHED_* sentinel — those are only produced when a user attaches
    a shapefile with unmatched features. The bundled paths don't
    involve user shapefiles.
    """
    from stablebound import BUNDLED_COUNTRIES, Lineage

    for code in BUNDLED_COUNTRIES:
        ln = Lineage(code)
        snap = ln.snapshot()
        offenders = [u for u in snap["unit_id"] if str(u).startswith("UNMATCHED_")]
        assert not offenders, f"{code}: sentinel IDs leaked: {offenders[:3]}"


def test_bundled_validity_start_year_consistent_with_baseline():
    """The BundledCountry.validity_start_year should equal the
    minimum year in the bundled baseline. Catches drift if the
    baseline is updated but the registry isn't.
    """
    from stablebound import BUNDLED_COUNTRIES, Lineage

    for code in BUNDLED_COUNTRIES:
        entry = BUNDLED_COUNTRIES[code]
        ln = Lineage(code)
        baseline = ln.baseline
        assert baseline is not None
        if "year" in baseline.columns:
            yrs = baseline["year"].dropna()
            if not yrs.empty:
                assert int(yrs.min()) == entry.validity_start_year, (
                    f"{code}: baseline min year {int(yrs.min())} != "
                    f"registry validity_start_year {entry.validity_start_year}"
                )
