"""Snapshot construction (paper Algorithm 1) and shapefile-year inference
(paper Algorithm 2).

A *snapshot* is the set of administrative units active in a given year.
It is derived purely from the relationship table by walking territorial
events in chronological order — Split / Merge / Redistribute events remove
their parents and add their children. NameChange and Coarse events leave
the active set unchanged (those rows describe metadata, not territorial
movement).

``infer_year`` validates that a shapefile's units match the snapshot for
some year in a candidate range, returning the year with smallest mismatch.
This is useful when a shapefile's effective vintage is uncertain — GADM
and GAUL releases often lag official boundary changes by years.
"""

from __future__ import annotations

from typing import Iterable

from .lineage import LineageGraph


def build_snapshot(
    graph: LineageGraph,
    year: int,
    additional_units: set[str] | None = None,
) -> set[str]:
    """Set of unit IDs active at the start of ``year`` (paper Algorithm 1).

    **Event-year convention:** ``event_year = T`` means the event occurs
    *during* year T. The parent is alive for the entire year T snapshot;
    the child first appears in the year T+1 snapshot. This matches the
    canonical FEWS NET / census admin-snapshot convention (1991 census
    has 466 districts → 1991 snapshot = pre-event-1991 = 466 districts;
    1991-event-induced creations show up in the 1992 snapshot).

    Algorithm: start with the initial active set (units alive before any
    event), then walk territorial events with ``event_year < year`` in
    chronological order, applying parent removal and child addition for
    each.

    The ``additional_units`` parameter extends the **initial** set, but
    with a non-obvious filter: only units that **don't appear anywhere
    in the relationship table** are treated as always-alive. The reason:

    - A unit that has NO events in the RT is genuinely static — it
      existed before the timeline starts and never changed. We need to
      seed it into the active set because there's no event that would
      otherwise bring it in. (India example: 136 of 467 1991-baseline
      districts never appear in the RT at all.)

    - A unit that HAS events in the RT is governed by those events. If
      we unconditionally added it to the initial set we'd see things
      like Korea Sejong (created 2012) in the 1986 snapshot — which is
      wrong, because Sejong didn't exist at 1986.

    The filter ``additional_units - rt_units`` enforces this distinction
    without forcing the caller to know which units have events.

    Typical caller pattern (in ``StableBoundary.build_boundaries``):

        extra_units = baseline_unit_ids | modern_shapefile_unit_ids
        snapshot = build_snapshot(graph, year, additional_units=extra_units)
    """
    # Start from the districts that were already there before anything
    # happened — the ones no recorded change ever created.
    active = set(graph.initial_units())

    # The caller may know about districts the lineage doesn't, from a map or
    # a baseline list. Take those on trust, since nothing here governs when
    # they came or went. Only ones the lineage says nothing about are added:
    # if a district is caught up in a real split or merge, the replay below
    # is the authority on whether it exists, not the caller.
    if additional_units:
        territorial = graph.territorial
        territorial_unit_ids = set(territorial["parent_id"]) | set(territorial["child_id"])
        always_alive = set(additional_units) - territorial_unit_ids
        active |= always_alive

    # Now replay history up to the year being asked about. Only changes that
    # move territory count; a district being renamed is still the same
    # district and doesn't affect who exists.
    #
    # Changes dated this very year are left out. A change recorded for a
    # given year happened during it, so it is the following year that shows
    # the result — ask for 2011 and you get the districts as they stood
    # going into 2011, before that year's changes took effect.
    territorial = graph.territorial
    relevant = territorial[territorial["event_year"] < year]
    if relevant.empty:
        return active

    # Apply a whole year's changes together rather than one at a time. Within
    # a single year the old districts go and the new ones arrive at the same
    # moment, and doing them one by one could briefly remove a district that
    # a later change that same year still needs.
    for y, events_y in relevant.groupby("event_year"):
        parents_removed = set(events_y["parent_id"])
        children_added = set(events_y["child_id"])
        active -= parents_removed
        active |= children_added
    return active


def infer_year(
    shapefile_units: set[str],
    graph: LineageGraph,
    candidate_years: Iterable[int],
    additional_units: set[str] | None = None,
) -> tuple[int, dict[int, int]]:
    """Identify the year whose snapshot best matches the shapefile (paper Algorithm 2).

    For each candidate year ``y``, this computes
    ``|shapefile_units △ snapshot(y)|`` (symmetric difference: count of
    units in one set but not the other). The year with the smallest
    mismatch is the most likely vintage of the shapefile.

    Returns:
        ``(best_year, mismatches_by_year)``. The dict has one entry per
        candidate year, useful for plotting the mismatch curve and for
        tie-break inspection.

    ``additional_units`` is forwarded to :func:`build_snapshot` and should
    normally be the baseline's unit ids. Without it, every baseline unit the
    relationship table never mentions is absent from *every* candidate
    snapshot and counts as a mismatch in all of them. That does not usually
    move the argmin, but it inflates the reported counts so far that the
    mismatch curve stops being readable: on India's shipped shapefile the
    chosen year reports 240 discrepancies unseeded and 12 seeded, so 95% of
    what a caller is shown is units that were never missing.

    Tie-breaking: when multiple years tie on the smallest mismatch count,
    the **earliest** wins. The reasoning: a shapefile assembled in year
    ``t`` most likely reflects boundaries as of ``t`` or a bit earlier
    rather than later (data takes time to propagate into the shapefile
    after a boundary change).
    """
    candidates = list(candidate_years)
    if not candidates:
        raise ValueError("infer_year requires at least one candidate year.")

    mismatches: dict[int, int] = {}
    for y in candidates:
        snap = build_snapshot(graph, y, additional_units=additional_units)
        # Symmetric difference: |A △ B| = |A - B| + |B - A|. Counts every
        # unit that's in exactly one of the two sets.
        mismatches[y] = len(shapefile_units ^ snap)

    # min by (count, year): primary sort by count ascending, ties broken
    # by year ascending — earliest year wins among ties.
    best = min(mismatches, key=lambda y: (mismatches[y], y))
    return best, mismatches
