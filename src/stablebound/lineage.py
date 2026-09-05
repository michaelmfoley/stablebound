"""Parse a relationship table into a year-keyed lineage graph.

The :class:`LineageGraph` is a thin wrapper around the validated event
DataFrame plus computed views (parent → children, child → parents,
multi-parent children, etc.). It's the central data structure passed to
every other algorithm in the package — snapshots, stable groups,
reconciliation, name history all read from it.

Two important behaviors live here:

1. ``LineageGraph.from_dataframe`` runs ``validate_lineage`` automatically
   at construction. Errors raise ``LineageDataError``; warnings/infos go
   to a logger. Researchers can opt out with ``validate=False``.

2. The "territorial" filter excludes ``NameChange`` and ``Coarse`` event
   types. These are non-territorial (they describe metadata changes, not
   territorial movement) and would otherwise pollute the parent→children
   adjacency graph that snapshot/group construction reads.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from .schemas import (
    RT_OPTIONAL_COLUMNS,
    RT_REQUIRED_COLUMNS,
    validate_relationship_table,
)

# Module-level logger — used for warning/info auto-validation results that
# shouldn't crash construction. Configure via ``logging.basicConfig`` to
# see them.
_LOGGER = logging.getLogger(__name__)

# Event types that move territory. NameChange and Coarse rows are excluded
# from this set — they describe metadata (rename, parent admin
# reassignment) without altering the unit's geometric extent. Snapshot
# walking, group construction, and reconciliation all key off this filter.
TERRITORIAL_EVENT_TYPES = ("Split", "Merge", "Redistribute")


@dataclass
class LineageGraph:
    """A year-keyed event graph parsed from a canonical relationship table.

    Attributes:
        events: DataFrame with the canonical RT columns (lowercased).
            ``event_year`` is cast to int. Original input row order is
            preserved (the ``reset_index(drop=True)`` step fixes the index
            but doesn't reshuffle).
    """

    events: pd.DataFrame

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        validate: bool = True,
    ) -> "LineageGraph":
        """Parse a canonical relationship table into a LineageGraph.

        Steps:
            1. Validate the schema (column names, event_type values).
            2. Coerce ``event_year`` to int.
            3. Drop any non-canonical columns the input might have had.
            4. (Optional, default on) Run ``validate_lineage`` and either
               raise on errors or log non-error findings.

        ``validate=True`` (the default) is the user-friendly path: errors
        in the lineage table surface as a clear ``LineageDataError`` with
        a row-by-row report. Pass ``validate=False`` only when you have a
        reason to skip — typically:

            - Adversarial unit tests that intentionally construct broken RTs.
            - Batch loads where the same RT has already been validated.
            - Pre-processing pipelines that catch errors at a different
              layer.
        """
        # Step 1: schema check (raises SchemaError for missing columns or
        # bad event_type values).
        validate_relationship_table(df)

        # Step 2: coerce types. ``astype(int)`` will fail loudly if any
        # event_year is non-integer-coercible, which is what we want.
        events = df.copy()
        events["event_year"] = events["event_year"].astype(int)

        # Step 3: project to canonical columns only. Inputs may have
        # extras (e.g. notes columns); we drop them so downstream code
        # doesn't accidentally rely on them.
        keep = [c for c in (*RT_REQUIRED_COLUMNS, *RT_OPTIONAL_COLUMNS) if c in events.columns]
        events = events[keep].reset_index(drop=True)

        graph = cls(events=events)

        # Step 4: data-quality validation. Local import to avoid a cycle
        # — validate.py imports LineageGraph from this module.
        if validate:
            from .validate import LineageDataError, format_issues, validate_lineage
            issues = validate_lineage(graph)
            errors = [i for i in issues if i.severity == "error"]
            other = [i for i in issues if i.severity != "error"]
            if errors:
                # Errors block construction. The exception message is the
                # full formatted report — researchers can paste it
                # directly into their RT review and act on it.
                raise LineageDataError(
                    "Lineage table has data errors that must be fixed:\n\n"
                    + format_issues(errors)
                )
            if other:
                # Warnings/infos don't block. One INFO line keeps the
                # signal without shouting at first-time users; the full
                # findings are accessible via Lineage.validation_report().
                # We log at INFO (not WARNING) so casual notebook use
                # stays quiet by default; researchers running with INFO
                # logging see the line; everyone can call
                # ``ln.validation_report()`` for the formatted detail.
                _LOGGER.info(
                    "Lineage loaded; %d validation finding(s). "
                    "Call ln.validation_report() for details.",
                    len(other),
                )
        return graph

    # --- Filtered views ---------------------------------------------

    @property
    def territorial(self) -> pd.DataFrame:
        """Events that move territory: Split, Merge, Redistribute.

        Used everywhere a unit's geometric extent matters — snapshot
        walking, adjacency-graph construction, group tracing.
        """
        return self.events[self.events["event_type"].isin(TERRITORIAL_EVENT_TYPES)]

    @property
    def name_changes(self) -> pd.DataFrame:
        """Events that rename a unit without territorial change."""
        return self.events[self.events["event_type"] == "NameChange"]

    @property
    def min_event_year(self) -> int | None:
        """Earliest ``event_year`` in the RT, or ``None`` if the RT is empty.

        Empty RTs are legitimate for static countries — the baseline
        carries the units and no territorial events have happened.
        Callers must handle ``None`` (typically by falling back to a
        baseline year).
        """
        if self.events.empty:
            return None
        return int(self.events["event_year"].min())

    @property
    def max_event_year(self) -> int | None:
        """Latest ``event_year`` in the RT, or ``None`` if the RT is empty."""
        if self.events.empty:
            return None
        return int(self.events["event_year"].max())

    # --- Unit-set queries -------------------------------------------

    def all_unit_ids(self) -> set[str]:
        """Every unit ID that appears anywhere in the RT, parent or child."""
        return set(self.events["parent_id"]) | set(self.events["child_id"])

    def initial_units(self) -> set[str]:
        """Units that exist before any recorded territorial event.

        Restricted to **territorial** events (Split / Merge /
        Redistribute). A unit that only appears in NameChange or Coarse
        rows is not added to the initial active set — those events are
        metadata-only and don't imply the unit was alive. Such units
        must be supplied via ``additional_units`` to
        :func:`stablebound.snapshot.build_snapshot` if they should
        appear in snapshots (typically via the baseline file).

        Rationale: India's bundled name_change_log carries 4 ADM1-level
        renames (e.g., Delhi → NCT of Delhi). Including those in
        ``initial_units`` polluted the ADM2 product's snapshots with 4
        spurious state-level IDs. Restricting to territorial events
        keeps the snapshot universe ADM-level-consistent.
        """
        ev = self.territorial
        territorial_units = set(ev["parent_id"]) | set(ev["child_id"])
        territorial_children = set(ev["child_id"])
        return territorial_units - territorial_children

    # --- Adjacency views (key inputs to snapshot/group code) ---------

    def adjacency_after(self, year: int) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Parent→children and child→parents for territorial events with
        ``event_year >= year``.

        Used by the group-building algorithm: starting from a base year,
        we want the lineage edges that take effect AFTER the base-year
        snapshot to trace forward. Under the canonical convention an
        event at ``event_year = T`` happens during T and shows up in the
        T+1 snapshot — so an event at event_year == base_year is
        post-base-year (its effect is in the base_year+1 snapshot, not
        baked into the base_year snapshot itself).

        Returns two dicts. Both have ``set`` values (not lists) because
        the lineage is sometimes a DAG, not a tree — a unit can have
        multiple parents (Redistribute) or multiple children (Split).
        """
        ev = self.territorial
        ev = ev[ev["event_year"] >= year]
        p2c: dict[str, set[str]] = defaultdict(set)
        c2p: dict[str, set[str]] = defaultdict(set)
        for parent, child in zip(ev["parent_id"], ev["child_id"]):
            p2c[parent].add(child)
            c2p[child].add(parent)
        # Cast to plain dict so callers don't accidentally insert keys.
        return dict(p2c), dict(c2p)

    def redistribute_children_after(self, year: int) -> set[str]:
        """Diagnostic: children with more than one parent in the post-``year``
        territorial graph.

        Structural identifier of "the multi-parent children" — units
        whose territory was drawn from multiple existing units. Examples:
        a redistribute child whose row is labeled ``event_type=Redistribute``,
        a Tirupati-style split where one new district is carved from
        multiple ADM2 parents (encoded as multi-row Splits sharing one
        ``child_id``), or a Merge ``A + B → C`` (C has parents A and B).

        These children force their ancestor base groups to merge into a
        single stable group via the Union-Find construction in
        ``build_stable_groups``. This method is currently kept as a
        diagnostic for researchers who want to surface "which events
        consolidated stable groups?"
        """
        ev = self.territorial
        # ``>= year`` matches ``adjacency_after``: an event at event_year
        # == base_year is post-base-year under the canonical convention.
        ev = ev[ev["event_year"] >= year]
        c2p: dict[str, set[str]] = defaultdict(set)
        for parent, child in zip(ev["parent_id"], ev["child_id"]):
            c2p[child].add(parent)
        # Multi-parent children only.
        return {c for c, parents in c2p.items() if len(parents) > 1}


def name_history(
    graph: LineageGraph,
    remap: dict[str, str],
    base_year: int,
    max_year: int,
) -> pd.DataFrame:
    """Audit table of every (stable_id, year, unit_id, name) within the window.

    For each year in ``[base_year, max_year]``, list each active unit (from
    the snapshot) along with its stable group and best-effort name as of
    that year. The intent: let researchers verify exactly which historical
    units (under which historical names) each stable group represents.
    Especially useful when a stable group bundles together units whose
    names changed mid-window.

    Output shape: long-form, one row per (stable_id, year, unit_id) triple.
    Sorted for stable display.
    """
    # Local import: snapshot.py imports lineage.py, so doing this at
    # module top would create a cycle.
    from .snapshot import build_snapshot

    rows = []
    for year in range(base_year, max_year + 1):
        active = build_snapshot(graph, year)
        for unit in active:
            rows.append(
                {
                    # Use remap[unit] when available; fall back to unit
                    # itself for units not in the remap (e.g. units alive
                    # in early years that the base-year remap doesn't
                    # cover).
                    "stable_id": remap.get(unit, unit),
                    "year": year,
                    "unit_id": unit,
                    "name": _name_of_unit_in_year(graph, unit, year),
                }
            )
    return (
        pd.DataFrame(rows, columns=["stable_id", "year", "unit_id", "name"])
        .sort_values(["stable_id", "year", "unit_id"])
        .reset_index(drop=True)
    )


def _name_of_unit_in_year(graph: LineageGraph, unit: str, year: int) -> str | None:
    """Best-effort name lookup for ``unit`` as of ``year``.

    The relationship table records names per-event, not per-year. To get
    "the name of this unit at year y", we look at the most relevant event
    row mentioning the unit and pull its name from there.

    Strategy (under the canonical event-year convention: event at year T
    happens during T, so the post-event name is effective starting at
    year T+1):

        1. Latest event with ``event_year < year`` mentioning ``unit``.
           - If as child: use ``child_name`` (the post-event / created name).
           - If as parent: use ``parent_name`` (its name at the moment of
             the event, which still applies up through ``year``).

        2. If no such past event exists: the unit pre-existed any event
           that affects it. Look ahead to the earliest event with
           ``event_year >= year`` where it appears as a parent —
           ``parent_name`` there is the unit's name as of that future
           event, and presumably also as of ``year``.

        3. Otherwise return None — no name source available.

    The function is intentionally conservative: it never invents a name,
    only reports what the RT contains.
    """
    ev = graph.events

    # Phase 1: most recent past event mentioning the unit. Strict ``<``
    # because under the canonical convention, an event at event_year == year
    # is "during year"; the post-event name doesn't take effect until year+1.
    past = ev[(ev["event_year"] < year) & ((ev["parent_id"] == unit) | (ev["child_id"] == unit))]
    if len(past) > 0:
        # idxmax() returns the index label of the first occurrence of the
        # max. Ties on year are broken by the input order, which is
        # preserved in our event table.
        idx = past["event_year"].idxmax()
        row = past.loc[idx]
        # Same row can have unit on both sides (NameChange where p_id == c_id).
        # Child side wins because it represents the post-event name.
        if row["child_id"] == unit:
            return row["child_name"]
        return row["parent_name"]

    # Phase 2: earliest future event (event_year >= year) where unit is a
    # parent — parent_name is the pre-event name, which is the name
    # effective during ``year``.
    future_as_parent = ev[(ev["event_year"] >= year) & (ev["parent_id"] == unit)]
    if len(future_as_parent) > 0:
        idx = future_as_parent["event_year"].idxmin()
        return ev.loc[idx, "parent_name"]

    # Phase 3: no name source.
    return None


def name_of_unit_in_year(graph: LineageGraph, unit: str, year: int) -> str | None:
    """Public best-effort name of ``unit`` as of ``year`` from the lineage.

    Thin wrapper over :func:`_name_of_unit_in_year` so downstream callers
    (e.g. the FEWS deliverable export) don't import a private symbol. Returns
    ``None`` when the relationship table carries no name source for the unit —
    it never invents a name. See :func:`_name_of_unit_in_year` for the exact
    latest-past-event / earliest-future-parent lookup strategy.
    """
    return _name_of_unit_in_year(graph, unit, year)


def apply_coarse_rename(
    graph: LineageGraph, coarse_id: object, name: str | None, year: int
) -> str | None:
    """Return ``name`` updated by any upper-admin rename in force at ``year``.

    A relationship table records the state name AS OF each event, and a baseline
    records it as of the start of the record. Both are correct and both are
    frozen, so without this a district reports whichever state name happened to
    be current when something last happened to it -- India's Odisha districts
    read "Orissa" in 2024, thirteen years after the rename, and Uttarakhand's
    read "Uttaranchal" seventeen years on.

    The rename rows are already in the event frame: ``merge_name_changes``
    folds the whole name-change log in, including its upper-admin rows, which
    match no district and are otherwise never read.

    Module-level rather than a closure because three separate paths need it --
    a district's own past events, the forward-looking fallback for a district
    that never changed, and the baseline fallback in the matcher's lookup.
    Fixing only one of them leaves the other two frozen, which is exactly how
    this was half-fixed the first time.
    """
    if coarse_id is None or (isinstance(coarse_id, float) and pd.isna(coarse_id)):
        return name
    ev = graph.events
    renames = ev[
        (ev["event_type"] == "NameChange")
        & (ev["child_id"] == coarse_id)
        & (ev["event_year"] <= year)
    ]
    if len(renames) == 0:
        return name
    return str(renames.loc[renames["event_year"].idxmax()]["child_name"])


def _coarse_name_of_unit_in_year(graph: LineageGraph, unit: str, year: int) -> str | None:
    """Best-effort coarse-name lookup for ``unit`` as of ``year``.

    Mirrors :func:`_name_of_unit_in_year` but pulls from
    ``child_coarse_name`` / ``parent_coarse_name`` (the optional
    upper-admin columns). Returns ``None`` if the RT doesn't carry
    coarse columns or the unit doesn't appear in any event with a
    coarse value. Used by the matcher to disambiguate homonyms via
    upper-admin (state/region) names.
    """
    ev = graph.events
    if "child_coarse_name" not in ev.columns and "parent_coarse_name" not in ev.columns:
        return None

    def _first_non_null(*candidates: object) -> str | None:
        for c in candidates:
            if c is not None and not (isinstance(c, float) and pd.isna(c)) and str(c).strip():
                return str(c)
        return None

    # Prefer the district's own past — anything that happened to it before
    # this year describes it as it already was.
    past = ev[(ev["event_year"] < year) & ((ev["parent_id"] == unit) | (ev["child_id"] == unit))]
    if len(past) > 0:
        # Walk the district's past newest-first rather than reading only the
        # single latest row. A synthetic NameChange row carries no coarse
        # columns at all, so when the most recent thing that happened to a
        # district was its own rename, reading just that row returns None and
        # the district loses its state entirely from that year on.
        for _, row in past.sort_values("event_year", ascending=False).iterrows():
            if row["child_id"] == unit:
                name = _first_non_null(row.get("child_coarse_name"),
                                       row.get("parent_coarse_name"))
                cid = row.get("child_coarse_id") if name else None
                if name is None:
                    cid = None
            else:
                name = _first_non_null(row.get("parent_coarse_name"),
                                       row.get("child_coarse_name"))
                cid = row.get("parent_coarse_id") if name else None
            if name is not None:
                return apply_coarse_rename(graph, cid, name, year)
        return None

    # Nothing in its past means the district has been there since the start
    # and simply hasn't changed yet. Its state is then whatever the first
    # future change to touch it says it was — and only changes where it is
    # the one being divided count, since a change that *creates* it would
    # belong to some other district that happens to share the identifier.
    future_as_parent = ev[(ev["event_year"] >= year) & (ev["parent_id"] == unit)]
    if len(future_as_parent) > 0:
        idx = future_as_parent["event_year"].idxmin()
        # Apply upper-admin renames here too. A district that never changed
        # reaches this branch, and those are exactly the ones whose state name
        # would otherwise stay frozen at whatever some later event happened to
        # record -- Orissa's untouched districts still read "Orissa" at 2024
        # when only the past-events branch applied the rename.
        return apply_coarse_rename(
            graph,
            ev.loc[idx].get("parent_coarse_id"),
            _first_non_null(ev.loc[idx].get("parent_coarse_name")),
            year,
        )
    # Nothing either way. Say "unknown" rather than guess — the matcher
    # treats that as a name it can't place and asks a human.
    return None
