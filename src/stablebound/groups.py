"""Stable-group construction (paper Algorithm 3) using Union-Find.

The job of this module: take a parsed lineage graph plus a base year, and
return a ``{unit_id → stable_id}`` mapping that says, for every unit (base-
year and modern alike), which stable polygon it belongs to. The dissolution
step downstream looks up each modern feature's stable_id here and unions
their geometries to produce the year's stable shapefile.

**Connected-component merge (paper Algorithm 3 as written).** The
algorithm walks each base unit's forward descendants through the
post-base-year territorial event DAG and unions every reachable unit into
the base's group. Because Union-Find is associative, a multi-parent child
that descends from multiple base units forces those base groups to merge
into a single stable group.

This is a deliberate choice: a multi-parent child means territory was
exchanged across what would otherwise be separate stable groups, so the
groups are no longer independent in any year of the analysis window.
Keeping them separate would force a downstream allocation assumption to
say where the multi-parent child's reported data belongs — which defeats
the stable boundary's "no allocation assumptions, only sums of reported
values" contract.

Worked example. Suppose A, B, D each contribute territory to a new
unit C in 2014; D also contributes to a new unit E; A and B also
contribute to a new unit F. The event DAG connects {A, B, D} via C, and
the post-2014 reorganization can no longer be decomposed into independent
sub-groups without inventing a map. The algorithm correctly merges
A, B, D, C, E, F into one stable group. Cost: the 2013 stable polygon
for that region is the union of A, B, and D rather than three separate
polygons. Benefit: data is summable across the merged group at every
year with no allocation step.

**Singleton fallback.** Every unit in ``additional_units`` (typically
modern-shapefile + baseline IDs) ends up in the remap, even if it
couldn't be unioned with any base unit. Such a unit becomes its own
stable_id (a singleton group). This guarantees geographic completeness
downstream — every modern shapefile feature gets a polygon at every
year. The tradeoff is that a post-base-year modern unit can appear as
its own polygon at pre-creation years (Korea Sejong, created 2012,
gets a polygon at 1986). See ``docs/methodology.md`` Algorithm 3 for
the full discussion.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .lineage import LineageGraph
from .snapshot import build_snapshot


class _UnionFind:
    """Standard Union-Find with path compression.

    Determinism note: ``union(x, y)`` always points the larger ID at the
    smaller one. As a consequence the final root of any group is its
    lexicographically smallest member. Two callers that build groups in
    different insertion orders will end up with the same roots — a key
    property for the canonical ``remap.json`` output that downstream stats
    aggregation must load (the paper's "never recompute the remap"
    invariant relies on this determinism across runs).
    """

    def __init__(self, items: Iterable[str]) -> None:
        # Initialize: every item is its own root.
        self.parent: dict[str, str] = {x: x for x in items}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        # Follow the chain of "belongs with" links until reaching a district
        # that points at itself. That one is the group's representative.
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Then relink everything passed along the way straight to it, so the
        # next lookup is immediate instead of walking the chain again. Long
        # chains build up quickly on a country with a lot of reorganisation.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Public contract: whichever identifier sorts first becomes the
        # group's representative. This is not a tidiness preference — it is
        # what makes the grouping reproducible. Merging in a different order
        # would otherwise give the same group a different name, and the file
        # recording those names is never rebuilt once written, so every later
        # result would disagree with the first run.
        if rx < ry:
            self.parent[ry] = rx
        else:
            self.parent[rx] = ry


def build_stable_groups(
    graph: LineageGraph,
    base_year: int,
    additional_units: set[str] | None = None,
) -> dict[str, str]:
    """Union-Find construction of stable groups (paper Algorithm 3).

    For each base-year unit, traces forward through territorial events
    occurring at or after the base year and unions every reachable
    descendant into the base unit's group. Multi-parent children are
    unioned into ALL of their parents' groups, which forces those groups
    to merge — a deliberate consequence (see module docstring).

    ``additional_units`` is forwarded to ``build_snapshot`` and also added
    to the Union-Find universe (see "Singleton fallback" in the module
    docstring). Pass the union of modern-shapefile IDs and baseline IDs.

    Returns:
        ``dict[unit_id → stable_id]``. The stable_id of a group is the
        lexicographically smallest unit ID in the group, for cross-run
        determinism.
    """
    # Step 1: figure out which units are alive at the base year. These are
    # the roots from which we trace forward. The ``additional_units`` flow
    # into the snapshot's initial set, but only those NOT in the RT (units
    # genuinely never involved in any event) — see snapshot.py for the
    # filter rationale.
    base_units = build_snapshot(graph, base_year, additional_units=additional_units)

    # Step 2: build the post-base-year adjacency graph. ``p2c`` maps each
    # parent to the set of its children in territorial events at or after
    # the base year; ``c2p`` is the inverse. Both exclude NameChange and
    # Coarse rows (those don't move territory).
    p2c, _c2p = graph.adjacency_after(base_year)

    # Step 3: follow each starting district forward through history and
    # collect everything it ever turned into — its children, their children,
    # and so on. Those all have to end up in one group, because no map drawn
    # at the start year can tell them apart.
    #
    # Taken in sorted order so the result is identical on every run.
    descendants_by_base: dict[str, set[str]] = {}
    for base in sorted(base_units):
        descendants_by_base[base] = _trace_forward(base, p2c)

    # Step 5: assemble the universe of units that need stable_ids. Three
    # sources: (a) base units, (b) descendants reached by tracing, and (c)
    # the singleton fallback — every additional_unit gets an entry even
    # if neither (a) nor (b) included it. Without (c), modern shapefile
    # features blocked by the safe rule would silently disappear from the
    # remap, leaving geographic gaps in the dissolved output.
    all_units = set(base_units)
    for desc in descendants_by_base.values():
        all_units.update(desc)
    if additional_units:
        all_units.update(additional_units)

    # Step 6: do the grouping. Every district starts in a group of its own,
    # then each starting district is joined with everything it became. A
    # district nothing ever joined it to simply stays in its own group of
    # one, which is the right answer for somewhere that never changed.
    uf = _UnionFind(sorted(all_units))
    for base, desc in descendants_by_base.items():
        for d in desc:
            uf.union(base, d)

    # Step 7: collect each group's members and pick the lex-smallest as
    # the stable_id. Building members_by_root then iterating it (rather
    # than calling find() directly inside the remap loop) keeps the output
    # stable_id deterministic — within a group, the smallest member is the
    # stable_id, regardless of which member happened to win the UF root.
    members_by_root: dict[str, set[str]] = defaultdict(set)
    for unit in all_units:
        members_by_root[uf.find(unit)].add(unit)

    remap: dict[str, str] = {}
    for root, members in members_by_root.items():
        canonical = min(members)
        for m in members:
            remap[m] = canonical
    return remap


def _trace_forward(
    base: str,
    p2c: dict[str, set[str]],
) -> set[str]:
    """DFS forward from ``base`` collecting all descendants in the
    post-base-year territorial DAG.

    Plain forward reachability: follow every ``p2c`` edge. Multi-parent
    children are reached from each of their parents independently, and
    the Union-Find at the call site collapses any base groups that share
    a common descendant — that's how cross-base merges happen.

    Implementation note: we use an explicit stack rather than recursion
    because the lineage graph can be deep on cascading splits (India has
    chains 5–6 events long), and Python's recursion limit is annoying to
    bump in a library context.
    """
    descendants: set[str] = {base}
    stack = [base]
    visited: set[str] = {base}

    while stack:
        u = stack.pop()
        for child in p2c.get(u, set()):
            if child in visited:
                continue
            visited.add(child)
            descendants.add(child)
            stack.append(child)

    return descendants
