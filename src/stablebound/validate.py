"""Lineage data-quality validator.

Walks a parsed :class:`LineageGraph` looking for structural issues that
would produce wrong or misleading stable boundaries downstream — duplicate
rows, ambiguous IDs, encoding mismatches, and so on.

Returns a list of :class:`LineageIssue` records with enough context that a
researcher can locate and fix the offending rows in their relationship
table.

The validator runs **automatically** when ``LineageGraph.from_dataframe()``
is called: errors raise :class:`LineageDataError` (blocking construction),
warnings and infos are logged via the ``stablebound.lineage`` logger.
Researchers can opt out with ``validate=False`` for unit tests or
pre-validated batch loads.

To inspect the full report manually::

    from stablebound.io import read_relationship_table
    from stablebound.lineage import LineageGraph
    from stablebound.validate import validate_lineage, format_issues

    rt = read_relationship_table("ADM2_LINEAGE.xlsx")
    graph = LineageGraph.from_dataframe(rt, validate=False)  # skip auto-check
    issues = validate_lineage(graph)
    print(format_issues(issues))

The validator does NOT modify the graph. It only inspects.

Each check function below corresponds to one row in the docstring's
finding categories table. Adding a new check: write a
``_check_<category>(graph) -> list[LineageIssue]`` function and register
it in ``validate_lineage``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .lineage import LineageGraph

Severity = str  # "error" | "warning" | "info"


class LineageDataError(ValueError):
    """Raised when a lineage table has data errors that must be fixed.

    The exception message is the formatted report from
    :func:`format_issues`; researchers can paste it directly into the
    relationship table review.
    """


@dataclass
class LineageIssue:
    """A single data-quality finding.

    Attributes:
        severity: ``"error"`` for clear data bugs, ``"warning"`` for
            structurally suspicious rows, ``"info"`` for semantic notes.
        category: short slug for grouping (e.g. ``"duplicate_row"``).
        message: one-line human-readable summary.
        detail: longer explanation including suggested fix.
        rows: pandas index labels of the offending event rows in
            ``graph.events``. Use ``graph.events.loc[issue.rows]`` to view.
        ids: unit IDs implicated. Useful for cross-referencing with the
            shapefile / stats.
    """

    severity: Severity
    category: str
    message: str
    detail: str
    rows: list[int] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)


def validate_lineage(graph: LineageGraph) -> list[LineageIssue]:
    """Run all checks on a lineage graph; return issues in execution order.

    Eight checks, registered in roughly increasing severity (errors first
    so they show up at the top of the formatted report):

        1. ``_check_duplicate_rows`` — error
        2. ``_check_self_referential_territorial`` — error
        3. ``_check_resurrected_units`` — error
        4. ``_check_namechange_with_distinct_ids`` — warning
        5. ``_check_multi_parent_splits`` — info
        6. ``_check_split_appears_to_be_rename`` — warning
        7. ``_check_orphan_post_event_units`` — info
        8. ``_check_namechange_unit_name_mismatch`` — warning

    Each returns a list (possibly empty) of LineageIssue records. We
    aggregate them all into one list rather than fail-fast — the caller
    typically wants to see every issue at once, not fix one and re-run.
    """
    # The checks are independent — none reads another's findings — so the
    # order here is purely about what a user should see first. Problems that
    # corrupt results come before ones that are merely suspicious.
    issues: list[LineageIssue] = []

    # The same change written down twice, which would be applied twice.
    issues.extend(_check_duplicate_rows(graph))
    # A split or merge where the district turns into itself. Nearly always a
    # rename that was filed under the wrong kind of change.
    issues.extend(_check_self_referential_territorial(graph))
    # A district that comes back after being dissolved.
    issues.extend(_check_resurrected_units(graph))
    # A rename that also swaps the district's identifier, which makes it a
    # different district as far as everything downstream is concerned.
    issues.extend(_check_namechange_with_distinct_ids(graph))
    # A new district drawing territory from several parents. Legitimate, but
    # also what a data-entry slip looks like, so it is worth surfacing.
    issues.extend(_check_multi_parent_splits(graph))
    # A "split" of one district into one district with the same name — a
    # rename in disguise. Same mistake as the second check, spotted by
    # matching names rather than matching identifiers.
    issues.extend(_check_split_appears_to_be_rename(graph))
    # A district that appears out of nowhere, with no change that created it.
    issues.extend(_check_orphan_post_event_units(graph))
    # The rename list and the lineage disagree about the district's old name.
    issues.extend(_check_namechange_unit_name_mismatch(graph))
    return issues


# --- Individual checks ---------------------------------------------------


def _check_resurrected_units(graph: LineageGraph) -> list[LineageIssue]:
    """A unit id that is created and consumed in the same year, or reused later.

    A unit id denotes a specific territorial extent, so once a territorial
    event consumes it the id must never come back. Two distinct pathologies
    violate that, both silent, and they are reported separately because the fix
    differs.

    **1. Transient unit (same year).** A unit is created and consumed within one
    event year — e.g. a 1983 Split creates it and a 1983 Redistribute consumes
    it. ``build_snapshot`` processes each year atomically
    (``active -= parents; active |= children``), which is correct for a unit
    that genuinely continues through a same-year event, but here it re-adds a
    unit that was only ever a stepping stone. The result is a **phantom alive
    forever alongside its own successors, double-counting their territory**.
    India hit exactly this with the Assam 2022-2023 transient districts and
    fixed it at source. The fix is to collapse the chain: emit the net event,
    not the intermediate.

    **2. Resurrection (later year).** ``A`` splits to ``B`` in 2010, then ``B``
    splits back to ``A`` in 2012 — directly, or around a longer loop. Usually a
    mistyped id. It loses data twice over: ``build_snapshot`` excludes ``A``
    from the always-alive seed (because it is now a territorial *child*), so
    ``A`` **vanishes from its own baseline year**; and ``build_stable_groups``
    can omit units on the loop entirely, taking their statistics with them.

    Neither raises. Both return a plausible answer of the wrong size.
    """
    terr = graph.territorial
    if terr.empty:
        return []

    issues: list[LineageIssue] = []

    # Problem one: a district that is created and dissolved in the same year,
    # so it never stands on its own in any year's picture. Find it by looking
    # for a district appearing on both sides of the same year's changes.
    for year, rows in terr.groupby("event_year"):
        both = set(rows["parent_id"].astype(str)) & set(rows["child_id"].astype(str))
        for uid in sorted(both):
            # A district turning into itself is a different mistake, caught
            # by its own check further down.
            if ((rows["parent_id"].astype(str) == uid)
                    & (rows["child_id"].astype(str) == uid)).any():
                continue
            successors = sorted(
                set(rows[rows["parent_id"].astype(str) == uid]["child_id"].astype(str))
            )
            issues.append(
                LineageIssue(
                    severity="error",
                    category="transient_unit",
                    message=(
                        f"unit {uid!r} is both created and consumed in {int(year)}, "
                        f"so it never exists in any snapshot on its own"
                    ),
                    detail=(
                        "Per-year snapshot updates are atomic, so this id is "
                        "removed and re-added in the same step and stays alive "
                        f"indefinitely alongside its successors {successors} — "
                        "double-counting their territory in every later year. "
                        "Collapse the chain to the net event rather than emitting "
                        "the intermediate unit (India's Assam 2022-2023 districts "
                        "were fixed this way at source)."
                    ),
                    ids=[uid, *successors],
                )
            )

    # Problem two: a district dissolved in one year and created again in a
    # later one. Note when each district was first dissolved...
    consumed_at: dict[str, int] = {}
    for _, row in terr.iterrows():
        p, y = str(row["parent_id"]), int(row["event_year"])
        if p not in consumed_at or y < consumed_at[p]:
            consumed_at[p] = y

    # ...then look for any that are brought back afterwards. Almost always a
    # mistyped identifier rather than a district genuinely being recreated.
    for idx, row in terr.iterrows():
        child, year = str(row["child_id"]), int(row["event_year"])
        if child == str(row["parent_id"]):
            continue
        prior = consumed_at.get(child)
        if prior is not None and prior < year:
            issues.append(
                LineageIssue(
                    severity="error",
                    category="resurrected_unit",
                    message=(
                        f"unit {child!r} was consumed by a territorial event in "
                        f"{prior} but is re-created as a child in {year}"
                    ),
                    detail=(
                        "A retired id must not reappear — normally a mistyped id. "
                        "The unit is dropped from the always-alive seed, so it "
                        "vanishes from snapshots before its first event, and units "
                        "on the loop can be left out of the stable grouping "
                        "entirely. Give the re-created unit a new id."
                    ),
                    rows=[idx],
                    ids=[child],
                )
            )
    return issues


def _check_duplicate_rows(graph: LineageGraph) -> list[LineageIssue]:
    """Exact-duplicate rows on (event_year, event_type, parent_id, child_id).

    These never represent valid lineage data — they're spreadsheet
    copy-paste errors that double-count an event in the adjacency dicts.
    India's RT had three of these (Baksa 2022 splits and a Jaipur 2024
    merge), each appearing as two identical rows.

    ``keep=False`` flags ALL members of a duplicate group (not just the
    second-and-later occurrence), so the user can see every duplicated
    row in the report.
    """
    ev = graph.events
    keys = ["event_year", "event_type", "parent_id", "child_id"]
    # ``duplicated(keep=False)`` marks every row that participates in a
    # duplicate group. ``keep="first"`` would mark only the 2nd, 3rd, …
    # occurrences — but for the report we want all of them visible.
    dup_mask = ev.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return []
    dup_rows = ev[dup_mask]
    issues: list[LineageIssue] = []
    # One issue per distinct duplicate group, not per duplicated row.
    for key, group in dup_rows.groupby(keys):
        year, etype, pid, cid = key
        issues.append(
            LineageIssue(
                severity="error",
                category="duplicate_row",
                message=(
                    f"{etype} {year}: {pid} → {cid} appears {len(group)} times"
                ),
                detail=(
                    f"The same event appears as {len(group)} identical rows in "
                    f"the relationship table (rows {sorted(group.index.tolist())}). "
                    f"This double-counts the event in adjacency dicts and may "
                    f"produce wrong stable groups. Fix: remove the extra "
                    f"{len(group) - 1} row(s) so the event appears exactly once."
                ),
                rows=sorted(group.index.tolist()),
                ids=[str(pid), str(cid)],
            )
        )
    return issues


def _check_self_referential_territorial(graph: LineageGraph) -> list[LineageIssue]:
    """parent_id == child_id with a territorial event_type.

    Split / Merge / Redistribute imply a territorial change, which by
    definition produces a new ID. A row with parent_id == child_id is
    almost certainly a rename mistakenly tagged territorial. NameChange
    (rename only) or Coarse (parent admin reassignment only) are the
    correct event types for these cases.

    No India row trips this check — the bundled lineage is clean on this
    axis. Included anyway because it's a common copy-paste error
    when researchers convert between schemas.
    """
    # Look for changes that move territory but name the same district on
    # both sides — a district turning into itself, which cannot happen.
    ev = graph.events
    mask = (
        ev["event_type"].isin(["Split", "Merge", "Redistribute"])
        & (ev["parent_id"] == ev["child_id"])
    )
    if not mask.any():
        return []
    # One report per offending record, so the message can point at the exact
    # row in the source file the user has to go and fix.
    bad = ev[mask]
    issues: list[LineageIssue] = []
    for idx, row in bad.iterrows():
        issues.append(
            LineageIssue(
                severity="error",
                category="self_referential_territorial",
                message=(
                    f"row {idx}: {row['event_type']} with parent_id == child_id "
                    f"({row['parent_id']})"
                ),
                detail=(
                    f"Row {idx} is a {row['event_type']} where parent and child "
                    f"are the same unit. Territorial events should change ID. "
                    f"If only the name changed, retag as NameChange. "
                    f"If only the parent admin level changed, retag as Coarse."
                ),
                rows=[idx],
                ids=[str(row["parent_id"])],
            )
        )
    return issues


def _check_namechange_with_distinct_ids(graph: LineageGraph) -> list[LineageIssue]:
    """NameChange rows where parent_id != child_id.

    The canonical schema's NameChange semantics is "same unit, new
    name" — both IDs should match. A row with distinct IDs implies the
    unit's ID changed, which is structurally a vintage successor (FEWS
    NET-style ``KR1982A105 → KR1986A105``).

    Flagged as warning, not error, because the algorithm doesn't crash
    — it just treats both IDs as separate units in the snapshot. If the
    intent was "same unit, vintage successor", the user should collapse
    to a single ID across the lineage. If the unit truly continues to
    exist under a new ID, the row is harmless.
    """
    # A rename should leave the district's identifier alone. Look for ones
    # that change it, which quietly turns a rename into two districts.
    ev = graph.events
    mask = (ev["event_type"] == "NameChange") & (ev["parent_id"] != ev["child_id"])
    if not mask.any():
        return []
    bad = ev[mask]
    issues: list[LineageIssue] = []
    for idx, row in bad.iterrows():
        issues.append(
            LineageIssue(
                severity="warning",
                category="namechange_distinct_ids",
                message=(
                    f"row {idx}: NameChange with distinct IDs "
                    f"({row['parent_id']} → {row['child_id']})"
                ),
                detail=(
                    f"Row {idx} is a NameChange but parent and child IDs differ. "
                    f"NameChange events are treated as non-territorial; the "
                    f"snapshot algorithm will keep both IDs alive. If the "
                    f"intent is 'same unit, new ID' (vintage successor), "
                    f"consider collapsing to a single ID across the lineage. "
                    f"If the unit truly continues to exist under a new ID, "
                    f"this row is harmless."
                ),
                rows=[idx],
                ids=[str(row["parent_id"]), str(row["child_id"])],
            )
        )
    return issues


def _check_multi_parent_splits(graph: LineageGraph) -> list[LineageIssue]:
    """Split events where the same child has multiple distinct parents.

    Structurally these are redistributes — territory from several
    parents combines into one new unit. India encodes a lot of them
    this way (43 cases): the 1991 Bokaro creation, the 1997 multi-parent
    UP/Uttarakhand splits, the 2022 AP reorganization, the 2023 Rajasthan
    reorganization.

    These are not defects. Group construction reads the edge set, not the
    event_type label, so a multi-parent child is handled the same however
    it is tagged. What it does do is force every one of its parents into
    the same stable group, which costs resolution — so each finding marks
    a place where the stable product is coarser because of a real
    administrative change rather than a modelling choice. Info severity,
    surfaced so that cost stays visible and locatable.
    """
    splits = graph.events[graph.events["event_type"] == "Split"]
    # Count distinct parents per child. A regular Split has 1 parent per
    # child (the parent_id is the same across all rows for that event);
    # multi-parent Splits have a child_id appearing under multiple
    # distinct parent_ids.
    grouped = splits.groupby("child_id")["parent_id"].nunique()
    multi = grouped[grouped > 1]
    if len(multi) == 0:
        return []
    issues: list[LineageIssue] = []
    for child_id in multi.index:
        rows = splits[splits["child_id"] == child_id]
        parents = sorted(rows["parent_id"].unique())
        years = sorted(rows["event_year"].unique())
        issues.append(
            LineageIssue(
                severity="info",
                category="multi_parent_split",
                message=(
                    f"child {child_id} has {len(parents)} parents in Split events"
                ),
                detail=(
                    f"Child {child_id!r} appears as a Split child of "
                    f"{len(parents)} distinct parents ({', '.join(parents[:5])}"
                    f"{'...' if len(parents) > 5 else ''}) in years {years}. "
                    f"This is structurally a Redistribute, and group "
                    f"construction treats it as one either way. Every parent "
                    f"listed here lands in the same stable group, so this is a "
                    f"place where the stable product loses resolution. Consider "
                    f"retagging event_type=Redistribute on these rows for clarity."
                ),
                rows=sorted(rows.index.tolist()),
                ids=[str(child_id), *parents],
            )
        )
    return issues


def _check_split_appears_to_be_rename(graph: LineageGraph) -> list[LineageIssue]:
    """1-to-1 Splits where parent_name == child_name.

    A Split that produces exactly one child with the same name as the
    parent is suspicious — either:
        (a) The unit really split into multiple children but the sibling
            row(s) are missing from the RT (the algorithm only sees the
            same-named successor, treating the event as a no-op); or
        (b) The row is actually a vintage rename, NOT a real split, and
            should be encoded as NameChange.

    Korea's 2012 Chungcheongbuk-Do (00002 → 00020) and Chungcheongnam-Do
    (00016 → 00021) are case (b): the parent kept its territory and got
    a new ID; the actual territorial change (Sejong's creation) is in
    separate Redistribute rows.

    Warning severity because either fix is the user's call — we can't
    distinguish (a) from (b) from the RT alone.
    """
    ev = graph.events
    splits = ev[ev["event_type"] == "Split"]
    # Group by (event_year, parent_id) — a single-child split is what
    # we're looking for. A multi-child split with same names would
    # indicate something different (and is rare in practice).
    grouped = splits.groupby(["event_year", "parent_id"])
    issues: list[LineageIssue] = []
    for (year, pid), group in grouped:
        if len(group) != 1:
            continue  # Multi-child split — proper territorial event.
        row = group.iloc[0]
        if row["parent_name"] != row["child_name"]:
            continue  # Different names — proper rename of carved-off child.
        if row["parent_id"] == row["child_id"]:
            continue  # Caught by _check_self_referential_territorial.
        issues.append(
            LineageIssue(
                severity="warning",
                category="split_looks_like_rename",
                message=(
                    f"row {row.name}: Split {year} with single child of same name "
                    f"({row['parent_name']!r}; {row['parent_id']} → {row['child_id']})"
                ),
                detail=(
                    f"Row {row.name} is a Split with exactly one child whose "
                    f"name matches the parent. If the unit really split into "
                    f"multiple children, the sibling row(s) are missing — add "
                    f"them so the algorithm can apportion territory. If the "
                    f"intent was a vintage rename (same territory, new ID), "
                    f"retag as NameChange (and align IDs if appropriate)."
                ),
                rows=[row.name],
                ids=[str(row["parent_id"]), str(row["child_id"])],
            )
        )
    return issues


def _check_orphan_post_event_units(graph: LineageGraph) -> list[LineageIssue]:
    """Units that only appear as territorial children with no further events.

    These are the "leaves" of the lineage DAG — units that were created
    by an event but never themselves split, merged, or redistributed.
    Most modern units fall in this category; the count is reported as a
    sanity-check against the modern shapefile.

    If the leaf count is much different from the modern shapefile's
    feature count, something's off (missing modern features, or
    intermediate units mistakenly tagged as leaves).
    """
    # A district at the end of the line is one that was created by some
    # change but never went on to be split or merged itself. Find them by
    # taking every district ever created and removing any that later changed.
    # These should be roughly the districts on a present-day map, which is
    # what makes the count worth reporting.
    ev = graph.events
    territorial = ev[ev["event_type"].isin(["Split", "Merge", "Redistribute"])]
    children = set(territorial["child_id"])
    parents = set(territorial["parent_id"])
    leaves = children - parents
    if not leaves:
        return []
    return [
        LineageIssue(
            severity="info",
            category="leaf_count",
            message=f"{len(leaves)} units appear only as territorial children (modern leaves)",
            detail=(
                f"{len(leaves)} unit IDs appear as the child of a territorial "
                f"event but never as the parent of one — these are modern-leaf "
                f"units. Sanity-check that the modern shapefile contains "
                f"features for these IDs (or for IDs that map to them)."
            ),
            rows=[],
            ids=sorted(leaves),
        )
    ]


# --- Stats × lineage consistency checks ---------------------------------
#
# These run on the stats DataFrame (long-form) against a parsed
# LineageGraph plus the modern shapefile's unit IDs. They surface the
# kinds of upstream-data-prep bugs that produce silently-wrong outputs:
#
#   - unit_ids that don't exist anywhere in the universe
#   - unit_ids that ceased per a territorial event but keep reporting
#     under the old code (causes data to land on wrong stable groups
#     via the late-reporting / cascade path)
#   - malformed sentinel values like "__FILTER__"
#
# Each check returns a list of LineageIssue. They aggregate into the same
# report format as the lineage checks. Researchers see all data-prep
# issues in one place.


def validate_stats_lineage_consistency(
    stats_df: pd.DataFrame,
    graph: LineageGraph,
    modern_unit_ids: Iterable[str] | None = None,
) -> list[LineageIssue]:
    """Cross-check stats unit_ids against the lineage and modern shapefile.

    Args:
        stats_df: long-form stats with the canonical schema.
        graph: parsed LineageGraph.
        modern_unit_ids: IDs in today's shapefile. If omitted, only the
            in-lineage check runs (units not in lineage are flagged as
            warnings without the modern fallback).

    Returns:
        list of LineageIssue. Three checks:

          1. ``stats_unit_id_malformed`` (error) — sentinel values or
             clearly invalid IDs.
          2. ``stats_unit_id_unknown`` (warning) — unit_ids that aren't
             in the lineage AND aren't in the modern shapefile.
          3. ``stats_post_cease_reporting`` (warning) — unit_ids that
             ceased per a territorial Split/Merge/Redistribute but
             still report stats for later years.
    """
    issues: list[LineageIssue] = []
    issues.extend(_check_stats_malformed_unit_ids(stats_df))
    issues.extend(_check_stats_unit_id_unknown(stats_df, graph, modern_unit_ids))
    issues.extend(_check_stats_post_cease_reporting(stats_df, graph))
    return issues


def _check_stats_malformed_unit_ids(stats_df: pd.DataFrame) -> list[LineageIssue]:
    """unit_ids that look like sentinel/control values, not real IDs.

    Catches things like ``"__FILTER__"`` (Delhi_Total control rows in
    the India source CSV), empty strings, and NaN. These rows have to
    be dropped or fixed at the data-prep layer before the package can
    do anything meaningful with them.
    """
    if stats_df.empty:
        return []
    # Three ways an identifier can be unusable: missing altogether, empty,
    # or a placeholder the source used to mark a row that isn't really a
    # district at all — country totals often arrive looking like this.
    uids = stats_df["unit_id"].astype(str)
    is_nan = stats_df["unit_id"].isna()
    is_blank = uids.str.strip() == ""
    is_sentinel = uids.str.startswith("__") & uids.str.endswith("__")
    bad = is_nan | is_blank | is_sentinel
    if not bad.any():
        return []
    # A missing identifier reads as the word "nan" once it is treated as
    # text, so list it once, under a label that reads as missing.
    bad_ids = sorted(set(uids[bad].unique()) - {"nan"})
    if is_nan.any() and "<NaN>" not in bad_ids:
        bad_ids.append("<NaN>")
    n_rows = int(bad.sum())
    return [
        LineageIssue(
            severity="error",
            category="stats_unit_id_malformed",
            message=f"{n_rows} stats rows have malformed unit_ids ({len(bad_ids)} distinct)",
            detail=(
                f"unit_id values that look like sentinel/control entries or "
                f"are missing entirely: {bad_ids[:8]}"
                f"{'...' if len(bad_ids) > 8 else ''}. "
                f"Drop these rows or assign canonical IDs upstream before "
                f"loading. Common cause: total/aggregate rows that snuck "
                f"into the stats CSV (e.g., 'Delhi_Total' coded as '__FILTER__')."
            ),
            rows=[],
            ids=bad_ids[:20],
        )
    ]


def _check_stats_unit_id_unknown(
    stats_df: pd.DataFrame,
    graph: LineageGraph,
    modern_unit_ids: Iterable[str] | None,
) -> list[LineageIssue]:
    """unit_ids that exist nowhere — not in lineage, not in modern shapefile.

    These rows are unallocatable. The package routes them to the
    late-reporting bucket on the stable side and to ``late_reporting.csv``
    on the modern side, but in both cases the data is effectively lost.
    Almost always indicates an upstream FNID assignment bug.
    """
    if stats_df.empty:
        return []
    # Set aside the unusable identifiers first; the check above already
    # reports those, and repeating them here would just be noise.
    uids = stats_df["unit_id"].dropna().astype(str)
    uids = uids[~(uids.str.startswith("__") & uids.str.endswith("__"))]
    uids = uids[uids.str.strip() != ""]

    # A district counts as known if it appears anywhere in the country's
    # history, or on the present-day map. Anything else has nowhere to go.
    rt_units = graph.all_unit_ids()
    modern = set(modern_unit_ids) if modern_unit_ids else set()
    known = rt_units | modern

    unknown_mask = ~uids.isin(known)
    unknown_ids = sorted(uids[unknown_mask].unique())
    if not unknown_ids:
        return []
    n_rows = int(unknown_mask.sum())
    return [
        LineageIssue(
            severity="warning",
            category="stats_unit_id_unknown",
            message=(
                f"{n_rows} stats rows reference {len(unknown_ids)} unit_id(s) "
                f"not in the lineage or modern shapefile"
            ),
            detail=(
                f"These unit_ids appear in stats but nowhere else in the "
                f"input universe. Their data has no destination in either "
                f"product. Common cause: FNID-assignment script is using "
                f"obsolete or post-event codes that the lineage hasn't been "
                f"updated to track. Sample IDs: "
                f"{unknown_ids[:6]}{'...' if len(unknown_ids) > 6 else ''}."
            ),
            rows=[],
            ids=unknown_ids[:20],
        )
    ]


def _check_stats_post_cease_reporting(
    stats_df: pd.DataFrame, graph: LineageGraph
) -> list[LineageIssue]:
    """unit_ids that report stats AFTER they ceased per a territorial event.

    A unit that's a parent of a Split/Merge/Redistribute ceases at that
    event year. Stats reported under that unit's ID for any later year
    are stale — they almost certainly belong to the unit's successor(s).
    The package handles them via late-reporting redistribution, but the
    redistribution can be wildly off if the data is voluminous (e.g.,
    Giridih (JH): 324 rows over 1997-2022 filed under a 1991-defunct
    code; the cascade redistributes most of it to Bokaro because Bokaro
    is the only post-1991 reporter for that variable).

    Reports as one issue per unit_id with a stale-row count and the
    cease year.
    """
    if stats_df.empty:
        return []
    terr = graph.territorial
    if terr.empty:
        return []

    # Work out when each district stopped existing. A district can be broken
    # up more than once in the records, so its real end is the first time it
    # happened — everything after that is describing a district already gone.
    cease_year: dict[str, int] = {}
    for _, row in terr.iterrows():
        uid = str(row["parent_id"])
        yr = int(row["event_year"])
        cease_year[uid] = min(cease_year.get(uid, yr), yr)

    # Narrow the statistics to just those districts before looking at years.
    # Checking each dissolved district against the full table one at a time
    # is the obvious way to write this and is far too slow to use — on a
    # country the size of India it does hundreds of millions of comparisons.
    ceased_ids = set(cease_year.keys())
    candidates = stats_df[stats_df["unit_id"].astype(str).isin(ceased_ids)].copy()
    if candidates.empty:
        return []
    candidates["year"] = candidates["year"].astype(int)
    candidates["unit_id"] = candidates["unit_id"].astype(str)

    # For each of those districts, collect anything filed for a year after it
    # ceased, and note how much there is and the span it covers.
    findings = []
    for uid, group in candidates.groupby("unit_id", sort=False):
        last_yr = cease_year.get(uid)
        if last_yr is None:
            continue
        bad = group[group["year"] > last_yr]
        if len(bad):
            findings.append({
                "unit_id": uid,
                "ceased_year": last_yr,
                "n_stale_rows": int(len(bad)),
                "first_stale_year": int(bad["year"].min()),
                "last_stale_year": int(bad["year"].max()),
            })
    if not findings:
        return []

    # Worst first, and only the top handful spelled out. A country with a
    # messy source can trip this on dozens of districts, and a report that
    # lists them all is one nobody reads.
    findings.sort(key=lambda r: -r["n_stale_rows"])
    summary_lines = [
        f"  {f['unit_id']}: {f['n_stale_rows']} rows in years "
        f"{f['first_stale_year']}-{f['last_stale_year']} "
        f"(unit ceased {f['ceased_year']})"
        for f in findings[:15]
    ]
    if len(findings) > 15:
        summary_lines.append(f"  ... +{len(findings) - 15} more")
    total_rows = sum(f["n_stale_rows"] for f in findings)
    return [
        LineageIssue(
            severity="warning",
            category="stats_post_cease_reporting",
            message=(
                f"{len(findings)} unit_id(s) report stats AFTER their "
                f"terminal territorial event ({total_rows} stale rows total)"
            ),
            detail=(
                "Each listed unit ceased to exist per a Split/Merge/"
                "Redistribute in the lineage, but stats keep being filed "
                "under the old code. The package redistributes these "
                "via late-reporting fractions, but if a successor child "
                "doesn't report, ALL the data flows to the other "
                "successor — which is wrong. Fix at the FNID-assignment "
                "step: the affected stats rows should be re-coded to the "
                "successor unit's ID (or the lineage's terminal-event "
                "year is incorrect).\n\n" + "\n".join(summary_lines)
            ),
            rows=[],
            ids=[f["unit_id"] for f in findings[:20]],
        )
    ]


# --- Stable-group contiguity check -------------------------------------
#
# Each stable group should be a spatially contiguous region — territorial
# events only happen between neighboring units. If a stable group's
# modern members aren't reachable from each other through shared borders,
# the lineage table is most likely wrong (a homonym district from
# another state got matched to the same canonical ID — e.g. "Hamirpur"
# exists in both Himachal Pradesh and Uttar Pradesh).
#
# We compute the spatial-neighbor graph from the modern shapefile once
# (geopandas spatial join, predicate='touches'), then for each stable
# group we check whether its members form a connected subgraph in that
# neighbor graph.


def validate_stable_group_contiguity(
    remap: dict[str, str],
    modern_gdf,
    id_column: str,
) -> list[LineageIssue]:
    """Flag stable groups whose modern members aren't spatially connected.

    Args:
        remap: ``{unit_id → stable_id}`` from ``build_stable_groups``.
        modern_gdf: GeoDataFrame of the modern shapefile.
        id_column: name of the unit_id column in ``modern_gdf``.

    Returns:
        List of LineageIssue (severity ``warning``, category
        ``stable_group_disconnected``) — one per disconnected group.
    """
    # Local imports — these are only needed for the geometric check.
    # Keeps the validators module light when used solely for lineage
    # validation (no shapefile available).
    import geopandas as gpd  # noqa: F401  (already imported in package)
    from shapely import make_valid

    if modern_gdf is None or len(modern_gdf) == 0:
        return []

    # Repair invalid geometries upfront — touches() can return wrong
    # answers on self-intersecting polygons. Keep only the id + geometry
    # columns to avoid name collisions during the self-sjoin.
    repaired = modern_gdf[[id_column, "geometry"]].copy()
    repaired["geometry"] = repaired["geometry"].apply(make_valid)
    # Use unit_id as index (drop the column) so the self-sjoin produces
    # ``index_right`` for the right side and the index for the left.
    repaired = repaired.set_index(id_column)

    # Build neighbor graph via a spatial join. predicate='touches' returns
    # pairs whose boundaries share any portion (without overlapping
    # interiors). Self-joins are filtered out below.
    try:
        joined = repaired.sjoin(repaired, predicate="touches", how="inner")
    except Exception as e:  # pragma: no cover — defensive on degenerate gdfs
        return [LineageIssue(
            severity="warning",
            category="stable_group_contiguity_skipped",
            message=f"could not compute spatial-neighbor graph ({e})",
            detail=("The contiguity check requires a clean modern shapefile. "
                    "Skipping; lineage validation is unaffected."),
        )]

    # joined index = left's unit_id; column 'index_right' = right's unit_id.
    # Build a {unit_id: set(neighbor unit_ids)} dict.
    neighbors: dict[str, set[str]] = defaultdict(set)
    right_col = "index_right"
    if right_col not in joined.columns:
        candidates = [c for c in joined.columns if c.endswith("_right")]
        if not candidates:  # pragma: no cover
            return []
        right_col = candidates[0]
    for left_id, right_id in zip(joined.index, joined[right_col]):
        if left_id == right_id:
            continue
        neighbors[left_id].add(right_id)

    # Group modern unit_ids by stable_id (only consider those that ARE in
    # modern — singletons in the remap that aren't modern are irrelevant
    # for spatial contiguity).
    modern_ids = set(repaired.index)
    members_by_stable: dict[str, set[str]] = defaultdict(set)
    for unit_id, stable_id in remap.items():
        if unit_id in modern_ids:
            members_by_stable[stable_id].add(unit_id)

    # A stable group is several of today's districts treated as one unit,
    # and they are supposed to form a single contiguous piece of ground. Test
    # that by starting from one member and walking to its neighbours, then
    # their neighbours, staying inside the group. If that reaches every
    # member, the group is one piece; if not, the group is in scattered
    # fragments, which normally means a name was matched to a district
    # somewhere else in the country entirely.
    issues: list[LineageIssue] = []
    for stable_id, members in members_by_stable.items():
        # A group of one is trivially in one piece.
        if len(members) <= 1:
            continue
        # Always start from the same member so the report reads identically
        # on repeat runs.
        seed = next(iter(sorted(members)))
        visited: set[str] = {seed}
        queue = [seed]
        while queue:
            u = queue.pop()
            for n in neighbors.get(u, ()):
                if n in members and n not in visited:
                    visited.add(n)
                    queue.append(n)
        if visited == members:
            continue

        # Some members were unreachable. Repeat the walk from each of those
        # in turn to find out how many separate pieces there are — a report
        # that says "three fragments" is far more diagnosable than one that
        # only says the group is broken.
        unvisited = members - visited
        components = [visited]
        remaining = set(unvisited)
        while remaining:
            seed2 = next(iter(sorted(remaining)))
            comp: set[str] = {seed2}
            q = [seed2]
            while q:
                u = q.pop()
                for n in neighbors.get(u, ()):
                    if n in members and n not in comp:
                        comp.add(n)
                        q.append(n)
            components.append(comp)
            remaining -= comp
        # Render every component (truncated to 5 ids each) so the
        # message stays useful when both components are size 1.
        comp_blurbs = []
        for comp in sorted(components, key=lambda c: (-len(c), sorted(c))):
            ids_str = ", ".join(sorted(comp)[:5])
            ellipsis = "…" if len(comp) > 5 else ""
            comp_blurbs.append(f"[{len(comp)}] {ids_str}{ellipsis}")
        issues.append(LineageIssue(
            severity="warning",
            category="stable_group_disconnected",
            message=(
                f"stable group {stable_id} has {len(components)} "
                f"spatially-disconnected components ({len(members)} total members)"
            ),
            detail=(
                "Members of this stable group are not all reachable from each "
                "other through shared borders in the modern shapefile. Common "
                "causes: (a) a homonym name in the relationship table matched "
                "to a unit in another part of the country (e.g., 'Hamirpur' "
                "exists in both Himachal Pradesh and Uttar Pradesh); (b) "
                "island or exclave geometry where the modern members are "
                "genuinely separated by water/another unit; (c) tiny gaps in "
                "the modern shapefile where boundaries don't quite meet. "
                "Components (size + sample ids): " + "; ".join(comp_blurbs) + "."
            ),
            ids=sorted(members)[:20],
        ))
    return issues


# --- Reporting ----------------------------------------------------------


def format_issues(issues: Iterable[LineageIssue]) -> str:
    """Render a human-readable report of issues, grouped by severity.

    Use cases: pretty-printing in a notebook, CI logs, the message of
    ``LineageDataError``, paper supplement. Severity ordering (errors
    first) makes scanning the report intuitive.

    For long row lists we truncate to 6 + count of remaining; for ID
    lists we show only when small (≤ 8) since IDs aren't always
    human-meaningful.
    """
    # Group by severity. defaultdict(list) gives us empty lists for
    # severities not present, simplifying the loop below.
    by_sev: dict[str, list[LineageIssue]] = defaultdict(list)
    for i in issues:
        by_sev[i.severity].append(i)

    lines: list[str] = []
    # Severity order: error → warning → info. Each gets a labeled
    # section header.
    order = [
        ("error", "Errors (must fix)"),
        ("warning", "Warnings (review)"),
        ("info", "Info (advisory)"),
    ]
    total = sum(len(v) for v in by_sev.values())
    lines.append(f"Lineage validation: {total} finding(s)")
    for sev, label in order:
        items = by_sev.get(sev, [])
        if not items:
            continue
        lines.append("")
        lines.append(f"=== {label} — {len(items)} ===")
        for i, issue in enumerate(items, 1):
            lines.append("")
            lines.append(f"[{sev.upper()} #{i}] {issue.category}")
            lines.append(f"  {issue.message}")
            lines.append(f"  → {issue.detail}")
            if issue.rows:
                # Truncate long row lists. The user can always re-call
                # validate_lineage and inspect issue.rows directly.
                head = issue.rows[:6]
                more = f" ... (+{len(issue.rows) - 6} more)" if len(issue.rows) > 6 else ""
                lines.append(f"  rows: {head}{more}")
            if issue.ids and len(issue.ids) <= 8:
                # Only include IDs when the list is short — long ID
                # lists clutter the report without adding much value.
                lines.append(f"  ids:  {issue.ids}")
    return "\n".join(lines)


def _normalize_for_compare(name: object) -> str:
    """Loose name comparison key: lowercase, collapse whitespace and dashes.

    Deliberately *not* :func:`stablebound.match.normalize_name` — importing the
    matcher here would put a name-matching dependency underneath the validator
    (and `lineage` already imports this module). This only has to catch
    "same district, cosmetic spelling difference", not do fuzzy matching.
    """
    s = str(name).strip().lower().replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def _check_namechange_unit_name_mismatch(graph: LineageGraph) -> list[LineageIssue]:
    """A NameChange whose unit is never known by either of its names — warning.

    A ``NameChange`` row asserts "unit U was called OLD, then NEW". If no
    *other* row in the lineage ever calls U by OLD or NEW, the row is very
    likely pointing at the wrong unit_id.

    This matters more than it looks, because
    :func:`stablebound.io.merge_name_changes` folds a separate name-change log
    into the graph as ``NameChange`` events. A mistyped id there renames an
    unrelated district in *every* name lookup — ``name_history``, per-year
    snapshot names, and the FEWS admin-definition workbooks.

    Two real examples from India's bundled log, both of which this check
    catches:

    - ``IN.ADM2.00481 'Bhabua' -> 'Kaimur'`` — ``00481`` is Garhwa
      (Jharkhand); the renamed district is ``00483`` (Bihar).
    - ``IN.ADM2.00799 'Saiha' -> 'Siaha'`` — ``00799`` is Palwal (Haryana);
      the renamed district is ``00704`` (Mizoram). This one reached the
      shipped deliverable: Palwal is labelled "Siaha" in every
      ``IN_Admin2_{2017..2025}.csv``.

    The message names the unit(s) that *do* carry those names, so the
    correction is in the report rather than left as an exercise.

    Warning, not error: a lineage may legitimately introduce a unit via a
    NameChange row alone, and a hard failure would block loading for a log
    that is merely imprecise. Rows with no corroborating evidence either way
    are skipped rather than guessed at.
    """
    ev = graph.events
    name_changes = ev[ev["event_type"] == "NameChange"]
    if name_changes.empty:
        return []

    # Collect every name the lineage gives each district, remembering which
    # record each name came from. Keeping track of the source matters because
    # a rename record has to be judged against what the *rest* of the lineage
    # says, not against its own claim — otherwise every record confirms
    # itself and the check finds nothing.
    per_row: dict[str, dict[int, set[str]]] = defaultdict(dict)
    for idx, e in ev.iterrows():
        per_row[str(e["parent_id"])].setdefault(idx, set()).add(
            _normalize_for_compare(e["parent_name"])
        )
        per_row[str(e["child_id"])].setdefault(idx, set()).add(
            _normalize_for_compare(e["child_name"])
        )

    def names_excluding(uid: str, skip: int) -> set[str]:
        return {n for i, ns in per_row[uid].items() if i != skip for n in ns}

    issues: list[LineageIssue] = []
    for idx, row in name_changes.iterrows():
        uid = str(row["parent_id"])
        old = _normalize_for_compare(row["parent_name"])
        new = _normalize_for_compare(row["child_name"])

        corroborating = names_excluding(uid, idx)
        if not corroborating:
            # The lineage mentions this district nowhere else, so there is
            # nothing to check the rename against. Silence beats a guess.
            continue
        # If either name matches what the rest of the lineage calls this
        # district, the rename is about the district it claims to be about.
        if old in corroborating or new in corroborating:
            continue

        # It doesn't match, so the rename has probably been filed against the
        # wrong district. Find who really goes by these names, so the report
        # can name the likely correct one instead of just flagging a problem.
        holders = sorted(
            u for u in per_row
            if u != uid and ({old, new} & names_excluding(u, idx))
        )
        hint = (
            f" {old!r}/{new!r} belong to {holders}, which is the likely correct "
            f"unit_id." if holders else
            " No unit in the lineage carries either name."
        )
        issues.append(
            LineageIssue(
                severity="warning",
                category="namechange_unit_name_mismatch",
                message=(
                    f"NameChange {int(row['event_year'])}: {uid} "
                    f"{row['parent_name']!r} -> {row['child_name']!r}, but no other "
                    f"row calls {uid} by either name"
                ),
                detail=(
                    f"{uid} is otherwise known as {sorted(corroborating)}."
                    + hint
                    + " If this row came from a name-change log, check its unit_id:"
                    " a mistyped id renames an unrelated unit in every name lookup,"
                    " including the FEWS admin-definition workbooks."
                ),
                rows=[idx],
                ids=[uid, *holders],
            )
        )
    return issues


# --- Cross-level checks --------------------------------------------------
#
# These need TWO graphs, so they are not part of `validate_lineage` (which
# takes one) and do not run at load time. `Lineage.validate_levels()` and
# `Lineage.export_fews()` call them.




def coarse_universe(
    graph: LineageGraph,
    coarse_graph: LineageGraph | None = None,
    *,
    baseline: pd.DataFrame | None = None,
    coarse_baseline: pd.DataFrame | None = None,
) -> set[str]:
    """Every upper-level unit that exists, from both levels' evidence.

    An upper-level lineage file records *events*, not membership: India's
    ``ADM1_LINEAGE.xlsx`` has 16 rows naming 21 states, but India has 32 —
    Kerala, Punjab and Gujarat never split, so they appear in it nowhere. The
    missing ones are recoverable only from the level below, where every
    district carries the state it belongs to.

    So the upper universe is the union of three sources, in decreasing
    authority: an explicit coarse baseline if one exists, the units named by
    upper-level events, and the ``coarse_id`` attribution carried on the lower
    level's baseline and events. ``fnid._admin1_universe`` derives the same set
    for FNID code assignment; this is that rule stated once, for reuse.
    """
    # No single source knows every state, so gather from all four and pool
    # the results. None of them is a fallback for another — they all run.
    universe: set[str] = set()

    # States named in the state-level lineage. Only ones that changed appear
    # there, so this alone misses most of the country.
    if coarse_graph is not None:
        universe |= set(coarse_graph.all_unit_ids())
    # A list of states, if the country supplied one outright. Rare, and the
    # only source that says which states exist rather than implying it.
    if coarse_baseline is not None and "unit_id" in coarse_baseline.columns:
        universe |= set(coarse_baseline["unit_id"].dropna().astype(str))
    # Every district records the state it belongs to, so the districts
    # between them name the states nobody else mentions. This is what finds a
    # state like Kerala, in India, which has never split.
    if baseline is not None and "coarse_id" in baseline.columns:
        universe |= set(baseline["coarse_id"].dropna().astype(str))
    # Districts created later aren't in that starting list, but the record of
    # their creation still names the state they were carved out of.
    for col in ("parent_coarse_id", "child_coarse_id"):
        if col in graph.events.columns:
            universe |= set(graph.events[col].dropna().astype(str))

    # A district with no state recorded leaves a blank behind; drop those so
    # we don't end up reporting a state whose name is nothing at all.
    return {u for u in universe if u}


def validate_coarse_references(
    graph: LineageGraph,
    coarse_graph: LineageGraph,
    *,
    baseline: pd.DataFrame | None = None,
    coarse_baseline: pd.DataFrame | None = None,
) -> list[LineageIssue]:
    """Check that the two admin levels agree about the upper level.

    A two-level country carries its admin1 attribution as ``coarse_id`` on the
    admin2 baseline and ``parent_coarse_id`` / ``child_coarse_id`` on admin2
    events. The FEWS deliverable keys every admin2 unit's code on the admin1 it
    originated in, so a disagreement between the levels ships as a district
    filed under the wrong state — invisible unless someone reads the output.

    **What cannot be checked, and why.** The obvious check — "every referenced
    ``coarse_id`` exists in the upper-level lineage" — is wrong, and an earlier
    version of this function that implemented it reported 22 false errors on
    India. The upper lineage records events, so a state that never split is
    absent from it by construction; being referenced-but-absent is the ordinary
    case for roughly two thirds of India's states. Worse, the fix is circular:
    the only other source for those units is the lower level's own attribution,
    so checking one against the other is vacuous. Without a genuine coarse
    baseline enumerating every upper unit, a dangling-reference check has no
    independent evidence to work from.

    **What is checked instead**, both with real signal:

    1. *Attribution coverage.* A lower unit with no ``coarse_id`` at all cannot
       be placed under any upper unit, so its FEWS code has no admin1 to key
       on. Reported as an error when it affects the baseline, since that is the
       set the deliverable enumerates.
    2. *Temporal consistency*, for the subset of upper units the upper lineage
       does describe. An admin2 event in year Y may not attribute its child to
       a state that the state-level file says had already ceased, or did not
       yet exist at Y+1. That is the real failure mode — a district filed under
       a state absent from the state-level file for that year — and it fires
       only where there is independent evidence to fire on.

    Pass ``coarse_baseline`` when the country has a real upper-level baseline;
    the dangling check then becomes sound and is reported at error severity.
    """
    issues: list[LineageIssue] = []
    ev = graph.events

    have_coarse_baseline = (
        coarse_baseline is not None and "unit_id" in coarse_baseline.columns
    )

    # Where each coarse_id is referenced from, for error messages.
    referenced: dict[str, set[str]] = defaultdict(set)
    for col in ("parent_coarse_id", "child_coarse_id"):
        if col not in ev.columns:
            continue
        for idx, val in ev[col].items():
            if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
                continue
            referenced[str(val)].add(f"event row {idx}")
    if baseline is not None and "coarse_id" in baseline.columns:
        for _, brow in baseline.iterrows():
            val = brow.get("coarse_id")
            if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
                continue
            referenced[str(val)].add(f"baseline {brow.get('unit_id')}")

    # --- 1. Attribution coverage ---------------------------------------
    if baseline is not None:
        if "coarse_id" not in baseline.columns:
            issues.append(
                LineageIssue(
                    severity="error",
                    category="baseline_has_no_coarse_attribution",
                    message=(
                        "the lower-level baseline has no 'coarse_id' column, so no "
                        "unit can be placed under an upper-level unit"
                    ),
                    detail=(
                        "The FEWS deliverable keys every admin2 code on the admin1 "
                        "it originated in. Add 'coarse_id' and 'coarse_name' to the "
                        "baseline, or export the upper level separately."
                    ),
                )
            )
        else:
            blank = baseline[baseline["coarse_id"].isna() | (baseline["coarse_id"] == "")]
            if len(blank):
                ids = [str(u) for u in blank.get("unit_id", pd.Series(dtype=str))]
                issues.append(
                    LineageIssue(
                        severity="error",
                        category="baseline_unit_without_coarse_id",
                        message=(
                            f"{len(blank)} baseline unit(s) have no coarse_id and "
                            "cannot be attributed to an upper-level unit"
                        ),
                        detail=(
                            f"Ids: {ids[:8]}"
                            + (f" (+{len(ids) - 8} more)" if len(ids) > 8 else "")
                            + ". Their FEWS codes have no admin1 to key on."
                        ),
                        ids=ids[:20],
                    )
                )

    if not referenced:
        issues.append(
            LineageIssue(
                severity="warning",
                category="no_coarse_references",
                message="the lower level carries no coarse_id references at all",
                detail=(
                    "Neither the baseline's 'coarse_id' column nor the events' "
                    "'parent_coarse_id'/'child_coarse_id' are populated, so the two "
                    "levels cannot be cross-checked."
                ),
            )
        )
        return issues

    # --- 2. Dangling references, only when a real baseline makes it sound ---
    if have_coarse_baseline:
        known = set(coarse_baseline["unit_id"].dropna().astype(str)) | set(
            coarse_graph.all_unit_ids()
        )
        dangling = sorted(set(referenced) - known)
        for cid in dangling:
            where = sorted(referenced[cid])
            issues.append(
                LineageIssue(
                    severity="error",
                    category="dangling_coarse_reference",
                    message=(
                        f"coarse_id {cid!r} is referenced by the lower level but "
                        "exists in neither the upper-level lineage nor its baseline"
                    ),
                    detail=(
                        f"Referenced from: {where[:6]}"
                        + (f" (+{len(where) - 6} more)" if len(where) > 6 else "")
                        + ". Most likely a typo; otherwise the upper level is "
                        "missing this unit."
                    ),
                    ids=[cid],
                )
            )

    # --- 3. Temporal consistency, where the upper level has evidence -------
    from .snapshot import build_snapshot

    # `build_snapshot` on a bare upper graph is nearly empty: `initial_units()`
    # returns only *territorial* units that are never a child, so India's ADM1
    # graph yields 7 states, and Orissa — which appears solely in a 2011
    # NameChange — is in no snapshot at all. That is deliberate (see the
    # build_snapshot docstring), and the prescribed caller pattern is to seed
    # the always-alive externals via `additional_units`. Here that seed is the
    # coarse universe, which turns the same call into a real 32-state 1991
    # snapshot. Skipping this step reported 84 false errors on India.
    universe = coarse_universe(
        graph, coarse_graph, baseline=baseline, coarse_baseline=coarse_baseline
    )
    if universe and not ev.empty:
        snap_cache: dict[int, set[str]] = {}

        def alive(year: int) -> set[str]:
            if year not in snap_cache:
                snap_cache[year] = set(
                    build_snapshot(coarse_graph, year, additional_units=universe)
                )
            return snap_cache[year]

        for idx, row in ev.iterrows():
            year = int(row["event_year"])
            # A parent is the unit in force *going into* the event year; a
            # child first appears the year after (see snapshot.build_snapshot
            # on the event-year convention). Check each against its own year.
            for col, when, who in (
                ("parent_coarse_id", year, "parent_id"),
                ("child_coarse_id", year + 1, "child_id"),
            ):
                if col not in ev.columns:
                    continue
                cid = row.get(col)
                if cid is None or (isinstance(cid, float) and pd.isna(cid)) or cid == "":
                    continue
                cid = str(cid)
                # A unit seeded as always-alive is in every snapshot, so this
                # is vacuous for never-changed states and only bites where the
                # upper level has genuine lifecycle evidence — a district filed
                # under Bihar after Bihar ceased in 2000, or under Telangana
                # before it existed.
                if cid in alive(when):
                    continue
                issues.append(
                    LineageIssue(
                        severity="error",
                        category="coarse_reference_not_alive",
                        message=(
                            f"event row {idx} ({row['event_type']} {year}) attributes "
                            f"{str(row.get(who))!r} to {cid!r}, which the upper-level "
                            f"lineage says is not in force at {when}"
                        ),
                        detail=(
                            "The upper level says this unit had ceased, or did not "
                            "yet exist, in that year. The FEWS deliverable would "
                            "file the unit under a state absent from the "
                            "state-level file for that vintage."
                        ),
                        rows=[idx],
                        ids=[str(row.get(who)), cid],
                    )
                )
    return issues


def validate_shapefile_lineage_consistency(
    shapefile_unit_ids: Iterable[str],
    graph: LineageGraph,
    year: int,
    baseline: pd.DataFrame | None = None,
) -> list[LineageIssue]:
    """Cross-check a shapefile's attached unit_ids against the lineage.

    Attaching ids to a map checks only that each feature GOT one. It does not
    ask whether the id is real, whether two polygons claim the same one, or
    whether the district it names still existed when the map was drawn. Those
    are the questions that catch a wrong id, because a wrong id is a perfectly
    well-formed string.

    The consequences are not cosmetic. An id belonging to a district 1,100 km
    away does not merely mislabel a polygon -- the polygon is then dissolved
    into that district's stable group, so the shipped geometry is wrong. India
    shipped exactly that: Chhattisgarh's Bilaspur carrying Himachal Pradesh's
    id, and a Raigarh polygon carrying an id the 2022 Sarangarh-Bilaigarh split
    had already retired.

    Args:
        shapefile_unit_ids: the ids attached to the map. Sentinel
            ``UNMATCHED_*`` values are ignored; they are already reported
            through :attr:`Lineage.unmatched_features`.
        graph: parsed LineageGraph.
        year: the map's vintage -- normally :attr:`Lineage.shapefile_year`.
        baseline: the lineage's baseline, if it has one. Needed for both
            directions: it seeds the snapshot (a district that appears in no
            event is otherwise absent from every snapshot) and it is part of
            the universe of ids the lineage knows about.

    Returns:
        list of LineageIssue. Four checks:

          1. ``shapefile_duplicate_id`` (warning) -- one id on several polygons.
          2. ``shapefile_unknown_id`` (error) -- an id the lineage has never
             heard of, at any year.
          3. ``shapefile_retired_id`` (warning) -- a real id, but not in force
             at ``year``.
          4. ``shapefile_unit_without_polygon`` (warning) -- the reverse
             direction: a district the lineage says exists, that no polygon
             claims.

    Checks 3 and 4 are the two halves of one mistake. When a polygon takes a
    retired id, the live district it should have taken is left with no polygon,
    so both fire together and name the two ends of the swap.
    """
    ids = [str(u) for u in shapefile_unit_ids
           if u is not None and str(u) not in ("", "None", "nan")]
    real = [u for u in ids if not u.startswith("UNMATCHED_")]
    if not real:
        return []

    baseline_ids: set[str] = set()
    if baseline is not None and len(baseline) > 0:
        baseline_ids = set(baseline["unit_id"].astype(str))

    issues: list[LineageIssue] = []
    issues.extend(_check_shapefile_duplicate_ids(real))
    known, retired = _check_shapefile_id_lifespan(real, graph, year, baseline_ids)
    issues.extend(known)
    issues.extend(retired)
    issues.extend(
        _check_shapefile_units_without_polygon(real, graph, year, baseline_ids)
    )
    return issues


def _check_shapefile_duplicate_ids(unit_ids: list[str]) -> list[LineageIssue]:
    """One unit_id attached to more than one polygon.

    Usually a homonym that was never separated: two districts share a name,
    the matcher could not tell them apart, and both took the first id. India's
    two Bilaspurs did this.

    Warning rather than error because it is occasionally legitimate. A district
    with detached territory -- an enclave, an island group -- may arrive as
    several features that genuinely are one district, and dissolving them is a
    reasonable thing for a map to leave to the user.
    """
    counts = defaultdict(int)
    for u in unit_ids:
        counts[u] += 1
    repeated = sorted(u for u, n in counts.items() if n > 1)
    if not repeated:
        return []
    shown = ", ".join(f"{u} (x{counts[u]})" for u in repeated[:5])
    return [
        LineageIssue(
            severity="warning",
            category="shapefile_duplicate_id",
            message=(
                f"{len(repeated)} unit_id(s) are attached to more than one "
                f"polygon: {shown}"
                + (" ..." if len(repeated) > 5 else "")
            ),
            detail=(
                "Two polygons claiming one district usually means a homonym "
                "went unresolved -- same name in two states, and both features "
                "took the first id. Re-run the match with a coarse_column "
                "naming the shapefile's state/province field, which separates "
                "them from the map's own attributes rather than from a "
                "hand-maintained list. If the repeated features are genuinely "
                "one district in several pieces (an enclave or island group), "
                "this is expected and can be ignored."
            ),
            ids=repeated,
        )
    ]


def _check_shapefile_id_lifespan(
    unit_ids: list[str],
    graph: LineageGraph,
    year: int,
    baseline_ids: set[str],
) -> tuple[list[LineageIssue], list[LineageIssue]]:
    """Split the ids into ones the lineage never knew and ones it has retired.

    Returned as two lists so the caller keeps them apart: an unknown id is a
    different problem from a retired one, and an id the lineage has never heard
    of cannot meaningfully also be "not in force at ``year``". Reporting both
    for the same id would double-count it.
    """
    from .snapshot import build_snapshot

    events = graph.events
    known = (baseline_ids
             | set(events["child_id"].dropna().astype(str))
             | set(events["parent_id"].dropna().astype(str)))
    alive = set(build_snapshot(graph, year, additional_units=baseline_ids))

    unknown = sorted({u for u in unit_ids if u not in known})
    retired = sorted({u for u in unit_ids if u in known and u not in alive})

    unknown_issues: list[LineageIssue] = []
    if unknown:
        unknown_issues.append(
            LineageIssue(
                severity="error",
                category="shapefile_unknown_id",
                message=(
                    f"{len(unknown)} unit_id(s) on the map appear nowhere in the "
                    f"lineage, at any year: {', '.join(unknown[:5])}"
                    + (" ..." if len(unknown) > 5 else "")
                ),
                detail=(
                    "The lineage has no record of these districts, so nothing "
                    "downstream can place them: they cannot be grouped, "
                    "aggregated, or traced. Either the ids were attached from a "
                    "different vintage of the lineage, or they were typed by "
                    "hand. Check them against the relationship table and the "
                    "baseline before building any product from this map."
                ),
                ids=unknown,
            )
        )

    retired_issues: list[LineageIssue] = []
    if retired:
        retired_issues.append(
            LineageIssue(
                severity="warning",
                category="shapefile_retired_id",
                message=(
                    f"{len(retired)} unit_id(s) on the map name districts the "
                    f"lineage says were not in force in {year}: "
                    f"{', '.join(retired[:5])}"
                    + (" ..." if len(retired) > 5 else "")
                ),
                detail=(
                    "A polygon is carrying an id that had already been retired "
                    "-- typically a parent that a later split replaced, where "
                    "the map kept the parent's name and the match followed the "
                    "name rather than the year. India's Raigarh did this: the "
                    "2022 split that created Sarangarh-Bilaigarh retired the "
                    "old id, but the 2024 map still carried it.\n"
                    "Check the shapefile_unit_without_polygon issue alongside "
                    "this one -- the district that SHOULD hold this polygon is "
                    "usually sitting in that list.\n"
                    "Warning, not error, because a deliberately historical map "
                    "will legitimately carry ids retired since. If that is what "
                    "this is, the vintage is what needs correcting, not the ids."
                ),
                ids=retired,
            )
        )
    return unknown_issues, retired_issues


def _check_shapefile_units_without_polygon(
    unit_ids: list[str],
    graph: LineageGraph,
    year: int,
    baseline_ids: set[str],
) -> list[LineageIssue]:
    """Districts the lineage says exist at ``year`` that no polygon claims.

    The reverse direction, and the half that is easy to forget: checking only
    the ids ON the map cannot see a district that is missing from it. This is
    the check that surfaced the Assam case, where districts the lineage created
    had no polygon at all.

    Note the asymmetry with the forward checks -- this one is about the map
    being incomplete, so it fires just as loudly when the ids present are all
    perfectly valid.
    """
    from .snapshot import build_snapshot

    alive = set(build_snapshot(graph, year, additional_units=baseline_ids))
    missing = sorted(alive - set(unit_ids))
    if not missing:
        return []
    return [
        LineageIssue(
            severity="warning",
            category="shapefile_unit_without_polygon",
            message=(
                f"{len(missing)} district(s) in force in {year} have no polygon "
                f"on the map: {', '.join(missing[:5])}"
                + (" ..." if len(missing) > 5 else "")
            ),
            detail=(
                "Either the map predates these districts and the inferred "
                "vintage is too late, or a polygon that should carry one of "
                "these ids is carrying something else instead -- see the "
                "shapefile_retired_id issue, which names the ids that were "
                "taken by mistake.\n"
                "Products built from this map will have no geometry for these "
                "districts, so any statistics reported against them have "
                "nowhere to go."
            ),
            ids=missing,
        )
    ]
