"""Name-matching utilities for shapefile and stats inputs.

The package's contract elsewhere is that ``unit_id`` columns are already
attached to user data. This module is the bridge: given a user's
shapefile or long-form stats table where rows are keyed by *name*
(district name, province name, etc.) rather than canonical ID, it
proposes a unit_id for each row by matching against the snapshot of
units the lineage says were alive at a target year.

Workflow (human-in-the-loop):

    1. Call :func:`propose_shapefile_mapping` or
       :func:`propose_stats_mapping`. Returns a :class:`MatchProposal`
       summarizing the best guess for each row.
    2. Save to CSV with ``proposal.to_csv("mapping.csv")`` and review
       the file in Excel. Fuzzy matches in the 0.80-0.90 band are
       flagged as "sketchy" for human verification; rows below 0.80 are
       reported as unmatched with their best candidate.
    3. Edit the CSV: correct any wrong proposed_unit_id values, fill
       in unmatched rows.
    4. Pass the reviewed CSV back to :func:`attach_shapefile_ids` or
       :func:`attach_stats_ids` to produce the canonical ID-attached
       artifact.

The matching pipeline runs in priority order:

    0. Homonym override by source row index (highest priority).
    1. Exact match on normalized name.
    2. Manual override (caller-supplied alias map).
    3. Fuzzy match (SequenceMatcher ratio >= ``fuzzy_threshold``).
    4. Unmatched (reports best candidate below threshold).

Lifted from the India 5-pass matcher in
``tools/india/prepare_shapefile.py`` with the country-specific
bits (normalizer rules, override dicts) made overrideable.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from .lineage import (
    LineageGraph,
    _coarse_name_of_unit_in_year,
    _name_of_unit_in_year,
    apply_coarse_rename,
)
from .snapshot import build_snapshot

_LOGGER = logging.getLogger(__name__)

# Default fuzzy thresholds. The "sketchy" band 0.80-0.90 captures matches
# that are plausible but human review can easily verify or refute.
DEFAULT_FUZZY_THRESHOLD = 0.80
DEFAULT_SKETCHY_THRESHOLD = 0.90

# Marker written into a proposal's ``notes`` (and appended to ``method``)
# whenever a year-aware match is remapped to a year-compatible *ancestor* — i.e.
# the row is aligned to an OLDER unit than its name first resolved to (a name
# reported before its unit existed routed back to the pre-split parent). This is
# exactly the case a reviewer must eyeball to be sure data isn't mistakenly
# attached to an old unit, so it is surfaced in three places: the ``method``
# column, the ``notes`` column, and ``MatchProposal.remapped`` / ``.summary()``.
ANCESTOR_WALK_TAG = "ancestor-walk"
# The same marker as it appears appended to ``method`` (``exact+ancestor_walk``).
# Two spellings exist because ``notes`` is prose and ``method`` is a token; they
# are defined together here so they cannot drift apart again. ``method`` is the
# authoritative one — see :attr:`MatchProposal.remapped`.
ANCESTOR_WALK_SUFFIX = ANCESTOR_WALK_TAG.replace("-", "_")

# Marker for a match made in a year the lineage does not actually describe —
# beyond its last recorded event. The unit is assumed to have persisted, which
# is the only reasonable default (statistics routinely outrun their
# relationship table), but it is an extrapolation and the reviewer should know
# which rows rest on it.
EXTRAPOLATED_TAG = "beyond-lineage"


# --- Default name normalization -----------------------------------------


def normalize_name(s: str) -> str:
    """Default name normalizer.

    Lowercases, removes diacritics via Unicode NFKD, collapses
    whitespace, strips punctuation. Generic enough for most Latin-
    script administrative names. Country-specific quirks (e.g., India's
    "Sri"/"Shri" honorifics or "District" suffix variations) should be
    handled by wrapping this function in a domain-specific normalizer
    passed via ``normalizer=...`` to the propose_* functions.
    """
    s = str(s).strip().lower()
    # Unicode normalization strips combining diacritics so "São Paulo"
    # and "Sao Paulo" collide. NFKD decomposes; we then drop combining
    # codepoints.
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    # Replace common word separators with a single space; punctuation
    # otherwise interferes with fuzzy similarity scoring.
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # Collapse whitespace runs to single spaces, then trim.
    return re.sub(r"\s+", " ", s).strip()


def _similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio. Pulled out so callers can override later."""
    return SequenceMatcher(None, a, b).ratio()


# --- Lookup construction ------------------------------------------------


# Lookup entry: (unit_id, original_name, coarse_name_or_None).
_LookupEntry = tuple[str, str, str | None]
_Lookup = dict[str, list[_LookupEntry]]


def _build_snapshot_lookup(
    graph: LineageGraph,
    year: int,
    *,
    baseline: pd.DataFrame | None = None,
    name_change_log: pd.DataFrame | None = None,
    normalizer: Callable[[str], str] = normalize_name,
) -> _Lookup:
    """Build {normalized_name → [(unit_id, name, coarse_name), …]} for the snapshot at ``year``.

    The lookup value is a *list* of candidates so homonyms (multiple
    units with the same normalized name at the same year — e.g.,
    Hamirpur in Himachal Pradesh vs Hamirpur in Uttar Pradesh) can be
    disambiguated downstream via the optional ``coarse_name`` (upper
    admin level: state, province, region).

    Sources for each candidate's coarse_name, in priority order:
    the baseline's ``coarse_name`` column (when present), then the
    most relevant RT event's ``child_coarse_name`` / ``parent_coarse_name``
    (via :func:`_coarse_name_of_unit_in_year`). ``None`` when no
    source has it.

    ``baseline`` and ``name_change_log`` work the same way they did
    before: baseline seeds units that don't appear in the RT, NCL
    extends the lookup so both old and new official names map to the
    same unit_id.
    """
    baseline_units = set(baseline["unit_id"].astype(str)) if baseline is not None else set()
    units = build_snapshot(graph, year, additional_units=baseline_units)

    baseline_names: dict[str, str] = {}
    baseline_coarses: dict[str, str] = {}
    baseline_coarse_ids: dict[str, str] = {}
    if baseline is not None:
        for _, row in baseline.iterrows():
            baseline_names[str(row["unit_id"])] = str(row["name"])
            if "coarse_name" in baseline.columns:
                cv = row.get("coarse_name")
                if cv is not None and not (isinstance(cv, float) and pd.isna(cv)):
                    baseline_coarses[str(row["unit_id"])] = str(cv)
            if "coarse_id" in baseline.columns:
                ci = row.get("coarse_id")
                if ci is not None and not (isinstance(ci, float) and pd.isna(ci)):
                    baseline_coarse_ids[str(row["unit_id"])] = str(ci)

    lookup: _Lookup = {}

    def _add(norm: str, entry: _LookupEntry) -> None:
        """Append a candidate, deduplicating by unit_id."""
        bucket = lookup.setdefault(norm, [])
        # The same district can be named more than once — by the lineage and
        # again by a rename record. List it once.
        if any(e[0] == entry[0] for e in bucket):
            return
        bucket.append(entry)

    # This builds the phone book the matcher searches: every name a district
    # has ever gone by, pointing back at the district.
    #
    # Work through the districts in a fixed order. Where two districts share
    # a name and nothing can separate them, the matcher takes whichever comes
    # first — so the order here decides that outcome, and leaving it to chance
    # would mean the same analysis gave different answers on different runs.
    for unit_id in sorted(units):
        name = _name_of_unit_in_year(graph, unit_id, year)
        if not name:
            name = baseline_names.get(unit_id)
        if not name:
            continue
        # Ask the lineage for the state AS OF this year before falling back to
        # the baseline. The baseline records the state a district was in at the
        # start of the record, so preferring it means a lookup built for 2024
        # hands back 1991 vocabulary: "Orissa" for three districts and
        # "Pondicherry U.T." for four, long after both were renamed. A modern
        # shapefile says Odisha and Puducherry, so those are exactly the
        # districts a state-based join would fail on.
        coarse = _coarse_name_of_unit_in_year(graph, unit_id, year)
        if not coarse:
            # A district that never appears in any event has no lineage coarse
            # at all, so the baseline is the only source -- and the baseline
            # records the state as of the start of the record. Three of
            # Odisha's districts reach this path, which is why they still read
            # "Orissa" in 2024 after the other two paths were fixed.
            coarse = apply_coarse_rename(
                graph, baseline_coarse_ids.get(unit_id),
                baseline_coarses.get(unit_id), year,
            )
        norm = normalizer(name)
        _add(norm, (unit_id, name, coarse))

    # Add every old name from the rename list, so a district answers to what
    # it used to be called as well as what it is called now.
    if name_change_log is not None:
        for _, row in name_change_log.iterrows():
            uid = str(row["unit_id"])
            # Skip renames of districts that didn't exist in this year.
            if uid not in units:
                continue
            # Note which state it was in, so an old name is still tied to the
            # right place when two districts share it.
            coarse = baseline_coarses.get(uid) or _coarse_name_of_unit_in_year(
                graph, uid, year
            )
            # Register both names, so the district answers to whichever the
            # user's file happens to use.
            for col in ("new_name", "old_name"):
                cand = str(row[col]) if pd.notna(row[col]) else ""
                if not cand:
                    continue
                _add(normalizer(cand), (uid, cand, coarse))

    # Do the same for renames recorded inside the lineage itself. A country
    # can state a rename either way — as a separate list, handled above, or
    # as a change in the lineage — and the two must behave identically.
    #
    # They did not, for a long time. A district renamed in 2016 answered to
    # its old name if the rename came from a list, and did not if the same
    # rename was written into the lineage. That silently penalised every
    # country whose records were written the second way.
    #
    # Both names work in both directions, which matters because a map
    # usually carries today's names while the statistics carry yesterday's,
    # and either can be the one being looked up.
    for _, ev in graph.name_changes.iterrows():
        # A rename names the same district on both sides, so either side
        # identifies it; the parent is used by convention.
        uid = str(ev["parent_id"])
        if uid not in units:
            continue
        coarse = baseline_coarses.get(uid) or _coarse_name_of_unit_in_year(
            graph, uid, year
        )
        # Before and after, same as above.
        for col in ("parent_name", "child_name"):
            cand = ev.get(col)
            cand = str(cand) if cand is not None and not pd.isna(cand) else ""
            if cand:
                _add(normalizer(cand), (uid, cand, coarse))

    return lookup


def _merge_lookups(lookups: Iterable[_Lookup]) -> _Lookup:
    """Merge per-year snapshot lookups into one union lookup.

    Candidates are merged bucket-by-bucket, deduplicating by
    (unit_id, original_name). Coarse names tagged to the same unit_id
    should agree across years (a state assignment doesn't usually
    change), so we keep the first observed. The input order therefore
    determines which coarse wins on a tie — callers pass per-year
    lookups in ascending year order.
    """
    # Fold each year's district list into one combined list, so a name can be
    # looked up without knowing which year it came from.
    union: _Lookup = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for per_year in lookups:
        for norm, candidates in per_year.items():
            bucket = union.setdefault(norm, [])
            seen_bucket = seen.setdefault(norm, set())
            for uid, name, coarse in candidates:
                # A district appears in every year it existed, so list each
                # district-and-name pairing once rather than thirty times.
                # The pairing, not the district alone: a district that was
                # renamed should still answer to both of its names.
                key = (uid, name)
                if key in seen_bucket:
                    continue
                seen_bucket.add(key)
                bucket.append((uid, name, coarse))
    return union


def _build_union_lookup(
    graph: LineageGraph,
    year_range: range,
    *,
    baseline: pd.DataFrame | None = None,
    name_change_log: pd.DataFrame | None = None,
    normalizer: Callable[[str], str] = normalize_name,
) -> _Lookup:
    """Union of snapshot lookups across a year range.

    Used by :func:`propose_stats_mapping` to handle stats rows whose
    canonical names vary over the analysis window (e.g., a district
    renamed mid-window: rows under the old name and rows under the
    new name both need to resolve to the same unit_id).
    """
    return _merge_lookups(
        _build_snapshot_lookup(
            graph,
            year,
            baseline=baseline,
            name_change_log=name_change_log,
            normalizer=normalizer,
        )
        for year in year_range
    )


def _infer_year_by_name(
    source_names: set[str],
    graph: LineageGraph,
    candidate_years: Iterable[int],
    *,
    baseline: pd.DataFrame | None = None,
    normalizer: Callable[[str], str] = normalize_name,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple[int, dict[int, int]]:
    """Pick the year whose snapshot names best match ``source_names`` (already normalized).

    For each candidate year, build the snapshot-name set (without the
    NCL — including renamed historical names would mute the year-of-
    validity signal we're trying to find). Score = symmetric-difference
    count after both exact and fuzzy matching at ``fuzzy_threshold``.
    Each side (source/snapshot) contributes any name without a
    counterpart in the other side.

    Returns ``(best_year, mismatches_by_year)``. Earliest year wins
    ties — same tie-break as :func:`stablebound.snapshot.infer_year`.

    Complexity: per year, O(|src| + |snap|) for the exact pass, then
    O(|src_unmatched| * |snap_unmatched|) for fuzzy. India scale
    (~700 features, ~600 snapshot names, ~32 years) runs in under a
    second.
    """
    candidates = list(candidate_years)
    if not candidates:
        raise ValueError("_infer_year_by_name requires at least one candidate year.")

    mismatches: dict[int, int] = {}
    for y in candidates:
        # No NCL here: we want the year-of-validity signal, not the
        # historical-aliases-also-count signal.
        lookup = _build_snapshot_lookup(
            graph, y, baseline=baseline, name_change_log=None, normalizer=normalizer
        )
        snap_names = set(lookup.keys())

        src_unmatched = source_names - snap_names
        snap_unmatched = snap_names - source_names

        # Fuzzy second pass: each remaining source picks its best
        # remaining snapshot name. A snapshot name claimed by one
        # source can't be claimed by another (keeps the metric
        # symmetric).
        # Names that didn't match outright may still be the same place spelled
        # differently, and a year shouldn't be penalised for that. Pair up
        # what is left by closeness before counting the year's score.
        fuzzy_src: set[str] = set()
        fuzzy_snap: set[str] = set()
        for src in src_unmatched:
            best_score = 0.0
            best_snap: str | None = None
            for snap in snap_unmatched:
                # One district can only answer for one name. Without this a
                # single district could soak up several unmatched names and
                # make a wrong year look like a good fit.
                if snap in fuzzy_snap:
                    continue
                score = _similarity(src, snap)
                if score > best_score:
                    best_score = score
                    best_snap = snap
            # Pair them off only if they are close enough to be believable.
            if best_score >= fuzzy_threshold and best_snap is not None:
                fuzzy_src.add(src)
                fuzzy_snap.add(best_snap)

        n_src_um = len(src_unmatched - fuzzy_src)
        n_snap_um = len(snap_unmatched - fuzzy_snap)
        mismatches[y] = n_src_um + n_snap_um

    best = min(mismatches, key=lambda y: (mismatches[y], y))
    return best, mismatches


# --- Year-aware ancestor walk (opt-in for propose_stats_mapping) --------
#
# Mirrors the year-validation logic in ``stablebound.india.matcher``
# (``_year_compatible`` / ``_walk_year_compat_ancestors`` /
# ``_earliest_created_ancestor``). Kept as an independent copy so the
# generic matcher doesn't import a country submodule; the algorithm and
# its deterministic tie-breaks are identical. When a name resolves to a
# unit_id that wasn't alive in the stats row's year (typically a
# post-split child id attached to pre-split data), we walk up the
# lineage's parent chain for a year-compatible ancestor.


def _parents_map_from_graph(graph: LineageGraph) -> dict[str, set[str]]:
    """{child_id: set(parent_ids)} from lineage events (no self-loops).

    NameChange events (parent_id == child_id) are skipped so only
    territorial parentage is walked.
    """
    out: dict[str, set[str]] = {}
    for _, e in graph.events.iterrows():
        pid, cid = e.get("parent_id"), e.get("child_id")
        if pd.notna(pid) and pd.notna(cid) and pid != cid:
            out.setdefault(str(cid), set()).add(str(pid))
    return out


def _year_alive_index_from_lookups(
    per_year_lookup: dict[int, _Lookup],
) -> dict[str, set[int]]:
    """{unit_id: set(years the id appears in a per-year snapshot lookup)}.

    NCL-added name entries point at ids already present in the year's
    snapshot, so this equals the set of years each id is genuinely alive.
    """
    out: dict[str, set[int]] = {}
    for year, lookup in per_year_lookup.items():
        for candidates in lookup.values():
            for uid, _name, _coarse in candidates:
                out.setdefault(uid, set()).add(year)
    return out


def _year_compatible(
    unit_id: str,
    year: int,
    year_alive_index: dict[str, set[int]],
    *,
    grace_post: int = 1,
    grace_pre: int = 1,
) -> bool:
    """True if ``unit_id`` was alive at ``year``.

    A 1-year grace on each side absorbs the snapshot boundary: a unit
    reporting one year past its terminal snapshot (a late report) or one
    year before its first snapshot (pre-baseline data) still counts.
    """
    alive = year_alive_index.get(unit_id)
    if not alive:
        return False
    if year in alive:
        return True
    if year > max(alive) and year <= max(alive) + grace_post:
        return True
    if year < min(alive) and year >= min(alive) - grace_pre:
        return True
    return False


def _walk_year_compat_ancestors(
    unit_id: str,
    year: int,
    parents_map: dict[str, set[str]],
    year_alive_index: dict[str, set[int]],
    *,
    max_depth: int = 10,
) -> list[str]:
    """BFS up ``parents_map``; return year-compatible ancestors found.

    Parents are visited in sorted order so the BFS is deterministic
    regardless of set iteration order (process hash seed).
    """
    # Climb the district's family tree looking for one that existed in the
    # year we care about. Work outwards a generation at a time rather than
    # following one branch to its end, so the closest relatives are found
    # first — a district's immediate parent is a far better guess than a
    # great-grandparent five reorganisations back.
    matches: list[str] = []
    seen = {unit_id}
    queue: list[tuple[str, int]] = [(unit_id, 0)]
    while queue:
        u, depth = queue.pop(0)
        # Stop after ten generations. Real administrative history never runs
        # that deep, so anything reaching this is a loop in the records, and
        # without the limit it would climb forever.
        if depth >= max_depth:
            continue
        # Parents are taken in a fixed order so that repeat runs of the same
        # analysis return the same answer.
        for p in sorted(parents_map.get(u, set())):
            if p in seen:
                continue
            seen.add(p)
            # Collect every ancestor that fits, rather than stopping at the
            # first — the caller needs to know when the choice was ambiguous.
            if _year_compatible(p, year, year_alive_index):
                matches.append(p)
            # Keep climbing past it regardless: a district that fits may
            # itself have a parent that fits, and both are candidates.
            queue.append((p, depth + 1))
    return matches


def _earliest_created_ancestor(
    candidates: list[str],
    year_alive_index: dict[str, set[int]],
) -> str:
    """Pick the ancestor created earliest (smallest min alive-year).

    Tie-break on the id string so the pick is deterministic when two
    candidates share the same earliest alive-year — without it, ``min()``
    returns whichever came first in iteration order, which varies with
    the process hash seed.
    """
    return min(
        candidates,
        key=lambda c: (min(year_alive_index.get(c, {9999})), c),
    )


# --- Proposal data class ------------------------------------------------


@dataclass
class MatchProposal:
    """Result of a propose_* call. Reviewable artifact.

    The ``proposals`` DataFrame has one row per source row, with
    columns:

        source_idx       : int, row index in the source (shapefile or stats).
        source_name      : str, the raw name from the source.
        source_year      : int | NA, only for stats matches.
        source_coarse    : str | NA, the upper-admin value this row came from.
                           Set only when ``coarse_column`` was passed. This is
                           what makes two same-named homonym rows tellable
                           apart — both by a human reading the CSV and by
                           ``attach_*_ids`` joining it back.
        proposed_unit_id : str | empty if unmatched.
        proposed_name    : str | empty if unmatched.
        method           : "homonym" | "exact" | "manual" | "fuzzy" | "unmatched",
                           optionally with a "+ancestor_walk" suffix when a
                           year-aware match was remapped to an older ancestor
                           unit (see :attr:`remapped`).
        score            : float, 1.0 for non-fuzzy methods, ratio for fuzzy,
                           best ratio for unmatched, NaN if no candidate.
        best_candidate   : str, only set when unmatched — best snapshot name
                           found below the fuzzy threshold.

    Use :meth:`to_csv` to write for human review. Edit the file in
    Excel, then load with :func:`read_mapping` and feed to
    :func:`attach_shapefile_ids` / :func:`attach_stats_ids`.
    """

    proposals: pd.DataFrame
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
    sketchy_threshold: float = DEFAULT_SKETCHY_THRESHOLD
    snapshot_unit_names: dict[str, str] = field(default_factory=dict)
    # Set by propose_shapefile_mapping when year=None and year_range is given.
    # None when the caller supplied an explicit year or used the modern-day
    # fallback. The mismatch dict maps each candidate year to its
    # symmetric-difference count for inspection / plotting.
    inferred_year: int | None = None
    year_inference_mismatches: dict[int, int] | None = None

    @property
    def matched(self) -> pd.DataFrame:
        return self.proposals[self.proposals["method"] != "unmatched"]

    @property
    def unmatched(self) -> pd.DataFrame:
        return self.proposals[self.proposals["method"] == "unmatched"]

    @property
    def sketchy(self) -> pd.DataFrame:
        """Fuzzy matches with score in [fuzzy_threshold, sketchy_threshold).

        These are matches that passed the fuzzy cutoff but are close
        enough to the cutoff that human verification is recommended.
        Reviewing this band is the primary value of the mapping CSV.
        """
        df = self.proposals
        return df[
            df["method"].astype(str).str.startswith("fuzzy")
            & (df["score"] < self.sketchy_threshold)
        ]

    @property
    def remapped(self) -> pd.DataFrame:
        """Rows the year-aware walk aligned to an older (ancestor) unit.

        Each matched a name, found that id wasn't alive in the row's year,
        and was remapped up the lineage to the year-compatible ancestor —
        i.e. data attached to an OLDER unit than the name first resolved to
        (a pre-split parent). Surfaced separately from :attr:`sketchy` because
        the underlying name match is often high-confidence, so it would not
        otherwise be flagged for the review a reviewer should give it.
        """
        df = self.proposals
        if "method" not in df.columns:
            return df.iloc[0:0]
        # Keyed on ``method``, not ``notes``: ``method`` is the authoritative
        # record and survives a reviewer clearing the free-text notes column
        # in Excel. (Both carry the tag; only this one is structural.)
        return df[df["method"].astype(str).str.contains(ANCESTOR_WALK_SUFFIX)]

    @property
    def ancestor_issues(self) -> pd.DataFrame:
        """Rows the year-aware walk could not resolve cleanly.

        ``ambiguous`` — several year-compatible ancestors existed and the
        earliest-created was chosen. ``no-ancestor`` — the picked unit wasn't
        alive that year and nothing upstream was either, so the row was
        dropped to unmatched rather than attached to a unit that did not
        exist. Both want a human look; neither is visible in
        :attr:`sketchy` (the underlying name match is usually confident) or
        in :attr:`remapped` (which lists only successful remaps).
        """
        df = self.proposals
        if "ancestor_status" not in df.columns:
            return df.iloc[0:0]
        return df[df["ancestor_status"].isin(["ambiguous", "no-ancestor"])]

    @property
    def extrapolated(self) -> pd.DataFrame:
        """Rows matched in years the lineage does not describe.

        Their year is past the last recorded event, so the unit_id rests on
        the assumption that nothing changed since. Worth a look when a country
        reports well beyond its relationship table's coverage — Philippines
        reports to 2024 from a lineage ending 2013.
        """
        df = self.proposals
        if "notes" not in df.columns:
            return df.iloc[0:0]
        return df[df["notes"].astype(str).str.contains(EXTRAPOLATED_TAG)]

    def to_csv(self, path: Path | str) -> None:
        """Write the proposal as a CSV ready for human review."""
        self.proposals.to_csv(Path(path), index=False)

    def summary(self) -> str:
        """One-screen text summary of match counts + sketchy + unmatched.

        Intended for printing or writing alongside the CSV. Counts by
        method, then lists sketchy matches and unmatched rows so a
        reviewer can act on the high-leverage cases without scrolling
        through the full CSV.
        """
        # Count rows by how they were matched. A row that was later moved to
        # an earlier district has that noted on the end of its method, so
        # trim back to the original method before counting — otherwise those
        # rows are tallied under a category of their own, quietly go missing
        # from the list below, and the numbers stop adding up to the total.
        base = (
            self.proposals["method"].astype(str).str.split("+").str[0]
        )
        counts = base.value_counts().to_dict()
        lines: list[str] = []
        lines.append(f"Total source rows:   {len(self.proposals)}")
        lines.append("")
        lines.append("Match counts by method:")
        for method in ("homonym", "exact", "manual", "fuzzy", "unmatched"):
            lines.append(f"  {method:<10s}  {counts.get(method, 0)}")
        # Anything unexpected still shows up rather than vanishing.
        for method in sorted(set(counts) - {"homonym", "exact", "manual", "fuzzy", "unmatched"}):
            lines.append(f"  {method:<10s}  {counts[method]}")
        # Everything from here down is the reviewer's to-do list, in the
        # order they should work through it: weak guesses, figures moved back
        # to an earlier district, years the lineage doesn't cover, walks that
        # failed outright, and finally names nothing could be found for. Each
        # section is skipped when empty, so a clean run gives a short report
        # rather than a page of zeroes.
        lines.append("")
        # Matches the package accepted but is not confident about. These come
        # first because they are the ones a reviewer can most easily fix.
        sketchy = self.sketchy
        lines.append(
            f"Sketchy fuzzy matches (score < {self.sketchy_threshold}): {len(sketchy)}"
        )
        if not sketchy.empty:
            for _, row in sketchy.sort_values("score").iterrows():
                lines.append(
                    f"  idx={row['source_idx']:<4}  score={row['score']:.2f}  "
                    f"{row['source_name']!r} -> {row['proposed_unit_id']} "
                    f"{row['proposed_name']!r}"
                )
        lines.append("")
        # Figures the package moved to an older district than the name
        # suggested. Worth a human eye: this is right far more often than
        # not, but when it is wrong the data lands on a district that had
        # already been dissolved.
        remapped = self.remapped
        lines.append(
            f"Year-remapped rows (aligned to an OLDER/ancestor unit — VERIFY "
            f"these are not mis-attached to a defunct unit): {len(remapped)}"
        )
        if not remapped.empty:
            for _, row in remapped.iterrows():
                yr = row.get("source_year")
                yr = "" if yr is None or pd.isna(yr) else int(yr)
                lines.append(
                    f"  {row['source_name']!r} ({yr}) -> {row['proposed_unit_id']} "
                    f"{row['proposed_name']!r}   [{row['notes']}]"
                )
        lines.append("")
        # Years the lineage says nothing about, because they fall after its
        # last recorded change. These matched on the assumption that nothing
        # has happened since, which is worth knowing before trusting them.
        extrap = self.extrapolated
        if not extrap.empty:
            yrs = pd.to_numeric(extrap["source_year"], errors="coerce").dropna()
            lines.append(
                f"Rows beyond the lineage's coverage: {len(extrap)}"
                + (f" (years {int(yrs.min())}-{int(yrs.max())})" if len(yrs) else "")
                + " — unit assumed unchanged; the relationship table does not "
                "describe these years."
            )
            lines.append("")
        # Where looking for an older district went wrong: either nothing
        # suitable was found and the figure was given up, or several fitted
        # and one was chosen. Both are judgement calls the reviewer inherits.
        issues = self.ancestor_issues
        if not issues.empty:
            counts = issues["ancestor_status"].value_counts().to_dict()
            lines.append(
                f"Ancestor-walk issues: {counts.get('no-ancestor', 0)} dropped "
                f"(not alive, no year-compatible ancestor), "
                f"{counts.get('ambiguous', 0)} ambiguous (earliest-created picked)"
            )
            for _, row in issues.head(20).iterrows():
                yr = row.get("source_year")
                yr = "" if yr is None or pd.isna(yr) else int(yr)
                lines.append(
                    f"  [{row['ancestor_status']}] {row['source_name']!r} ({yr})"
                    f"   {row['notes']}"
                )
            if len(issues) > 20:
                lines.append(f"  … and {len(issues) - 20} more")
            lines.append("")
        # Names nothing could be found for. Listed last but read first by
        # most people, with the nearest miss beside each one, since that is
        # usually enough to see what went wrong.
        unmatched = self.unmatched
        lines.append(f"Unmatched rows: {len(unmatched)}")
        if not unmatched.empty:
            for _, row in unmatched.sort_values("score", ascending=False).iterrows():
                score_repr = f"{row['score']:.2f}" if pd.notna(row["score"]) else "  -"
                lines.append(
                    f"  idx={row['source_idx']:<4}  src={row['source_name']!r:<35s}  "
                    f"best={row['best_candidate']!r} (score={score_repr})"
                )
        return "\n".join(lines)

    def write_review(self, path: Path | str) -> None:
        """Write :meth:`summary` text to a file."""
        Path(path).write_text(self.summary() + "\n")


# --- The matching pipeline ----------------------------------------------


def _pick_by_coarse(
    candidates: list[_LookupEntry],
    src_coarse: str | None,
    normalizer: Callable[[str], str],
) -> _LookupEntry | None:
    """From a list of homonym candidates, return the one whose coarse
    name normalizes to the source's coarse value, or ``None`` if no
    unique match exists.

    Returns:
        - The single matching entry if exactly one candidate's coarse
          normalizes to ``src_coarse``.
        - ``None`` if no candidates match, OR multiple match (ambiguous),
          OR the source row didn't carry a coarse value.

    The caller treats ``None`` as "couldn't disambiguate; fall through
    to fuzzy".
    """
    if not src_coarse:
        return None
    target = normalizer(src_coarse)
    if not target:
        return None
    matches = [c for c in candidates if c[2] and normalizer(c[2]) == target]
    if len(matches) == 1:
        return matches[0]
    return None


def _match_rows(
    source_rows: list[tuple[int, str, int | None, str | None]],
    snap_lookup: _Lookup,
    *,
    homonym_overrides: dict[int, tuple[str, str]] | None,
    manual_overrides: dict[str, str] | None,
    normalizer: Callable[[str], str],
    fuzzy_threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Run the 4-pass matcher (homonym → exact → manual → fuzzy → unmatched).

    Returns ``(rows, unresolved_homonyms)``. ``rows`` is the proposal
    list; ``unresolved_homonyms`` records every source row that
    exact/manual-matched a multi-candidate bucket without
    coarse-column resolution, so the caller can surface a single
    warning at the end. Each unresolved-homonym entry is a dict with
    ``source_idx``, ``source_name``, ``picked_unit_id``,
    ``alternative_unit_ids``.

    When ``snap_lookup[norm]`` has multiple candidates (a homonym):

      - If ``source_coarse`` resolves to exactly one of them, use it
        as an exact match (no warning).
      - Otherwise pick the *first* candidate AND record an unresolved-
        homonym entry. The row's ``notes`` field surfaces the
        ambiguity into the proposal CSV so it survives review.
    """
    homonyms = homonym_overrides or {}
    manuals = manual_overrides or {}

    # An override VALUE is a lineage *name*, not a unit_id — it is normalized
    # and looked up in the snapshot name index. Passing an id is a natural
    # guess (it was the first thing tried during the Philippines dry run) and
    # used to fail silently: the match count simply did not move, leaving the
    # user to wonder whether the override had been read at all. Check up front
    # and say so.
    dead = sorted(
        {
            f"{src!r} -> {tgt!r}"
            for src, tgt in manuals.items()
            if normalizer(str(tgt)) not in snap_lookup
        }
    )
    if dead:
        warnings.warn(
            f"{len(dead)} manual_overrides entry(ies) name a target that does "
            f"not exist in the snapshot and will have no effect: {dead[:5]}"
            + (f" (+{len(dead) - 5} more)" if len(dead) > 5 else "")
            + ". The VALUE must be a unit NAME as it appears in the lineage, "
            "not a unit_id — e.g. {'national capital region': 'Metro Manila'}, "
            "not {'national capital region': 'PH.ADM1.00044'}.",
            UserWarning,
            stacklevel=3,
        )

    matched_unit_ids: set[str] = set()
    unresolved: list[dict] = []

    def remaining() -> _Lookup:
        # The districts still up for grabs when guessing at leftover names.
        # A confident match is allowed to claim the same district twice —
        # two pieces of a map can legitimately be the same district — but a
        # guess is not, because a guess competing against a name we already
        # matched properly is far more likely to be the wrong one.
        out: _Lookup = {}
        for k, candidates in snap_lookup.items():
            keep = [c for c in candidates if c[0] not in matched_unit_ids]
            if keep:
                out[k] = keep
        return out

    def _resolve_bucket(
        src_idx: int, src_name: str, candidates: list[_LookupEntry],
        src_coarse: str | None,
    ) -> tuple[_LookupEntry, str]:
        """Pick one candidate from a bucket; return (picked, notes).

        notes is a non-empty string only when the bucket had >1
        candidate and coarse-column resolution failed — i.e. a silent
        first-wins pick the reviewer should know about.
        """
        # Several districts can share a name. If the source told us which
        # state the row belongs to, that settles it.
        picked = _pick_by_coarse(candidates, src_coarse, normalizer)
        if picked is not None:
            return picked, ""
        # Only one district goes by this name, so there is nothing to settle.
        if len(candidates) <= 1:
            return candidates[0], ""

        # Genuinely ambiguous. Take the first, but say so loudly: this is a
        # coin toss the reviewer has to see, and the note tells them the two
        # ways to settle it. Silently picking one is how data ends up filed
        # under a district on the other side of the country.
        first = candidates[0]
        alt_ids = [c[0] for c in candidates if c[0] != first[0]]
        unresolved.append({
            "source_idx": src_idx,
            "source_name": src_name,
            "picked_unit_id": first[0],
            "alternative_unit_ids": alt_ids,
        })
        notes = (
            f"homonym: {len(candidates)} candidates {[c[0] for c in candidates]}; "
            f"picked first; pass coarse_column or homonym_overrides[{src_idx}] to disambiguate"
        )
        return first, notes

    results: list[dict] = []
    pending_fuzzy: list[tuple[int, str, str, int | None, str | None]] = []

    # Names are matched in descending order of how much we trust the answer:
    # a decision a human already made, then an exact name, then a translation
    # a human supplied, and only then a guess. The first three are settled
    # here; anything that reaches the bottom is set aside for the guessing
    # pass below, so that guessing happens only once every confident match
    # has staked its claim.
    for src_idx, src_name, src_year, src_coarse in source_rows:
        norm = normalizer(src_name)

        # A reviewer already looked at this exact row and said which district
        # it is. Nothing outranks that.
        if src_idx in homonyms:
            sid, sname = homonyms[src_idx]
            matched_unit_ids.add(sid)
            results.append(
                _row(src_idx, src_name, src_year, src_coarse, sid, sname, "homonym", 1.0, "")
            )
            continue

        # The name matches a district outright, once both have been tidied
        # into a comparable form.
        if norm in snap_lookup:
            candidates = snap_lookup[norm]
            picked, notes = _resolve_bucket(src_idx, src_name, candidates, src_coarse)
            sid, sname, _ = picked
            matched_unit_ids.add(sid)
            results.append(
                _row(src_idx, src_name, src_year, src_coarse, sid, sname, "exact", 1.0, "", notes)
            )
            continue

        # The user gave us a translation for this name. Look up what they
        # said it should be, and take it only if that actually names a
        # district — a translation pointing at nothing is caught earlier.
        if norm in manuals:
            override_norm = normalizer(manuals[norm])
            if override_norm in snap_lookup:
                candidates = snap_lookup[override_norm]
                picked, notes = _resolve_bucket(src_idx, src_name, candidates, src_coarse)
                sid, sname, _ = picked
                matched_unit_ids.add(sid)
                results.append(
                    _row(
                        src_idx, src_name, src_year, src_coarse,
                        sid, sname, "manual", 1.0, "", notes,
                    )
                )
                continue

        pending_fuzzy.append((src_idx, src_name, norm, src_year, src_coarse))

    # Now the leftovers. For each, find the closest district name still
    # unclaimed and score how alike they are — this is what catches spelling
    # differences and transliterations between two sources.
    for src_idx, src_name, norm, src_year, src_coarse in pending_fuzzy:
        best_score = 0.0
        best_match: _LookupEntry | None = None
        for snap_norm, candidates in remaining().items():
            score = _similarity(norm, snap_norm)
            if score > best_score:
                # If that name belongs to several districts, prefer the one
                # in the right state, as above.
                picked = _pick_by_coarse(candidates, src_coarse, normalizer) or candidates[0]
                best_score = score
                best_match = picked
        # Close enough is a match; anything less is reported as unmatched
        # along with what it nearly matched, so a reviewer can judge it. The
        # package never quietly accepts a weak guess.
        if best_score >= fuzzy_threshold and best_match is not None:
            sid, sname, _ = best_match
            matched_unit_ids.add(sid)
            results.append(
                _row(src_idx, src_name, src_year, src_coarse, sid, sname, "fuzzy", best_score, "")
            )
        else:
            best_name = best_match[1] if best_match else ""
            score = best_score if best_match else float("nan")
            results.append(
                _row(src_idx, src_name, src_year, src_coarse, "", "", "unmatched", score, best_name)
            )

    return results, unresolved


def _row(
    src_idx: int,
    src_name: str,
    src_year: int | None,
    src_coarse: str | None,
    unit_id: str,
    snap_name: str,
    method: str,
    score: float,
    best_candidate: str,
    notes: str = "",
) -> dict:
    """Standard proposal-row shape. Centralized so the columns stay aligned.

    ``source_coarse`` records *which* upper-admin value produced this row. It
    is what makes homonyms distinguishable: when a proposal is keyed per
    ``(name, coarse)``, two same-named rows are otherwise identical both to a
    human reviewing the CSV and to :func:`attach_stats_ids` joining it back.
    """
    return {
        "source_idx": src_idx,
        "source_name": src_name,
        "source_year": src_year if src_year is not None else pd.NA,
        "source_coarse": src_coarse if src_coarse is not None else pd.NA,
        "proposed_unit_id": unit_id,
        "proposed_name": snap_name,
        "method": method,
        "score": score,
        "best_candidate": best_candidate,
        # Outcome of the year-aware ancestor walk. "not-needed" for every
        # row the walk didn't touch (all rows, in non-year-aware modes).
        # See :func:`_apply_ancestor_walk` for the other three values.
        "ancestor_status": "not-needed",
        "notes": notes,
    }


# --- Public API: propose_* ----------------------------------------------


_GdfOrPath = Union[gpd.GeoDataFrame, Path, str]
_DfOrPath = Union[pd.DataFrame, Path, str]


def _coerce_gdf(shapefile: _GdfOrPath) -> gpd.GeoDataFrame:
    if isinstance(shapefile, gpd.GeoDataFrame):
        return shapefile
    return gpd.read_file(Path(shapefile))


def _coerce_df(stats: _DfOrPath) -> pd.DataFrame:
    if isinstance(stats, pd.DataFrame):
        return stats
    return pd.read_csv(Path(stats))


def propose_shapefile_mapping(
    shapefile: _GdfOrPath,
    graph: LineageGraph,
    name_column: str,
    *,
    coarse_column: str | None = None,
    year: int | None = None,
    year_range: Iterable[int] | None = None,
    baseline: pd.DataFrame | None = None,
    name_change_log: pd.DataFrame | None = None,
    manual_overrides: dict[str, str] | None = None,
    homonym_overrides: dict[int, tuple[str, str]] | None = None,
    normalizer: Callable[[str], str] = normalize_name,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    sketchy_threshold: float = DEFAULT_SKETCHY_THRESHOLD,
) -> MatchProposal:
    """Propose canonical ``unit_id`` for every feature in a shapefile.

    Year selection:

      - ``year=<int>`` — use that exact year's snapshot.
      - ``year=None`` (default) and ``year_range`` provided — infer
        the year whose snapshot names best match the shapefile (exact
        + fuzzy via :func:`_infer_year_by_name`). The
        :class:`Lineage` wrapper passes ``year_range=self.years``
        automatically, so callers going through ``ln.propose_shapefile_mapping``
        get inference for free.
      - ``year=None`` and no ``year_range`` — falls back to
        ``graph.max_event_year + 1`` (the modern-day snapshot).

    The override dicts (``manual_overrides``, ``homonym_overrides``)
    are how country-specific quirks enter the pipeline — they remain
    user data, not bundled with the package.

    ``manual_overrides`` maps a **normalized source name** to a **lineage
    name**, not to a unit_id::

        {"national capital region": "Metro Manila"}   # correct
        {"national capital region": "PH.ADM1.00044"}  # WRONG — no effect

    The value is normalized and looked up in the snapshot's name index, so an
    id never resolves. Passing one used to change nothing and say nothing; an
    override whose target is not in the snapshot now raises a ``UserWarning``
    naming the offending entries.

    ``coarse_column`` (optional) names a shapefile column carrying the
    upper-admin name (state, province, region). When provided, exact-
    match homonyms are disambiguated automatically: a feature whose
    name matches two candidates picks the one whose lineage
    ``coarse_name`` (state) matches the shapefile's coarse value.
    When the shapefile has homonyms but no ``coarse_column`` was
    passed, the matcher emits a UserWarning naming the affected rows
    and the candidate unit_ids so the reviewer can act on them.
    """
    gdf = _coerce_gdf(shapefile)
    if name_column not in gdf.columns:
        raise KeyError(
            f"name_column {name_column!r} not present in shapefile columns: {list(gdf.columns)}"
        )
    if coarse_column is not None and coarse_column not in gdf.columns:
        raise KeyError(
            f"coarse_column {coarse_column!r} not present in shapefile columns: "
            f"{list(gdf.columns)}"
        )

    # A map file rarely says which year it depicts, but the answer matters:
    # match against the wrong year and districts that hadn't been created
    # yet, or had already gone, are compared against ones that had.
    #
    # So if the caller didn't say, work it out — try each year in turn and
    # keep whichever one's district list the map's names agree with best.
    inferred_year: int | None = None
    inference_mismatches: dict[int, int] | None = None
    if year is None:
        if year_range is not None:
            source_names_normed = {normalizer(str(n)) for n in gdf[name_column]}
            inferred_year, inference_mismatches = _infer_year_by_name(
                source_names_normed,
                graph,
                year_range,
                baseline=baseline,
                normalizer=normalizer,
                fuzzy_threshold=fuzzy_threshold,
            )
            year = inferred_year
            _LOGGER.info(
                "Inferred shapefile year %d (mismatch=%d); "
                "pass year=<int> to override.",
                inferred_year,
                inference_mismatches[inferred_year],
            )
        else:
            # No range to search, so fall back to the year after the last
            # recorded change — the present-day picture.
            #
            # Some countries have never changed at all. Japan's 47
            # prefectures and Brunei's 4 districts have been fixed for the
            # whole period covered here, so there is no "last change" to
            # count from, and every year looks the same anyway. Use the year
            # the baseline describes instead.
            if graph.max_event_year is None:
                year = (
                    int(baseline["year"].min())
                    if baseline is not None and "year" in baseline.columns
                    else 0
                )
            else:
                year = graph.max_event_year + 1

    # Build the list of names to match against: every district that existed
    # in that year, under every name it has ever gone by.
    snap_lookup = _build_snapshot_lookup(
        graph,
        year,
        baseline=baseline,
        name_change_log=name_change_log,
        normalizer=normalizer,
    )

    # Pull out each shape's name, along with the state it sits in if the map
    # records one — that is what separates two districts sharing a name.
    if coarse_column is not None:
        coarses = gdf[coarse_column].astype(str).tolist()
    else:
        coarses = [None] * len(gdf)
    source_rows = [
        (int(idx), str(name), None, coarse)
        for idx, name, coarse in zip(gdf.index, gdf[name_column], coarses)
    ]
    # Run every shape's name down the matching ladder. This is where the
    # actual work happens; everything above prepares its two inputs and
    # everything below packages what it returns.
    rows, unresolved = _match_rows(
        source_rows,
        snap_lookup,
        homonym_overrides=homonym_overrides,
        manual_overrides=manual_overrides,
        normalizer=normalizer,
        fuzzy_threshold=fuzzy_threshold,
    )

    # Some names matched several districts with nothing to separate them, and
    # the first was taken. That is a coin toss deciding where a district's
    # data lands, so warn rather than leave it in a column nobody opens — and
    # spell out both ways to settle it, with the offending rows filled in
    # ready to paste back.
    if unresolved:
        lines = [
            f"{len(unresolved)} homonym(s) without coarse disambiguation "
            f"(silently picked first; review proposal `notes` column):"
        ]
        for u in unresolved:
            lines.append(
                f"  source_idx={u['source_idx']}  {u['source_name']!r}  "
                f"-> picked {u['picked_unit_id']}; also matched "
                f"{u['alternative_unit_ids']}"
            )
        lines.append("To disambiguate, re-run with one of:")
        lines.append("  (a) coarse_column='<your shapefile state/province column>'")
        lines.append(
            "  (b) homonym_overrides={"
            + ", ".join(f"{u['source_idx']}: (..., ...)" for u in unresolved[:3])
            + (", ...}" if len(unresolved) > 3 else "}")
        )
        warnings.warn("\n".join(lines), UserWarning, stacklevel=2)

    # Hand back the proposals in the map's own row order, so a reviewer
    # opening the CSV alongside the map file reads them side by side.
    proposals = pd.DataFrame(rows).sort_values("source_idx").reset_index(drop=True)
    # The district names travel with the proposal so a reviewer correcting a
    # row can see what each candidate is actually called, rather than being
    # handed a bare identifier to look up themselves.
    return MatchProposal(
        proposals=proposals,
        fuzzy_threshold=fuzzy_threshold,
        sketchy_threshold=sketchy_threshold,
        snapshot_unit_names={
            uid: name
            for candidates in snap_lookup.values()
            for uid, name, _ in candidates
        },
        inferred_year=inferred_year,
        year_inference_mismatches=inference_mismatches,
    )


def propose_stats_mapping(
    stats: _DfOrPath,
    graph: LineageGraph,
    name_column: str,
    *,
    coarse_column: str | None = None,
    year_column: str | None = None,
    year_range: range | None = None,
    year_aware: bool = False,
    baseline: pd.DataFrame | None = None,
    name_change_log: pd.DataFrame | None = None,
    manual_overrides: dict[str, str] | None = None,
    normalizer: Callable[[str], str] = normalize_name,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    sketchy_threshold: float = DEFAULT_SKETCHY_THRESHOLD,
) -> MatchProposal:
    """Propose canonical ``unit_id`` for every row in a long-form stats table.

    Distinct from shapefile matching in two ways:

    1. Many rows per name. Names typically repeat across (year,
       season, variable). The proposal is keyed by ``(source_name)``
       — one mapping row per *unique name* in the stats file, not per
       data row. The caller applies the mapping back to the full
       stats DataFrame.
    2. Year-tolerant lookup. The lookup is built from a *union* of
       snapshots across the year range, so a unit's old name and new
       name both resolve to the same unit_id.

    ``coarse_column`` (optional) names a stats column carrying the
    upper-admin name. When two units share a normalized name (a
    homonym), the coarse value is used to disambiguate per (name,
    coarse) pair: each unique (source_name, source_coarse)
    combination becomes one proposal row instead of just per name.

    ``year_aware=True`` (requires ``year_column``) switches to *year-
    correct* matching: each name is matched against the snapshot for its
    own year, and a pick that wasn't alive that year is walked up the
    lineage to its year-compatible ancestor (the behaviour the India
    matcher uses so, e.g., a name reported before a split resolves to the
    pre-split parent, not a post-split child). The proposal is then keyed
    per unique (name, year[, coarse]) and carries ``source_year``; pass
    the same ``year_column`` to :func:`attach_stats_ids` to join per
    (name, year).

    Any such ancestor remap — data aligned to an OLDER unit than the name
    first resolved to — is flagged loudly: ``method`` gains an
    ``+ancestor_walk`` suffix, the ``notes`` column records the original id
    and the remap, and :attr:`MatchProposal.remapped` /
    :meth:`MatchProposal.summary` list every one, so a reviewer can confirm
    nothing is mis-attached to a defunct unit.

    Old↔new name aliasing comes from **both** sources: the
    ``name_change_log`` argument and NameChange events recorded in the lineage
    itself. The two encodings of the same fact behave identically, in both
    directions — a historical name matched against a modern year and a modern
    name matched against a historical year both resolve. (Until 2026-07-27
    only the log aliased, so a country following the canonical schema matched
    worse than one that also kept a separate log.)

    Aliasing is not a licence to match anything: it adds the unit's other
    known names to the lookup, so an unrelated name still comes back
    ``unmatched`` rather than being absorbed.
    """
    df = _coerce_df(stats)
    if name_column not in df.columns:
        raise KeyError(
            f"name_column {name_column!r} not present in stats columns: {list(df.columns)}"
        )
    if coarse_column is not None and coarse_column not in df.columns:
        raise KeyError(
            f"coarse_column {coarse_column!r} not present in stats columns: "
            f"{list(df.columns)}"
        )

    # Work out which years to consider. The statistics themselves are the
    # best guide, since matching against years nobody reported is wasted
    # effort; failing that, fall back to the span the lineage describes.
    if year_range is None:
        if year_column is not None and year_column in df.columns:
            yrs = df[year_column].dropna().astype(int)
            if not yrs.empty:
                year_range = range(int(yrs.min()), int(yrs.max()) + 1)
        if year_range is None:
            year_range = range(graph.min_event_year, graph.max_event_year + 2)

    # Two ways to do this. The year-aware path matches each figure against
    # the districts of its own year and is the right choice for a country
    # whose boundaries moved a lot; the simpler path below pools all years
    # together. The year-aware one needs to know which column holds the
    # year, so refuse rather than silently falling back to the other.
    if year_aware:
        if year_column is None or year_column not in df.columns:
            raise ValueError(
                "propose_stats_mapping(year_aware=True) requires a year_column "
                "present in the stats so each name is matched against the "
                "snapshot for its own year."
            )
        return _propose_stats_year_aware(
            df,
            graph,
            name_column,
            coarse_column=coarse_column,
            year_column=year_column,
            year_range=year_range,
            baseline=baseline,
            name_change_log=name_change_log,
            manual_overrides=manual_overrides,
            normalizer=normalizer,
            fuzzy_threshold=fuzzy_threshold,
            sketchy_threshold=sketchy_threshold,
        )

    # Statistics span many years at once, so unlike a map there is no single
    # year to match against. Pool every district that existed at any point in
    # the period and match against all of them together. That is the
    # trade-off this path makes: it will find a name from any era, but it
    # cannot tell that a name belonged to a different district back then.
    # Callers who need that ask for the year-aware path above.
    snap_lookup = _build_union_lookup(
        graph,
        year_range,
        baseline=baseline,
        name_change_log=name_change_log,
        normalizer=normalizer,
    )

    # A statistics file repeats the same district for every year and crop, so
    # match each distinct name once rather than tens of thousands of times.
    # Where the file names the state too, treat a name in two states as two
    # separate things to look up, so a reviewer sees one row for each.
    if coarse_column is None:
        # No state recorded, so each distinct name is looked up once and two
        # districts sharing a name cannot be told apart.
        unique_names = df[name_column].dropna().astype(str).unique()
        source_rows = [(i, name, None, None) for i, name in enumerate(unique_names)]
    else:
        # With the state, the same name in two states becomes two separate
        # things to look up, and the reviewer gets a row for each.
        pairs = (
            df[[name_column, coarse_column]]
            .dropna(subset=[name_column])
            .astype(str)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        source_rows = [
            (i, row[name_column], None, row[coarse_column])
            for i, row in pairs.iterrows()
        ]

    # Send them down the same matching ladder the map path uses. Reviewer
    # decisions per row are not offered here: a statistics file has no stable
    # row numbering to hang them off, so ambiguity is settled by the state
    # column or reported.
    rows, unresolved = _match_rows(
        source_rows,
        snap_lookup,
        homonym_overrides=None,
        manual_overrides=manual_overrides,
        normalizer=normalizer,
        fuzzy_threshold=fuzzy_threshold,
    )

    # Names that matched more than one district and couldn't be told apart
    # were settled by taking the first. Warn loudly: this is the failure mode
    # that quietly files one district's harvest under another's.
    if unresolved:
        lines = [
            f"{len(unresolved)} homonym(s) in stats without coarse disambiguation "
            f"(silently picked first; review proposal `notes` column):"
        ]
        for u in unresolved:
            lines.append(
                f"  source_idx={u['source_idx']}  {u['source_name']!r}  "
                f"-> picked {u['picked_unit_id']}; also matched "
                f"{u['alternative_unit_ids']}"
            )
        lines.append(
            "To disambiguate, re-run with coarse_column='<your stats "
            "state/province column>'."
        )
        warnings.warn("\n".join(lines), UserWarning, stacklevel=2)

    proposals = pd.DataFrame(rows).sort_values("source_idx").reset_index(drop=True)
    return MatchProposal(
        proposals=proposals,
        fuzzy_threshold=fuzzy_threshold,
        sketchy_threshold=sketchy_threshold,
        snapshot_unit_names={
            uid: name
            for candidates in snap_lookup.values()
            for uid, name, _ in candidates
        },
    )


def _apply_ancestor_walk(
    row: dict,
    year: int,
    graph: LineageGraph,
    year_alive_index: dict[str, set[int]],
    parents_map: dict[str, set[str]],
) -> None:
    """If ``row``'s picked unit_id wasn't alive in ``year``, remap it to
    the year-compatible ancestor (mutates ``row`` in place).

    Records the outcome in ``ancestor_status``:

    ``not-needed``
        The pick was already alive that year; nothing to do.
    ``ok``
        Walked to exactly one year-compatible ancestor.
    ``ambiguous``
        Several year-compatible ancestors; the earliest-created one was
        chosen and all candidates are listed in ``notes``. Deterministic,
        but a reviewer should confirm the pick.
    ``no-ancestor``
        The unit was not alive that year and no year-compatible ancestor
        exists. **The row is dropped to unmatched.** Keeping the id would
        attach data to a unit that did not exist — a wrong answer nobody
        sees. An empty cell the reviewer fills is strictly better; the
        rejected id is preserved in ``notes`` and ``best_candidate``.
    """
    # Nothing to reconsider if the name didn't match anything to begin with.
    uid = row["proposed_unit_id"]
    if not uid or row["method"] == "unmatched":
        return
    # The district we landed on did exist in this year, so the match stands.
    if _year_compatible(uid, year, year_alive_index):
        return

    # It didn't exist yet. This is the ordinary case for old statistics: a
    # figure reported in 1995 under a name that today belongs to a district
    # created in 2011. Walk back up the family tree looking for the district
    # that held this territory at the time.
    candidates = _walk_year_compat_ancestors(uid, year, parents_map, year_alive_index)
    if not candidates:
        # No ancestor fits either, so we have no idea where this figure
        # belongs. Throw the match away rather than keep it: attaching data
        # to a district that did not exist is a wrong answer nobody would
        # ever notice, while an empty cell is a question the reviewer
        # answers. The rejected guess is kept in the notes for them.
        rejected_name = row["proposed_name"]
        drop = (
            f"{ANCESTOR_WALK_TAG}: {uid} ({rejected_name}) not alive in {year} and "
            f"no year-compatible ancestor exists; dropped to unmatched"
        )
        row["ancestor_status"] = "no-ancestor"
        row["notes"] = f"{row['notes']}; {drop}" if row.get("notes") else drop
        row["best_candidate"] = rejected_name
        row["proposed_unit_id"] = ""
        row["proposed_name"] = ""
        row["method"] = "unmatched"
        return
    # Several ancestors can fit — territory is often reorganised more than
    # once. Take the oldest, since that is the one that held the territory
    # longest ago and so is likeliest to be what an old figure referred to.
    chosen = _earliest_created_ancestor(candidates, year_alive_index)
    if chosen == uid:
        return
    # More than one fit means the choice was a judgement. Flag it so the
    # reviewer sees which alternatives were passed over.
    if len(candidates) > 1:
        row["ancestor_status"] = "ambiguous"
        row["notes"] = (
            f"{row['notes']}; " if row.get("notes") else ""
        ) + f"{ANCESTOR_WALK_TAG}: {len(candidates)} year-compatible ancestors " \
            f"{sorted(candidates)}; picked earliest-created {chosen}"
    else:
        row["ancestor_status"] = "ok"
    note = (
        f"{ANCESTOR_WALK_TAG}: {uid} not alive in {year}; "
        f"remapped to year-compatible ancestor {chosen}"
    )
    # Point the row at the older district instead, and relabel it with what
    # that district was called back then rather than its modern name — the
    # reviewer is checking a historical figure and needs the historical name
    # to recognise it. Notes are appended to, never replaced, so an earlier
    # warning on this row survives.
    row["proposed_unit_id"] = chosen
    row["proposed_name"] = _name_of_unit_in_year(graph, chosen, year) or row["proposed_name"]
    row["notes"] = f"{row['notes']}; {note}" if row.get("notes") else note
    # Make the remap visible in the method column too, so it can't pass as a
    # silent high-confidence match when the review CSV is scanned by method.
    if ANCESTOR_WALK_SUFFIX not in row["method"]:
        row["method"] = f"{row['method']}+{ANCESTOR_WALK_SUFFIX}"


def _propose_stats_year_aware(
    df: pd.DataFrame,
    graph: LineageGraph,
    name_column: str,
    *,
    coarse_column: str | None,
    year_column: str,
    year_range: range,
    baseline: pd.DataFrame | None,
    name_change_log: pd.DataFrame | None,
    manual_overrides: dict[str, str] | None,
    normalizer: Callable[[str], str],
    fuzzy_threshold: float,
    sketchy_threshold: float,
) -> MatchProposal:
    """Year-aware backend for :func:`propose_stats_mapping`.

    Two behaviours ported from the India matcher:

    1. **Per-year matching.** Each (name, year) is matched against the
       snapshot for *that* year first; the union of all years fills gaps
       for names absent from that year's snapshot. A name that maps to
       different ids in different years therefore resolves correctly per
       year instead of collapsing to a single id.
    2. **Ancestor walk.** After a pick, if the unit_id wasn't alive in
       the row's year, walk up the lineage's parent chain to the
       year-compatible ancestor (earliest-created on ties).

    Keyed per unique (name, year[, coarse]); rows carry ``source_year``.
    """
    # The alive-index and union must span BOTH the lineage's event window and
    # every year the stats actually contain.
    #
    # The lineage side keeps units created or ended outside the stats span
    # visible to the ancestor walk. The stats side matters just as much and
    # was missing: a country whose statistics outrun its relationship table
    # (Philippines reports to 2024 from a lineage ending 2013) had no
    # alive-index entry for those later years, so every unit read as "not
    # alive", the walk found no ancestor — because no ancestor is alive in a
    # year the index does not cover either — and the row was dropped. A unit
    # is presumed to persist past the last event; absence of evidence of
    # change is not evidence of death.
    lo, hi = year_range.start, year_range.stop
    if not graph.events.empty:
        lo = min(lo, graph.min_event_year)
        hi = max(hi, graph.max_event_year + 2)
    stats_years = pd.to_numeric(df[year_column], errors="coerce").dropna()
    if len(stats_years):
        lo = min(lo, int(stats_years.min()))
        hi = max(hi, int(stats_years.max()) + 1)
    full_range = range(lo, hi)

    # Build the district list for every year up front. Everything downstream
    # is derived from these — the pooled list used as a fallback, and the
    # record of which years each district was around for — so the lineage is
    # read once per year rather than once per name looked up.
    per_year_lookup: dict[int, _Lookup] = {
        y: _build_snapshot_lookup(
            graph,
            y,
            baseline=baseline,
            name_change_log=name_change_log,
            normalizer=normalizer,
        )
        for y in full_range
    }
    # Three things are derived from those lists, each answering a different
    # question later on: every district of any year pooled together, used
    # when a year's own list doesn't recognise a name; which years each
    # district was around for, used to tell whether a match is possible; and
    # who each district was carved out of, used to walk back when it isn't.
    union = _merge_lookups(per_year_lookup[y] for y in full_range)
    year_alive_index = _year_alive_index_from_lookups(per_year_lookup)
    parents_map = _parents_map_from_graph(graph)

    merged_cache: dict[int, _Lookup] = {}

    def _merged_for_year(y: int) -> _Lookup:
        """Year ``y``'s lookup with union candidates filling absent names."""
        # That year's districts first, then every other district as a
        # fallback for names the year itself doesn't recognise. So a name is
        # answered by the right year where possible, and still found rather
        # than dropped where not — the ancestor walk sorts out the rest.
        m = merged_cache.get(y)
        if m is None:
            m = dict(per_year_lookup.get(y, {}))
            for norm, cands in union.items():
                m.setdefault(norm, cands)
            merged_cache[y] = m
        return m

    # Reduce the statistics to the distinct things that need looking up: each
    # district name, in each year it was reported, and in each state if the
    # file says. A national file repeats the same name for every crop and
    # season, so this turns hundreds of thousands of rows into thousands.
    subset_cols = [name_column, year_column] + (
        [coarse_column] if coarse_column is not None else []
    )
    sub = df[subset_cols].dropna(subset=[name_column, year_column]).copy()
    # Rows whose year cannot be read as a number are set aside — without a
    # year there is no way to know which districts to compare against.
    sub[year_column] = pd.to_numeric(sub[year_column], errors="coerce")
    sub = sub.dropna(subset=[year_column])
    sub[year_column] = sub[year_column].astype(int)
    sub[name_column] = sub[name_column].astype(str)
    if coarse_column is not None:
        sub[coarse_column] = sub[coarse_column].astype(str)
    # Sorted by year then name, so the reviewed file reads in a sensible
    # order and comes out the same way on every run.
    uniq = (
        sub.drop_duplicates(subset=subset_cols)
        .sort_values([year_column, name_column])
        .reset_index(drop=True)
    )
    # Each row's position here becomes its reference number in the proposal,
    # which is how a reviewer's edits find their way back to the right row.
    uniq = uniq.reset_index(drop=True)

    # This is what makes this path different from the plain one: rather than
    # matching every name against one pooled list of districts, work through
    # the years one at a time and match each year's names against the
    # districts that existed in that year.
    all_rows: list[dict] = []
    all_unresolved: list[dict] = []
    for y, grp in uniq.groupby(year_column, sort=True):
        year = int(y)
        # Each name carries its year along with it, so the record of what was
        # decided says which year's districts it was decided against — and so
        # the reviewed file can be joined back on year as well as name.
        source_rows = [
            (
                int(idx),
                row[name_column],
                year,
                row[coarse_column] if coarse_column is not None else None,
            )
            for idx, row in grp.iterrows()
        ]
        rows_y, unresolved_y = _match_rows(
            source_rows,
            _merged_for_year(year),
            homonym_overrides=None,
            manual_overrides=manual_overrides,
            normalizer=normalizer,
            fuzzy_threshold=fuzzy_threshold,
        )
        # Matching by name can still land on a district that didn't exist
        # yet, because the list carries historical names too. This is the
        # correction: move each figure back to whichever district actually
        # held that ground in the year the figure is from.
        for row in rows_y:
            _apply_ancestor_walk(row, year, graph, year_alive_index, parents_map)
        all_rows.extend(rows_y)
        all_unresolved.extend(unresolved_y)

    # Same ambiguous-name warning as the other paths, gathered across all
    # years so a name that is ambiguous in twenty of them is reported once
    # per year rather than being lost in a per-year message.
    if all_unresolved:
        lines = [
            f"{len(all_unresolved)} homonym(s) in stats without coarse "
            f"disambiguation (silently picked first; review proposal `notes` "
            f"column):"
        ]
        for u in all_unresolved:
            lines.append(
                f"  source_idx={u['source_idx']}  {u['source_name']!r}  "
                f"-> picked {u['picked_unit_id']}; also matched "
                f"{u['alternative_unit_ids']}"
            )
        lines.append(
            "To disambiguate, re-run with coarse_column='<your stats "
            "state/province column>'."
        )
        warnings.warn("\n".join(lines), UserWarning, stacklevel=3)

    proposals = pd.DataFrame(all_rows).sort_values("source_idx").reset_index(drop=True)
    # Years past the lineage's last event are not described by it. Matching
    # there assumes no further administrative change — reasonable, but an
    # assumption, so say which rows depend on it rather than presenting them
    # as equally attested.
    if not graph.events.empty and len(proposals):
        # The lineage stops describing the country after its last recorded
        # change. Statistics often run past that, so mark those rows: their
        # districts are assigned on the assumption nothing changed since,
        # which is usually right and occasionally badly wrong.
        described_through = int(graph.max_event_year) + 1
        yrs = pd.to_numeric(proposals["source_year"], errors="coerce")
        beyond = yrs.notna() & (yrs > described_through)
        if beyond.any():
            lo, hi = int(yrs[beyond].min()), int(yrs[beyond].max())
            # Add the note without wiping whatever the row already said.
            notes = proposals["notes"].astype(str)
            tag = proposals.loc[beyond].apply(
                lambda r: (
                    f"{EXTRAPOLATED_TAG}: {int(r['source_year'])} is past the "
                    f"lineage's last event ({int(graph.max_event_year)}); unit "
                    "assumed unchanged"
                ),
                axis=1,
            )
            proposals.loc[beyond, "notes"] = [
                f"{n}; {x}" if n else x
                for n, x in zip(notes[beyond], tag)
            ]
            warnings.warn(
                f"{int(beyond.sum())} of {len(proposals)} mapping row(s) fall in "
                f"years {lo}-{hi}, past the lineage's last recorded event "
                f"({int(graph.max_event_year)}). Their unit_ids are assigned on "
                "the assumption that no further administrative changes "
                "occurred — the relationship table does not cover those years. "
                "Check whether a newer one exists; affected rows are tagged "
                f"{EXTRAPOLATED_TAG!r} in the proposal's `notes` column.",
                UserWarning,
                stacklevel=3,
            )

    return MatchProposal(
        proposals=proposals,
        fuzzy_threshold=fuzzy_threshold,
        sketchy_threshold=sketchy_threshold,
        snapshot_unit_names={
            uid: name
            for candidates in union.values()
            for uid, name, _ in candidates
        },
    )


# --- Public API: attach back ---------------------------------------------


def read_mapping(path: Path | str) -> pd.DataFrame:
    """Read a (possibly user-edited) mapping CSV back into a DataFrame.

    Required columns: ``source_name``, ``proposed_unit_id``. Other
    columns (method, score, etc.) are preserved but not required —
    convenient for users who hand-edit the CSV and only care about
    the name → unit_id mapping.

    The ``source_idx`` column (positional index in the original
    source) is preserved when present and used by
    :func:`attach_shapefile_ids` / :func:`attach_stats_ids` to do a
    position-based join. Position-based join is homonym-safe; name-
    based join is a fallback for hand-built mappings that don't carry
    source_idx.
    """
    # Only two columns are genuinely required: the name that was looked up
    # and the district a reviewer settled on. Everything else the proposal
    # wrote out is there to help them decide and can be deleted freely, so
    # check for those two and accept whatever else survived editing.
    df = pd.read_csv(Path(path))
    required = {"source_name", "proposed_unit_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"mapping CSV missing required columns: {sorted(missing)}. "
            f"Have: {list(df.columns)}"
        )
    return df


def attach_shapefile_ids(
    shapefile: _GdfOrPath,
    mapping: pd.DataFrame | Path | str,
    name_column: str,
    *,
    id_column: str = "unit_id",
    coarse_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Attach a canonical ``unit_id`` column to a shapefile via a mapping.

    Two join strategies, in priority order:

    1. **Positional** — if ``mapping`` carries a ``source_idx``
       column (proposals from :func:`propose_shapefile_mapping`
       always do), each mapping row attaches to the shapefile row at
       the same positional index. Homonym-safe: two features with
       the same name keep their distinct proposed unit_ids.

    2. **Name-based** — fallback when ``source_idx`` is absent
       (e.g., a user-built mapping written by hand). Lossy for
       homonyms; the last name in the mapping wins.

    Rows whose source_name doesn't appear in the mapping get
    ``unit_id = NaN``. Rows whose mapping has an empty
    ``proposed_unit_id`` (the user left an unmatched row blank during
    review) also get ``NaN`` — both cases surface in the resulting
    GeoDataFrame as missing IDs that the rest of the pipeline will
    reject loudly.

    The shapefile MUST be in its original row order when this is
    called (positional join only works if the indices match the
    proposal's). Reordering inside the mapping CSV is fine; reordering
    inside the shapefile (e.g., resaving via QGIS) is not.
    """
    gdf = _coerce_gdf(shapefile).copy()
    if isinstance(mapping, (str, Path)):
        mapping = read_mapping(mapping)

    # The reviewed file normally remembers which row of the map each
    # decision was about, so match them up by position. That is exact even
    # when two districts share a name, because row 12 is row 12 regardless
    # of what it is called.
    if "source_idx" in mapping.columns:
        idx_to_id = dict(
            zip(mapping["source_idx"].astype(int), mapping["proposed_unit_id"])
        )
        gdf[id_column] = [idx_to_id.get(i) for i in range(len(gdf))]
    else:
        # A hand-built file may not have that column, leaving only the name
        # to go on. That cannot tell two same-named districts apart, so if
        # any names repeat, say so rather than quietly merging them.
        # A state column settles homonyms that a name alone cannot: two
        # districts called Bilaspur are one in Himachal Pradesh and one in
        # Chhattisgarh, and joining on name alone silently gives both the
        # same id -- which does not merely mislabel the polygon, it dissolves
        # it into the wrong stable group 1,100 km away.
        has_coarse = (
            "source_coarse" in mapping.columns
            and mapping["source_coarse"].notna().any()
        )
        use_coarse = (
            has_coarse and coarse_column is not None and coarse_column in gdf.columns
        )
        if use_coarse:
            key = list(
                zip(mapping["source_name"].astype(str),
                    mapping["source_coarse"].astype(str))
            )
            pair_to_id = dict(zip(key, mapping["proposed_unit_id"]))
            gdf[id_column] = [
                pair_to_id.get((str(n), str(c)))
                for n, c in zip(gdf[name_column], gdf[coarse_column])
            ]
        else:
            if mapping["source_name"].duplicated().any():
                import warnings as _w
                hint = (
                    " The mapping carries source_coarse: pass "
                    "coarse_column=<your state/province column> to use it."
                    if has_coarse else
                    " Regenerate the mapping via propose_shapefile_mapping() "
                    "to get a source_idx column, or hand-edit one in."
                )
                _w.warn(
                    "Mapping has duplicate source_name values and no "
                    "source_idx column; the name-based join will collapse "
                    "homonyms to a single unit_id." + hint,
                    UserWarning, stacklevel=2,
                )
            name_to_id = dict(
                zip(mapping["source_name"].astype(str), mapping["proposed_unit_id"])
            )
            gdf[id_column] = gdf[name_column].astype(str).map(name_to_id)

    # A reviewer who clears a cell in Excel leaves an empty string behind,
    # not a missing value. Treat the two the same, so "I don't know" reads
    # as unmatched rather than as a district whose name is blank.
    gdf.loc[gdf[id_column] == "", id_column] = pd.NA
    return gdf


def attach_stats_ids(
    stats: _DfOrPath,
    mapping: pd.DataFrame | Path | str,
    name_column: str,
    *,
    id_column: str = "unit_id",
    year_column: str | None = None,
    coarse_column: str | None = None,
) -> pd.DataFrame:
    """Attach a canonical ``unit_id`` column to a stats DataFrame.

    Mirrors :func:`attach_shapefile_ids` but for long-form stats. Stats
    rows aren't positionally keyed, so the join is by value: the mapping
    is per *unique key*, not per row. Caller can subsequently drop the
    ``name_column`` if the rest of the pipeline only needs the ID.

    The join key is built from whichever of ``source_name``,
    ``source_year`` and ``source_coarse`` the mapping carries *and* the
    caller has named a stats column for:

    - Pass ``year_column`` when the mapping came from
      ``propose_stats_mapping(year_aware=True)`` — the mapping is then per
      (name, year), so a name resolving to different ids in different years
      attaches correctly.
    - Pass ``coarse_column`` when the mapping came from
      ``propose_stats_mapping(coarse_column=...)`` — the mapping is then per
      (name, coarse), so homonyms stay distinguishable.

    Passing either against a mapping that doesn't carry the corresponding
    column is harmless; the key simply omits it.

    Raises:
        ValueError: if the resulting key does not uniquely determine
            ``proposed_unit_id``. That happens when a mapping was built with
            a richer key than the join is using — e.g. a year-aware mapping
            joined name-only, where one name has a different id per year.
            Silently collapsing such a mapping (last value wins) would
            misattribute data with no warning, so this is an error rather
            than a fallback.
    """
    df = _coerce_df(stats).copy()
    if isinstance(mapping, (str, Path)):
        mapping = read_mapping(mapping)

    m = mapping.copy()
    m["_name"] = m["source_name"].astype(str)

    # Work out what to join on. A reviewed mapping can be keyed on the name
    # alone, or on the name plus the year, or plus the state — depending on
    # how it was built. Both sides have to agree: the mapping has to carry
    # the extra detail, and the caller has to have said which of their own
    # columns holds it.
    has_year = "source_year" in m.columns and m["source_year"].notna().any()
    has_coarse = "source_coarse" in m.columns and m["source_coarse"].notna().any()
    use_year = has_year and year_column is not None and year_column in df.columns
    use_coarse = has_coarse and coarse_column is not None and coarse_column in df.columns

    key_cols = ["_name"]
    if use_year:
        m["_yr"] = pd.to_numeric(m["source_year"], errors="coerce")
        m = m.dropna(subset=["_yr"])
        m["_yr"] = m["_yr"].astype(int)
        key_cols.append("_yr")
    if use_coarse:
        m["_coarse"] = m["source_coarse"].astype(str)
        key_cols.append("_coarse")

    # Before joining anything, check the chosen key actually settles which
    # district each row belongs to. If a key points at more than one, the
    # mapping is finer-grained than the join being asked for — a mapping
    # that names a different district per year, joined on name alone, would
    # silently take whichever came last and file years of data under the
    # wrong district. Refuse, and say which detail is missing.
    collisions = m.groupby(key_cols, dropna=False)["proposed_unit_id"].nunique()
    ambiguous = collisions[collisions > 1]
    if len(ambiguous):
        missing = []
        if has_year and not use_year:
            missing.append("year_column=<your year column> (mapping has source_year)")
        if has_coarse and not use_coarse:
            missing.append(
                "coarse_column=<your state/province column> (mapping has source_coarse)"
            )
        hint = (
            "Pass " + " and ".join(missing) + " to attach_stats_ids()."
            if missing
            else "The mapping itself is inconsistent: the same key is assigned "
            "more than one proposed_unit_id. Fix the reviewed CSV."
        )
        sample = list(ambiguous.index[:5])
        raise ValueError(
            f"{len(ambiguous)} mapping key(s) resolve to more than one "
            f"proposed_unit_id under the join key {key_cols}; attaching would "
            f"silently misattribute data. Sample keys: {sample!r}. {hint}"
        )

    # The key is settled, so turn the reviewed decisions into a lookup...
    id_by_key = dict(zip(m[key_cols].itertuples(index=False, name=None), m["proposed_unit_id"]))

    # ...and assemble the matching key from the caller's own data, using the
    # same pieces in the same order so the two sides line up.
    parts: list = [df[name_column].astype(str)]
    if use_year:
        parts.append(pd.to_numeric(df[year_column], errors="coerce"))
    if use_coarse:
        parts.append(df[coarse_column].astype(str))

    def _lookup(key: tuple) -> object:
        # A NaN year can't match an int key; treat it as unmatched.
        if use_year and pd.isna(key[1]):
            return pd.NA
        if use_year:
            key = (key[0], int(key[1])) + tuple(key[2:])
        return id_by_key.get(key, pd.NA)

    df[id_column] = [_lookup(k) for k in zip(*parts)]
    df.loc[df[id_column] == "", id_column] = pd.NA
    return df


__all__ = [
    "MatchProposal",
    "propose_shapefile_mapping",
    "propose_stats_mapping",
    "attach_shapefile_ids",
    "attach_stats_ids",
    "read_mapping",
    "normalize_name",
    "DEFAULT_FUZZY_THRESHOLD",
    "DEFAULT_SKETCHY_THRESHOLD",
]
