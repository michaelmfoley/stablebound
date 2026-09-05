"""Convert old-format relationship tables into canonical lineage + baseline.

The package's canonical RT schema is event-only: one row per boundary
event. But many publishers (and the legacy HarvestStat workflow) supply
"old-format" relationship tables that mix two row kinds in one CSV:

  - ``category == "hierarchical"`` rows — define which units exist in
    each snapshot year (the steady-state).
  - ``category == "temporal"`` rows — define how units evolve between
    snapshot years (``relationship_type ∈ {successor, split, merge}``).

This module converts that format into the canonical lineage DataFrame
(directly consumable by :class:`LineageGraph`) plus a baseline snapshot
(directly consumable by :mod:`stablebound.snapshot`).

Ported from the legacy HarvestStat ``rt_to_lineage`` conversion script.
Behaviour is pinned by the hand-authored legacy-dialect fixture
``tests/fixtures/rt_convert/relationshiptable_XX.csv`` and its canonical twin
``tests/fixtures/synthetic/legacy_admin1/`` (see the README beside the CSV).
"""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Tuple

import pandas as pd

__all__ = [
    "convert_relationship_table_to_lineage",
    "read_legacy_relationship_table",
]

ID_WIDTH = 5  # zero-padded numeric part: 00001


def _make_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:0{ID_WIDTH}d}"


def _extract_name(raw: object) -> object:
    """Strip ", CountryName" suffix from a unit name; leave NaN alone."""
    if pd.isna(raw):
        return raw
    return str(raw).split(",")[0].strip()


def _extract_year_from_fnid(fnid: object) -> int | None:
    """Pull the first 4-digit run out of an fnid like 'KR1982A105' → 1982."""
    m = re.search(r"(\d{4})", str(fnid))
    return int(m.group(1)) if m else None


def read_legacy_relationship_table(path: str | Path) -> pd.DataFrame:
    """Read an old-format RT CSV with category/relationship_type columns.

    Adds derived columns (``from_name``, ``to_name``, ``from_year``,
    ``to_year``) used by the converter. Returns the augmented DataFrame
    unchanged otherwise.
    """
    df = pd.read_csv(path)
    df["from_name"] = df["from_unit_name"].apply(_extract_name)
    df["to_name"] = df["to_unit_name"].apply(_extract_name)
    df["from_year"] = df["fnid_from_unit"].apply(_extract_year_from_fnid)
    df["to_year"] = df["fnid_to_unit"].apply(_extract_year_from_fnid)
    return df


def _build_snapshots(
    df: pd.DataFrame, admin_level: int
) -> "OrderedDict[int, list[dict]]":
    """Group hierarchical rows at the target admin level into year → unit list.

    Hierarchical rows carry ``relationship_type`` values like ``admin1_0``
    or ``admin2_1`` — the first digit is the unit's admin level, the
    second is the parent's. A single FEWS RT can mix levels (India ships
    both ADM1 and ADM2 in one file), so the level filter is required.
    """
    rel = df["relationship_type"].astype(str).str.lower()
    hier_mask = (df["category"] == "hierarchical") & rel.str.startswith(
        f"admin{admin_level}_"
    )
    hier = df[hier_mask]
    snapshots: dict[int, list[dict]] = defaultdict(list)
    for _, row in hier.iterrows():
        snapshots[row["from_year"]].append(
            {
                "name": row["from_name"],
                "fnid": row["fnid_from_unit"],
                "from_unit": row["from_unit"],
            }
        )
    return OrderedDict(sorted(snapshots.items()))


def _filter_temporal_to_level(
    df: pd.DataFrame, valid_fnids: set
) -> pd.DataFrame:
    """Keep temporal rows whose both endpoints are units at the target level.

    Also lowercases ``relationship_type`` so downstream comparisons can
    match the canonical lowercase forms FEWS files use inconsistently.
    """
    temporal = df[df["category"] == "temporal"].copy()
    temporal["relationship_type"] = (
        temporal["relationship_type"].astype(str).str.lower()
    )
    return temporal[
        temporal["fnid_from_unit"].isin(valid_fnids)
        & temporal["fnid_to_unit"].isin(valid_fnids)
    ].copy()


def _extract_events(temporal: pd.DataFrame) -> pd.DataFrame:
    """Classify temporal rows into NameChange / Split / Merge / Redistribute.

    Two ways a rename can appear in a FEWS RT (both observed in the
    India file): an explicit ``relationship_type == "name change"`` row,
    or a 1-to-1 ``successor`` row where ``from_name != to_name``. Both
    are emitted as ``NameChange`` events; pure 1-to-1 successors with
    matching names are dropped as carry-forwards.
    """
    if temporal.empty:
        return pd.DataFrame()

    children_per_parent = temporal.groupby("from_unit")["to_unit"].nunique()
    parents_per_child = temporal.groupby("to_unit")["from_unit"].nunique().to_dict()

    rows = []
    for _, row in temporal.iterrows():
        parent_unit = row["from_unit"]
        child_unit = row["to_unit"]
        rel_type = row["relationship_type"]
        from_name = row["from_name"]
        to_name = row["to_name"]
        n_children = children_per_parent.get(parent_unit, 1)
        n_parents = parents_per_child.get(child_unit, 1)

        # Explicit rename row.
        if rel_type == "name change":
            event_type = "NameChange"
        # 1-to-1 successor: rename if names differ, else carry-forward.
        elif rel_type == "successor" and n_children == 1 and n_parents == 1:
            if pd.notna(from_name) and pd.notna(to_name) and from_name != to_name:
                event_type = "NameChange"
            else:
                continue
        elif n_parents > 1 and n_children > 1:
            event_type = "Redistribute"
        elif n_parents > 1:
            event_type = "Merge"
        else:
            event_type = "Split"

        rows.append(
            {
                "from_year": row["from_year"],
                "to_year": row["to_year"],
                "event_type": event_type,
                "parent_fnid": row["fnid_from_unit"],
                "child_fnid": row["fnid_to_unit"],
                "parent_name": from_name,
                "child_name": to_name,
                "parent_unit": parent_unit,
                "child_unit": child_unit,
            }
        )
    return pd.DataFrame(rows)


def _assign_ids(
    snapshots: "OrderedDict[int, list[dict]]",
    events_df: pd.DataFrame,
    id_prefix: str,
) -> Tuple[dict, int]:
    """Assign dense IDs: baseline gets 1..N, new event children get N+1, N+2, ..."""
    years = list(snapshots.keys())
    baseline_year = years[0]
    fnid_to_id: dict = {}
    next_id = 1

    for unit in sorted(snapshots[baseline_year], key=lambda u: u["name"]):
        fnid_to_id[unit["fnid"]] = _make_id(id_prefix, next_id)
        next_id += 1
    baseline_count = next_id - 1

    if not events_df.empty:
        for _, ev in events_df.sort_values(["to_year", "child_name"]).iterrows():
            fnid = ev["child_fnid"]
            if fnid not in fnid_to_id:
                fnid_to_id[fnid] = _make_id(id_prefix, next_id)
                next_id += 1

    # Successors that never appeared in events still need IDs.
    for year in years[1:]:
        for unit in sorted(snapshots[year], key=lambda u: u["name"]):
            if unit["fnid"] not in fnid_to_id:
                fnid_to_id[unit["fnid"]] = _make_id(id_prefix, next_id)
                next_id += 1

    return fnid_to_id, baseline_count


def _resolve_successor_chains(temporal: pd.DataFrame, fnid_to_id: dict) -> None:
    """Collapse 1-to-1 successor and name-change chains to a single ID.

    A renamed unit (``relationship_type == "name change"`` or a
    1-to-1 successor with differing names) is still the same logical
    unit — both fnids must map to the same canonical ID so the eventual
    NameChange event satisfies ``parent_id == child_id``.
    """
    # The source format gives a district a fresh identifier every time it
    # publishes a new edition, even when nothing about the district changed.
    # Left alone that would read as a district ceasing to exist and an
    # identical one appearing beside it, once per edition.
    #
    # So follow the chain and give every edition the same identifier — but
    # only where the district genuinely just continued. Count how many ways
    # each district leads and is led into first.
    children_per_parent = temporal.groupby("from_unit")["to_unit"].nunique().to_dict()
    parents_per_child = temporal.groupby("to_unit")["from_unit"].nunique().to_dict()
    chainable = temporal[
        temporal["relationship_type"].isin({"successor", "name change"})
    ]
    for _, row in chainable.iterrows():
        # One in and one out means the same district carried on, possibly
        # under a new name. Anything else is a real split or merge and has
        # to keep separate identifiers, or the change would vanish.
        if (
            children_per_parent.get(row["from_unit"], 1) == 1
            and parents_per_child.get(row["to_unit"], 1) == 1
            and row["fnid_from_unit"] in fnid_to_id
        ):
            fnid_to_id[row["fnid_to_unit"]] = fnid_to_id[row["fnid_from_unit"]]


def _build_lineage(events_df: pd.DataFrame, fnid_to_id: dict) -> pd.DataFrame:
    """Emit the canonical event-only lineage DataFrame (lowercase columns)."""
    cols = [
        "event_year",
        "event_type",
        "parent_id",
        "parent_name",
        "child_id",
        "child_name",
    ]
    if events_df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, ev in events_df.iterrows():
        rows.append(
            {
                # event_year is the year the change HAPPENED, matching the
                # hand-authored lineages (India stores 2000 for Jharkhand,
                # created 15 Nov 2000, and Jharkhand first appears in
                # snapshot(2001)). A FEWS temporal row's `to_year` is that
                # same year, so it is used directly.
                #
                # DO NOT "correct" this to `to_year - 1`. A FEWS *vintage
                # file* labelled V lists the units in force AFTER year V's
                # changes, so it lines up with `snapshot(V+1)`, not
                # `snapshot(V)`. That one-year gap is FEWS's file-naming
                # convention, not an error here — subtracting a year to make
                # the vintage comparison line up silently shifts every
                # derived event a year before the date it actually occurred,
                # and breaks consistency with the hand-authored lineages.
                "event_year": int(ev["to_year"]),
                "event_type": ev["event_type"],
                "parent_id": fnid_to_id.get(ev["parent_fnid"], "???"),
                "parent_name": ev["parent_name"],
                "child_id": fnid_to_id.get(ev["child_fnid"], "???"),
                "child_name": ev["child_name"],
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["event_year", "parent_id", "child_name"]).reset_index(drop=True)


def _build_baseline(
    snapshots: "OrderedDict[int, list[dict]]",
    fnid_to_id: dict,
) -> pd.DataFrame:
    """Emit baseline snapshot at the earliest year (lowercase canonical columns)."""
    baseline_year = next(iter(snapshots))
    rows = []
    for unit in sorted(snapshots[baseline_year], key=lambda u: u["name"]):
        rows.append(
            {
                "year": int(baseline_year),
                "unit_id": fnid_to_id.get(unit["fnid"], "???"),
                "name": unit["name"],
            }
        )
    return pd.DataFrame(rows, columns=["year", "unit_id", "name"]).sort_values(
        "unit_id"
    ).reset_index(drop=True)


def convert_relationship_table_to_lineage(
    old_rt: pd.DataFrame,
    *,
    country: str,
    admin_level: int = 1,
    return_fnid_map: bool = False,
):
    """Convert old-format hierarchical+temporal RT to canonical lineage + baseline.

    Parameters
    ----------
    old_rt : pd.DataFrame
        Output of :func:`read_legacy_relationship_table` (or an equivalent
        DataFrame with columns ``label, from_unit, to_unit, relationship_type,
        category, from_unit_name, to_unit_name, fnid_from_unit, fnid_to_unit``
        plus the derived ``from_name, to_name, from_year, to_year`` columns).
    country : str
        ISO-2 country code, used as ID prefix (e.g. ``"XX"`` →
        ``"XX.ADM1.00001"``).
    admin_level : int, default 1
        Admin level (1, 2, 3, ...). Controls only the ID prefix segment
        (``"ADM1"`` / ``"ADM2"`` / ``"ADM3"``); column names on outputs
        are always the canonical lowercase ones. Pakistan's FEWS RT
        goes to ADM3; deeper levels are accepted but untested.

    return_fnid_map : bool, default False
        Also return ``{fnid: unit_id}``. The mapping is built internally
        anyway; returning it lets a caller attach ids to statistics that
        already carry FNIDs — which FEWS-sourced stats generally do — and
        skip name matching entirely. Kept opt-in so the 2-tuple return of
        existing callers is unchanged.

    Returns
    -------
    (lineage_df, baseline_df) : tuple of pd.DataFrame
        ``lineage_df`` has columns
        ``(event_year, event_type, parent_id, parent_name, child_id, child_name)``
        — directly consumable by :meth:`LineageGraph.from_dataframe`.
        ``baseline_df`` has columns ``(year, unit_id, name)`` at the
        earliest snapshot year — directly consumable by
        :mod:`stablebound.snapshot`.
    """
    if not isinstance(admin_level, int) or admin_level < 1:
        raise ValueError(
            f"admin_level must be a positive int (1, 2, 3, ...), got {admin_level!r}"
        )
    id_prefix = f"{country}.ADM{admin_level}."

    snapshots = _build_snapshots(old_rt, admin_level)
    if not snapshots:
        raise ValueError(
            f"No hierarchical rows at admin_level={admin_level} found in input. "
            f"Expected at least one row with category == 'hierarchical' and "
            f"relationship_type starting with 'admin{admin_level}_'."
        )

    valid_fnids = {u["fnid"] for units in snapshots.values() for u in units}
    temporal = _filter_temporal_to_level(old_rt, valid_fnids)
    events_df = _extract_events(temporal)
    fnid_to_id, _baseline_count = _assign_ids(snapshots, events_df, id_prefix)
    _resolve_successor_chains(temporal, fnid_to_id)

    lineage_df = _build_lineage(events_df, fnid_to_id)
    baseline_df = _build_baseline(snapshots, fnid_to_id)
    if return_fnid_map:
        return lineage_df, baseline_df, dict(fnid_to_id)
    return lineage_df, baseline_df
