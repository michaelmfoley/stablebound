"""FEWS NET unit-definition file generation.

For every ``(country, admin level, year)`` we emit a CSV listing the
admin units active at that year along with their FNID and admin-name
context, in the layout of FEWS's distributed admin-definition files.

Schema:

- Admin1 file::

    FNID, EFF_YEAR, COUNTRY, admin0, admin1

- Admin2 file (extension)::

    FNID, EFF_YEAR, COUNTRY, admin0, admin1, admin2

The package's :func:`stablebound.snapshot.build_snapshot` gives the
active admin2 set per year. To turn that into rows we also need the
admin1 attribution per (year, admin2) — derived here by walking events
forward from the baseline.

Public API:

- ``build_admin1_attribution_table(graph, baseline) -> pd.DataFrame``
- ``build_unit_defs_table(graph, baseline, code_map, *, iso, admin0, level, year)``
- ``write_unit_defs_files(graph, baseline, admin1_code_map, admin2_code_map, *,
  iso, admin0, years, levels, out_dir)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .fnid import build_fnid
from .lineage import LineageGraph, _name_of_unit_in_year
from .snapshot import build_snapshot


# ---------------------------------------------------------------------------
# Per-(year, admin2) admin1 attribution
# ---------------------------------------------------------------------------

def build_admin1_attribution_table(
    graph: LineageGraph,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(year, admin2_id) → (admin1_id, admin1_name) walking events forward.

    Algorithm:
      1. Seed the attribution from ``baseline.csv`` — every admin2 row
         carries its initial-year ``coarse_id``/``coarse_name``.
      2. Walk events in chronological order. For each event row,
         set ``state[child_id] = (child_coarse_id, child_coarse_name)``
         (post-event admin1 of the child) and drop the parent if it's a
         territorial event (Split / Merge / Redistribute — the parent
         ceases). NameChange and Coarse events keep the parent alive but
         still update the attribution if the row's child_id == parent_id
         (which it does for both NameChange and Coarse).
      3. After processing all events with ``event_year < year``, the
         current ``state`` is the attribution at the start of ``year``.

    Returns a long-form DataFrame with columns
    ``year, unit_id, admin1_id, admin1_name`` covering every year from
    ``baseline.year.min()`` through ``graph.max_event_year + 1``. Years
    with no change reuse the prior year's state.

    Note: we don't generate a full year × unit grid here — the caller
    intersects with the snapshot when emitting unit_defs rows.
    """
    if "coarse_id" not in baseline.columns:
        raise ValueError(
            "baseline must include 'coarse_id' / 'coarse_name' columns "
            "to build admin1 attribution"
        )

    # State: unit_id → (admin1_id, admin1_name). Seeded from baseline.
    # Track which state each district belongs to, updating it as the country
    # is reorganised, so that any year can be asked about afterwards. Start
    # from how things stood at the beginning.
    state: dict[str, tuple[str, str]] = {}
    if "year" in baseline.columns and not baseline["year"].isna().all():
        base_year = int(baseline["year"].min())
    else:
        base_year = graph.min_event_year - 1
    for _, row in baseline.iterrows():
        cid = row.get("coarse_id")
        # A district with no state recorded is left out rather than filed
        # under a guess.
        if pd.isna(cid):
            continue
        cname = row.get("coarse_name")
        state[str(row["unit_id"])] = (
            str(cid),
            str(cname) if pd.notna(cname) else str(cid),
        )

    territorial_types = {"Split", "Merge", "Redistribute"}

    rows: list[dict] = []
    # Snapshot the baseline year first.
    for unit_id, (admin1_id, admin1_name) in state.items():
        rows.append({
            "year": base_year,
            "unit_id": unit_id,
            "admin1_id": admin1_id,
            "admin1_name": admin1_name,
        })

    # Walk events year by year. We process all events for a given year
    # together so the post-year-T state captures all of T's events at once.
    ev = graph.events.sort_values(["event_year", "event_type"], kind="stable")
    if len(ev) == 0:
        return pd.DataFrame(rows)

    for year, events_y in ev.groupby("event_year"):
        # First, drop parents of territorial events (they cease).
        for _, row in events_y.iterrows():
            if row["event_type"] in territorial_types:
                pid = str(row["parent_id"])
                cid_unit = str(row["child_id"])
                # Only drop if the parent isn't reused as the child (rare,
                # but safe — NameChange has parent_id == child_id and is
                # excluded from this branch anyway).
                if pid != cid_unit and pid in state:
                    del state[pid]

        # Then, apply the post-event admin1 of each child.
        for _, row in events_y.iterrows():
            cid_unit = str(row["child_id"])
            child_admin1_id = row.get("child_coarse_id")
            child_admin1_name = row.get("child_coarse_name")
            if pd.isna(child_admin1_id):
                # Fall back to the current attribution (for events whose
                # child_coarse_* columns are blank — shouldn't happen for
                # India, but be safe).
                continue
            state[cid_unit] = (
                str(child_admin1_id),
                str(child_admin1_name)
                if pd.notna(child_admin1_name)
                else str(child_admin1_id),
            )

        # Snapshot the post-(year T) state. This represents the
        # attribution at the START of year T+1.
        for unit_id, (admin1_id, admin1_name) in state.items():
            rows.append({
                "year": int(year) + 1,
                "unit_id": unit_id,
                "admin1_id": admin1_id,
                "admin1_name": admin1_name,
            })

    out = pd.DataFrame(rows)
    # Coalesce: for years with no events, the most recent prior year's
    # state still applies. The caller asks for a specific year; we'll
    # forward-fill on demand.
    return out


def _attribution_for_year(
    attribution: pd.DataFrame,
    year: int,
) -> dict[str, tuple[str, str]]:
    """Pick the attribution rows effective at the start of ``year``.

    Strategy: for each unit, take its row from the largest ``year``
    column value <= ``year``. If that row's unit was dropped in a later
    event, it won't appear in any year > drop_year, so the lookup is
    naturally pruned.
    """
    rel = attribution[attribution["year"] <= year]
    if rel.empty:
        return {}
    # Last row per unit_id wins (pandas keeps the latest after sorting).
    rel = rel.sort_values("year")
    last = rel.drop_duplicates(subset="unit_id", keep="last")
    return {
        str(row["unit_id"]): (str(row["admin1_id"]), str(row["admin1_name"]))
        for _, row in last.iterrows()
    }


# ---------------------------------------------------------------------------
# Unit-defs DataFrame builder
# ---------------------------------------------------------------------------

def build_unit_defs_table(
    graph: LineageGraph,
    baseline: pd.DataFrame,
    code_map: pd.DataFrame,
    *,
    iso: str,
    admin0: str,
    level: int,
    year: int,
    additional_units: set[str] | None = None,
    attribution: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the unit-defs DataFrame for one ``(level, year)`` pair.

    Parameters:
        graph: LineageGraph from the package's RT.
        baseline: baseline DataFrame (canonical schema with
            ``coarse_id``/``coarse_name`` columns).
        code_map: ``build_admin1_code_map(...)`` if level=1,
            ``build_admin2_code_map(...)`` if level=2.
        iso: 2-char country code (e.g. "IN").
        admin0: country name (e.g. "India").
        level: 1 or 2.
        year: target year.
        additional_units: optional set of always-alive units to seed into
            the snapshot (matches ``build_snapshot``'s contract — units
            not in the RT but present in baseline).
        attribution: optional pre-computed attribution table from
            ``build_admin1_attribution_table``. Cached across years to
            speed up bulk emission. If None, recomputed.

    Returns a DataFrame with columns:
      - level=1: ``FNID, EFF_YEAR, COUNTRY, admin0, admin1``
      - level=2: ``FNID, EFF_YEAR, COUNTRY, admin0, admin1, admin2``

    Sorted ascending by FNID (mirrors VN_Admin1_1991.csv).
    """
    if level not in (1, 2):
        raise ValueError(f"unsupported level={level}; expected 1 or 2")

    if attribution is None:
        attribution = build_admin1_attribution_table(graph, baseline)

    # Always-alive seed for build_snapshot: the baseline's unit_ids that
    # the RT never mentions. This matches StableBoundary's pattern.
    if additional_units is None:
        additional_units = set(baseline["unit_id"].astype(str))

    active_admin2 = build_snapshot(graph, year=year, additional_units=additional_units)
    attr_at_year = _attribution_for_year(attribution, year)

    if level == 1:
        # Active admin1s = the set of admin1_ids that have at least one
        # active admin2 with that admin1 at year T.
        admin1_ss = dict(zip(code_map["ADMIN1_ID"].astype(str), code_map["SS"].astype(str)))
        admin1_name = dict(zip(
            code_map["ADMIN1_ID"].astype(str),
            code_map["ADMIN1_NAME"].astype(str),
        ))

        active_admin1: dict[str, str] = {}  # admin1_id -> admin1_name
        for unit_id in active_admin2:
            if unit_id in attr_at_year:
                a1_id, a1_name = attr_at_year[unit_id]
                active_admin1[a1_id] = a1_name
            # If a unit has no attribution at this year, skip it — we
            # can't place it under any admin1.

        rows = []
        for admin1_id in sorted(active_admin1):
            ss = admin1_ss.get(admin1_id)
            if ss is None:
                # Code map is missing this admin1 — should not happen if
                # baseline + lineage are consistent. Skip with no FNID
                # rather than silently emit a malformed one.
                continue
            rows.append({
                "FNID": build_fnid(iso, year, 1, ss),
                "EFF_YEAR": year,
                "COUNTRY": iso,
                "admin0": admin0,
                "admin1": admin1_name.get(admin1_id, active_admin1[admin1_id]),
            })
        return pd.DataFrame(
            rows, columns=["FNID", "EFF_YEAR", "COUNTRY", "admin0", "admin1"]
        ).sort_values("FNID").reset_index(drop=True)

    # level == 2
    # Build FNID lookup keyed by admin2_id.
    cm = code_map.set_index("ADMIN2_ID")
    # Modern/most-recent name per unit — the fallback for units with no event
    # history (they never changed, so their one name applies to every year).
    latest_name = dict(zip(
        code_map["ADMIN2_ID"].astype(str),
        code_map["ADMIN2_NAME"].astype(str),
    ))
    # Units that appear in any event are the only ones whose name can vary by
    # year; resolve those year-accurately (splits, renames from the name-change
    # log), fall back to the modern name otherwise. Gating avoids a per-year
    # event walk for the many territorially-static units.
    units_with_events = (
        set(graph.events["parent_id"].astype(str))
        | set(graph.events["child_id"].astype(str))
    )

    rows = []
    for unit_id in sorted(active_admin2):
        if unit_id not in cm.index:
            # No FNID code for this admin2 (shouldn't happen if origin
            # resolution covered the baseline). Skip.
            continue
        ss = str(cm.at[unit_id, "SS"])
        dd = str(cm.at[unit_id, "DD"])
        if unit_id in attr_at_year:
            a1_id, a1_name = attr_at_year[unit_id]
        else:
            # Fall back to origin admin1 if the per-year attribution
            # doesn't cover this unit (defensive — same value as the
            # SS-implied admin1 for territorially-static units).
            a1_id = str(cm.at[unit_id, "ORIGIN_ADMIN1_ID"])
            a1_name = str(cm.at[unit_id, "ORIGIN_ADMIN1_NAME"])
        if unit_id in units_with_events:
            name = _name_of_unit_in_year(graph, unit_id, year) or latest_name.get(unit_id, "")
        else:
            name = latest_name.get(unit_id, "")
        rows.append({
            "FNID": build_fnid(iso, year, 2, ss, dd),
            "EFF_YEAR": year,
            "COUNTRY": iso,
            "admin0": admin0,
            "admin1": a1_name,
            "admin2": name,
        })
    return pd.DataFrame(
        rows, columns=["FNID", "EFF_YEAR", "COUNTRY", "admin0", "admin1", "admin2"]
    ).sort_values("FNID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Admin1-only unit-defs (single-level lineages: each unit is an admin1)
# ---------------------------------------------------------------------------

def build_admin1_defs_table(
    graph: LineageGraph,
    baseline: pd.DataFrame,
    code_map: pd.DataFrame,
    *,
    iso: str,
    admin0: str,
    year: int,
    additional_units: set[str] | None = None,
) -> pd.DataFrame:
    """Admin1 unit-defs for a *single-level* lineage (each unit is an admin1).

    :func:`build_unit_defs_table` (level=1) places each active admin2 under
    its attributed admin1, which needs the ``coarse_id`` attribution that
    admin1-only countries don't have. This variant bypasses
    attribution entirely: every unit active at ``year`` is emitted directly as
    an admin1 row (snapshot unit → ``A1{SS}`` → year-accurate name), mirroring
    the level-2 name resolution (year-accurate for units touched by events,
    the code-map name otherwise).

    ``code_map`` is a :func:`stablebound.fnid.build_admin1_only_code_map`
    output (``ADMIN1_ID`` = unit id). Columns match the level-1
    :func:`build_unit_defs_table`: ``FNID, EFF_YEAR, COUNTRY, admin0, admin1``.
    """
    ss = dict(zip(code_map["ADMIN1_ID"].astype(str), code_map["SS"].astype(str)))
    latest = dict(zip(
        code_map["ADMIN1_ID"].astype(str),
        code_map["ADMIN1_NAME"].astype(str),
    ))
    if additional_units is None:
        additional_units = set(baseline["unit_id"].astype(str))
    active = build_snapshot(graph, year=year, additional_units=additional_units)
    units_with_events = (
        set(graph.events["parent_id"].astype(str))
        | set(graph.events["child_id"].astype(str))
    )

    rows = []
    for unit_id in sorted(active):
        s = ss.get(unit_id)
        if s is None:
            # No code for this unit (baseline/lineage inconsistency) — skip
            # rather than emit a malformed FNID.
            continue
        if unit_id in units_with_events:
            name = _name_of_unit_in_year(graph, unit_id, year) or latest.get(unit_id, "")
        else:
            name = latest.get(unit_id, "")
        rows.append({
            "FNID": build_fnid(iso, year, 1, s),
            "EFF_YEAR": year,
            "COUNTRY": iso,
            "admin0": admin0,
            "admin1": name,
        })
    return pd.DataFrame(
        rows, columns=["FNID", "EFF_YEAR", "COUNTRY", "admin0", "admin1"]
    ).sort_values("FNID").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bulk write
# ---------------------------------------------------------------------------

def write_unit_defs_files(
    graph: LineageGraph,
    baseline: pd.DataFrame,
    *,
    admin1_code_map: pd.DataFrame | None = None,
    admin2_code_map: pd.DataFrame | None = None,
    iso: str,
    admin0: str,
    years: Iterable[int],
    levels: Iterable[int] = (1, 2),
    out_dir: Path | str,
) -> list[Path]:
    """Write ``{ISO}_Admin{LEVEL}_{YEAR}.csv`` for every (level, year).

    Files are written into per-level subfolders:
    ``out_dir/admin1/`` and ``out_dir/admin2/``. The directories are
    created if they don't exist.

    Returns the list of written paths in deterministic order.
    """
    out_dir = Path(out_dir)
    levels = list(levels)
    years = list(years)

    if 1 in levels and admin1_code_map is None:
        raise ValueError("admin1_code_map is required when level=1 is requested")
    if 2 in levels and admin2_code_map is None:
        raise ValueError("admin2_code_map is required when level=2 is requested")

    # Compute the attribution once — cache it across years.
    attribution = build_admin1_attribution_table(graph, baseline)
    additional_units = set(baseline["unit_id"].astype(str))

    # One file per admin level per year — the receiving system wants each
    # year's list of units as its own document rather than one combined
    # table, so a year can be replaced without touching the others.
    written: list[Path] = []
    for level in sorted(levels):
        if level == 1:
            cm = admin1_code_map
        else:
            cm = admin2_code_map
        sub = out_dir / f"admin{level}"
        sub.mkdir(parents=True, exist_ok=True)
        for year in sorted(years):
            df = build_unit_defs_table(
                graph, baseline, cm,
                iso=iso, admin0=admin0,
                level=level, year=year,
                additional_units=additional_units,
                attribution=attribution,
            )
            path = sub / f"{iso}_Admin{level}_{year}.csv"
            df.to_csv(path, index=False)
            written.append(path)
    return written
