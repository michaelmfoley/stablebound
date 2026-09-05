"""Country-agnostic builders for the three FEWS "upload_ready" deliverables.

Given a lineage (relationship table) + baseline + statistics for any country,
produce:

1. ``{ISO}_Admin_Definitions_{year}.xlsx`` — per-year workbook, two tabs
   ``{iso}_admin1_{year}`` / ``{iso}_admin2_{year}``, columns
   ``FNID, EFF_YEAR, COUNTRY_CODE, admin0, admin1[, admin2]``, sorted by FNID.
2. ``{ISO}_GeographicUnitRelationship.csv`` — dense per-consecutive-year-pair
   relationship table (``successor / split / merge / redistribute / name change``)
   at both admin levels.
3. ``{ISO}_AgStats_*.xlsx`` — statistics with a canonical ``FNID`` column stamped
   from the same code map.

These are pure, stateless transforms over the existing package building blocks
(:mod:`stablebound.fnid`, :mod:`stablebound.unit_defs`, :mod:`stablebound.snapshot`,
:class:`stablebound.lineage.LineageGraph`). This module is **additive**; it does
not change any existing behaviour.

**Matching is deliberately out of scope.** Admin definitions and the relationship
table need no name matching (pure lineage → FNID). Only the AgStats file needs
name → id resolution, which is country-specific; the caller performs it upstream
(India: ``stablebound.india.match_stats_to_lineage``; other countries:
``propose_stats_mapping`` → review → ``attach_stats_ids``) and passes statistics
that already carry a ``unit_id`` column. :func:`attach_stats_fnids` only stamps
FNIDs onto already-matched rows.

It replaces three India one-off scripts (an upload-package builder, a
relationship exporter and a stats re-stamper, none of them part of this
repository) with parameterized functions — and drops their now-fixed-at-source hacks (the
Gujarat ``A212`` checks and the Assam ``fix_assam_2023`` dedup). The generic
:func:`validate_no_within_admin1_duplicate` replaces the Assam dedup: it raises
on any duplicate name so a source problem is surfaced, never silently patched.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import pandas as pd

from .fnid import (
    assign_fnids,
    build_admin1_code_map,
    build_admin1_only_code_map,
    build_admin2_code_map,
    build_fnid,
    validate_fnid,
)
from .lineage import LineageGraph, name_of_unit_in_year  # noqa: F401 (re-export intent)
from .unit_defs import (
    build_admin1_attribution_table,
    build_admin1_defs_table,
    build_unit_defs_table,
)

RELATIONSHIP_COLUMNS = [
    "relationship_type",
    "from_unit_name",
    "to_unit_name",
    "fnid_from_unit",
    "fnid_to_unit",
]

#: The 11-column dialect FEWS distributes (and that
#: :func:`stablebound.rt_convert.read_legacy_relationship_table` reads).
#: Distinct from :data:`RELATIONSHIP_COLUMNS`, which is the 5-column upload
#: format — see :func:`build_legacy_relationship_table` for why both exist.
LEGACY_RELATIONSHIP_COLUMNS = [
    "label",
    "from_unit",
    "to_unit",
    "relationship_type",
    "category",
    "from_unit_name",
    "to_unit_name",
    "fnid_from_unit",
    "fnid_to_unit",
    "can_aggregate_to_from",
    "can_aggregate_from_to",
]


class DeliverableValidationError(ValueError):
    """Raised when a deliverable fails a pre-write invariant.

    Currently used for within-admin1 duplicate names — the condition FEWS
    rejects on upload (a unit name must be unique within its parent). Surfacing
    it loudly is intentional: it means a *source* lineage problem to fix, not
    something to silently dedup away.
    """


@dataclass
class RelationshipLevelInput:
    """One admin level's inputs to :func:`build_relationship_table`.

    Attributes:
        level: 1 or 2.
        event_graph: LineageGraph carrying this level's territorial +
            name-change events (in this level's id space). For India admin2
            this is the ADM2 graph; for India admin1 it is the ADM1_LINEAGE +
            level-1 name-change-log graph.
        code_map: ``build_admin{level}_code_map(...)`` output (id → SS[, DD]).
        defs_by_year: ``{year: build_unit_defs_table(level, year)}`` — supplies
            the active-id set and year-accurate names per year.
    """

    level: int
    event_graph: LineageGraph
    code_map: pd.DataFrame
    defs_by_year: Mapping[int, pd.DataFrame]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rename_country_column(df: pd.DataFrame) -> pd.DataFrame:
    """``COUNTRY`` → ``COUNTRY_CODE`` and cast EFF_YEAR to int (FEWS schema)."""
    out = df.rename(columns={"COUNTRY": "COUNTRY_CODE"})
    if "EFF_YEAR" in out.columns:
        out["EFF_YEAR"] = out["EFF_YEAR"].astype(int)
    return out


def _code_of_fnid(fnid: str, level: int) -> str:
    """The id-stable code slice of an FNID: SS at level 1, SS+DD at level 2.

    Parses rather than slices. A positional slice on a malformed id returns a
    short string instead of failing, which would silently produce a wrong
    reverse lookup; ``validate_fnid`` raises instead, and refuses levels
    beyond 2 rather than guessing their code width.
    """
    return validate_fnid(fnid, level=level).code


def _ids_and_names_from_defs(
    defs_df: pd.DataFrame, code_map: pd.DataFrame, level: int
) -> dict[str, str]:
    """Recover ``{unit_id: year_accurate_name}`` from a unit-defs table.

    The unit-defs table is keyed by year-stamped FNID; invert the id-stable
    code (SS[+DD]) back to the unit id via ``code_map``. The name is the
    table's ``admin{level}`` column (already year-accurate).
    """
    if level == 1:
        rev = dict(zip(code_map["SS"].astype(str), code_map["ADMIN1_ID"].astype(str)))
        name_col = "admin1"
    else:
        rev = dict(
            zip(
                (code_map["SS"].astype(str) + code_map["DD"].astype(str)),
                code_map["ADMIN2_ID"].astype(str),
            )
        )
        name_col = "admin2"
    out: dict[str, str] = {}
    for fnid, name in zip(defs_df["FNID"].astype(str), defs_df[name_col].astype(str)):
        uid = rev.get(_code_of_fnid(fnid, level))
        if uid is not None:
            out[uid] = name
    return out


def validate_no_within_admin1_duplicate(
    a1: pd.DataFrame, a2: pd.DataFrame | None, year: int
) -> None:
    """Raise :class:`DeliverableValidationError` on any duplicate unit name.

    (1) duplicate admin1 name in the admin1 table; (2) duplicate
    ``(admin1, admin2)`` pair in the admin2 table. Replaces the India-specific
    ``fix_assam_2023`` dedup — surfaces the collision instead of dropping rows.
    """
    dup1 = a1["admin1"][a1["admin1"].duplicated()].unique()
    if len(dup1):
        raise DeliverableValidationError(
            f"{year}: duplicate admin1 name(s): {list(dup1)}"
        )
    if a2 is not None:
        dups = a2[a2.duplicated(subset=["admin1", "admin2"], keep=False)]
        if not dups.empty:
            raise DeliverableValidationError(
                f"{year}: duplicate admin2 name within an admin1:\n"
                f"{dups[['FNID', 'admin1', 'admin2']].to_string(index=False)}"
            )


def _write_two_tab_workbook(
    a1: pd.DataFrame,
    a2: pd.DataFrame | None,
    out_path: Path,
    *,
    sheet_a1: str,
    sheet_a2: str,
) -> Path:
    """Write a per-year admin-definition workbook (admin1 tab first)."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        a1.to_excel(xw, sheet_name=sheet_a1, index=False)
        if a2 is not None:
            a2.to_excel(xw, sheet_name=sheet_a2, index=False)
    return out_path


# ---------------------------------------------------------------------------
# deliverable 1: admin-definition workbooks
# ---------------------------------------------------------------------------

def write_admin_definitions_workbooks(
    defs_by_level_year: Mapping[int, Mapping[int, pd.DataFrame]],
    *,
    iso: str,
    years: Sequence[int],
    levels: Sequence[int],
    out_dir: Path | str,
) -> list[Path]:
    """Write one ``{ISO}_Admin_Definitions_{year}.xlsx`` per year.

    ``defs_by_level_year[level][year]`` is a ``build_unit_defs_table`` output.
    Per year: rename ``COUNTRY→COUNTRY_CODE``, sort ascending by FNID, validate
    no within-admin1 duplicate names, write the two-tab workbook (admin1 first;
    admin2 tab omitted for admin1-only countries).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    has_admin2 = 2 in levels
    for year in years:
        a1 = (
            _rename_country_column(defs_by_level_year[1][year])
            .sort_values("FNID")
            .reset_index(drop=True)
        )
        a2 = None
        if has_admin2:
            a2 = (
                _rename_country_column(defs_by_level_year[2][year])
                .sort_values("FNID")
                .reset_index(drop=True)
            )
        validate_no_within_admin1_duplicate(a1, a2, year)
        out = out_dir / f"{iso}_Admin_Definitions_{year}.xlsx"
        _write_two_tab_workbook(
            a1, a2, out,
            sheet_a1=f"{iso}_admin1_{year}",
            sheet_a2=f"{iso}_admin2_{year}",
        )
        written.append(out)
    return written


# ---------------------------------------------------------------------------
# deliverable 2: relationship table
# ---------------------------------------------------------------------------

def build_relationship_table(
    *,
    iso: str,
    years: Sequence[int],
    level_inputs: Sequence[RelationshipLevelInput],
    name_style: Literal["bare", "with_country"] = "bare",
    admin0: str | None = None,
) -> pd.DataFrame:
    """Build the dense per-year-pair relationship table across all levels.

    For each level (in the order given — pass admin2 before admin1 to match the
    India file's row grouping), for each consecutive pair (Y, Y+1):

    * territorial events at Y → ``split``/``merge``/``redistribute`` rows
      (names from the event row);
    * name-change events at Y (unit active both years) → ``name change`` rows;
    * every unit active in both years, not touched by a territorial event and
      not name-changed → ``successor`` row.

    Output is sorted deterministically (the source row order is set-iteration
    order and thus non-reproducible; the row *set* is what matters for FEWS).
    """
    if name_style == "with_country" and not admin0:
        raise ValueError("name_style='with_country' requires admin0")

    year_set = set(years)
    pairs = [(y, y + 1) for y in years if (y + 1) in year_set]

    def styled(name: object) -> str:
        s = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name)
        return f"{s}, {admin0}" if name_style == "with_country" else s

    rows: list[tuple] = []
    for li in level_inputs:
        level = li.level
        # Look up each unit's short code. A state needs one code; a district
        # needs two — its state's, plus its own within that state.
        cm = li.code_map
        if level == 1:
            ss = dict(zip(cm["ADMIN1_ID"].astype(str), cm["SS"].astype(str)))
            dd: dict[str, str] | None = None
        else:
            ss = dict(zip(cm["ADMIN2_ID"].astype(str), cm["SS"].astype(str)))
            dd = dict(zip(cm["ADMIN2_ID"].astype(str), cm["DD"].astype(str)))

        def fnid(u: str, y: int) -> str:
            return build_fnid(iso, y, level, ss[u], dd[u] if dd is not None else "")

        active = {
            y: _ids_and_names_from_defs(li.defs_by_year[y], cm, level) for y in years
        }
        terr = li.event_graph.territorial
        names = li.event_graph.name_changes

        for (y, y1) in pairs:
            ay, ay1 = active[y], active[y1]
            ev = terr[terr["event_year"] == y]
            nc = names[names["event_year"] == y]
            touched = set(ev["parent_id"].astype(str)) | set(ev["child_id"].astype(str))

            for _, e in ev.iterrows():
                p, c = str(e["parent_id"]), str(e["child_id"])
                if p in ss and c in ss:
                    rows.append((
                        str(e["event_type"]).lower(),
                        styled(e["parent_name"]), styled(e["child_name"]),
                        fnid(p, y), fnid(c, y1),
                    ))

            nc_units: set[str] = set()
            for _, e in nc.iterrows():
                u = str(e["parent_id"])
                if u in ss and u in ay and u in ay1:
                    rows.append((
                        "name change",
                        styled(e["parent_name"]), styled(e["child_name"]),
                        fnid(u, y), fnid(u, y1),
                    ))
                    nc_units.add(u)

            for u in sorted((set(ay) & set(ay1)) - touched - nc_units):
                if u in ss:
                    rows.append((
                        "successor",
                        styled(ay[u]), styled(ay1[u]),
                        fnid(u, y), fnid(u, y1),
                    ))

    df = pd.DataFrame(rows, columns=RELATIONSHIP_COLUMNS)
    return (
        df.sort_values(["fnid_from_unit", "fnid_to_unit", "relationship_type"])
        .reset_index(drop=True)
    )


def write_relationship_table(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Write ``{ISO}_GeographicUnitRelationship.csv``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# deliverable 2b: the legacy 11-column dialect (round-trip / interchange)
# ---------------------------------------------------------------------------
#
# Public contract: a FEWS *vintage* labelled V lists the units in force AFTER
# year V's changes, so it corresponds to package ``snapshot(V + 1)``.
#
# This is a file-naming convention, not an off-by-one. It was established
# against real data: Korea's distributed RT has vintages 1982/1986/1989/1997/2012
# holding 13/14/15/16/17 units, and a lineage with event years 1986/1989/1997/2012
# gives snapshot(1987)=14, snapshot(1990)=15, snapshot(1998)=16, snapshot(2013)=17
# — matching at V+1 and at no other offset. The pinned check now runs on the
# hand-authored ``tests/fixtures/rt_convert/relationshiptable_XX.csv`` and its
# canonical twin ``tests/fixtures/synthetic/legacy_admin1/``. ``rt_convert``
# reads the dialect back with ``event_year = to_year``, which is the same
# statement from the other side.
#
# Consequence, and the reason this is not a relabelling of the 5-column table:
# the two dialects stamp the *same* event with different years. India's split of
# Bihar (event_year 2000) is ``IN2000A105 -> IN2001A134`` in the 5-column upload
# file (package snapshot years, child stamped the year it appears) but belongs to
# vintage 2000 here (child stamped the event year). Emitting one from the other
# by string substitution would silently shift every child by a year.


def _legacy_label(kind: str, from_name: str, to_name: str, level: int, parent: str) -> str:
    if kind == "hierarchical":
        return f"{from_name} is an Admin {level} within {parent}"
    return f"{from_name} {kind} {to_name}"


def build_legacy_relationship_table(
    *,
    iso: str,
    admin0: str,
    years: Sequence[int],
    level_inputs: Sequence[RelationshipLevelInput],
    attribution: pd.DataFrame | None = None,
    name_style: Literal["bare", "with_country"] = "with_country",
) -> pd.DataFrame:
    """Serialize the lineage in the 11-column dialect FEWS distributes.

    Two dialects exist because import and export were built for different
    consumers, and until this function there was **no way to test the export at
    all**: ``read_legacy_relationship_table`` could not read
    ``write_relationship_table``'s output, so the package's FEWS deliverable had
    no round trip and no correctness proof beyond eyeballing. This is a second
    serialization of data ``build_deliverables`` already computes, not a second
    pipeline, and it exists primarily so the round trip can be asserted.

    Two structural differences from the 5-column upload format:

    * **Vintages, not every year.** A vintage is emitted for the first year and
      for each event year; between vintages nothing changes, so the compression
      is lossless. India's 29 consecutive years collapse to the years where
      something actually happened.
    * **Hierarchical rows.** Each vintage lists every unit with its parent
      (``admin{L}_{L-1}``), which the 5-column format omits entirely. This is
      what carries membership — the upper level's own file records only events,
      so without these rows a reader cannot recover which units exist.

    ``from_unit`` / ``to_unit`` are surrogate integers, one per
    ``(vintage, level, unit)``. They are not decorative: ``rt_convert`` groups on
    them to tell a 1→2 split from two independent successors, so a scheme that
    reused an id across vintages would silently merge unrelated events. Values
    are assigned densely in a deterministic order; only their grouping is
    meaningful, and FEWS reassigns them on ingest.

    Args:
        years: package snapshot years, as passed to :func:`build_deliverables`.
            Vintage ``V`` draws its content from package year ``V + 1``, so the
            emitted labels span ``years[0] - 1 .. years[-1] - 1``.
        attribution: ``build_admin1_attribution_table`` output, required when a
            level-2 input is present to place each district under its admin1.
        name_style: the distributed files use ``"Name, Country"``, so that is
            the default here (the 5-column writer defaults to bare names).

    Returns:
        DataFrame with :data:`LEGACY_RELATIONSHIP_COLUMNS`, sorted
        deterministically.
    """
    years = list(years)
    if not years:
        raise ValueError("years must be non-empty")
    year_set = set(years)
    by_level = {li.level: li for li in level_inputs}
    if 2 in by_level and attribution is None:
        raise ValueError(
            "a level-2 input needs `attribution` (build_admin1_attribution_table "
            "output) to place each admin2 unit under its admin1"
        )

    def styled(name: object) -> str:
        s = "" if name is None or (isinstance(name, float) and pd.isna(name)) else str(name)
        return f"{s}, {admin0}" if name_style == "with_country" else s

    # --- vintages -------------------------------------------------------
    # Content of vintage V is package year V+1, so V ranges over
    # years[0]-1 .. years[-1]-1. Content changes only across an event year,
    # so those are exactly the vintages worth emitting.
    event_years: set[int] = set()
    for li in level_inputs:
        ev = li.event_graph.events
        if not ev.empty:
            event_years |= {int(y) for y in ev["event_year"]}

    # The base vintage's label is the one genuinely ambiguous choice here: with
    # no preceding event, any label whose content matches is defensible. Prefer
    # `years[0]`, which is what the distributed files use (Korea's base vintage
    # is 1982, not 1981) and which makes a round trip preserve `min_year`. That
    # is only sound when nothing happens in years[0] — otherwise years[0] is
    # itself a change vintage and the base has to sit a year earlier.
    lo = years[0] - 1 if years[0] in event_years else years[0]

    # Events at or before `lo` predate the window and are already baked into
    # the base vintage's content. Events needing content from beyond the last
    # requested year cannot be placed either — vintage V reads package year
    # V+1, so the final year has nothing to read.
    #
    # Both exclusions are legitimate: a caller exporting 2015-2018 of a country
    # with events through 2025 wants a window, not an error. But the loss must
    # not be silent — Korea's 2012 split vanished this way while this function
    # was being written, leaving a well-formed file that was quietly missing an
    # event (the regression now runs on the synthetic legacy fixture).
    # `build_deliverables` extends by one year precisely so the default path
    # never trips this.
    needed = {e for e in event_years if lo < e and (e + 1) in year_set}
    dropped = sorted(e for e in event_years if lo < e and (e + 1) not in year_set)
    if dropped:
        warnings.warn(
            f"{len(dropped)} event year(s) fall outside the emittable vintage "
            f"range and are omitted from the legacy relationship table: "
            f"{dropped[:6]}{'...' if len(dropped) > 6 else ''}. A vintage V lists "
            f"the units in force after year V's changes, i.e. package "
            f"snapshot(V+1), so representing event year E requires E+1 in "
            f"`years` (currently {years[0]}..{years[-1]}). Extend the window to "
            f"{max(dropped) + 1} to include them all.",
            UserWarning,
            stacklevel=2,
        )
    vintages = sorted({lo} | needed)

    # --- per-level lookups ----------------------------------------------
    codes: dict[int, dict[str, tuple[str, str]]] = {}
    active: dict[int, dict[int, dict[str, str]]] = {}
    for lvl, li in by_level.items():
        cm = li.code_map
        if lvl == 1:
            codes[lvl] = {
                str(u): (str(ss), "")
                for u, ss in zip(cm["ADMIN1_ID"], cm["SS"])
            }
        else:
            codes[lvl] = {
                str(u): (str(ss), str(dd))
                for u, ss, dd in zip(cm["ADMIN2_ID"], cm["SS"], cm["DD"])
            }
        # Which units each published edition contains. An edition labelled
        # with a given year describes the units as they stood going into the
        # following one, which is why the year is stepped forward here —
        # a naming convention of the format, not an error.
        active[lvl] = {
            v: _ids_and_names_from_defs(li.defs_by_year[v + 1], cm, lvl)
            for v in vintages
            if (v + 1) in year_set
        }

    def fnid_at(lvl: int, unit: str, vintage: int) -> str:
        ss, dd = codes[lvl][unit]
        return build_fnid(iso, vintage, lvl, ss, dd)

    # admin2 -> admin1 for a given package year.
    attr: dict[int, dict[str, tuple[str, str]]] = {}
    if attribution is not None:
        for y, sub in attribution.groupby("year"):
            attr[int(y)] = {
                str(u): (str(a), str(n))
                for u, a, n in zip(sub["unit_id"], sub["admin1_id"], sub["admin1_name"])
            }

    # --- surrogate ids ---------------------------------------------------
    # One per (vintage, level, unit), plus one per vintage for the country.
    surrogate: dict[tuple[int, int, str], int] = {}
    country_surrogate: dict[int, int] = {}
    counter = 1
    for v in vintages:
        country_surrogate[v] = counter
        counter += 1
        for lvl in sorted(by_level):
            for u in sorted(active[lvl].get(v, {})):
                surrogate[(v, lvl, u)] = counter
                counter += 1

    rows: list[dict] = []

    # --- hierarchical rows ----------------------------------------------
    # Write out the tree for each edition: the country at the top, its states
    # beneath, and each state's districts beneath that. Every row names its
    # own parent, which is how the receiving system rebuilds the hierarchy.
    for v in vintages:
        country_fnid = f"{iso}{v}A0"
        for lvl in sorted(by_level):
            names = active[lvl].get(v, {})
            for u in sorted(names):
                # A state hangs directly off the country.
                if lvl == 1:
                    parent_name, parent_fnid = admin0, country_fnid
                    parent_surrogate = country_surrogate[v]
                else:
                    a1 = attr.get(v + 1, {}).get(u)
                    if a1 is None or a1[0] not in codes.get(1, {}):
                        # No admin1 to hang this district on in this vintage;
                        # skip rather than invent a parent. validate_levels()
                        # is what reports this as a data problem.
                        continue
                    parent_name = a1[1]
                    parent_fnid = fnid_at(1, a1[0], v)
                    parent_surrogate = surrogate.get((v, 1, a1[0]), country_surrogate[v])
                rows.append(
                    {
                        "label": _legacy_label(
                            "hierarchical", names[u], parent_name, lvl, parent_name
                        ),
                        "from_unit": surrogate[(v, lvl, u)],
                        "to_unit": parent_surrogate,
                        "relationship_type": f"admin{lvl}_{lvl - 1}",
                        "category": "hierarchical",
                        "from_unit_name": styled(names[u]),
                        "to_unit_name": styled(parent_name) if lvl > 1 else parent_name,
                        "fnid_from_unit": fnid_at(lvl, u, v),
                        "fnid_to_unit": parent_fnid,
                        "can_aggregate_to_from": False,
                        "can_aggregate_from_to": True,
                    }
                )

    # --- temporal rows ---------------------------------------------------
    # Vintages are exactly the event years, so the pair (V_prev, V) is driven
    # by the events at year V and nothing else.
    for prev, cur in zip(vintages, vintages[1:]):
        for lvl in sorted(by_level):
            li = by_level[lvl]
            ay, ay1 = active[lvl].get(prev, {}), active[lvl].get(cur, {})
            terr = li.event_graph.territorial
            ncs = li.event_graph.name_changes
            ev = terr[terr["event_year"] == cur]
            nc = ncs[ncs["event_year"] == cur]
            touched = set(ev["parent_id"].astype(str)) | set(ev["child_id"].astype(str))

            def add(kind, p, c, pname, cname, agg_to_from, agg_from_to):
                rows.append(
                    {
                        "label": _legacy_label(kind, pname, cname, lvl, ""),
                        "from_unit": surrogate[(prev, lvl, p)],
                        "to_unit": surrogate[(cur, lvl, c)],
                        "relationship_type": kind,
                        "category": "temporal",
                        "from_unit_name": styled(pname),
                        "to_unit_name": styled(cname),
                        "fnid_from_unit": fnid_at(lvl, p, prev),
                        "fnid_to_unit": fnid_at(lvl, c, cur),
                        "can_aggregate_to_from": agg_to_from,
                        "can_aggregate_from_to": agg_from_to,
                    }
                )

            for _, e in ev.iterrows():
                p, c = str(e["parent_id"]), str(e["child_id"])
                if p in ay and c in ay1:
                    add(
                        str(e["event_type"]).lower(),
                        p, c, e["parent_name"], e["child_name"],
                        True, False,
                    )

            nc_units: set[str] = set()
            for _, e in nc.iterrows():
                u = str(e["parent_id"])
                if u in ay and u in ay1:
                    add("name change", u, u, e["parent_name"], e["child_name"], True, True)
                    nc_units.add(u)

            for u in sorted((set(ay) & set(ay1)) - touched - nc_units):
                add("successor", u, u, ay[u], ay1[u], True, True)

    df = pd.DataFrame(rows, columns=LEGACY_RELATIONSHIP_COLUMNS)
    return (
        df.sort_values(
            ["category", "fnid_from_unit", "fnid_to_unit", "relationship_type"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )


def write_legacy_relationship_table(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Write the 11-column dialect to ``relationshiptable_{ISO}.csv``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# deliverable 3: AgStats FNID attach + workbook
# ---------------------------------------------------------------------------

def attach_stats_fnids(
    stats: pd.DataFrame,
    code_map: pd.DataFrame,
    *,
    iso: str,
    level: int = 2,
    unit_id_col: str,
    year_col: str,
    fnid_col: str = "FNID",
    keep_old_where_unmatched: bool = True,
) -> pd.DataFrame:
    """Stamp canonical FNIDs onto statistics that already carry a unit id.

    Thin wrapper over :func:`stablebound.fnid.assign_fnids`. When
    ``keep_old_where_unmatched`` and a prior ``fnid_col`` exists, keep the old
    value where the code map has no row for the unit (mirrors the restamp
    convention). Does not mutate the input.
    """
    stamped = assign_fnids(
        stats, code_map, iso=iso, level=level,
        unit_id_col=unit_id_col, year_col=year_col, fnid_col="_fnid_new",
    )["_fnid_new"].astype("string")
    stamped.index = stats.index
    out = stats.copy()
    if keep_old_where_unmatched and fnid_col in out.columns:
        old = out[fnid_col].astype("string")
        out[fnid_col] = stamped.where(stamped.str.len() > 0, old)
    else:
        out[fnid_col] = stamped.where(stamped.str.len() > 0, "")
    return out


def write_agstats_workbook(
    stats: pd.DataFrame,
    out_path: Path | str,
    *,
    sheet_name: str,
    scratch_dir: Path | str | None = None,
) -> Path:
    """Write the AgStats workbook (single sheet).

    Large workbooks (~40 MB) can time out writing directly to a project dir;
    pass ``scratch_dir`` to write to fast local scratch first, then copy.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch_dir) / out_path.name if scratch_dir is not None else None
    if scratch is not None and scratch.resolve() != out_path.resolve():
        stats.to_excel(scratch, sheet_name=sheet_name, index=False)
        shutil.copy2(scratch, out_path)
    else:
        stats.to_excel(out_path, sheet_name=sheet_name, index=False)
    return out_path


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def build_deliverables(
    *,
    iso: str,
    admin0: str,
    years: Iterable[int],
    out_dir: Path | str,
    adm2_graph: LineageGraph | None,
    admin1_graph: LineageGraph,
    baseline: pd.DataFrame,
    stats: pd.DataFrame | None = None,
    stats_unit_id_col: str = "ADM2_ID",
    stats_year_col: str = "Year",
    stats_sheet_name: str | None = None,
    stats_out_name: str | None = None,
    name_style: Literal["bare", "with_country"] = "bare",
    scratch_dir: Path | str | None = None,
    legacy_relationship: bool = False,
) -> dict[str, list[Path]]:
    """Emit all three deliverables into ``out_dir``.

    ``adm2_graph`` present → an admin2 country (levels 1 and 2); ``None`` →
    admin1-only (level 1). ``admin1_graph`` supplies the admin1 *events* for the
    relationship table (for India this is the separate ADM1_LINEAGE + level-1
    name-change-log graph; for a single-level admin2 country pass ``adm2_graph``;
    for an admin1-only country pass the country's own graph). Both code maps and
    the unit-defs tables are derived from the admin2 graph when present, else the
    admin1 graph. ``stats`` (already carrying ``stats_unit_id_col``) is optional;
    when given, the AgStats workbook is written too.

    ``legacy_relationship`` additionally writes ``relationshiptable_{ISO}.csv``
    in the 11-column dialect FEWS distributes — the interchange and round-trip
    format, *not* part of the upload set. Off by default so the three shipped
    deliverables stay exactly three.

    Returns ``{'admin_definitions': [...], 'relationship': [...], 'agstats': [...]}``,
    plus ``'legacy_relationship'`` when that flag is set.
    """
    years = list(years)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_admin2 = adm2_graph is not None
    levels = (1, 2) if has_admin2 else (1,)
    graph_for_defs = adm2_graph if has_admin2 else admin1_graph

    defs: dict[int, dict[int, pd.DataFrame]] = {lvl: {} for lvl in levels}
    if has_admin2:
        # Two-level country: admin2 units attributed up to admin1 via coarse_id.
        cm1 = build_admin1_code_map(graph_for_defs, baseline, iso=iso)
        cm2: pd.DataFrame | None = build_admin2_code_map(graph_for_defs, baseline, iso=iso)
        attribution = build_admin1_attribution_table(graph_for_defs, baseline)
        for lvl in levels:
            cm = cm1 if lvl == 1 else cm2
            for y in years:
                defs[lvl][y] = build_unit_defs_table(
                    graph_for_defs, baseline, cm,
                    iso=iso, admin0=admin0, level=lvl, year=y, attribution=attribution,
                )
    else:
        # Admin1-only country: each unit is its own admin1 — no coarse layer,
        # so bypass the attribution path (which requires coarse_id).
        cm1 = build_admin1_only_code_map(graph_for_defs, baseline, iso=iso)
        cm2 = None
        attribution = None  # no coarse layer to attribute up to
        for y in years:
            defs[1][y] = build_admin1_defs_table(
                graph_for_defs, baseline, cm1,
                iso=iso, admin0=admin0, year=y,
            )

    result: dict[str, list[Path]] = {}

    # (1) admin definitions
    result["admin_definitions"] = write_admin_definitions_workbooks(
        defs, iso=iso, years=years, levels=levels, out_dir=out_dir,
    )

    # (2) relationship table — admin2 first (matches India row grouping)
    level_inputs: list[RelationshipLevelInput] = []
    if has_admin2:
        level_inputs.append(RelationshipLevelInput(2, adm2_graph, cm2, defs[2]))
    level_inputs.append(RelationshipLevelInput(1, admin1_graph, cm1, defs[1]))
    rel = build_relationship_table(
        iso=iso, years=years, level_inputs=level_inputs,
        name_style=name_style, admin0=admin0,
    )
    rel_path = out_dir / f"{iso}_GeographicUnitRelationship.csv"
    write_relationship_table(rel, rel_path)
    result["relationship"] = [rel_path]

    # (2b) the same lineage in the dialect FEWS distributes. Written alongside,
    # never instead of, the upload file — India's frozen bundle is compared
    # byte-for-byte and must not gain or lose a row.
    if legacy_relationship:
        # A vintage reads its content from the *next* package year, so an event
        # in the final requested year needs one year beyond the window to be
        # representable. Build that year's defs here rather than making every
        # caller know the convention.
        tail = years[-1] + 1
        legacy_years = [*years, tail]
        legacy_inputs = []
        for li in level_inputs:
            defs_tail = (
                build_unit_defs_table(
                    graph_for_defs, baseline, li.code_map,
                    iso=iso, admin0=admin0, level=li.level, year=tail,
                    attribution=attribution,
                )
                if has_admin2
                else build_admin1_defs_table(
                    graph_for_defs, baseline, li.code_map,
                    iso=iso, admin0=admin0, year=tail,
                )
            )
            legacy_inputs.append(
                RelationshipLevelInput(
                    li.level, li.event_graph, li.code_map,
                    {**li.defs_by_year, tail: defs_tail},
                )
            )
        legacy = build_legacy_relationship_table(
            iso=iso, admin0=admin0, years=legacy_years, level_inputs=legacy_inputs,
            attribution=attribution if has_admin2 else None,
        )
        legacy_path = out_dir / f"relationshiptable_{iso}.csv"
        write_legacy_relationship_table(legacy, legacy_path)
        result["legacy_relationship"] = [legacy_path]

    # (3) AgStats (optional)
    if stats is not None:
        code_map = cm2 if has_admin2 else cm1
        level = 2 if has_admin2 else 1
        stamped = attach_stats_fnids(
            stats, code_map, iso=iso, level=level,
            unit_id_col=stats_unit_id_col, year_col=stats_year_col,
        )
        sheet = stats_sheet_name or "ag_stats"
        name = stats_out_name or f"{iso}_AgStats.xlsx"
        ag_path = write_agstats_workbook(
            stamped, out_dir / name, sheet_name=sheet, scratch_dir=scratch_dir,
        )
        result["agstats"] = [ag_path]

    return result
