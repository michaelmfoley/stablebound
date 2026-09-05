"""Readiness report for a country you are about to onboard.

One call, the four files a user actually has, and a graded answer to "will this
work, and what will I have to fix first?" — before committing to a full run.

The problem it solves: today a new country is a sequence of separate steps
(convert the relationship table, build the graph, match the shapefile, match the
statistics, export), each of which can fail in a way that only shows up two
steps later. Someone onboarding their sixth Asian country wants the bad news up
front, ordered by how much work it implies.

Grading rubric, stated here so it is not a mystery:

    PASS     nothing to do; this section will not cause trouble
    WARN     usable, but a human should look — sketchy matches, thin coverage
    FAIL     will produce wrong or missing output if run as-is
    SKIP     not applicable, or the input needed was not supplied

The overall grade is the worst section grade. A FAIL is never a refusal to
proceed — it is a statement that proceeding will cost you correctness.

Deliberately *not* an accuracy audit. Everything here is provable from the files
supplied; nothing depends on whether a name or date is historically right.

    from stablebound import assess_country

    report = assess_country(
        relationship_table="lineage.csv",
        baseline="baseline.csv",
        shapefile="modern.geojson",
        shapefile_name_column="ADM1_EN",
        stats="stats.csv",
    )
    print(report)
    report.grade          # "PASS" | "WARN" | "FAIL"
    report.sections[2]    # individual graded sections
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .lineage import LineageGraph
from .snapshot import build_snapshot
from .validate import validate_lineage

GRADES = ("PASS", "WARN", "FAIL", "SKIP")
_ORDER = {"SKIP": -1, "PASS": 0, "WARN": 1, "FAIL": 2}

#: A shapefile this many times larger (or smaller) than the lineage is almost
#: certainly the wrong admin level rather than a matching failure — GAUL L1 has
#: 17 Philippine regions where the FEWS RT has ~80 provinces, and someone
#: following the examples will attach L1 and see a match rate near zero.
_LEVEL_MISMATCH_RATIO = 1.8


@dataclass
class Section:
    """One graded finding, with the evidence that produced it."""

    name: str
    grade: str
    summary: str
    detail: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grade not in GRADES:
            raise ValueError(f"grade must be one of {GRADES}, got {self.grade!r}")


@dataclass
class ReadinessReport:
    """Graded sections plus the inputs they were derived from."""

    country: str
    sections: list[Section] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Worst section grade; SKIP sections do not count against it."""
        real = [s.grade for s in self.sections if s.grade != "SKIP"]
        if not real:
            return "SKIP"
        return max(real, key=lambda g: _ORDER[g])

    @property
    def failures(self) -> list[Section]:
        return [s for s in self.sections if s.grade == "FAIL"]

    @property
    def warnings(self) -> list[Section]:
        return [s for s in self.sections if s.grade == "WARN"]

    def __str__(self) -> str:
        mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "----"}
        width = 66
        lines = [
            "=" * width,
            f"StableBound readiness: {self.country}",
            "=" * width,
            "",
        ]
        for s in self.sections:
            lines.append(f"[{mark[s.grade]}]  {s.name}")
            lines.append(f"         {s.summary}")
            for d in s.detail:
                lines.append(f"           - {d}")
            lines.append("")
        lines.append("-" * width)
        lines.append(f"OVERALL: {self.grade}")
        if self.failures:
            lines.append("")
            lines.append("Fix before running:")
            for s in self.failures:
                lines.append(f"  * {s.name}: {s.summary}")
        if self.warnings:
            lines.append("")
            lines.append("Worth a look:")
            for s in self.warnings:
                lines.append(f"  * {s.name}: {s.summary}")
        lines.append("=" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "country": self.country,
            "grade": self.grade,
            "sections": [
                {"name": s.name, "grade": s.grade, "summary": s.summary,
                 "detail": s.detail, "data": s.data}
                for s in self.sections
            ],
        }


# --- dialect detection ---------------------------------------------------

#: Columns unique to the legacy 11-column FEWS dialect.
_LEGACY_MARKERS = {"category", "relationship_type", "fnid_from_unit"}


def detect_dialect(table: pd.DataFrame) -> str:
    """``"canonical"`` or ``"legacy"``. A user should not have to know.

    The two formats are told apart by the columns they carry, not by filename
    or by asking: the legacy FEWS table has ``category`` / ``relationship_type``
    / ``fnid_*``, the canonical one has ``event_year`` / ``event_type``.
    """
    cols = {str(c).lower() for c in table.columns}
    if _LEGACY_MARKERS & cols:
        return "legacy"
    if {"event_year", "event_type"} <= cols:
        return "canonical"
    return "unknown"


def _read(path_or_df) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    p = Path(path_or_df)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers. Generated lineages often ship UPPERCASE."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


# --- the sections --------------------------------------------------------


def _section_dialect(raw: pd.DataFrame) -> tuple[Section, str]:
    dialect = detect_dialect(raw)
    upper = [c for c in raw.columns if str(c) != str(c).lower()]
    if dialect == "unknown":
        return Section(
            "Relationship table dialect", "FAIL",
            "could not identify the format",
            ["expected either the canonical schema (event_year, event_type, "
             "parent_id, ...) or a legacy FEWS table (category, "
             "relationship_type, fnid_from_unit)",
             f"columns present: {sorted(str(c) for c in raw.columns)[:8]}"],
            {"dialect": dialect},
        ), dialect
    detail = []
    grade = "PASS"
    if upper:
        grade = "WARN"
        detail.append(
            f"{len(upper)} column name(s) are not lowercase "
            f"(e.g. {upper[:3]}); the canonical loader is case-sensitive, so "
            "these are normalised here but will need lowercasing at source"
        )
    if dialect == "legacy":
        detail.append(
            "legacy FEWS table — use Lineage.from_legacy_rt(); note a vintage "
            "labelled V lists the units in force AFTER year V, i.e. package "
            "snapshot(V+1)"
        )
    return Section(
        "Relationship table dialect", grade,
        f"{dialect} format, {len(raw)} row(s)", detail, {"dialect": dialect},
    ), dialect


def _section_lineage_health(graph: LineageGraph) -> Section:
    issues = validate_lineage(graph)
    errors = [i for i in issues if i.severity == "error"]
    warns = [i for i in issues if i.severity == "warning"]
    if errors:
        grade, summary = "FAIL", f"{len(errors)} error(s), {len(warns)} warning(s)"
    elif warns:
        grade, summary = "WARN", f"{len(warns)} warning(s)"
    else:
        grade, summary = "PASS", f"clean ({len(graph.events)} events)"
    detail = [f"[{i.severity}] {i.category}: {i.message}" for i in (errors + warns)[:6]]
    if len(errors) + len(warns) > 6:
        detail.append(f"...and {len(errors) + len(warns) - 6} more")
    return Section("Lineage health", grade, summary, detail,
                   {"n_errors": len(errors), "n_warnings": len(warns),
                    "categories": sorted({i.category for i in issues})})


def _section_coverage(graph: LineageGraph, baseline: pd.DataFrame) -> Section:
    """Baseline and events must describe the same universe and era."""
    detail: list[str] = []
    grade = "PASS"
    if "unit_id" not in baseline.columns:
        return Section("Baseline coverage", "FAIL",
                       "baseline has no 'unit_id' column", [], {})
    seed = set(baseline["unit_id"].astype(str))
    base_year = int(baseline["year"].min()) if "year" in baseline.columns else None

    ev = graph.events
    if not ev.empty and base_year is not None:
        early = int((ev["event_year"] < base_year).sum())
        if early:
            grade = "FAIL"
            detail.append(
                f"{early} event(s) predate the baseline year {base_year}; their "
                "effect is already baked into the baseline, so applying them "
                "again double-counts"
            )
    known = seed | set(ev["child_id"].astype(str)) if not ev.empty else seed
    orphans = sorted(set(ev["parent_id"].astype(str)) - known) if not ev.empty else []
    if orphans:
        grade = "FAIL"
        detail.append(
            f"{len(orphans)} event parent(s) appear in neither the baseline nor "
            f"as a child of an earlier event, e.g. {orphans[:3]} — these units "
            "exist nowhere and their events cannot be applied"
        )
    summary = f"{len(seed)} baseline unit(s)"
    if base_year is not None:
        summary += f" at {base_year}"
    if not ev.empty:
        summary += f", events {int(ev['event_year'].min())}-{int(ev['event_year'].max())}"
    return Section("Baseline coverage", grade, summary, detail,
                   {"n_baseline": len(seed), "base_year": base_year,
                    "n_orphans": len(orphans)})


def _match_bands(proposal, label: str, n_source: int) -> Section:
    """Shared grading for a shapefile or stats match proposal."""
    props = proposal.proposals
    base = props["method"].astype(str).str.split("+").str[0]
    counts = base.value_counts().to_dict()
    unmatched = counts.get("unmatched", 0)
    fuzzy = counts.get("fuzzy", 0)
    sketchy = len(proposal.sketchy)
    matched = len(props) - unmatched
    rate = matched / len(props) if len(props) else 0.0

    if rate < 0.80:
        grade = "FAIL"
    elif unmatched or sketchy:
        grade = "WARN"
    else:
        grade = "PASS"

    detail = [f"exact {counts.get('exact', 0)}, fuzzy {fuzzy}, "
              f"unmatched {unmatched}"]
    if sketchy:
        detail.append(f"{sketchy} fuzzy match(es) below the sketchy threshold — "
                      "review these by hand before trusting them")
    if unmatched:
        names = props[base == "unmatched"]["source_name"].astype(str).head(5).tolist()
        detail.append(f"unmatched names include: {names}")
    return Section(label, grade, f"{matched}/{len(props)} matched ({rate:.0%})",
                   detail, {"rate": rate, "unmatched": unmatched,
                            "sketchy": sketchy, "counts": counts})


def _section_shapefile(graph, shapefile, name_column, *, year, baseline,
                       coarse_column) -> Section:
    if shapefile is None or name_column is None:
        return Section("Shapefile match", "SKIP",
                       "no shapefile (or no name column) supplied", [], {})
    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover
        return Section("Shapefile match", "SKIP", "geopandas not installed", [], {})
    from .match import propose_shapefile_mapping

    gdf = shapefile if isinstance(shapefile, gpd.GeoDataFrame) else gpd.read_file(shapefile)
    if name_column not in gdf.columns:
        return Section(
            "Shapefile match", "FAIL",
            f"name column {name_column!r} not in the shapefile",
            [f"columns available: {sorted(map(str, gdf.columns))[:10]}"], {})

    # Wrong-admin-level check first: a large count gap explains a bad match rate
    # far better than "the names don't match", and the fix is different.
    # Count how many units the country's records expect, and compare against
    # how many shapes the map actually has. A map of the wrong administrative
    # tier — provinces where the records describe regions, say — matches
    # almost nothing, and the resulting report would blame the names when the
    # real problem is that the user opened the wrong file.
    seed = set(baseline["unit_id"].astype(str)) if baseline is not None else set()
    probe_year = year or (
        int(graph.max_event_year) + 1 if not graph.events.empty else None
    )
    expected = (
        len(build_snapshot(graph, probe_year, additional_units=seed))
        if probe_year
        else len(seed)
    )
    ratio = (len(gdf) / expected) if expected else 0.0
    level_note = None
    if expected and (ratio > _LEVEL_MISMATCH_RATIO or ratio < 1 / _LEVEL_MISMATCH_RATIO):
        level_note = (
            f"shapefile has {len(gdf)} features but the lineage expects about "
            f"{expected} units at {probe_year} — a {ratio:.1f}x gap usually means "
            "the wrong admin level, not a matching problem (GAUL L1 has 17 "
            "Philippine regions where the relationship table has ~80 provinces)"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proposal = propose_shapefile_mapping(
            gdf, graph, name_column=name_column, year=probe_year,
            baseline=baseline, coarse_column=coarse_column,
        )
    section = _match_bands(proposal, "Shapefile match", len(gdf))
    if level_note:
        section.grade = "FAIL"
        section.detail.insert(0, level_note)
        section.data["probable_wrong_admin_level"] = True
    section.data["n_features"] = len(gdf)
    section.data["n_expected"] = expected
    return section


def _section_stats(graph, stats, *, name_column, year_column, coarse_column,
                   baseline) -> Section:
    if stats is None:
        return Section("Statistics match", "SKIP", "no statistics supplied", [], {})
    df = _read(stats)
    if name_column is None:
        # FEWS-sourced statistics usually carry an FNID, which skips matching.
        fnid_like = [c for c in df.columns if "fnid" in str(c).lower()]
        if fnid_like:
            return Section(
                "Statistics match", "PASS",
                f"carries {fnid_like[0]!r} — join by FNID, no name matching needed",
                ["use Lineage.attach_stats_by_fnid(); this is exact and skips "
                 "the most expensive part of onboarding"],
                {"fnid_column": fnid_like[0]})
        return Section("Statistics match", "SKIP",
                       "no stats name column supplied", [], {})
    from .match import propose_stats_mapping

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proposal = propose_stats_mapping(
            df, graph, name_column=name_column, year_column=year_column,
            coarse_column=coarse_column, baseline=baseline, year_aware=bool(year_column),
        )
    section = _match_bands(proposal, "Statistics match", len(df))
    issues = getattr(proposal, "ancestor_issues", None)
    if issues is not None and len(issues):
        section.detail.append(
            f"{len(issues)} row(s) needed an ancestor walk that was ambiguous or "
            "found nothing — these are dropped rather than guessed"
        )
        if section.grade == "PASS":
            section.grade = "WARN"
    extrapolated = getattr(proposal, "extrapolated", None)
    if extrapolated is not None and len(extrapolated):
        section.detail.append(
            f"{len(extrapolated)} row(s) fall outside the span the lineage "
            "describes; ids were assigned by extrapolation"
        )
        if section.grade == "PASS":
            section.grade = "WARN"
    return section


def _section_completeness_forecast(graph, stats, baseline, *, name_column,
                                   year_column) -> Section:
    """How complete will the output be — answerable before a full run.

    Per year: how many of the units alive that year appear in the statistics at
    all. This is the cheap version of the completeness report, computed from the
    inputs rather than from a finished aggregation, so a user can decide whether
    a country is worth running before spending the time.
    """
    if stats is None or year_column is None:
        return Section("Completeness forecast", "SKIP",
                       "needs statistics with a year column", [], {})
    df = _read(stats)
    if year_column not in df.columns:
        return Section("Completeness forecast", "SKIP",
                       f"no {year_column!r} column in the statistics", [], {})
    seed = set(baseline["unit_id"].astype(str)) if baseline is not None else set()
    id_col = next((c for c in ("unit_id", "ADM2_ID", "ADM1_ID") if c in df.columns), None)
    if id_col is None and name_column not in df.columns:
        return Section("Completeness forecast", "SKIP",
                       "statistics carry neither unit ids nor the name column", [], {})

    years = sorted({int(y) for y in pd.to_numeric(df[year_column], errors="coerce").dropna()})
    rows = []
    for y in years:
        alive = build_snapshot(graph, y, additional_units=seed)
        if not alive:
            continue
        sub = df[pd.to_numeric(df[year_column], errors="coerce") == y]
        if id_col:
            reporting = set(sub[id_col].astype(str)) & alive
        else:
            reporting = set()  # name-keyed: needs the matcher, so skip the count
        rows.append((y, len(reporting), len(alive)))
    if not rows or all(r[1] == 0 for r in rows):
        return Section("Completeness forecast", "SKIP",
                       "statistics are name-keyed; run the matcher first", [], {})

    worst = min(rows, key=lambda r: r[1] / r[2])
    mean = sum(r[1] / r[2] for r in rows) / len(rows)
    grade = "PASS" if mean >= 0.9 else "WARN" if mean >= 0.6 else "FAIL"
    return Section(
        "Completeness forecast", grade,
        f"mean {mean:.0%} of live units report, across {len(rows)} year(s)",
        [f"worst year {worst[0]}: {worst[1]}/{worst[2]} units reporting "
         f"({worst[1] / worst[2]:.0%})",
         "this counts units present in the statistics at all, not values — it "
         "is an upper bound on the completeness of the final product"],
        {"mean": mean, "worst_year": worst[0], "per_year": rows},
    )


def _section_fews(graph, baseline, *, admin_level) -> Section:
    """Will the FEWS deliverable build, and does the code space fit?"""
    from .fnid import _CODE_WIDTH

    detail: list[str] = []
    grade = "PASS"
    if admin_level >= 3:
        return Section(
            "FEWS exportability", "FAIL",
            f"admin level {admin_level} is not supported",
            ["FNID code width for level 3+ is unconfirmed; levels 1 and 2 work "
             "today (see stablebound.fnid._CODE_WIDTH)"],
            {"admin_level": admin_level})

    has_coarse = baseline is not None and "coarse_id" in baseline.columns
    if admin_level == 2 and not has_coarse:
        grade = "FAIL"
        detail.append(
            "an admin2 lineage needs 'coarse_id'/'coarse_name' on the baseline: "
            "every admin2 FNID is keyed on the admin1 the unit originated in, "
            "and there is no honest way to synthesise one"
        )

    # Code-space capacity: predict FNIDOverflowError rather than hit it at write.
    width = _CODE_WIDTH.get(admin_level)
    if width and baseline is not None and "unit_id" in baseline.columns:
        seed = set(baseline["unit_id"].astype(str))
        ev = graph.events
        universe = seed | (set(ev["child_id"].astype(str)) if not ev.empty else set())
        if admin_level == 1:
            capacity = 100 ** (width // 2)
            if len(universe) > capacity:
                grade = "FAIL"
                detail.append(
                    f"{len(universe)} distinct units exceed the {width}-character "
                    f"level-{admin_level} code space (capacity {capacity})"
                )
            else:
                detail.append(
                    f"{len(universe)} unit(s) fit the {width}-character code "
                    f"space (capacity {capacity})"
                )
    return Section("FEWS exportability", grade,
                   f"admin level {admin_level}" + ("" if grade == "PASS" else " — blocked"),
                   detail, {"admin_level": admin_level, "has_coarse": has_coarse})


# --- entry point ---------------------------------------------------------


def assess_country(
    *,
    relationship_table,
    baseline=None,
    shapefile=None,
    shapefile_name_column: str | None = None,
    shapefile_coarse_column: str | None = None,
    stats=None,
    stats_name_column: str | None = None,
    stats_year_column: str | None = None,
    stats_coarse_column: str | None = None,
    country: str = "(unnamed)",
    admin_level: int = 1,
    year: int | None = None,
) -> ReadinessReport:
    """Grade a country's four input files before committing to a full run.

    Every argument except ``relationship_table`` is optional; a section whose
    input is missing grades SKIP rather than failing, so this is useful with
    only a lineage in hand and gets more informative as you supply more.

    Accepts paths or in-memory frames. The relationship table may be in either
    dialect — canonical or legacy FEWS — and is detected, not asked about.

    Args:
        relationship_table: canonical lineage or legacy FEWS table.
        baseline: units alive at the base year. Strongly recommended; without
            it, units governed by no event are invisible.
        shapefile: modern geometry, with ``shapefile_name_column`` naming the
            column to match on.
        stats: statistics. Supply ``stats_year_column`` to enable the year-aware
            matcher and the completeness forecast.
        admin_level: 1 or 2. Controls the FEWS exportability checks.
        year: the year to match the shapefile against. Defaults to the year
            after the last event, i.e. the modern snapshot.

    Returns:
        A :class:`ReadinessReport`. ``print(report)`` renders the text form;
        ``report.grade`` is the worst section grade.
    """
    report = ReadinessReport(country=country)

    raw = _read(relationship_table)
    dialect_section, dialect = _section_dialect(raw)
    report.sections.append(dialect_section)
    if dialect == "unknown":
        return report

    if dialect == "legacy":
        from .rt_convert import (
            convert_relationship_table_to_lineage,
            read_legacy_relationship_table,
        )

        prepared = (
            relationship_table
            if isinstance(relationship_table, pd.DataFrame)
            else read_legacy_relationship_table(relationship_table)
        )
        lin_df, derived_baseline = convert_relationship_table_to_lineage(
            prepared, country=country[:2].upper() if country != "(unnamed)" else "XX",
            admin_level=admin_level,
        )
        table = lin_df
        base_df = _read(baseline) if baseline is not None else derived_baseline
    else:
        table = _normalise_columns(raw)
        base_df = _normalise_columns(_read(baseline)) if baseline is not None else None

    try:
        graph = LineageGraph.from_dataframe(table, validate=False)
    except Exception as exc:  # noqa: BLE001
        report.sections.append(Section(
            "Lineage health", "FAIL", f"could not build the graph: {exc}", [], {}))
        return report

    report.sections.append(_section_lineage_health(graph))
    if base_df is not None:
        report.sections.append(_section_coverage(graph, base_df))
    else:
        report.sections.append(Section(
            "Baseline coverage", "WARN", "no baseline supplied",
            ["units that never appear in an event are invisible without one, so "
             "snapshots and grouping will be incomplete"], {}))

    report.sections.append(_section_shapefile(
        graph, shapefile, shapefile_name_column, year=year, baseline=base_df,
        coarse_column=shapefile_coarse_column))
    report.sections.append(_section_stats(
        graph, stats, name_column=stats_name_column, year_column=stats_year_column,
        coarse_column=stats_coarse_column, baseline=base_df))
    report.sections.append(_section_completeness_forecast(
        graph, stats, base_df, name_column=stats_name_column,
        year_column=stats_year_column))
    report.sections.append(_section_fews(graph, base_df, admin_level=admin_level))
    return report
