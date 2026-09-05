"""User-facing :class:`StableBoundary` orchestrator.

Wires the algorithm modules (snapshot, groups, dissolve, reconcile,
stats) into a small API. Researchers should rarely need to call the
underlying algorithm functions directly — they construct a
:class:`Lineage`, attach a shapefile, instantiate
:class:`StableBoundary`, and call ``build_boundaries`` /
``aggregate_stats``.

    from stablebound import Lineage, StableBoundary
    ln = Lineage("IN")
    ln.attach_shapefile("modern.shp", mapping="mapping.csv",
                        name_column="DISTRICT_NAME")
    sb = StableBoundary(ln, target_year=1991)
    sb.build_boundaries()                          # geometries
    sb.aggregate_stats(stats="ag_stats.csv",       # stats
                      extensive=["rice_area_ha", "rice_production_mt"],
                      intensive={"yield": ("rice_production_mt", "rice_area_ha")})
    df = sb.get_stats(variable="rice_area_ha", year=2018)

Outputs land in ``output_dir`` (see ``docs/USAGE.md`` "Outputs"). The
same data is available in-memory through the getter methods so
notebook users never need to read from disk.

State management note: the class uses ``_lazy`` attributes for
intermediate results (modern_gdf, remaps, stats outputs). Methods
populate these on first use; downstream methods reuse them. The
``refresh=True`` flag on ``build_boundaries`` and ``aggregate_stats``
forces a recomputation rather than reading from disk cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .data_dict import DataDictionary
from .dissolve import dissolve
from .groups import build_stable_groups
from .lineage import name_history
from .lineage_class import Lineage
from .match import attach_stats_ids
from .reconcile import reconcile
from .schemas import STATS_REQUIRED_COLUMNS, validate_stats
from .snapshot import build_snapshot, infer_year
from .stats import aggregate

# Canonical id column name on the attached shapefile after
# Lineage.attach_shapefile(...). The legacy Config used to let users
# name their own column; the new design enforces "unit_id" by
# convention so downstream code doesn't have to plumb it through.
ID_COLUMN = "unit_id"

# Schema version embedded in ``summary.json``. Bump whenever the
# summary's shape changes so on-disk caches from older versions get
# rejected and rebuilt. The disk cache short-circuit reads this and
# falls through to a fresh build on mismatch.
#
# Schema versions:
#   1 — initial.
#   2 — adds ``n_unmatched_features``; ``stable_*.geojson`` gains an
#       ``is_matched`` column; unmatched_features.geojson sidecar.
#   3 — cache validation policy tightens (target_year + country_code
#       must match the on-disk summary). Bump forces a rebuild so
#       legacy caches from v0.1.x-pre-cache-fix don't sneak through.
STABLE_SUMMARY_SCHEMA = 3


class StableBoundary:
    """User-facing orchestrator for the stable boundary product.

    The lineage carries the parsed relationship table, baseline, and
    modern shapefile. The constructor decides the analysis window
    (``target_year``, ``max_year``) and writes outputs to
    ``output_dir``. The class itself is mostly orchestration — every
    transformation lives in a dedicated algorithm module.
    """

    def __init__(
        self,
        lineage: Lineage,
        *,
        target_year: int | None = None,
        max_year: int | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.lineage = lineage
        # ``target_year`` is the year stable boundaries are pinned to —
        # the canonical "base year" the paper refers to. Defaults to
        # the earliest valid year for the lineage (validity_start_year
        # for bundled countries, min_event_year otherwise).
        self.target_year = target_year if target_year is not None else lineage.min_year
        # ``max_year`` caps the analysis window. Default: the inferred
        # shapefile vintage if attached, else lineage.lineage.max_event_year.
        # Resolved at build time so the lineage doesn't need a
        # shapefile attached at constructor time (it does, by the time
        # build_boundaries runs).
        self._explicit_max_year = max_year
        self.output_dir = Path(output_dir) if output_dir else Path("./stablebound_out")

        # Lazy state. Populated by build_boundaries / aggregate_stats.
        self._modern_gdf: gpd.GeoDataFrame | None = None
        self._all_remaps: dict[int, dict[str, str]] | None = None
        self._max_year: int | None = None
        self._stats_agg: pd.DataFrame | None = None
        self._reconcile_flags: pd.DataFrame | None = None
        self._name_history: pd.DataFrame | None = None
        self._summary: dict | None = None
        # Validation findings stashed for the validation_report() API.
        # build_boundaries() populates contiguity_issues; aggregate_stats
        # populates stats_issues.
        self._contiguity_issues: list = []
        self._stats_issues: list = []

    # --- Geometry pipeline ----------------------------------------------

    def build_boundaries(self, refresh: bool = False) -> None:
        """Build stable boundaries for every year in [target_year, max_year].

        Pipeline:
            1. Short-circuit if already cached on disk and refresh=False.
            2. Pull the parsed LineageGraph + modern shapefile from
               the lineage.
            3. Determine max_year (constructor override OR shapefile
               vintage via Algorithm 2).
            4. Assemble ``additional_units`` from baseline + modern
               shapefile.
            5. For each year in [target_year, max_year]: build year-
               specific remap, dissolve modern shapefile, write per-
               year geojson.
            6. Write canonical ``remap.json`` (target-year remap),
               per-year snapshots, name history, summary.
            7. Cache results in-memory for subsequent calls.

        ``refresh=True`` forces step 5 onward to re-run even if the
        disk cache exists.
        """
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        if not refresh and self._has_cached_boundaries(out):
            self._load_cached_boundaries(out)
            return

        graph = self.lineage.lineage
        modern_gdf = self.lineage.shapefile
        if ID_COLUMN not in modern_gdf.columns:
            raise RuntimeError(
                f"Attached shapefile is missing the {ID_COLUMN!r} "
                "column. Lineage.attach_shapefile should populate it; "
                "did you bypass the attach helper?"
            )

        # Step 3: max_year. Three sources, in order:
        # - Constructor override (self._explicit_max_year).
        # - The lineage's inferred shapefile_year (if attached, which
        #   it must be by now).
        # - Fallback to graph.max_event_year + 1 if no shapefile_year
        #   could be inferred (e.g., all IDs were unmatched).
        if self._explicit_max_year is not None:
            max_year = self._explicit_max_year
            vintage_report = (
                f"max_year={max_year} taken from constructor (no inference performed)\n"
            )
        else:
            try:
                inferred = self.lineage.shapefile_year
                # Take the later of the inferred vintage and the source's
                # coverage end. Year inference minimises symmetric difference
                # and breaks ties toward the EARLIEST year, so a country whose
                # units have been stable for decades infers a vintage far in
                # the past: Bangladesh's ids are identical from 1986 onward,
                # so it inferred 1985 and produced three boundary files for a
                # 2025 shapefile. Coverage end is the honest upper bound —
                # the last year the source actually affirms the unit set.
                # Guessing the map's year from its district names can only
                # narrow things down to a span where the districts were
                # unchanged, and it reports the earliest year of that span.
                # If the country's records are known to run later than that,
                # prefer the later date: the map is equally consistent with
                # it, and stopping early would throw away years of results.
                coverage = self.lineage.coverage_end_year
                if coverage is not None and coverage > inferred:
                    max_year = coverage
                    vintage_report = (
                        f"max_year={max_year} from the source's coverage end "
                        f"(inferred shapefile year was {inferred}; the unit set "
                        f"is unchanged between the two, so inference tied and "
                        f"took the earliest)\n"
                    )
                else:
                    max_year = inferred
                    vintage_report = (
                        f"max_year={max_year} taken from inferred shapefile year\n"
                    )
            except RuntimeError:
                # No shapefile_year available; fall back.
                max_year = graph.max_event_year + 1
                vintage_report = (
                    f"max_year={max_year} (fallback: graph.max_event_year + 1; "
                    "shapefile_year could not be inferred)\n"
                )
        # Asking for years the source does not cover is legitimate — it just
        # cannot be confirmed. Say so rather than implying the boundaries are
        # attested for those years.
        coverage = self.lineage.coverage_end_year
        if coverage is not None and max_year > coverage:
            import warnings as _w

            _w.warn(
                f"max_year={max_year} extends past this lineage's coverage end "
                f"({coverage}). Boundaries for {coverage + 1}-{max_year} assume "
                "no further administrative changes; the source data does not "
                "confirm that. Check whether a newer relationship table exists.",
                UserWarning,
                stacklevel=2,
            )
            vintage_report += (
                f"WARNING: max_year={max_year} exceeds coverage end {coverage}; "
                f"years {coverage + 1}-{max_year} are assumed unchanged.\n"
            )
        (out / "shapefile_vintage_report.txt").write_text(vintage_report)

        # Step 4: assemble ``additional_units``. Baseline + modern IDs
        # INCLUDING sentinels — unmatched features (UNMATCHED_<idx>)
        # flow through as singleton stable groups so they appear in
        # the output rather than vanishing.
        modern_unit_ids = set(modern_gdf[ID_COLUMN].dropna().astype(str))
        baseline_unit_ids: set[str] = set()
        if self.lineage.baseline is not None:
            baseline_unit_ids = set(self.lineage.baseline["unit_id"].astype(str))
        extra_units = baseline_unit_ids | modern_unit_ids

        # Step 5: per-year stable groups + dissolutions.
        # Print one progress heads-up for non-trivial year ranges so
        # notebook users see something is happening on long runs. Per-
        # year geometry dissolution is the slowest step on large
        # countries (~17s on India for a 28-year window). Gated at
        # year-count >= 15 so small examples (Exampleland's 11 years)
        # don't see the message — they finish in well under a second
        # and the heads-up is misleading.
        import sys as _sys
        import time as _time
        _year_count = max_year - self.target_year + 1
        _verbose = _year_count >= 15
        _t_start = _time.perf_counter()
        if _verbose:
            print(
                f"[stablebound] building stable boundaries for "
                f"{_year_count} years ({self.target_year}-{max_year})...",
                file=_sys.stderr,
                flush=True,
            )

        all_remaps: dict[int, dict[str, str]] = {}
        for year in range(self.target_year, max_year + 1):
            remap = build_stable_groups(graph, year, additional_units=extra_units)
            all_remaps[year] = remap
            stable_gdf = dissolve(modern_gdf, remap, ID_COLUMN)
            # Annotate matched-vs-unmatched. Sentinel stable_ids start
            # with "UNMATCHED_"; the user can filter them in downstream
            # analysis or display them with a distinct style.
            stable_gdf["is_matched"] = ~stable_gdf["stable_id"].astype(str).str.startswith(
                "UNMATCHED_"
            )
            # GeoJSON over Shapefile: (a) easy to diff against legacy
            # outputs and inspect by hand; (b) no 10-char column-name
            # limit; (c) carries CRS metadata cleanly.
            stable_gdf.to_file(out / f"stable_{year}.geojson", driver="GeoJSON")

        if _verbose:
            print(
                f"[stablebound] boundaries built in "
                f"{_time.perf_counter() - _t_start:.1f}s "
                f"({_year_count} years).",
                file=_sys.stderr,
                flush=True,
            )

        # Step 6a: write down which districts were grouped with which.
        #
        # Public contract: this file is written once and read forever after.
        # Every figure that comes out of this package is filed under a group
        # named here, so recomputing it later — even by the same code, even
        # from the same inputs — would risk renaming a group and quietly
        # breaking comparability with everything already published.
        base_remap = all_remaps[self.target_year]
        with (out / "remap.json").open("w") as f:
            json.dump(base_remap, f, indent=2, sort_keys=True)

        # Step 6a-bis: spatial contiguity check on the target-year remap.
        # Surfaces homonym name-matching bugs in the relationship table.
        # Issues are cached on the instance; see validation_report().
        from .validate import validate_stable_group_contiguity
        self._contiguity_issues = validate_stable_group_contiguity(
            base_remap, modern_gdf, ID_COLUMN
        )
        if self._contiguity_issues:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Contiguity check produced %d finding(s). "
                "Call sb.validation_report() for details.",
                len(self._contiguity_issues),
            )

        # Step 6b: snapshots per year.
        snapshots = {
            year: sorted(build_snapshot(graph, year, additional_units=extra_units))
            for year in range(self.target_year, max_year + 1)
        }
        with (out / "snapshots.json").open("w") as f:
            json.dump(snapshots, f, indent=2, sort_keys=True)

        # Step 6c: name history.
        nh = name_history(graph, base_remap, self.target_year, max_year)
        nh.to_csv(out / "name_history.csv", index=False)

        # Step 6c-bis: unmatched features sidecar.
        # If the attached shapefile carried any sentinel UNMATCHED_<idx>
        # IDs, mirror the lineage's unmatched_features manifest into
        # the product output dir so a user inspecting `_out/` sees
        # what fell out of the matcher.
        unmatched = self.lineage.unmatched_features
        if not unmatched.empty:
            (out / "unmatched_features.geojson").unlink(missing_ok=True)
            unmatched.to_file(
                out / "unmatched_features.geojson", driver="GeoJSON"
            )

        # Step 6d: a short record of what this run was, kept alongside the
        # results. It is what lets a later run recognise its own output and
        # reuse it, and what tells a user opening the folder months later
        # which country and which years they are looking at.
        summary = {
            "_schema": STABLE_SUMMARY_SCHEMA,
            "country_code": self.lineage.country_code,
            "n_events": int(len(graph.events)),
            "n_stable_groups_at_base": len(set(base_remap.values())),
            "n_modern_units": int(len(modern_gdf)),
            "n_unmatched_features": int(len(unmatched)),
            "year_range": [self.target_year, max_year],
        }
        with (out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        # Step 7: in-memory cache.
        self._modern_gdf = modern_gdf
        self._all_remaps = all_remaps
        self._max_year = max_year
        self._name_history = nh
        self._summary = summary

    def get_boundary(self, year: int) -> gpd.GeoDataFrame:
        """Return the dissolved stable shapefile for ``year``.

        Reads from disk every call — GeoDataFrames can be large and
        notebook users typically only need one or two years at a time.
        """
        if year is None:
            raise ValueError("year is required.")
        path = self.output_dir / f"stable_{year}.geojson"
        if not path.exists():
            raise FileNotFoundError(
                f"No stable boundary cached for year={year}. "
                "Did you call build_boundaries()?"
            )
        return gpd.read_file(path)

    def infer_shapefile_year(self) -> tuple[int, dict[int, int]]:
        """Validate the modern shapefile's vintage against the lineage.

        Wraps :func:`stablebound.snapshot.infer_year`. Useful as a
        standalone diagnostic when adopting a new shapefile.
        """
        graph = self.lineage.lineage
        modern_gdf = self.lineage.shapefile
        units = set(modern_gdf[ID_COLUMN].dropna().astype(str))
        candidates = range(self.target_year, graph.max_event_year + 5)
        base = set(self.lineage.baseline["unit_id"].astype(str))
        return infer_year(units, graph, candidates, additional_units=base)

    # --- Stats pipeline -------------------------------------------------

    def aggregate_stats(
        self,
        stats: str | Path | pd.DataFrame,
        *,
        mapping: str | Path | pd.DataFrame | None = None,
        name_column: str | None = None,
        year_column: str | None = None,
        coarse_column: str | None = None,
        data_dict: DataDictionary | None = None,
        stats_columns: dict[str, str] | None = None,
        extensive: list[str] | None = None,
        intensive: dict[str, tuple[str, str]] | None = None,
        ignore: list[str] | None = None,
        reconcile_mode: str = "off",
        reconcile_tau_drop: float = 0.15,
        reconcile_tau_sum: float = 0.15,
        reconcile_window: int = 3,
        refresh: bool = False,
    ) -> None:
        """Run reconciliation + aggregation.

        Stats can be a path or a DataFrame. If ``mapping`` is provided
        (a CSV path or DataFrame from
        :meth:`Lineage.propose_stats_mapping`), it's joined onto the
        stats to attach ``unit_id``s. If ``stats_columns`` is provided,
        columns are renamed to canonical names before validation.

        ``data_dict`` (optional) is shorthand for the four kwargs
        ``stats_columns`` / ``extensive`` / ``intensive`` / ``ignore``;
        passing it overrides any individually-passed kwargs. When
        ``data_dict`` and an individual kwarg are both passed, the
        individual kwarg wins (lets the user override one field of a
        shared DataDictionary inline).

        The reconcile diagnostic is currently disabled:
        ``reconcile_mode`` must be ``"off"`` (the default) and any other
        value raises ``NotImplementedError``. No
        ``reconciliation_flags.csv`` is written (and any stale file
        from a prior run is deleted). Stats × lineage validation
        still runs regardless of mode.
        """
        # 2026-06-04: reconcile is disabled by default and the active /
        # flag-only modes raise. The published implementation of the
        # window-averaged drop and sum-jump tests flags ordinary post-event
        # reporting handoffs as suspected double-counting at a ~99% rate on
        # the India dataset, and `merge` / `subtract` modes delete
        # legitimate child rows in response. Use mode="off" until the
        # diagnostic is rebuilt around per-year (rather than window-averaged)
        # overlap detection. The analysis behind this is in the authors'
        # diagnostic notebook (not part of this repository).
        if reconcile_mode != "off":
            raise NotImplementedError(
                f"reconcile_mode={reconcile_mode!r} is disabled. "
                "The current implementation flags clean post-event reporting "
                "handoffs as double-counting at a ~99% rate on real data, "
                "and the active modes (merge/subtract) delete legitimate "
                "post-event child rows. Use reconcile_mode='off' until the "
                "diagnostic is reworked. See the Known Issues section of "
                "the README."
            )
        # Resolve the data dictionary. The individual kwargs win when
        # passed; data_dict supplies defaults for any kwargs left None.
        dd = data_dict or DataDictionary()
        stats_columns = stats_columns if stats_columns is not None else dd.stats_columns
        extensive = extensive if extensive is not None else dd.extensive
        intensive = intensive if intensive is not None else dd.intensive
        ignore = ignore if ignore is not None else dd.ignore

        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # No early short-circuit on prior runs. A user calling
        # aggregate_stats again expects the passed stats to be used;
        # the previous behavior was to silently return the in-memory
        # cache regardless. If the user wants the disk cache hot, they
        # can call get_stats() (which lazy-loads from disk).

        # Geometry-first: stats aggregation needs the remap from
        # build_boundaries. Run it if it hasn't been run yet.
        if self._all_remaps is None:
            self.build_boundaries(refresh=False)

        graph = self.lineage.lineage
        max_year = self._max_year  # type: ignore[assignment]

        # Step 1: load + preprocess stats.
        stats_df = _load_stats(
            stats,
            mapping=mapping,
            name_column=name_column,
            stats_columns=stats_columns,
            extensive=extensive,
            intensive=intensive,
            ignore=ignore,
            target_year=self.target_year,
            mapping_year_column=year_column,
            mapping_coarse_column=coarse_column,
        )

        # Step 2: cross-check stats against the lineage. Surfaces
        # upstream FNID-assignment bugs. Errors raise; warnings/infos log.
        from .validate import (
            LineageDataError,
            format_issues,
            validate_stats_lineage_consistency,
        )
        modern_unit_ids = set(
            self.lineage.shapefile[ID_COLUMN].dropna().astype(str)
        )
        stats_issues = validate_stats_lineage_consistency(
            stats_df, graph, modern_unit_ids=modern_unit_ids
        )
        stats_errors = [i for i in stats_issues if i.severity == "error"]
        stats_other = [i for i in stats_issues if i.severity != "error"]
        if stats_errors:
            raise LineageDataError(
                "Stats × lineage validation found errors:\n\n"
                + format_issues(stats_errors)
            )
        # Stash non-error findings for validation_report().
        self._stats_issues = stats_other
        if stats_other:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Stats × lineage produced %d finding(s). "
                "Call sb.validation_report() for details.",
                len(stats_other),
            )

        # Step 3: reconcile (via shared helper; persists flags file +
        # remap changes; off mode returns empty flags + unmodified inputs).
        reconciled, modified_remap, flags = self._run_reconcile(
            stats_df,
            mode=reconcile_mode,
            tau_drop=reconcile_tau_drop,
            tau_sum=reconcile_tau_sum,
            window=reconcile_window,
        )

        # Step 4: aggregate (group-by + sum + late-reporting flags), and
        # derive the declared intensives in the same call. Pass
        # modified_remap — if merge mode merged any stable groups, the
        # aggregation needs the post-merge remap. Graph is passed so the
        # aggregator can compute per-year snapshots for snapshot-aware
        # late-reporting detection. Each intensive (yield) is computed
        # from only the units that reported both of its inputs in a cell;
        # see ``stablebound.stats.aggregate``.
        agg = aggregate(
            reconciled, modified_remap, graph, self.target_year, max_year,
            intensive=(intensive or None),
        )

        # Step 6: write aggregated stats.
        agg.to_csv(out / "stats_aggregated.csv", index=False)
        # Dataset-level companion to the per-row columns. Free to produce and
        # it is the artifact that goes into a paper appendix, so write it
        # rather than making every user call the method.
        from .completeness import summarize as _summarize
        try:
            _summarize(agg).to_csv(out / "completeness_report.csv", index=False)
        except (ValueError, KeyError):
            # A frame with no completeness columns (e.g. everything filtered
            # out) simply gets no report; never fail the run over a summary.
            pass


        self._stats_agg = agg

    def reconcile_stats(
        self,
        stats: str | Path | pd.DataFrame,
        *,
        mode: str = "off",
        mapping: str | Path | pd.DataFrame | None = None,
        name_column: str | None = None,
        year_column: str | None = None,
        coarse_column: str | None = None,
        data_dict: DataDictionary | None = None,
        stats_columns: dict[str, str] | None = None,
        extensive: list[str] | None = None,
        intensive: dict[str, tuple[str, str]] | None = None,
        ignore: list[str] | None = None,
        tau_drop: float = 0.15,
        tau_sum: float = 0.15,
        window: int = 3,
    ) -> pd.DataFrame:
        """Run reconciliation diagnostics WITHOUT aggregating.

        Same setup as :meth:`aggregate_stats` up through the reconcile
        step: builds boundaries if not yet built, loads + validates
        the stats frame, and calls :func:`reconcile` at the requested
        mode. Writes ``reconciliation_flags.csv`` (or deletes any
        stale one when ``mode="off"``). Returns the flags DataFrame
        for notebook inspection.

        Useful when:

        - The user wants to inspect potential reporting bugs without
          committing to aggregation.
        - The user prefers ``aggregate_stats(reconcile_mode="off")``
          for the math and runs this separately for diagnostics.
        - Researchers iterating on ``tau_drop``/``tau_sum`` thresholds
          want a tight feedback loop without re-aggregating.

        Both reconciliation and breakpoint detection are intentionally
        toggleable — neither will ever be perfectly solved, and the
        package shouldn't force unfinished diagnostics on users who'd
        rather skip them.
        """
        # Reconcile is disabled; see the aggregate_stats note above.
        if mode != "off":
            raise NotImplementedError(
                f"reconcile mode={mode!r} is disabled. "
                "Use mode='off' until the diagnostic is reworked. "
                "See the Known Issues section of the README."
            )
        dd = data_dict or DataDictionary()
        stats_columns = stats_columns if stats_columns is not None else dd.stats_columns
        extensive = extensive if extensive is not None else dd.extensive
        intensive = intensive if intensive is not None else dd.intensive
        ignore = ignore if ignore is not None else dd.ignore

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Geometry-first: need the remap.
        if self._all_remaps is None:
            self.build_boundaries(refresh=False)

        stats_df = _load_stats(
            stats,
            mapping=mapping,
            name_column=name_column,
            stats_columns=stats_columns,
            extensive=extensive,
            intensive=intensive,
            ignore=ignore,
            target_year=self.target_year,
            mapping_year_column=year_column,
            mapping_coarse_column=coarse_column,
        )

        # Stats × lineage validation always runs — errors should not
        # be silenceable by the reconcile off-switch.
        from .validate import (
            LineageDataError,
            format_issues,
            validate_stats_lineage_consistency,
        )
        modern_unit_ids = set(
            self.lineage.shapefile[ID_COLUMN].dropna().astype(str)
        )
        stats_issues = validate_stats_lineage_consistency(
            stats_df, self.lineage.lineage, modern_unit_ids=modern_unit_ids
        )
        stats_errors = [i for i in stats_issues if i.severity == "error"]
        stats_other = [i for i in stats_issues if i.severity != "error"]
        if stats_errors:
            raise LineageDataError(
                "Stats × lineage validation found errors:\n\n"
                + format_issues(stats_errors)
            )
        self._stats_issues = stats_other
        if stats_other:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Stats × lineage produced %d finding(s). "
                "Call sb.validation_report() for details.",
                len(stats_other),
            )

        _reconciled, _new_remap, flags = self._run_reconcile(
            stats_df, mode=mode, tau_drop=tau_drop, tau_sum=tau_sum, window=window,
        )
        return flags

    def _run_reconcile(
        self,
        stats_df: pd.DataFrame,
        *,
        mode: str,
        tau_drop: float,
        tau_sum: float,
        window: int,
    ) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
        """Shared reconcile invocation: runs reconcile(), persists the
        flags CSV (or removes a stale one when mode='off'), persists
        any remap changes from merge mode, and stashes the flags on the
        instance. Called from both :meth:`aggregate_stats` and
        :meth:`reconcile_stats` so the disk + in-memory state stays
        consistent regardless of entry point.
        """
        graph = self.lineage.lineage
        base_remap = self._all_remaps[self.target_year]  # type: ignore[index]

        reconciled, modified_remap, flags = reconcile(
            stats_df,
            graph,
            base_remap,
            mode=mode,
            tau_drop=tau_drop,
            tau_sum=tau_sum,
            window=window,
        )

        flags_path = self.output_dir / "reconciliation_flags.csv"
        if mode == "off":
            # Remove stale flags from a prior run so the user isn't
            # misled by out-of-date diagnostics.
            flags_path.unlink(missing_ok=True)
        else:
            flags.to_csv(flags_path, index=False)

        # Persist remap changes from merge mode.
        if modified_remap != base_remap:
            with (self.output_dir / "remap.json").open("w") as f:
                json.dump(modified_remap, f, indent=2, sort_keys=True)
            self._all_remaps[self.target_year] = modified_remap  # type: ignore[index]

        self._reconcile_flags = flags
        return reconciled, modified_remap, flags

    def completeness_report(
        self, *, by=None, as_text: bool = False, worst_n: int = 3
    ):
        """How complete is this dataset, one row per (variable, year)?

        The per-row completeness columns in ``stats_aggregated.csv`` are the
        ground truth; this is the scannable view — the thing to read before
        deciding a country is ready to publish, and a reasonable paper
        appendix table as-is.

        ``by="stable_id"`` re-groups to find chronically incomplete *units*
        rather than chronically incomplete *years*. ``as_text=True`` returns
        a printable block instead of a DataFrame.
        """
        from .completeness import format_report, summarize

        summary = summarize(self.get_stats(), by=by, worst_n=worst_n)
        return format_report(summary, "Stable completeness") if as_text else summary

    def export_fews(self, out_dir, **kwargs) -> dict:
        """Write the FEWS upload files. Delegates to :meth:`Lineage.export_fews`.

        Here because a user who has built a product has ``sb`` in hand, not
        ``ln``; the deliverable depends only on the lineage, not on anything
        the boundary build produces.
        """
        return self.lineage.export_fews(out_dir, **kwargs)

    def get_stats(self, **filters) -> pd.DataFrame:
        """Filter the aggregated stats. Supported keys: variable, year,
        years, season, stable_id. Unknown keys raise ``ValueError``
        to catch typos (``varible="rice"`` used to silently return
        the unfiltered frame).
        """
        allowed = {"variable", "year", "years", "season", "stable_id"}
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(
                f"get_stats: unknown filter key(s) {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}."
            )
        # Read the results from disk the first time and keep them, since a
        # user exploring results usually asks several questions in a row.
        if self._stats_agg is None:
            self._stats_agg = self._load_stats_agg()
        df = self._stats_agg
        # Apply whichever filters were asked for, narrowing as it goes. Any
        # combination is allowed and omitting one means "all of them".
        if "variable" in filters:
            df = df[df["variable"] == filters["variable"]]
        if "year" in filters:
            df = df[df["year"] == filters["year"]]
        if "years" in filters:
            df = df[df["year"].isin(filters["years"])]
        if "season" in filters:
            df = df[df["season"] == filters["season"]]
        if "stable_id" in filters:
            df = df[df["stable_id"] == filters["stable_id"]]
        return df.reset_index(drop=True)

    def get_reconciliation_flags(self) -> pd.DataFrame:
        """The full reconciliation flags table."""
        if self._reconcile_flags is None:
            path = self.output_dir / "reconciliation_flags.csv"
            if not path.exists():
                raise FileNotFoundError(
                    "No reconciliation flags cached. Call aggregate_stats() first."
                )
            self._reconcile_flags = pd.read_csv(path)
        return self._reconcile_flags

    def get_name_history(self) -> pd.DataFrame:
        """The (stable_id, year, unit_id, name) audit table."""
        if self._name_history is None:
            path = self.output_dir / "name_history.csv"
            if not path.exists():
                raise FileNotFoundError(
                    "No name history cached. Call build_boundaries() first."
                )
            self._name_history = pd.read_csv(path)
        return self._name_history

    def validation_report(self) -> str:
        """Formatted summary of all validation findings encountered.

        Combines three sources:

        1. Lineage findings from the underlying :class:`Lineage`
           (lazy-loaded via ``Lineage.validation_issues``).
        2. Stable-group contiguity findings recorded during
           :meth:`build_boundaries` (the "homonym name-matching bugs"
           check).
        3. Stats × lineage findings recorded during
           :meth:`aggregate_stats` (post-cease reporting, unknown
           unit_ids, etc.).

        Sections that have no findings are reported as such; the
        method always returns a non-empty string.
        """
        sections: list[str] = []
        lineage_issues = list(self.lineage.validation_issues)
        sections.append(_format_section("Lineage", lineage_issues))
        sections.append(_format_section("Stable-group contiguity", self._contiguity_issues))
        sections.append(_format_section("Stats × lineage", self._stats_issues))
        # Unmatched features: count + preview, sourced from the lineage.
        sections.append(_format_unmatched_section(self.lineage.unmatched_features))
        return "\n\n".join(sections)

    def summary(self) -> dict:
        """Counts and metadata from the last build_boundaries run."""
        if self._summary is None:
            path = self.output_dir / "summary.json"
            if not path.exists():
                raise FileNotFoundError(
                    "No summary cached. Call build_boundaries() first."
                )
            self._summary = json.loads(path.read_text())
        return self._summary

    # --- Internal helpers ------------------------------------------------

    def _has_cached_boundaries(self, out: Path) -> bool:
        """Disk-cache detection: both files exist AND the on-disk
        summary matches our current run.

        Matching rules:

        - Schema version equals the code's
          :data:`STABLE_SUMMARY_SCHEMA`.
        - ``country_code`` equals the current lineage's.
        - ``year_range`` matches this build's ``[target_year, max_year]``
          (the requested max_year, or — when not explicit — the
          shapefile-inferred year; matches what we'd write today).

        Any mismatch triggers a fresh rebuild rather than hydrating
        stale outputs.
        """
        # Earlier results can be reused, but only after checking they are
        # actually the same job. Each test below rejects a different way the
        # cached copy could be stale, cheapest first.
        remap_p = out / "remap.json"
        summary_p = out / "summary.json"
        # Both halves have to be there; one without the other is a run that
        # was interrupted partway.
        if not (remap_p.exists() and summary_p.exists()):
            return False
        # Unreadable is treated as absent rather than fatal — the run simply
        # redoes the work.
        try:
            summary = json.loads(summary_p.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        # Written by an older version of the package that recorded different
        # things, so its contents cannot be trusted here.
        if summary.get("_schema") != STABLE_SUMMARY_SCHEMA:
            return False
        # A different country's results left in the same folder.
        if summary.get("country_code") != self.lineage.country_code:
            return False
        # year_range check: the cached target_year must match this
        # instance's target_year. We don't compare max_year here
        # because a user constructing with a larger max_year on the
        # same dir might legitimately want to extend the existing
        # build — but that would currently rebuild anyway via the
        # build loop. Conservatively reject any year_range mismatch.
        disk_range = summary.get("year_range")
        if (
            not isinstance(disk_range, list)
            or len(disk_range) != 2
            or int(disk_range[0]) != self.target_year
        ):
            return False
        # If the user passed an explicit max_year, require an exact
        # match to that. (When max_year is None, infer-time defaults
        # may differ across runs — but we still expect identical
        # build inputs in that case, so check the disk value is at
        # least >= our likely upper bound.)
        if self._explicit_max_year is not None:
            if int(disk_range[1]) != self._explicit_max_year:
                return False
        return True

    def _load_cached_boundaries(self, out: Path) -> None:
        """Hydrate in-memory cache from a previous build_boundaries run.

        Only the target-year remap is cached on disk; per-year remaps
        for years > target_year live in the per-year geojson files
        and are re-read on demand via :meth:`get_boundary`.
        """
        with (out / "remap.json").open() as f:
            base_remap = json.load(f)
        self._summary = json.loads((out / "summary.json").read_text())
        max_year = int(self._summary["year_range"][1])
        self._max_year = max_year
        self._all_remaps = {self.target_year: base_remap}

    def _load_stats_agg(self) -> pd.DataFrame:
        """Lazy-load the aggregated stats from disk."""
        path = self.output_dir / "stats_aggregated.csv"
        if not path.exists():
            raise FileNotFoundError(
                "No aggregated stats cached. Call aggregate_stats() first."
            )
        return pd.read_csv(path)


# --- Module-level helpers ------------------------------------------------


def _format_section(title: str, issues: list) -> str:
    """Render one section of a validation_report() output.

    A section with no issues still appears (with the line "no
    findings") so the user knows the check was run. Sections with
    issues delegate to :func:`stablebound.validate.format_issues`.
    """
    if not issues:
        return f"== {title} ==\nno findings"
    from .validate import format_issues
    return f"== {title} ({len(issues)} finding(s)) ==\n{format_issues(issues)}"


def _format_unmatched_section(unmatched_gdf) -> str:
    """Render the 'Unmatched features' section of validation_report().

    Lists count + the first few offending names. Empty section
    reports "no findings" so users know the check ran.
    """
    n = len(unmatched_gdf)
    if n == 0:
        return "== Unmatched features ==\nno findings"
    preview_lines = []
    for _, row in unmatched_gdf.head(10).iterrows():
        src_idx = row.get("source_idx", "?")
        src_name = row.get("source_name", "?")
        preview_lines.append(f"  source_idx={src_idx}  source_name={src_name!r}")
    suffix = f"\n  ... ({n - 10} more)" if n > 10 else ""
    return (
        f"== Unmatched features ({n} feature(s)) ==\n"
        + "\n".join(preview_lines)
        + suffix
    )


# --- Module-level stats preprocessing ------------------------------------


def _load_stats(
    stats: str | Path | pd.DataFrame,
    *,
    mapping: str | Path | pd.DataFrame | None,
    name_column: str | None,
    stats_columns: dict[str, str],
    extensive: list[str] | None,
    intensive: dict[str, tuple[str, str]],
    ignore: list[str],
    target_year: int,
    mapping_year_column: str | None = None,
    mapping_coarse_column: str | None = None,
) -> pd.DataFrame:
    """Load + preprocess stats for either :class:`StableBoundary` or
    :class:`ModernBoundary`.

    Centralized here so the two products share the same column-rename
    / mapping-attach / variable-filter / schema-validate pipeline.

    Steps:
        1. Read CSV if a path was passed.
        2. Apply ``stats_columns`` renames so canonical column names appear.
        3. If a mapping is provided, attach ``unit_id`` via the matcher.
        4. Drop variables in ``ignore``.
        5. Filter to ``extensive`` ∪ intensive-inputs (if extensive is non-None).
        6. Lowercase + validate against the canonical stats schema.

    Emits ``UserWarning`` if declared variables don't appear in the
    stats (typo detection) or if the filtered frame ends up empty.
    Silently dropping every row was the previous behavior and it
    masked typos.
    """
    import warnings as _w

    # Step 1.
    if isinstance(stats, (str, Path)):
        df = pd.read_csv(Path(stats))
    else:
        df = stats.copy()

    # Step 2: column renames.
    if stats_columns:
        df = df.rename(columns=stats_columns)

    # Step 3: attach unit_id via mapping if provided. Mapping rows
    # with blank proposed_unit_id leave NaN unit_id behind; drop
    # those rows BEFORE schema validation (which would otherwise
    # reject them as malformed). Emit a UserWarning so the
    # transparency is loud, and write them to disk later.
    if mapping is not None:
        if name_column is None:
            raise ValueError(
                "name_column is required when mapping is provided "
                "(it tells aggregate_stats which stats column the "
                "mapping's source_name values match)."
            )
        # A year-aware (or coarse-keyed) mapping has several rows per name.
        # Joining it name-only would silently keep whichever row came last,
        # so the column names must be plumbed through; attach_stats_ids
        # raises rather than collapsing if they are missing.
        df = attach_stats_ids(
            df,
            mapping,
            name_column=name_column,
            year_column=mapping_year_column,
            coarse_column=mapping_coarse_column,
        )
        unmatched_stats_mask = df["unit_id"].isna()
        if unmatched_stats_mask.any():
            n_unmatched = int(unmatched_stats_mask.sum())
            unmatched_names = df.loc[unmatched_stats_mask, name_column]
            sample = list(dict.fromkeys(unmatched_names.astype(str)))[:5]
            _w.warn(
                f"Dropped {n_unmatched} stats row(s) whose mapping had a "
                f"blank proposed_unit_id (sample names: {sample!r}). "
                "Inspect or supply IDs and re-run.",
                UserWarning,
                stacklevel=3,
            )
            df = df.loc[~unmatched_stats_mask].copy()

    # Step 4: drop ignored variables.
    if ignore:
        df = df[~df["variable"].isin(ignore)]

    # Step 5: filter to declared extensives + intensive inputs, with a
    # diagnostic warning when the declaration doesn't match the data.
    if extensive is not None:
        keep = set(extensive)
        intensive_inputs: set[str] = set()
        for num, den in (intensive or {}).values():
            intensive_inputs.add(num)
            intensive_inputs.add(den)
        keep |= intensive_inputs

        present = set(df["variable"].unique()) if "variable" in df.columns else set()
        missing = keep - present
        if missing:
            _w.warn(
                f"Declared extensive/intensive variables not present in stats: "
                f"{sorted(missing)}. Variables actually in stats: "
                f"{sorted(present)}.",
                UserWarning,
                stacklevel=3,
            )
        df = df[df["variable"].isin(keep)]
        if df.empty:
            _w.warn(
                "Stats filter produced an empty frame; aggregate_stats will "
                "have nothing to aggregate. Check that the variable names in "
                "your `extensive` / `intensive` declarations match those in "
                "the stats file.",
                UserWarning,
                stacklevel=3,
            )

    # Step 6: normalize column names + validate.
    df.columns = [str(c).strip().lower() for c in df.columns]
    validate_stats(df, base_year=target_year)
    return df[list(STATS_REQUIRED_COLUMNS)].reset_index(drop=True)


__all__ = ["StableBoundary", "ID_COLUMN"]
