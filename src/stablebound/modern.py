"""User-facing :class:`ModernBoundary` orchestrator.

Parallel to :class:`StableBoundary` for the second product the
package ships: history rescaled onto today's geometry. Where the
stable product fixes geometry at the target year and projects
history forward, the modern product fixes geometry at the modern
shapefile and rescales history backward — for every modern unit,
the user gets a full time series on the present-day extent.

    from stablebound import Lineage, ModernBoundary
    ln = Lineage("IN")
    ln.attach_shapefile("modern.shp", mapping="mapping.csv",
                        name_column="DISTRICT_NAME")
    mb = ModernBoundary(ln)
    mb.aggregate_stats(stats="ag_stats.csv",
                       extensive=["rice_area_ha", "rice_production_mt"],
                       intensive={"yield": ("rice_production_mt", "rice_area_ha")},
                       modern_window_default=5)

Outputs (under ``output_dir / "modern/"``):

    stats_modern.csv    — long-form per-modern-unit time series
    event_fractions.csv — audit of every per-event, per-variable fraction
    late_reporting.csv  — reports stranded on units with no modern destination
    summary.json        — counts + run metadata

The class deliberately mirrors :class:`StableBoundary` rather than
extending it. The two products are conceptually parallel but
operationally distinct (different output schema, no remap.json
dependence, different caching semantics).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from .boundary import ID_COLUMN, _load_stats
from .data_dict import DataDictionary
from .lineage_class import Lineage
from .modern_algorithm import build_modern_ledger
from .stats import derive_intensive

# Schema version for the modern product's ``summary.json``. Same role
# as ``STABLE_SUMMARY_SCHEMA`` in boundary.py — bump when the summary
# shape changes; disk caches with an older schema are rebuilt.
#
# Schema versions:
#   1 — initial.
#   2 — adds ``n_unmatched_features`` field; output gains an
#       ``unmatched_features.geojson`` sidecar.
#   3 — cache invalidation policy bump (aggregate_stats no longer
#       short-circuits on file existence; getters still lazy-load
#       from disk).
MODERN_SUMMARY_SCHEMA = 3


class ModernBoundary:
    """User-facing orchestrator for the modern boundary product.

    Single-pipeline API: :meth:`aggregate_stats` does all of the
    algorithmic work in one call — load stats, validate against the
    lineage, run :func:`build_modern_ledger`, derive intensives, and
    write the output files. Unlike :class:`StableBoundary` there is no
    separate ``build_boundaries`` step because the modern product's
    geometry is just the attached shapefile (no dissolution required).

    Inspection: :meth:`get_modern_stats`, :meth:`get_event_fractions`,
    :meth:`get_late_reporting`, :meth:`validation_report`,
    :meth:`summary`.
    """

    # Subdirectory under output_dir where modern artifacts land.
    OUTPUT_SUBDIR = "modern"

    def __init__(
        self,
        lineage: Lineage,
        *,
        target_year: int | None = None,
        max_year: int | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.lineage = lineage
        # target_year here is the *floor* of the stats window (earliest
        # year of reported data to consider). Modern doesn't pin
        # geometry to it; it just bounds the temporal window. Defaults
        # to lineage.min_year.
        self.target_year = target_year if target_year is not None else lineage.min_year
        self._explicit_max_year = max_year
        self.output_dir = Path(output_dir) if output_dir else Path("./stablebound_out")

        # Lazy state.
        self._stats_modern: pd.DataFrame | None = None
        self._fractions: pd.DataFrame | None = None
        self._late_reporting: pd.DataFrame | None = None
        self._summary: dict | None = None
        self._max_year: int | None = None
        # Validation findings stashed for the validation_report() API.
        self._stats_issues: list = []

    # --- Pipeline -------------------------------------------------------

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
        modern_window: dict[str, int] | None = None,
        modern_window_default: int = 5,
        refresh: bool = False,
    ) -> None:
        """Run the modern-boundary pipeline.

        Steps:
            1. Short-circuit if cached on disk and refresh=False.
            2. Load + preprocess stats (renames, mapping, filter,
               schema-validate).
            3. Cross-check stats against the lineage.
            4. Determine max_year and per-unit modern areas.
            5. Run the algorithm core (build_modern_ledger).
            6. Apply derive_intensive if intensives declared.
            7. Write outputs + populate the cache.

        ``modern_window`` is the per-variable post-event window length;
        ``modern_window_default`` (default 5 years) applies to any
        variable not listed.
        """
        # Resolve data dictionary (individual kwargs win over data_dict).
        dd = data_dict or DataDictionary()
        stats_columns = stats_columns if stats_columns is not None else dd.stats_columns
        extensive = extensive if extensive is not None else dd.extensive
        intensive = intensive if intensive is not None else dd.intensive
        ignore = ignore if ignore is not None else dd.ignore
        modern_window = modern_window if modern_window is not None else {}

        out = self.output_dir / self.OUTPUT_SUBDIR
        out.mkdir(parents=True, exist_ok=True)

        # No early short-circuit on prior disk state. A user calling
        # aggregate_stats with new stats / data dictionary / window
        # parameters expects new output, not the cached version.
        # The getters (get_modern_stats, get_event_fractions, etc.)
        # still lazy-load from disk on first read of a fresh process.

        graph = self.lineage.lineage
        modern_gdf = self.lineage.shapefile
        if ID_COLUMN not in modern_gdf.columns:
            raise RuntimeError(
                f"Attached shapefile is missing the {ID_COLUMN!r} column."
            )

        # Step 1: load + preprocess stats. Shared helper with
        # StableBoundary keeps the data dictionary semantics identical
        # across products.
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

        # Step 2: cross-check stats against the lineage.
        modern_unit_ids = set(modern_gdf[ID_COLUMN].dropna().astype(str))
        from .validate import (
            LineageDataError,
            format_issues,
            validate_stats_lineage_consistency,
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
                "Call mb.validation_report() for details.",
                len(stats_other),
            )

        # Step 3: max_year. Explicit override wins; otherwise pick the
        # larger of (last lineage event) and (last stats year) so post-
        # event observations aren't silently truncated. A user with
        # stats running through 2030 gets output through 2030 even if
        # the lineage's last event is 2024.
        if self._explicit_max_year is not None:
            max_year = self._explicit_max_year
        else:
            stats_max = (
                int(stats_df["year"].max())
                if not stats_df.empty
                else int(graph.max_event_year)
            )
            max_year = max(int(graph.max_event_year), stats_max)

        # Step 4: per-modern-unit area for the area-fallback fraction tier.
        modern_areas = {
            str(row[ID_COLUMN]): float(row.geometry.area)
            for _, row in modern_gdf.iterrows()
            if row.geometry is not None and not row.geometry.is_empty
        }

        # Step 5: algorithm core.
        sm, fr, late = build_modern_ledger(
            graph,
            stats_df,
            modern_unit_ids=modern_unit_ids,
            base_year=self.target_year,
            max_year=max_year,
            window_per_var=modern_window,
            default_window=modern_window_default,
            modern_areas=modern_areas,
        )

        # Step 6: intensives (optional).
        if intensive:
            sm = self._derive_intensive_for_modern(sm, intensive)

        # Step 7: write outputs.
        sm.to_csv(out / "stats_modern.csv", index=False)
        # Dataset-level companion to the per-row columns. Free to produce and
        # it is the artifact that goes into a paper appendix, so write it
        # rather than making every user call the method.
        from .completeness import summarize as _summarize
        try:
            _summarize(sm).to_csv(out / "completeness_report.csv", index=False)
        except (ValueError, KeyError):
            # A frame with no completeness columns (e.g. everything filtered
            # out) simply gets no report; never fail the run over a summary.
            pass

        if not fr.empty:
            fr = fr.copy()
            fr["computed_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        fr.to_csv(out / "event_fractions.csv", index=False)
        late.to_csv(out / "late_reporting.csv", index=False)

        # Unmatched sidecar (mirrors the stable product).
        unmatched = self.lineage.unmatched_features
        if not unmatched.empty:
            (out / "unmatched_features.geojson").unlink(missing_ok=True)
            unmatched.to_file(
                out / "unmatched_features.geojson", driver="GeoJSON"
            )

        n_nan_cells = int(sm["has_nan_fraction"].sum()) if not sm.empty else 0
        summary = {
            "_schema": MODERN_SUMMARY_SCHEMA,
            "country_code": self.lineage.country_code,
            "target_year": self.target_year,
            "max_year": int(max_year),
            "n_events_processed": int(len(graph.territorial)),
            "n_modern_units": int(len(modern_unit_ids)),
            "n_unmatched_features": int(len(unmatched)),
            "n_fractions_recorded": int(len(fr)),
            "n_nan_fraction_cells": n_nan_cells,
            "n_late_reporting_rows": int(len(late)),
            "default_window": modern_window_default,
            "window_per_var": dict(modern_window),
        }
        with (out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        # Hold on to the results as well as writing them out, so a user who
        # asks a follow-up question straight away is answered from memory
        # rather than by reading back what was just written.
        self._stats_modern = sm
        self._fractions = fr
        self._late_reporting = late
        self._summary = summary
        self._max_year = int(max_year)

    # --- Getters --------------------------------------------------------

    def completeness_report(self, *, by=None, as_text: bool = False):
        """Provenance summary, one row per (variable, year).

        Deliberately *not* a completeness ratio: a modern value is a
        redistribution, not an aggregation, so there is no "expected members"
        denominator to divide by. What this reports instead is how many
        historical units fed each cell, how thin the weakest fraction's
        evidence was, and the least-trustworthy fraction tier in play.
        """
        from .completeness import format_report, summarize

        summary = summarize(self.get_modern_stats(), by=by)
        return format_report(summary, "Modern provenance") if as_text else summary

    def get_modern_stats(self, **filters) -> pd.DataFrame:
        """Filter the modern-stats frame. Supported keys: variable,
        year, years, season, modern_id. Unknown keys raise
        ``ValueError`` so typos surface immediately rather than
        returning an unfiltered frame.
        """
        allowed = {"variable", "year", "years", "season", "modern_id"}
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(
                f"get_modern_stats: unknown filter key(s) {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}."
            )
        if self._stats_modern is None:
            self._stats_modern = self._load_csv("stats_modern.csv")
        # Narrow the results by whichever filters were given; leaving one out
        # means "all of them". Same shape as the stable product's getter, so
        # the two read alike.
        df = self._stats_modern
        if "variable" in filters:
            df = df[df["variable"] == filters["variable"]]
        if "year" in filters:
            df = df[df["year"] == filters["year"]]
        if "years" in filters:
            df = df[df["year"].isin(filters["years"])]
        if "season" in filters:
            df = df[df["season"] == filters["season"]]
        if "modern_id" in filters:
            df = df[df["modern_id"] == filters["modern_id"]]
        return df.reset_index(drop=True)

    def get_event_fractions(self) -> pd.DataFrame:
        """Audit table: one row per (event, child, variable, season)."""
        if self._fractions is None:
            self._fractions = self._load_csv("event_fractions.csv")
        return self._fractions

    def get_late_reporting(self) -> pd.DataFrame:
        """Reports stranded on units with no modern destination."""
        if self._late_reporting is None:
            self._late_reporting = self._load_csv("late_reporting.csv")
        return self._late_reporting

    def validation_report(self) -> str:
        """Formatted summary of all validation findings.

        Combines the underlying :class:`Lineage` findings with the
        stats × lineage findings recorded during this product's
        :meth:`aggregate_stats`. Modern doesn't run the stable
        product's contiguity check (no remap to check), so the
        report has two sections instead of three.
        """
        from .boundary import _format_section, _format_unmatched_section
        sections: list[str] = []
        sections.append(_format_section("Lineage", list(self.lineage.validation_issues)))
        sections.append(_format_section("Stats × lineage", self._stats_issues))
        sections.append(_format_unmatched_section(self.lineage.unmatched_features))
        return "\n\n".join(sections)

    def summary(self) -> dict:
        """Run metadata + counts from the last aggregate_stats run."""
        if self._summary is None:
            path = self.output_dir / self.OUTPUT_SUBDIR / "summary.json"
            if not path.exists():
                raise FileNotFoundError(
                    "No modern summary cached. Call aggregate_stats() first."
                )
            self._summary = json.loads(path.read_text())
        return self._summary

    # --- Intensives helper ---------------------------------------------

    @staticmethod
    def _derive_intensive_for_modern(
        sm: pd.DataFrame,
        intensive: dict[str, tuple[str, str]],
    ) -> pd.DataFrame:
        """Recompute intensives on the modern-stats frame.

        :func:`stablebound.stats.derive_intensive` is written for the
        stable product's schema (``stable_id`` + ``late_reporting``
        join keys). The modern frame uses ``modern_id`` and has the
        additional columns ``fraction_method`` and
        ``late_report_redistributed``. We rename briefly so the existing
        helper applies, then restore.
        """
        df = sm.rename(columns={"modern_id": "stable_id"}).copy()
        df["late_reporting"] = False
        if "sources" in df.columns:
            df["constituent_ids"] = df["sources"]
        if "n_constituents" not in df.columns:
            df["n_constituents"] = pd.NA

        with_intensive = derive_intensive(df, intensive)

        out = with_intensive.rename(columns={"stable_id": "modern_id"})
        out = out.drop(columns=["late_reporting", "n_constituents"], errors="ignore")
        if "constituent_ids" in out.columns:
            if "sources" in out.columns:
                out["sources"] = out["sources"].fillna(out["constituent_ids"])
            else:
                out["sources"] = out["constituent_ids"]
            out = out.drop(columns=["constituent_ids"])

        # Derived intensive rows have no provenance of their own — they are
        # recomputed from already-cascaded extensives — so these stay NA
        # rather than inheriting a misleading figure from one input.
        for col in ("lineage_depth", "has_nan_fraction", "n_sources",
                    "min_n_common_observations",
                    "fraction_method", "late_report_redistributed"):
            if col not in out.columns:
                out[col] = pd.NA

        cols = [
            "year", "season", "variable", "modern_id", "value",
            "sources", "n_sources", "min_n_common_observations",
            "lineage_depth", "has_nan_fraction",
            "fraction_method", "late_report_redistributed",
        ]
        return out[[c for c in cols if c in out.columns]].reset_index(drop=True)

    # --- Internal helpers ----------------------------------------------

    def _has_cache(self, out: Path) -> bool:
        """Require summary.json + stats_modern.csv on disk AND a
        matching schema version on the summary. See
        :data:`MODERN_SUMMARY_SCHEMA`.
        """
        summary_p = out / "summary.json"
        if not (summary_p.exists() and (out / "stats_modern.csv").exists()):
            return False
        try:
            disk_schema = json.loads(summary_p.read_text()).get("_schema")
        except (OSError, json.JSONDecodeError):
            return False
        return disk_schema == MODERN_SUMMARY_SCHEMA

    def _hydrate_from_disk(self, out: Path) -> None:
        """Populate the in-memory cache from a previous aggregate_stats run."""
        self._summary = json.loads((out / "summary.json").read_text())
        self._max_year = int(self._summary["max_year"])
        # low_memory=False — intensive rows have NaN lineage_depth which
        # confuses pandas about column dtype.
        # Reload a previous run's results so the object behaves as though it
        # had just produced them. The main results file must be there, but
        # the two supporting ones are allowed to be missing: a country with
        # no boundary changes produces no shares and no stragglers, and an
        # empty table is the honest answer rather than an error.
        self._stats_modern = pd.read_csv(out / "stats_modern.csv", low_memory=False)
        for attr, fname in (
            ("_fractions", "event_fractions.csv"),
            ("_late_reporting", "late_reporting.csv"),
        ):
            path = out / fname
            if path.exists():
                setattr(self, attr, pd.read_csv(path, low_memory=False))
            else:
                setattr(self, attr, pd.DataFrame())

    def _load_csv(self, name: str) -> pd.DataFrame:
        """Lazy CSV loader for the get_* methods."""
        path = self.output_dir / self.OUTPUT_SUBDIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found under {path.parent}. "
                "Call aggregate_stats() first."
            )
        return pd.read_csv(path, low_memory=False)


__all__ = ["ModernBoundary"]
