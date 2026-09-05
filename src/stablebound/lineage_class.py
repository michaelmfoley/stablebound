"""User-facing :class:`Lineage` class.

A :class:`Lineage` wraps the canonical relationship table, baseline,
and (optional) name change log for a country, exposes inspection
methods (snapshot, year range), and supports attaching a modern
shapefile via the human-in-the-loop matcher in :mod:`stablebound.match`.

Construction:

    from stablebound import Lineage

    # Bundled country: zero-config.
    ln = Lineage("IN")

    # Custom country: pass paths explicitly.
    ln = Lineage(
        "PT",
        relationship_table_path="my_country/rt.csv",
        baseline_path="my_country/baseline.csv",
    )

Inspection (no shapefile required):

    ln.lineage              # LineageGraph
    ln.snapshot()           # DataFrame of current-day units
    ln.snapshot(year=2010)  # DataFrame at a specific year
    ln.years                # range of years the lineage covers

Shapefile attachment (human-in-the-loop name matching):

    proposal = ln.propose_shapefile_mapping(
        "modern.shp", name_column="DISTRICT_NAME",
    )
    proposal.to_csv("mapping.csv")     # user reviews / edits in Excel
    ln.attach_shapefile("modern.shp", mapping="mapping.csv",
                        name_column="DISTRICT_NAME")
    ln.shapefile_year                   # inferred from matches

The class is mutable on purpose: attach_shapefile updates state so the
products downstream (:class:`StableBoundary`, :class:`ModernBoundary`)
can read shapefile + shapefile_year off the lineage without
re-plumbing them through every constructor.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from .data import BUNDLED_COUNTRIES
from .io import (
    merge_name_changes,
    read_baseline,
    read_name_change_log,
    read_relationship_table,
)
from .lineage import LineageGraph, _name_of_unit_in_year
from .match import (
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_SKETCHY_THRESHOLD,
    MatchProposal,
    attach_shapefile_ids,
    attach_stats_ids,
    normalize_name,
    propose_shapefile_mapping as _propose_shapefile_mapping,
    propose_stats_mapping as _propose_stats_mapping,
)
from .snapshot import build_snapshot, infer_year

_LOGGER = logging.getLogger(__name__)

_GdfOrPath = Union[gpd.GeoDataFrame, Path, str]
_DfOrPath = Union[pd.DataFrame, Path, str]


class Lineage:
    """Country-level lineage + shapefile facade. Lazy-loads file inputs.

    Construction validates paths exist and (for custom countries) that
    a relationship table is provided. Heavy reads happen on first
    attribute access: ``self.lineage`` parses the RT and runs
    validation; ``self.baseline`` reads the baseline CSV; etc. A user
    who only wants ``ln.years`` doesn't pay for full RT parsing.

    Bundled-country shortcut: ``Lineage("IN")`` resolves to the paths
    in :data:`stablebound.BUNDLED_COUNTRIES`. Any explicit path argument
    overrides the bundled default for that file only.

    State that mutates after construction:

        - ``_shapefile`` / ``_shapefile_year`` — populated by
          :meth:`attach_shapefile`.

    Cached lazy loads (read-once):

        - ``_relationship_table`` / ``_baseline`` / ``_name_change_log``
        - ``_lineage`` (the parsed LineageGraph)
    """

    def __init__(
        self,
        country_code: str,
        *,
        relationship_table_path: Path | str | None = None,
        baseline_path: Path | str | None = None,
        name_change_log_path: Path | str | None = None,
        max_year: int | None = None,
    ) -> None:
        self.country_code = country_code.upper()

        # A country either ships with the package or the caller brings their
        # own files. Which one decides everything set below.
        bundled = BUNDLED_COUNTRIES.get(self.country_code)
        if bundled is not None:
            # For a country we ship, the files are already known. A caller can
            # still point at their own version of any one of them — to try a
            # corrected boundary record without editing the installed package,
            # which is the usual way improvements start.
            self._rt_path = (
                Path(relationship_table_path)
                if relationship_table_path
                else bundled.lineage_path
            )
            self._baseline_path = Path(baseline_path) if baseline_path else bundled.baseline_path
            self._ncl_path = (
                Path(name_change_log_path)
                if name_change_log_path
                else bundled.name_change_log_path
            )
            # These come only from the shipped list. They are the accumulated
            # judgements about the country that no file states outright: from
            # when the data can be trusted, how far it has been kept current,
            # whether a second tier of boundaries exists, and how that
            # country's names have to be tidied before they will match.
            self._validity_start_year: int | None = bundled.validity_start_year
            self._coverage_end_year: int | None = bundled.coverage_end_year
            self._admin_level: int = bundled.admin_level
            self._coarse_rt_path: Path | None = bundled.coarse_lineage_path
            self._notes: str = bundled.notes
        else:
            if relationship_table_path is None:
                raise ValueError(
                    f"country_code {country_code!r} is not bundled with "
                    f"StableBound; relationship_table_path is required. "
                    f"Bundled countries: {sorted(BUNDLED_COUNTRIES.keys())}"
                )
            # A country we don't ship brings its own files, and loses
            # everything the built-in list would have told us: when its data
            # becomes trustworthy, when it stops, and whether a second admin
            # level exists. Note the admin level is *assumed* to be 1 rather
            # than worked out, so a user's own district-level country is
            # treated as if it were state-level until they say otherwise.
            self._rt_path = Path(relationship_table_path)
            self._baseline_path = Path(baseline_path) if baseline_path else None
            self._ncl_path = Path(name_change_log_path) if name_change_log_path else None
            self._validity_start_year = None
            self._coverage_end_year = None
            self._admin_level = 1
            self._coarse_rt_path = None
            self._notes = ""

        # Files aren't read until they're needed, but check now that they
        # exist. A mistyped path should complain here rather than surfacing
        # much later, inside whatever step happened to need it first.
        for label, p in (
            ("relationship_table_path", self._rt_path),
            ("baseline_path", self._baseline_path),
            ("name_change_log_path", self._ncl_path),
        ):
            if p is not None and not p.exists():
                raise FileNotFoundError(f"{label} not found: {p}")

        self.max_year: int | None = max_year

        self._init_lazy_state()

    def _init_lazy_state(self) -> None:
        """Set every lazily-populated attribute to its empty value.

        Called by ``__init__`` and by :meth:`from_legacy_rt`, which builds its
        instance with ``cls.__new__`` and so never runs ``__init__``. Keeping
        the list in one place is not tidiness: adding a field and forgetting to
        mirror it here leaves a half-initialised object that raises
        ``AttributeError`` only on the path that touches it. That has happened
        twice.
        """
        self._relationship_table: pd.DataFrame | None = None
        self._baseline: pd.DataFrame | None = None
        self._name_change_log: pd.DataFrame | None = None
        self._lineage: LineageGraph | None = None
        self._validation_issues: list | None = None
        self._coarse_lineage: LineageGraph | None = None

        # Set by attach_shapefile.
        self._shapefile: gpd.GeoDataFrame | None = None
        self._shapefile_year: int | None = None
        self._unmatched_features: gpd.GeoDataFrame | None = None
        self._shapefile_issues: list | None = None
        # Only populated by from_legacy_rt — a canonical lineage has no FNIDs.
        self._fnid_map: dict[str, str] | None = None

    # --- Alternative constructors ---------------------------------------

    @classmethod
    def from_legacy_rt(
        cls,
        relationship_table_path: Path | str,
        *,
        country: str,
        admin_level: int = 1,
        name_change_log_path: Path | str | None = None,
        max_year: int | None = None,
    ) -> "Lineage":
        """Construct a Lineage from a FEWS-style relationship table.

        FEWS NET ships RTs in a legacy hierarchical+temporal CSV format
        (``category`` column with ``hierarchical``/``temporal`` rows)
        rather than the canonical event-only format that
        :meth:`__init__` expects. This classmethod runs the conversion
        in memory: snapshots come from the ``hierarchical`` rows at the
        requested ``admin_level``; events come from the ``temporal``
        rows. Renames are detected from both
        ``relationship_type == "name change"`` rows and 1-to-1
        ``successor`` rows with differing names.

        A baseline path is not accepted — the baseline is derived from
        the earliest hierarchical snapshot (the FEWS format encodes the
        baseline implicitly).

        ``name_change_log_path`` is optional but recommended for
        countries where the FEWS RT is known to be incomplete on
        renames (India in particular: the bundled FEWS RT encodes only
        ~12 renames out of dozens).
        """
        from .rt_convert import (
            convert_relationship_table_to_lineage,
            read_legacy_relationship_table,
        )

        rt_path = Path(relationship_table_path)
        if not rt_path.exists():
            raise FileNotFoundError(f"relationship_table_path not found: {rt_path}")
        ncl_path = Path(name_change_log_path) if name_change_log_path else None
        if ncl_path is not None and not ncl_path.exists():
            raise FileNotFoundError(f"name_change_log_path not found: {ncl_path}")

        old_rt = read_legacy_relationship_table(rt_path)
        lineage_df, baseline_df, fnid_map = convert_relationship_table_to_lineage(
            old_rt, country=country, admin_level=admin_level, return_fnid_map=True
        )

        obj = cls.__new__(cls)
        obj.country_code = country.upper()
        obj._rt_path = rt_path
        obj._baseline_path = None
        obj._ncl_path = ncl_path
        obj._init_lazy_state()
        obj._admin_level = admin_level
        obj._coarse_rt_path = None
        obj._validity_start_year = (
            int(baseline_df["year"].iloc[0]) if not baseline_df.empty else None
        )
        obj._notes = f"converted from legacy FEWS RT at {rt_path}"
        obj._fnid_map = fnid_map
        # The RT's latest hierarchical vintage is the last year the source
        # affirms the unit set. Anything after that is unattested — the same
        # meaning as BundledCountry.coverage_end_year.
        try:
            _vintages = pd.to_numeric(
                old_rt.get("from_year"), errors="coerce"
            ).dropna()
            obj._coverage_end_year = int(_vintages.max()) if len(_vintages) else None
        except (AttributeError, TypeError, ValueError):
            obj._coverage_end_year = None
        obj.max_year = max_year

        # Prefill caches so the lazy properties don't hit disk for the
        # RT or baseline. NCL still gets read here if supplied so the
        # merged RT cache is consistent with the public NCL property.
        if ncl_path is not None:
            ncl = read_name_change_log(ncl_path)
            obj._name_change_log = ncl
            obj._relationship_table = merge_name_changes(lineage_df, ncl)
        else:
            obj._name_change_log = None
            obj._relationship_table = lineage_df
        obj._baseline = baseline_df
        obj._lineage = None
        obj._validation_issues = None
        obj._shapefile = None
        obj._shapefile_year = None
        obj._unmatched_features = None
        obj._shapefile_issues = None
        return obj

    # --- Lazy data loads -------------------------------------------------

    @property
    def relationship_table(self) -> pd.DataFrame:
        """Canonical relationship table. Name changes already merged in."""
        if self._relationship_table is None:
            rt = read_relationship_table(self._rt_path)
            # Fold the name change log into the RT as NameChange rows so
            # the resulting LineageGraph sees them in one place. A country
            # may ship without a separate NCL.
            if self._ncl_path is not None:
                ncl = read_name_change_log(self._ncl_path)
                self._name_change_log = ncl
                rt = merge_name_changes(rt, ncl)
            self._relationship_table = rt
        return self._relationship_table

    @property
    def baseline(self) -> pd.DataFrame | None:
        """Baseline snapshot (units alive at the earliest valid year).

        ``None`` for custom countries that didn't supply one. Bundled
        countries always have a baseline. Alternative constructors
        (e.g. :meth:`from_legacy_rt`) prefill the cache with a derived
        baseline and leave ``_baseline_path`` unset — those cases are
        served from the cache.
        """
        if self._baseline is not None:
            return self._baseline
        if self._baseline_path is None:
            return None
        # validity_start_year is used as the filter year if the
        # baseline file carries multi-year rows. India's bundled
        # baseline is single-year (1991).
        self._baseline = read_baseline(
            self._baseline_path,
            base_year=self._validity_start_year,
        )
        return self._baseline

    @property
    def admin_level(self) -> int:
        """Admin level of the working lineage (2 for India's districts)."""
        return self._admin_level

    @property
    def max_level(self) -> int:
        """Deepest level this lineage describes. Same as :attr:`admin_level`."""
        return self._admin_level

    def graph(self, level: int | None = None) -> LineageGraph:
        """The :class:`LineageGraph` for ``level`` (default: the working level).

        A country whose upper admin units also change needs their events too:
        India's states split (Uttarakhand 2000, Telangana 2014) and the FEWS
        relationship table carries rows at both levels. Those live in a
        separate file in their own id space, registered as
        ``BundledCountry.coarse_lineage_path``.

        The working level's graph is :attr:`lineage` unchanged. The coarse
        level is built on demand, with the name-change log filtered to that
        level so a state rename is not injected into the district graph.
        """
        # Usually the caller wants the level this country was loaded at.
        if level is None or level == self._admin_level:
            return self.lineage

        # The rest turns away requests we can't serve, each for a different
        # reason, so the message names the actual mistake.

        # There is nothing below the top level but the country itself.
        if self._admin_level == 1:
            raise ValueError(
                f"{self.country_code} is admin level 1; there is no level below "
                f"it (level {level} would be the country itself). Its own graph "
                "serves both levels of the FEWS deliverable."
            )
        # We only track one level up — districts know their state, and that
        # is as far as it goes.
        if level != self._admin_level - 1:
            raise ValueError(
                f"{self.country_code} describes admin level {self._admin_level}"
                + (f" and {self._admin_level - 1}" if self._coarse_rt_path else "")
                + f"; asked for {level}."
            )
        # A fair question, but this country never supplied that file.
        if self._coarse_rt_path is None:
            raise ValueError(
                f"{self.country_code} has no level-{level} lineage. Register one "
                "as coarse_lineage_path, or pass its graph explicitly."
            )

        # Read it once and keep it. Renames are filtered to this level first,
        # because a country may keep both levels' renames in one file and a
        # state's rename means nothing to a district.
        if self._coarse_lineage is None:
            rt = read_relationship_table(self._coarse_rt_path)
            ncl = self._ncl_for_level(level)
            if ncl is not None and len(ncl):
                rt = merge_name_changes(rt, ncl)
            self._coarse_lineage = LineageGraph.from_dataframe(rt)
        return self._coarse_lineage

    def _ncl_for_level(self, level: int) -> pd.DataFrame | None:
        """Name-change-log rows for one admin level, or all of them.

        India's log covers both levels and carries a ``level`` column. Logs
        without one are returned whole — filtering on a column that is not
        there would silently drop every row.
        """
        ncl = self.name_change_log
        if ncl is None or "level" not in ncl.columns:
            return ncl
        keep = ncl[ncl["level"].astype(int) == level]
        return keep[["event_year", "unit_id", "old_name", "new_name"]].reset_index(
            drop=True
        )

    @property
    def fnid_map(self) -> dict[str, str] | None:
        """``{fnid: unit_id}`` for lineages built from a FEWS RT, else ``None``.

        FEWS-sourced statistics usually already carry an ``FNID`` column, so
        this is the shortcut past name matching entirely — see
        :meth:`attach_stats_by_fnid`. Canonical lineages have no FNIDs and
        return ``None``.
        """
        return self._fnid_map

    def attach_stats_by_fnid(
        self,
        stats: _DfOrPath,
        *,
        fnid_column: str = "FNID",
        id_column: str = "unit_id",
    ) -> pd.DataFrame:
        """Attach ``unit_id`` to stats that already carry FNIDs.

        Exact id-to-id join — no fuzzy matching, no review CSV, no aliases.
        Where it applies it removes the single most expensive step of
        onboarding a country: India needed roughly 150 hand-curated name
        aliases, and Philippines, Sri Lanka and India all publish statistics
        with an FNID column already present.

        Rows that do not resolve get ``<NA>`` and are reported rather than
        dropped silently. The two reasons are reported **separately**, because
        they call for completely different responses:

        * *No FNID in the row at all* — a source-data gap. Nothing about the
          lineage will fix it; those observations simply are not attributed to
          a unit upstream. On the Philippines this is 28.7% of rows (34,263 of
          119,310), and reporting it as a lineage mismatch sent a reader
          hunting for a vintage problem that did not exist.
        * *An FNID the lineage does not contain* — usually a genuine vintage
          mismatch between the statistics and the relationship table, and worth
          investigating. On the Philippines this is zero: every one of the
          85,047 rows that carries an FNID resolves.
        """
        import warnings as _w

        if self._fnid_map is None:
            raise RuntimeError(
                f"{self.country_code} has no FNID map. Only lineages built with "
                "Lineage.from_legacy_rt() carry one; a canonical lineage has no "
                "FNIDs to join on. Use propose_stats_mapping() instead."
            )
        df = stats if isinstance(stats, pd.DataFrame) else pd.read_csv(Path(stats))
        df = df.copy()
        if fnid_column not in df.columns:
            raise KeyError(
                f"fnid_column {fnid_column!r} not in stats columns: {list(df.columns)}"
            )
        # Work out which rows have no identifier at all BEFORE looking
        # anything up. A missing value reads as the word "nan" once it is
        # treated as text, and would then look like an identifier the country
        # has never heard of. That is exactly how a hole in the source
        # statistics once got reported as the two files being out of step.
        raw = df[fnid_column]
        blank = raw.isna() | raw.astype(str).str.strip().isin({"", "nan", "None", "<NA>"})
        df[id_column] = raw.astype(str).map(self._fnid_map)
        df.loc[blank, id_column] = pd.NA

        # Two different failures, counted apart, because they call for
        # completely different responses: rows with no identifier are a gap
        # in the source, while rows with one nobody recognises usually mean
        # the statistics and the boundary records are different vintages.
        n_blank = int(blank.sum())
        unresolved = df[id_column].isna() & ~blank
        n_unresolved = int(unresolved.sum())

        if n_blank:
            _w.warn(
                f"{n_blank} of {len(df)} stats row(s) ({n_blank / len(df):.1%}) "
                f"carry no {fnid_column} at all and cannot be attributed to a "
                "unit. They keep a null unit_id rather than being dropped. This "
                "is a gap in the source statistics, not a lineage problem — no "
                "change to the relationship table will resolve it.",
                UserWarning,
                stacklevel=2,
            )
        if n_unresolved:
            sample = sorted({str(f) for f in raw[unresolved]})[:5]
            _w.warn(
                f"{n_unresolved} stats row(s) carry an FNID absent from the "
                f"{self.country_code} lineage (sample: {sample!r}). They keep a "
                "null unit_id rather than being dropped. This usually means the "
                "statistics and the relationship table are different vintages.",
                UserWarning,
                stacklevel=2,
            )
        return df

    @property
    def coverage_end_year(self) -> int | None:
        """Last year the source data affirms this country's unit set.

        ``None`` for custom countries — the package cannot know how current a
        user-supplied lineage is. See
        :attr:`stablebound.data.BundledCountry.coverage_end_year`.
        """
        return self._coverage_end_year

    @property
    def name_change_log(self) -> pd.DataFrame | None:
        """Standalone name-change log, or ``None`` if not provided."""
        if self._ncl_path is None:
            return None
        if self._name_change_log is None:
            # The RT loader populates ``self._name_change_log`` as a side
            # effect; touching ``self.relationship_table`` first forces
            # the cache. We do the same thing explicitly here for the
            # case where the user reads ``name_change_log`` first.
            _ = self.relationship_table
        return self._name_change_log

    @property
    def lineage(self) -> LineageGraph:
        """Parsed :class:`LineageGraph` (with name changes merged in)."""
        if self._lineage is None:
            self._lineage = LineageGraph.from_dataframe(self.relationship_table)
        return self._lineage

    @property
    def validation_issues(self) -> "list":
        """All :class:`~stablebound.validate.LineageIssue` findings.

        Lazy: runs :func:`stablebound.validate.validate_lineage` on
        first access and caches the result. The list is empty if the
        RT is structurally clean. Errors are surfaced as exceptions
        at lineage-load time (via :meth:`LineageGraph.from_dataframe`),
        so anything here is severity ``warning`` or ``info``.
        """
        if self._validation_issues is None:
            from .validate import validate_lineage
            self._validation_issues = validate_lineage(self.lineage)
        return self._validation_issues

    def validation_report(self) -> str:
        """Formatted text summary of all lineage validation findings.

        Intended for printing or writing to a file. Wraps
        :func:`stablebound.validate.format_issues` over
        :attr:`validation_issues`. Use this instead of digging through
        log output when investigating warnings on a country's lineage.
        """
        from .validate import format_issues
        issues = self.validation_issues
        if not issues:
            return f"Lineage({self.country_code}): no validation findings."
        return (
            f"Lineage({self.country_code}): {len(issues)} validation "
            f"finding(s).\n\n" + format_issues(issues)
        )

    # --- Derived metadata -----------------------------------------------

    @property
    def validity_start_year(self) -> int | None:
        """Earliest year for which the bundled product is valid.

        For bundled countries, comes from
        :data:`stablebound.BUNDLED_COUNTRIES`. For custom countries
        this is ``None`` (the package can't infer it).
        """
        return self._validity_start_year

    @property
    def notes(self) -> str:
        """Free-text notes from the bundled-country registry, if any."""
        return self._notes

    @property
    def min_year(self) -> int:
        """Lowest year the lineage can describe.

        Resolution order:

        1. ``validity_start_year`` from the bundled registry (when set).
        2. The minimum value in the baseline's ``year`` column (when a
           baseline is provided and it carries a year column).
        3. The lineage graph's ``min_event_year`` (when the RT is non-
           empty).

        Raises ``ValueError`` if none of these are inferable — a custom
        country with no validity year, no baseline year column, and an
        empty RT has no lower bound.
        """
        # Best answer: someone decided where this country's data becomes
        # trustworthy and wrote it down. No file can be inspected for that.
        if self._validity_start_year is not None:
            return self._validity_start_year
        # Failing that, the year the baseline itself claims to describe.
        if self.baseline is not None and "year" in self.baseline.columns:
            yrs = self.baseline["year"].dropna()
            if not yrs.empty:
                return int(yrs.min())
        # Last resort: the first year anything changed. Weaker, because a
        # country existed for some time before its first recorded change.
        min_ev = self.lineage.min_event_year
        if min_ev is not None:
            return min_ev
        # Guessing here would silently shorten every result the user gets.
        raise ValueError(
            f"Cannot infer min_year for {self.country_code!r}: no "
            "validity_start_year, no baseline year column, and an empty "
            "relationship table. Pass `max_year` to the constructor or "
            "supply a baseline with a year column."
        )

    @property
    def years(self) -> range:
        """Inclusive range of years this lineage spans.

        Upper bound: constructor ``max_year`` if supplied; otherwise
        ``lineage.max_event_year + 1`` (the "current-day" snapshot)
        when the RT is non-empty; otherwise the same as ``min_year``
        (a single-year window for empty-RT static countries).
        """
        max_ev = self.lineage.max_event_year
        if max_ev is not None:
            # A change recorded for a given year takes effect during it, so
            # the units it creates only show up the year after. Run one year
            # past the last change or the newest districts never appear.
            upper = max_ev + 1
        elif self.max_year is not None:
            upper = self.max_year
        else:
            # No changes and no ceiling given: a country whose boundaries
            # never moved, so the range is the single baseline year.
            upper = self.min_year
        # A ceiling the caller set always wins, including over the year we
        # just derived from the last change.
        if self.max_year is not None:
            upper = min(upper, self.max_year)
        return range(self.min_year, upper + 1)

    # --- Snapshot inspection -------------------------------------------

    def snapshot(self, year: int | None = None) -> pd.DataFrame:
        """DataFrame of canonical (unit_id, unit_name) pairs at ``year``.

        ``year`` defaults to ``shapefile_year`` if a shapefile is
        attached, else the latest year in :attr:`years` (the current-
        day snapshot). The returned frame has columns ``unit_id``,
        ``unit_name``, ``year`` (constant), sorted by ``unit_id``.

        Use this to (a) sanity-check the lineage at the year you care
        about, and (b) bootstrap a stats file's ``unit_id`` column —
        join your name-keyed stats against this DataFrame.
        """
        # Default to the year of whatever map is attached, so the answer
        # lines up with the boundaries the user is looking at.
        if year is None:
            year = self._shapefile_year if self._shapefile_year is not None else max(self.years)

        baseline_unit_ids: set[str] = set()
        baseline_names: dict[str, str] = {}
        if self.baseline is not None:
            baseline_unit_ids = set(self.baseline["unit_id"].astype(str))
            baseline_names = dict(
                zip(self.baseline["unit_id"].astype(str), self.baseline["name"].astype(str))
            )

        # The lineage only records changes, so a district that never split or
        # merged is mentioned nowhere in it. Handing over the baseline's
        # districts as a starting point is what keeps those in the answer —
        # otherwise we would list only the places that changed.
        units = build_snapshot(self.lineage, year, additional_units=baseline_unit_ids)
        rows: list[dict] = []
        # Sorted so two runs of the same analysis produce the same file.
        for uid in sorted(units):
            # The lineage knows about renames, so ask it what this district
            # was called in this particular year.
            name = _name_of_unit_in_year(self.lineage, uid, year)
            # No answer means the district never changed and so was never
            # named in a change; the baseline is the only place it is written.
            if not name:
                name = baseline_names.get(uid)
            rows.append({"unit_id": uid, "unit_name": name, "year": year})
        return pd.DataFrame(rows, columns=["unit_id", "unit_name", "year"])

    # --- Shapefile attachment (human-in-the-loop) -----------------------

    def _resolve_normalizer(
        self, user_normalizer: Callable[[str], str] | None
    ) -> Callable[[str], str]:
        """Pick the normalizer for matching: user override → bundled → default.

        Order:
            1. Explicit user-supplied ``normalizer`` kwarg.
            2. Bundled per-country normalizer (e.g., India's
               district/honorific stripping). Discovered via
               ``BUNDLED_COUNTRIES[country_code].normalizer``.
            3. The package-default :func:`normalize_name`.
        """
        if user_normalizer is not None:
            return user_normalizer
        bundled = BUNDLED_COUNTRIES.get(self.country_code)
        if bundled is not None and bundled.normalizer is not None:
            return bundled.normalizer
        return normalize_name

    @property
    def default_normalizer(self) -> Callable[[str], str]:
        """The normalizer that would be used if none was passed explicitly.

        For bundled countries this may be a country-specific
        normalizer (India strips "district" / honorifics); otherwise
        it's the package default :func:`normalize_name`.
        """
        return self._resolve_normalizer(None)

    def propose_shapefile_mapping(
        self,
        shapefile: _GdfOrPath,
        name_column: str,
        *,
        coarse_column: str | None = None,
        year: int | None = None,
        manual_overrides: dict[str, str] | None = None,
        homonym_overrides: dict[int, tuple[str, str]] | None = None,
        normalizer: Callable[[str], str] | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        sketchy_threshold: float = DEFAULT_SKETCHY_THRESHOLD,
    ) -> MatchProposal:
        """Propose a unit_id for every feature in a shapefile.

        Thin wrapper over :func:`stablebound.match.propose_shapefile_mapping`
        that fills in the lineage's graph, baseline, and name change log
        automatically, and resolves a country-appropriate default
        normalizer (see :attr:`default_normalizer`).

        ``coarse_column`` (optional) names a shapefile column carrying
        the upper-admin name (state, province). When provided, exact-
        match homonyms (Hamirpur in two states, etc.) are disambiguated
        automatically without needing ``homonym_overrides`` entries.

        Workflow: save the returned proposal to CSV, review in Excel,
        then pass back via :meth:`attach_shapefile`.
        """
        return _propose_shapefile_mapping(
            shapefile,
            self.lineage,
            name_column=name_column,
            coarse_column=coarse_column,
            year=year,
            year_range=self.years,
            baseline=self.baseline,
            name_change_log=self.name_change_log,
            manual_overrides=manual_overrides,
            homonym_overrides=homonym_overrides,
            normalizer=self._resolve_normalizer(normalizer),
            fuzzy_threshold=fuzzy_threshold,
            sketchy_threshold=sketchy_threshold,
        )

    def propose_stats_mapping(
        self,
        stats: _DfOrPath,
        name_column: str,
        *,
        coarse_column: str | None = None,
        year_column: str | None = None,
        year_aware: bool = False,
        manual_overrides: dict[str, str] | None = None,
        normalizer: Callable[[str], str] | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        sketchy_threshold: float = DEFAULT_SKETCHY_THRESHOLD,
    ) -> MatchProposal:
        """Propose unit_ids for unique names in a stats DataFrame.

        ``coarse_column`` (optional) names a stats column with the
        upper-admin name (state, province). With it, the proposal is
        keyed per (name, coarse) pair instead of per name alone, so
        homonyms get distinct rows.

        ``year_aware=True`` (requires ``year_column``) matches each name
        against the snapshot for *its own year* and walks the lineage back
        to a year-compatible ancestor when the match wasn't alive yet — the
        behaviour lineage-heavy countries need, so a name reported before a
        split resolves to the pre-split parent rather than a post-split
        child. The proposal is then keyed per (name, year[, coarse]), so
        pass the same ``year_column`` (and ``coarse_column``) to
        :meth:`attach_stats_ids` or to a product's ``aggregate_stats``.
        """
        return _propose_stats_mapping(
            stats,
            self.lineage,
            name_column=name_column,
            coarse_column=coarse_column,
            year_column=year_column,
            year_aware=year_aware,
            year_range=self.years,
            baseline=self.baseline,
            name_change_log=self.name_change_log,
            manual_overrides=manual_overrides,
            normalizer=self._resolve_normalizer(normalizer),
            fuzzy_threshold=fuzzy_threshold,
            sketchy_threshold=sketchy_threshold,
        )

    def attach_shapefile(
        self,
        shapefile: _GdfOrPath,
        *,
        mapping: pd.DataFrame | Path | str | None = None,
        name_column: str | None = None,
        id_column: str = "unit_id",
        coarse_column: str | None = None,
        on_unmatched: str = "keep",
        validate: bool = True,
    ) -> gpd.GeoDataFrame:
        """Attach unit_ids to a shapefile and store it on the lineage.

        Two paths to acquire IDs:

        - ``mapping`` is provided: the shapefile is joined against the
          mapping CSV (output of :meth:`propose_shapefile_mapping`,
          possibly user-edited) using ``name_column``. Mapping rows
          with a blank ``proposed_unit_id`` produce NaN unit_id on
          the corresponding shapefile feature.
        - ``mapping`` is ``None``: the shapefile is assumed to already
          have a column named ``id_column`` (default ``"unit_id"``).

        Unmatched-feature policy (``on_unmatched``):

        - ``"keep"`` (default): features with NaN unit_id get sentinel
          IDs ``UNMATCHED_<source_idx>``, pass through the products
          as singleton stable groups, and appear in the output with
          ``is_matched=False``. Inspect via :attr:`unmatched_features`.
        - ``"drop"``: features with NaN unit_id are removed from the
          attached shapefile. They're still recorded on
          :attr:`unmatched_features` for transparency, but they
          don't appear in products.
        - ``"error"``: raise ``ValueError`` with the count and first
          few offending names.

        After attachment, :attr:`shapefile_year` is populated from
        :func:`stablebound.snapshot.infer_year` over the matched
        (non-sentinel) units.

        With ``validate=True`` (the default) the attached ids are then
        cross-checked against the lineage at that vintage: ids on more than
        one polygon, ids the lineage has never heard of, ids already retired
        by that year, and districts in force with no polygon at all. Findings
        are reported as a single ``UserWarning`` and kept on
        :attr:`shapefile_issues`; nothing is raised, because a deliberately
        historical map fails these for a good reason. See
        :meth:`shapefile_report`.
        """
        if on_unmatched not in {"keep", "drop", "error"}:
            raise ValueError(
                f"on_unmatched must be 'keep', 'drop', or 'error'; "
                f"got {on_unmatched!r}"
            )

        # Two ways in. Either a reviewed decision file is supplied and the
        # districts are read off it, or the map already carries district
        # identifiers because the user attached them some other way.
        if mapping is not None:
            if name_column is None:
                raise ValueError(
                    "name_column is required when mapping is provided "
                    "(it tells attach which shapefile column the "
                    "mapping's source_name matches)."
                )
            gdf = attach_shapefile_ids(
                shapefile, mapping, name_column=name_column, id_column=id_column,
                coarse_column=coarse_column,
            )
        else:
            from .match import _coerce_gdf
            gdf = _coerce_gdf(shapefile).copy()
            if id_column not in gdf.columns:
                raise ValueError(
                    f"No mapping provided and shapefile is missing "
                    f"{id_column!r}. Either pass mapping=... or attach "
                    f"unit_ids to your shapefile beforehand."
                )

        # --- Unmatched-feature handling ------------------------------
        # Identify rows whose unit_id is still null (the mapping had a
        # blank proposed_unit_id, or the user supplied a shapefile with
        # some NaN values). Apply the on_unmatched policy.
        unmatched_mask = gdf[id_column].isna()
        n_unmatched = int(unmatched_mask.sum())

        if n_unmatched > 0:
            # Keep a copy of the shapes that were left out, noting where each
            # sat in the original file so a user can find it again after
            # anything below has filtered the map.
            unmatched_record = gdf.loc[unmatched_mask].copy()
            unmatched_record = unmatched_record.reset_index(names="source_idx")
            # Give each one a readable label. Preferably the name column the
            # caller pointed at; failing that, whatever the first descriptive
            # column turns out to be, since a list of row numbers is no help
            # to someone trying to work out what went missing.
            if name_column and name_column in unmatched_record.columns:
                unmatched_record["source_name"] = unmatched_record[name_column].astype(str)
            else:
                candidates = [
                    c for c in unmatched_record.columns
                    if c not in ("geometry", id_column, "source_idx")
                ]
                if candidates:
                    unmatched_record["source_name"] = (
                        unmatched_record[candidates[0]].astype(str)
                    )
                else:
                    unmatched_record["source_name"] = ""

            # Three ways to handle them, and the choice belongs to the user
            # because it depends on what the leftovers are. Refuse outright
            # is the safe default for a country being set up, where an
            # unmatched shape usually means the matching went wrong.
            if on_unmatched == "error":
                sample = unmatched_record["source_name"].head(5).tolist()
                raise ValueError(
                    f"attach_shapefile: {n_unmatched} feature(s) have no "
                    f"unit_id. First few: {sample!r}. "
                    "Pass on_unmatched='keep' to flag them as singletons, "
                    "or on_unmatched='drop' to remove them."
                )
            # Throw them away — right when the leftovers are genuinely not
            # wanted, such as a map that covers neighbouring countries too.
            if on_unmatched == "drop":
                _LOGGER.info(
                    "attach_shapefile: dropped %d unmatched feature(s); "
                    "inspect ln.unmatched_features.",
                    n_unmatched,
                )
                gdf = gdf.loc[~unmatched_mask].copy()
            else:
                # Or keep them, each under a placeholder identifier of its
                # own. They then travel through the rest of the pipeline as
                # districts in their own right rather than vanishing, which
                # keeps the map whole and leaves the anomaly visible in the
                # output instead of silently absent from it.
                sentinel_ids = [
                    f"UNMATCHED_{i}" for i in unmatched_record["source_idx"]
                ]
                gdf.loc[unmatched_mask, id_column] = sentinel_ids
                _LOGGER.info(
                    "attach_shapefile: %d unmatched feature(s) tagged "
                    "with sentinel IDs (UNMATCHED_<idx>); inspect "
                    "ln.unmatched_features.",
                    n_unmatched,
                )

            self._unmatched_features = gpd.GeoDataFrame(
                unmatched_record, crs=getattr(gdf, "crs", None)
            )
        else:
            self._unmatched_features = gpd.GeoDataFrame(
                {"source_idx": [], "source_name": [], "geometry": []},
                geometry="geometry",
                crs=getattr(gdf, "crs", None),
            )

        # Rename custom id_column to canonical "unit_id" — products
        # always read from "unit_id" by design (see boundary.py
        # ID_COLUMN). Deferred until after the unmatched-handling
        # block above so masks and slicing still work on the original
        # column name.
        if id_column != "unit_id":
            gdf = gdf.rename(columns={id_column: "unit_id"})

        # Infer the shapefile's effective year from the matched units
        # (excluding sentinels — they're not real lineage IDs).
        real_units = {
            u for u in gdf["unit_id"].astype(str)
            if not u.startswith("UNMATCHED_")
        }
        if real_units:
            # Seed the baseline, or every baseline unit the relationship table
            # never mentions counts as a mismatch in every candidate year --
            # 240 reported discrepancies against 12 real ones on India.
            year, _ = infer_year(
                real_units, self.lineage, list(self.years),
                additional_units=set(self.baseline["unit_id"].astype(str)),
            )
            self._shapefile_year = year

            # Now that the vintage is known, ask whether the ids actually
            # belong on this map. Attachment above only checked that every
            # feature GOT an id; these checks ask whether the id is real,
            # unique, and in force in that year -- the questions a wrong id
            # passes, because a wrong id is a well-formed string.
            if validate:
                from .validate import validate_shapefile_lineage_consistency
                # The COLUMN, not real_units: real_units is a set, and a set
                # cannot show that two polygons claim one id -- deduplicating
                # is precisely what hides the duplicate.
                self._shapefile_issues = validate_shapefile_lineage_consistency(
                    gdf["unit_id"].tolist(), self.lineage, year,
                    baseline=self.baseline,
                )
                self._warn_about_shapefile_issues()
        # If every feature is unmatched, leave _shapefile_year unset.

        self._shapefile = gdf
        return gdf

    def _warn_about_shapefile_issues(self) -> None:
        """Surface attach-time findings once, as a single warning.

        One warning rather than one per finding: the four checks fire in
        correlated pairs (a polygon on a retired id leaves a live district
        with no polygon), so reporting them separately would make one mistake
        look like several.
        """

        issues = self._shapefile_issues or []
        if not issues:
            return
        errors = [i for i in issues if i.severity == "error"]
        head = (f"attach_shapefile: {len(issues)} finding(s) on the attached map"
                + (f", {len(errors)} of them errors" if errors else ""))
        body = "\n".join(f"  [{i.severity}] {i.message}" for i in issues)
        warnings.warn(
            f"{head}:\n{body}\n"
            "Inspect ln.shapefile_issues, or print ln.shapefile_report(). "
            "Pass validate=False to attach_shapefile to skip these checks.",
            UserWarning,
            stacklevel=3,
        )

    @property
    def shapefile_issues(self) -> "list":
        """Findings from cross-checking the attached map against the lineage.

        Populated by :meth:`attach_shapefile` (unless it was called with
        ``validate=False``). Empty list when the map is clean; raises nothing
        either way -- these are reported, never enforced, because a map whose
        vintage genuinely sits outside the lineage will fail them for a good
        reason.
        """
        return self._shapefile_issues or []

    def shapefile_report(self) -> str:
        """Formatted text summary of the attached map's findings.

        The counterpart to :meth:`validation_report`, for the shapefile
        rather than the relationship table.
        """
        from .validate import format_issues
        if self._shapefile is None:
            return "No shapefile attached."
        issues = self.shapefile_issues
        if not issues:
            year = self._shapefile_year
            return (f"Lineage({self.country_code}) shapefile"
                    + (f" (vintage {year})" if year else "")
                    + ": no findings.")
        return (
            f"Lineage({self.country_code}) shapefile (vintage "
            f"{self._shapefile_year}): {len(issues)} finding(s).\n\n"
            + format_issues(issues)
        )

    @property
    def unmatched_features(self) -> gpd.GeoDataFrame:
        """Shapefile features that didn't receive a real unit_id at attach.

        Empty until :meth:`attach_shapefile` runs with unmatched
        features (under either ``on_unmatched="keep"`` or
        ``"drop"``). Columns include ``source_idx``, ``source_name``,
        and ``geometry``.

        With ``on_unmatched="keep"`` the same features also appear in
        :attr:`shapefile` with sentinel ``UNMATCHED_<idx>`` unit_ids;
        with ``"drop"`` they appear only here.
        """
        if self._unmatched_features is None:
            return gpd.GeoDataFrame(
                {"source_idx": [], "source_name": [], "geometry": []},
                geometry="geometry",
            )
        return self._unmatched_features

    @property
    def shapefile(self) -> gpd.GeoDataFrame:
        """The attached shapefile. Raises until :meth:`attach_shapefile` runs."""
        if self._shapefile is None:
            raise RuntimeError(
                "No shapefile attached to this Lineage. Call "
                "attach_shapefile(...) first."
            )
        return self._shapefile

    @property
    def shapefile_year(self) -> int:
        """Inferred year of the attached shapefile.

        Raises until :meth:`attach_shapefile` runs.
        """
        if self._shapefile_year is None:
            raise RuntimeError(
                "No shapefile attached (or no IDs matched, so the year "
                "could not be inferred)."
            )
        return self._shapefile_year

    # --- Stats helpers ---------------------------------------------------

    def attach_stats_ids(
        self,
        stats: _DfOrPath,
        mapping: pd.DataFrame | Path | str,
        *,
        name_column: str,
        id_column: str = "unit_id",
        year_column: str | None = None,
        coarse_column: str | None = None,
    ) -> pd.DataFrame:
        """Convenience: attach unit_ids to a stats frame via a reviewed mapping.

        No state is mutated on the lineage — the product call
        (``aggregate_stats(stats=...)``) is where stats actually
        plug into the pipeline. This helper just keeps the API
        symmetric with :meth:`attach_shapefile`.

        Pass the same ``year_column`` / ``coarse_column`` you used to build
        the proposal. A mapping keyed per (name, year) or (name, coarse) has
        several rows per name; joining it name-only would silently keep
        whichever row happened to come last, so
        :func:`stablebound.match.attach_stats_ids` raises instead of
        collapsing.
        """
        return attach_stats_ids(
            stats,
            mapping,
            name_column=name_column,
            id_column=id_column,
            year_column=year_column,
            coarse_column=coarse_column,
        )

    def validate_levels(self) -> list:
        """Check the admin levels agree about which upper units exist.

        Returns :class:`stablebound.validate.LineageIssue` records — empty
        when clean. Only meaningful for a country with a registered
        ``coarse_lineage_path``; single-level countries return ``[]``.
        """
        from .validate import validate_coarse_references

        if self._admin_level < 2 or self._coarse_rt_path is None:
            return []
        return validate_coarse_references(
            self.lineage,
            self.graph(level=self._admin_level - 1),
            baseline=self.baseline,
        )

    # --- FEWS deliverables -----------------------------------------------

    def export_fews(
        self,
        out_dir: Path | str,
        *,
        years: range | list[int] | None = None,
        admin0: str | None = None,
        iso: str | None = None,
        stats: pd.DataFrame | None = None,
        stats_unit_id_col: str = "unit_id",
        stats_year_col: str = "year",
        stats_out_name: str | None = None,
        stats_sheet_name: str | None = None,
        name_style: str = "bare",
        scratch_dir: Path | str | None = None,
        legacy_relationship: bool = False,
    ) -> dict[str, list[Path]]:
        """Write the three FEWS upload files for this country.

        Produces per-year ``{ISO}_Admin_Definitions_{year}.xlsx``,
        ``{ISO}_GeographicUnitRelationship.csv`` and — when ``stats`` is given
        — ``{ISO}_AgStats.xlsx``.

        This is the facade over :func:`stablebound.fews_export.build_deliverables`,
        which needs the per-level graphs, both code maps and the baseline
        assembled by hand. Doing that by hand is ~80 lines per country (the
        original India deliverable script did exactly that), and getting the
        name-change-log split wrong there injects a state's rename into the
        district graph.

        ``stats`` must already carry ``stats_unit_id_col`` — matching is out of
        scope here. Use :meth:`attach_stats_by_fnid` for FEWS-sourced
        statistics, or :meth:`propose_stats_mapping` + review otherwise.

        Args:
            years: Years to emit definitions for. Defaults to
                ``validity_start_year..coverage_end_year`` — the span the
                source actually describes, rather than an arbitrary window.
            admin0: Country display name for the ``admin0`` column. Defaults
                to the country code.
            iso: Override the ISO2 prefix; defaults to ``country_code``.
            name_style: ``"bare"`` (India's convention) or ``"with_country"``,
                which appends ``", {admin0}"`` as older FEWS files do.
            legacy_relationship: also write ``relationshiptable_{ISO}.csv`` in
                the 11-column dialect FEWS distributes. This is the interchange
                and round-trip format, not a fourth upload file — see
                :func:`stablebound.fews_export.build_legacy_relationship_table`.

        Returns:
            ``{"admin_definitions": [...], "relationship": [...],
            "agstats": [...]}``, plus ``"legacy_relationship"`` when that flag
            is set.
        """
        from .fews_export import build_deliverables

        iso = (iso or self.country_code).upper()
        admin0 = admin0 or iso
        if years is None:
            start = self.min_year
            end = self.coverage_end_year or self.lineage.max_event_year
            if end is None or end < start:
                raise ValueError(
                    f"cannot infer a year range for {iso}: no coverage_end_year "
                    "and no events. Pass years= explicitly."
                )
            years = range(int(start), int(end) + 1)

        if self._admin_level >= 2:
            # The admin2 code map keys each district on its ORIGIN admin1, so
            # without an upper layer there is nothing to build the admin1 tab
            # from — and no honest way to fake one. A district-level lineage
            # published with no division structure is exactly this case.
            # Emitting its districts as "admin1" would misdescribe the
            # hierarchy in a file handed to FEWS, so this refuses instead
            # (regression: tests/test_export_fews.py::
            # test_admin2_without_an_upper_layer_is_refused_clearly).
            ev = self.lineage.events
            has_event_coarse = (
                "child_coarse_id" in ev.columns and ev["child_coarse_id"].notna().any()
            )
            has_baseline_coarse = (
                self.baseline is not None and "coarse_id" in self.baseline.columns
            )
            if not (has_event_coarse or has_baseline_coarse):
                raise ValueError(
                    f"{iso} is an admin-level-{self._admin_level} lineage with no "
                    "upper-level attribution: its baseline has no 'coarse_id' "
                    "column and its events carry no 'child_coarse_id'. The FEWS "
                    "deliverable needs both levels — every admin2 unit's code is "
                    "keyed on the admin1 it originated in. Supply coarse_id / "
                    "coarse_name on the baseline (and ideally a "
                    "coarse_lineage_path for the upper level's own events), or "
                    "export the upper level separately."
                )

        # A dangling coarse_id would attribute districts to a state absent
        # from the state-level file — wrong hierarchy, shipped silently.
        level_issues = [i for i in self.validate_levels() if i.severity == "error"]
        if level_issues:
            from .validate import LineageDataError, format_issues

            raise LineageDataError(
                "cross-level validation failed; the deliverable would "
                "misattribute units:\n\n" + format_issues(level_issues)
            )

        has_coarse = self._coarse_rt_path is not None and self._admin_level > 1
        if self._admin_level > 2:
            raise NotImplementedError(
                f"{iso} is admin level {self._admin_level}; the FEWS deliverable "
                "supports levels 1 and 2 only. The code width for deeper levels "
                "is unconfirmed — see stablebound.fnid._CODE_WIDTH."
            )

        # A two-level country needs its upper level's events; a single-level
        # one uses its own graph for both, which is what build_deliverables
        # expects when there is no separate coarse lineage.
        adm2_graph = self.lineage if self._admin_level == 2 else None
        admin1_graph = self.graph(level=1) if has_coarse else self.lineage

        return build_deliverables(
            iso=iso,
            admin0=admin0,
            years=years,
            out_dir=Path(out_dir),
            adm2_graph=adm2_graph,
            admin1_graph=admin1_graph,
            baseline=self.baseline,
            stats=stats,
            stats_unit_id_col=stats_unit_id_col,
            stats_year_col=stats_year_col,
            stats_out_name=stats_out_name,
            stats_sheet_name=stats_sheet_name,
            name_style=name_style,
            scratch_dir=scratch_dir,
            legacy_relationship=legacy_relationship,
        )

    # --- Repr ------------------------------------------------------------

    def __repr__(self) -> str:
        """Progressive disclosure: show what's already been loaded.

        A fresh ``Lineage`` shows just the identity (country code,
        validity year, shapefile state). Once the user has accessed
        the graph (via ``ln.lineage``), the repr starts showing the
        year range and event count too. This keeps printing a Lineage
        cheap on fresh objects while surfacing richer info once
        loading has happened.
        """
        parts = [f"country_code={self.country_code!r}"]
        if self._validity_start_year is not None:
            parts.append(f"validity_start_year={self._validity_start_year}")
        if self._lineage is not None:
            yrs = self.years
            parts.append(f"years={yrs.start}..{yrs.stop - 1}")
            parts.append(f"events={len(self._lineage.events)}")
        parts.append("shapefile attached" if self._shapefile is not None else "no shapefile")
        return f"Lineage({', '.join(parts)})"


__all__ = ["Lineage"]
