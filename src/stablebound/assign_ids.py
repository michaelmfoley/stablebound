"""Mint canonical unit ids for a lineage that only carries names.

The package's own schema wants an id on every side of every event
(``parent_id`` / ``child_id`` in the relationship table, ``unit_id`` in
the baseline). Authoring a lineage, though, is done in names: "Bihar
split into Bihar and Jharkhand in 2000". This module bridges the two.
Give it a baseline whose ids are already right and a relationship table
whose id cells are blank, and it fills the blanks by replaying the
events in order, exactly the way the package's own snapshot walk would
read them afterwards:

* a parent is looked up by name (and coarse name, when the files carry
  one) among the units alive at the start of its event year;
* every child of a territorial event (``Split`` / ``Merge`` /
  ``Redistribute``) gets a fresh id, numbered after the highest id in
  use — the id names a territorial extent, so a changed extent is a new
  id even when the name continues (India's Chennai, Vietnam's Dong Nai);
* one child name in one year under one coarse unit is one id, however
  many parent rows feed it (multi-parent merges and redistributes);
* ``NameChange`` and ``Coarse`` keep the parent's id, as the schema
  requires (``parent_id == child_id``).

Two modes. ``"fill"`` (the default) trusts every id already present and
mints only for blank cells, so inserting a newly discovered event into
an existing lineage costs one new id and moves nothing else — mappings
keyed on ids stay valid. ``"rebuild"`` discards every id that is not a
baseline id and re-mints chronologically, which is what the original
India id-assignment script did; use it for a first pass or when the
numbering itself should be tidy.

Coarse (parent-level) ids are resolved, never minted: from the
baseline's ``coarse_id`` / ``coarse_name`` columns and, when an already
id'd coarse-level lineage is supplied, from that lineage's snapshots —
so a district moving into a state created in the same year is attached
to the new state's id. For a country with two changing levels, run this
on the coarse level first (its own coarse is the country) and pass the
result as ``coarse_lineage`` for the fine level.

Three places mint ids in this codebase, each for a different input:

* this module — the package's canonical schema with names only;
* :mod:`stablebound.rt_convert` — the legacy FEWS relationship-table
  dialect, where FNIDs are the names;
* the authoring skill's ``build_lineage.py`` — a JSON event spec.

Do not add a fourth; extend one of these.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .io import read_baseline, read_relationship_table
from .lineage import LineageGraph, name_of_unit_in_year
from .match import normalize_name
from .schemas import (
    NCL_REQUIRED_COLUMNS,
    RT_OPTIONAL_COLUMNS,
    RT_REQUIRED_COLUMNS,
    validate_relationship_table,
)
from .snapshot import build_snapshot
from .validate import LineageIssue, format_issues, validate_lineage

__all__ = ["IdAssignment", "IdAssignmentError", "assign_unit_ids"]

TERRITORIAL_EVENT_TYPES = ("Split", "Merge", "Redistribute")
MODES = ("fill", "rebuild")

# An id is "anything, then a run of digits": ``IN.ADM2.00001`` -> (``IN.ADM2.``,
# ``00001``); the synthetic fixtures' ``P.001`` -> (``P.``, ``001``).
_ID_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")
_LEVEL_RE = re.compile(r"ADM(\d)")

Normalizer = Callable[[str], str]


class IdAssignmentError(ValueError):
    """Raised when ids cannot be assigned from the names given.

    The message is a row-by-row report: every parent name that resolves to
    no living unit (or to more than one), every coarse name with no id, and
    the fix each one needs. All failures are collected before raising so a
    lineage with twenty typos is one round trip, not twenty.
    """


@dataclass
class IdAssignment:
    """The filled files plus an account of what was done.

    Attributes:
        lineage: the relationship table with every id cell filled; same
            columns and row order as the input.
        baseline: the baseline as read (ids were required, so unchanged).
        name_change_log: the log with blank ``unit_id`` cells filled, or
            ``None`` when no log was given.
        minted: one row per newly created id — ``unit_id``, ``name``,
            ``coarse_name``, ``event_year``, ``event_type``, ``row`` (the
            input row label that first needed it).
        id_prefix, id_width: the id format used, inferred or given.
        mode: ``"fill"`` or ``"rebuild"``.
        n_resolved: how many blank parent/child cells were filled with an
            *existing* id by name lookup (as opposed to minted).
        warnings: things that did not stop assignment but a human should
            read — an id that disagrees with its name, a name-change-log
            row that could not be placed.
        issues: the package validator's findings on the filled lineage.
            Not raised here: a lineage can be structurally odd and still be
            worth inspecting now that it loads.
    """

    lineage: pd.DataFrame
    baseline: pd.DataFrame
    name_change_log: pd.DataFrame | None
    minted: pd.DataFrame
    id_prefix: str
    id_width: int
    mode: str
    n_resolved: int
    warnings: list[str] = field(default_factory=list)
    issues: list[LineageIssue] = field(default_factory=list)

    def report(self) -> str:
        """Human-readable summary: format, counts, minted ids, warnings, issues."""
        lines = [
            "Unit id assignment",
            "==================",
            f"mode            : {self.mode}",
            f"id format       : {self.id_prefix}{'#' * self.id_width}",
            f"baseline units  : {len(self.baseline)}",
            f"lineage rows    : {len(self.lineage)}",
            f"ids minted      : {len(self.minted)}",
            f"cells resolved  : {self.n_resolved} (filled with an existing id by name)",
            "",
        ]
        if len(self.minted):
            lines += ["Minted ids", "----------", self.minted.to_string(index=False), ""]
        if self.warnings:
            lines += ["Warnings", "--------", *[f"- {w}" for w in self.warnings], ""]
        if self.issues:
            lines += ["Validator findings on the filled lineage", "-" * 40,
                      format_issues(self.issues), ""]
        else:
            lines += ["Validator findings on the filled lineage: none", ""]
        return "\n".join(lines)

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write ``lineage.csv``, ``baseline.csv``, ``name_change_log.csv``
        (when present) and ``id_assignment_report.txt`` into ``out_dir``.

        Returns the paths written, keyed by file stem.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        self.lineage.to_csv(out / "lineage.csv", index=False)
        written["lineage"] = out / "lineage.csv"
        self.baseline.to_csv(out / "baseline.csv", index=False)
        written["baseline"] = out / "baseline.csv"
        if self.name_change_log is not None:
            self.name_change_log.to_csv(out / "name_change_log.csv", index=False)
            written["name_change_log"] = out / "name_change_log.csv"
        (out / "id_assignment_report.txt").write_text(self.report(), encoding="utf-8")
        written["report"] = out / "id_assignment_report.txt"
        return written


# --- Small helpers ----------------------------------------------------------


def _is_blank(value: object) -> bool:
    """True for None, NaN, empty and whitespace-only strings, and the text 'nan'."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _clean(value: object) -> str | None:
    return None if _is_blank(value) else str(value).strip()


def _infer_format(
    baseline_ids: list[str], id_prefix: str | None, id_width: int | None
) -> tuple[str, int]:
    """The (prefix, digit width) every baseline id shares, unless overridden."""
    if id_prefix is not None and id_width is not None:
        return id_prefix, int(id_width)
    prefixes: set[str] = set()
    widths: set[int] = set()
    unparsed: list[str] = []
    for uid in baseline_ids:
        m = _ID_RE.match(uid)
        if m is None:
            unparsed.append(uid)
            continue
        prefixes.add(m.group("prefix"))
        widths.add(len(m.group("num")))
    problems: list[str] = []
    if unparsed:
        problems.append(f"ids that do not end in digits: {unparsed[:6]}")
    if len(prefixes) > 1:
        problems.append(f"more than one prefix: {sorted(prefixes)[:6]}")
    if len(widths) > 1:
        problems.append(f"more than one digit width: {sorted(widths)}")
    if problems and (id_prefix is None or id_width is None):
        raise IdAssignmentError(
            "cannot infer the id format from the baseline ("
            + "; ".join(problems)
            + "). Pass id_prefix= and id_width= explicitly, e.g. "
            "id_prefix='XX.ADM1.', id_width=5."
        )
    if not prefixes:
        raise IdAssignmentError("the baseline has no unit ids to infer a format from.")
    prefix = id_prefix if id_prefix is not None else next(iter(prefixes))
    width = int(id_width) if id_width is not None else next(iter(widths))
    return prefix, width


def _number_of(uid: str | None, prefix: str) -> int | None:
    """The numeric part of ``uid`` when it uses ``prefix``; else None."""
    if uid is None or not uid.startswith(prefix):
        return None
    rest = uid[len(prefix):]
    return int(rest) if rest.isdigit() else None


def _load_rt(lineage: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(lineage, pd.DataFrame):
        df = lineage.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        validate_relationship_table(df)
        keep = [c for c in (*RT_REQUIRED_COLUMNS, *RT_OPTIONAL_COLUMNS) if c in df.columns]
        return df[keep]
    return read_relationship_table(lineage)


def _load_baseline(baseline: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(baseline, pd.DataFrame):
        df = baseline.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "unit_id" not in df.columns or "name" not in df.columns:
            raise IdAssignmentError(
                "the baseline needs unit_id and name columns; the baseline is where ids "
                "come from, so it cannot itself be names-only."
            )
        return df
    return read_baseline(baseline)


def _load_ncl(ncl: pd.DataFrame | Path | str | None) -> pd.DataFrame | None:
    """Read a name-change log without the schema check, which would reject
    the blank ``unit_id`` cells this module exists to fill."""
    if ncl is None:
        return None
    if isinstance(ncl, pd.DataFrame):
        df = ncl.copy()
    else:
        p = Path(ncl)
        df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in NCL_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IdAssignmentError(f"name-change log is missing columns: {missing}")
    return df


# --- The living set, replayed year by year -----------------------------------


class _Alive:
    """Who exists right now, findable by (normalised name, normalised coarse name).

    Kept as three small maps rather than one keyed dict so that a rename or
    a coarse reassignment is an update in place, and so that a name shared
    by two living units is detected (returned as two candidates) instead of
    one silently shadowing the other.
    """

    def __init__(self, norm: Normalizer, use_coarse: bool) -> None:
        self.norm = norm
        self.use_coarse = use_coarse
        self.name_of: dict[str, str] = {}
        self.coarse_of: dict[str, str | None] = {}
        self.by_name: dict[str, set[str]] = defaultdict(set)

    def add(self, uid: str, name: str, coarse: str | None) -> None:
        self.name_of[uid] = name
        self.coarse_of[uid] = coarse
        self.by_name[self.norm(name)].add(uid)

    def remove(self, uid: str) -> None:
        name = self.name_of.pop(uid, None)
        self.coarse_of.pop(uid, None)
        if name is not None:
            self.by_name[self.norm(name)].discard(uid)

    def rename(self, uid: str, new_name: str) -> None:
        if uid not in self.name_of:
            return
        self.by_name[self.norm(self.name_of[uid])].discard(uid)
        self.name_of[uid] = new_name
        self.by_name[self.norm(new_name)].add(uid)

    def recoarse(self, uid: str, new_coarse: str | None) -> None:
        if uid in self.coarse_of:
            self.coarse_of[uid] = new_coarse

    def rename_coarse(self, old: str, new: str) -> int:
        """A coarse unit was renamed: every unit filed under ``old`` now sits
        under ``new``. Returns how many units moved."""
        want = self.norm(old)
        moved = 0
        for uid, co in self.coarse_of.items():
            if co is not None and self.norm(co) == want:
                self.coarse_of[uid] = new
                moved += 1
        return moved

    def __contains__(self, uid: str) -> bool:
        return uid in self.name_of

    def _same_coarse(self, uid: str, coarse: str) -> bool:
        have = self.coarse_of.get(uid)
        return have is not None and self.norm(have) == self.norm(coarse)

    def candidates(self, name: str, coarse: str | None) -> tuple[list[str], bool]:
        """Living units called ``name``, and whether the coarse name disagreed.

        The coarse name breaks ties; it does not veto. A name carried by one
        living unit resolves to that unit even when the row's coarse name
        differs (the flag says so), because the coarse column is descriptive
        and the coarse level may itself have been renamed. Two living units
        with the name are narrowed by coarse; if none matches, both are
        returned so the caller reports the ambiguity.
        """
        ids = sorted(self.by_name.get(self.norm(name), ()))
        if not self.use_coarse or coarse is None or not ids:
            return ids, False
        if len(ids) == 1:
            return ids, not self._same_coarse(ids[0], coarse)
        narrowed = [u for u in ids if self._same_coarse(u, coarse)]
        return (narrowed, False) if narrowed else (ids, True)

    def nearest(self, name: str, coarse: str | None, n: int = 3) -> list[str]:
        """Living names closest to ``name`` (same coarse unit first), for error text."""
        from .match import _similarity

        target = self.norm(name)
        scored: list[tuple[float, str]] = []
        for uid, nm in self.name_of.items():
            if self.use_coarse and coarse is not None:
                have = self.coarse_of.get(uid)
                if have is None or self.norm(str(have)) != self.norm(coarse):
                    continue
            scored.append((_similarity(target, self.norm(nm)), f"{nm} ({uid})"))
        scored.sort(reverse=True)
        return [label for _, label in scored[:n]]


class _CoarseResolver:
    """Coarse name -> coarse id, from the baseline and (optionally) a coarse lineage."""

    def __init__(
        self,
        baseline: pd.DataFrame,
        coarse_lineage: pd.DataFrame | Path | str | None,
        norm: Normalizer,
    ) -> None:
        self.norm = norm
        self.base_by_name: dict[str, set[str]] = defaultdict(set)
        self.base_names: dict[str, str] = {}
        if "coarse_id" in baseline.columns and "coarse_name" in baseline.columns:
            for cid, cname in zip(baseline["coarse_id"], baseline["coarse_name"]):
                c_id, c_name = _clean(cid), _clean(cname)
                if c_id is None or c_name is None:
                    continue
                self.base_by_name[norm(c_name)].add(c_id)
                self.base_names.setdefault(c_id, c_name)
        self.graph: LineageGraph | None = None
        if coarse_lineage is not None:
            self.graph = LineageGraph.from_dataframe(_load_rt(coarse_lineage), validate=False)
        self._snapshots: dict[int, dict[str, set[str]]] = {}
        # Renames of coarse units recorded in a name-change log rather than in
        # the coarse lineage: normalised new name -> normalised old name.
        self.aliases: dict[str, str] = {}

    def rename(self, old: str, new: str) -> None:
        self.aliases[self.norm(new)] = self.norm(old)
        if self.norm(old) in self.base_by_name:
            self.base_by_name[self.norm(new)] |= self.base_by_name[self.norm(old)]

    def _lookup(self, table: dict[str, set[str]], key: str) -> list[str]:
        seen: set[str] = set()
        while key not in table and key in self.aliases and key not in seen:
            seen.add(key)
            key = self.aliases[key]
        return sorted(table.get(key, ()))

    def _at(self, year: int) -> dict[str, set[str]]:
        """Normalised coarse name -> ids alive at the start of ``year``."""
        assert self.graph is not None
        if year not in self._snapshots:
            ids = build_snapshot(self.graph, year, additional_units=set(self.base_names))
            table: dict[str, set[str]] = defaultdict(set)
            for uid in ids:
                nm = name_of_unit_in_year(self.graph, uid, year) or self.base_names.get(uid)
                if nm:
                    table[self.norm(nm)].add(uid)
            self._snapshots[year] = table
        return self._snapshots[year]

    def resolve(self, name: str, year: int) -> tuple[str | None, list[str]]:
        """(id, candidates): id is set only when exactly one candidate exists."""
        key = self.norm(name)
        if self.graph is not None:
            found = self._lookup(self._at(year), key)
            if found:
                return (found[0] if len(found) == 1 else None), found
        found = self._lookup(self.base_by_name, key)
        return (found[0] if len(found) == 1 else None), found


# --- The assignment ---------------------------------------------------------


def assign_unit_ids(
    lineage: pd.DataFrame | Path | str,
    baseline: pd.DataFrame | Path | str,
    *,
    name_change_log: pd.DataFrame | Path | str | None = None,
    coarse_lineage: pd.DataFrame | Path | str | None = None,
    mode: str = "fill",
    id_prefix: str | None = None,
    id_width: int | None = None,
    normalizer: Normalizer | None = None,
) -> IdAssignment:
    """Fill the id columns of a relationship table from its names.

    Args:
        lineage: relationship table (DataFrame or CSV/XLSX path) in the
            package's schema, with some or all ``parent_id`` / ``child_id``
            (and ``*_coarse_id``) cells blank.
        baseline: baseline snapshot with ``unit_id`` and ``name`` filled;
            ``coarse_id`` / ``coarse_name`` when the country has a coarse level.
        name_change_log: optional log (``event_year, unit_id, old_name,
            new_name`` [, ``level``]); blank ``unit_id`` cells are filled and
            the renames take part in the replay so later events may use the
            new name.
        coarse_lineage: optional, already id'd relationship table of the
            coarse level, used to resolve coarse names year by year.
        mode: ``"fill"`` keeps existing ids and mints only for blanks;
            ``"rebuild"`` discards every non-baseline id first.
        id_prefix, id_width: the id format; inferred from the baseline when
            omitted (``IN.ADM2.00001`` -> ``"IN.ADM2."``, 5).
        normalizer: name normaliser for lookups; default
            :func:`stablebound.match.normalize_name`.

    Returns:
        An :class:`IdAssignment` with the filled frames and a report.

    Raises:
        IdAssignmentError: a name that cannot be placed, an ambiguous name,
            an id format that cannot be inferred, or a bad ``mode``.
    """
    if mode not in MODES:
        raise IdAssignmentError(f"mode must be one of {MODES}, got {mode!r}")
    norm: Normalizer = normalizer or normalize_name

    rt = _load_rt(lineage)
    base = _load_baseline(baseline)
    ncl = _load_ncl(name_change_log)

    base_ids = [str(u).strip() for u in base["unit_id"]]
    base_names = [str(n).strip() for n in base["name"]]
    base_coarse = (
        [_clean(c) for c in base["coarse_name"]] if "coarse_name" in base.columns
        else [None] * len(base)
    )
    prefix, width = _infer_format(base_ids, id_prefix, id_width)
    level_match = _LEVEL_RE.search(prefix)
    my_level: int | None = int(level_match.group(1)) if level_match else None

    has_coarse_cols = all(c in rt.columns for c in RT_OPTIONAL_COLUMNS)
    use_coarse = has_coarse_cols and "coarse_name" in base.columns
    id_cols = ["parent_id", "child_id"] + (
        ["parent_coarse_id", "child_coarse_id"] if has_coarse_cols else []
    )

    # Work on plain records: one dict per row, blanks normalised to None.
    index = list(rt.index)
    rows: list[dict] = rt.to_dict("records")
    for r in rows:
        r["event_year"] = int(r["event_year"])
        r["event_type"] = str(r["event_type"]).strip()
        for c in id_cols:
            r[c] = _clean(r.get(c))
        for c in ("parent_name", "child_name", "parent_coarse_name", "child_coarse_name"):
            if c in r:
                r[c] = _clean(r.get(c))

    baseline_id_set = set(base_ids)
    if mode == "rebuild":
        for r in rows:
            for c in ("parent_id", "child_id"):
                if r[c] is not None and r[c] not in baseline_id_set:
                    r[c] = None

    used = [n for uid in base_ids if (n := _number_of(uid, prefix)) is not None]
    for r in rows:
        for c in ("parent_id", "child_id"):
            n = _number_of(r[c], prefix)
            if n is not None:
                used.append(n)
    next_n = max(used, default=0) + 1

    def mint() -> str:
        nonlocal next_n
        uid = f"{prefix}{next_n:0{width}d}"
        next_n += 1
        return uid

    alive = _Alive(norm, use_coarse)
    for uid, name, coarse in zip(base_ids, base_names, base_coarse):
        alive.add(uid, name, coarse)

    coarse_resolver = _CoarseResolver(base, coarse_lineage, norm) if has_coarse_cols else None

    # Name-change-log rows, grouped by year, with blanks normalised.
    ncl_rows: list[dict] = []
    if ncl is not None:
        for label, rec in zip(ncl.index, ncl.to_dict("records")):
            rec["_label"] = label
            rec["event_year"] = int(rec["event_year"])
            rec["unit_id"] = _clean(rec.get("unit_id"))
            rec["old_name"] = _clean(rec.get("old_name"))
            rec["new_name"] = _clean(rec.get("new_name"))
            ncl_rows.append(rec)
    if mode == "rebuild":
        # The log's ids for this level follow the numbering being discarded;
        # they are re-resolved from old_name. Coarse-level ids are kept — the
        # coarse level is not re-minted here.
        for rec in ncl_rows:
            uid = rec["unit_id"]
            if uid is not None and _number_of(uid, prefix) is not None \
                    and uid not in baseline_id_set:
                rec["unit_id"] = None

    errors: list[str] = []
    warnings: list[str] = []
    minted: list[dict] = []
    n_resolved = 0

    def resolve_parent(name: str, coarse: str | None,
                       pending: dict[str, str],
                       same_year: dict[tuple[str, str | None], str]) -> tuple[str | None, str]:
        """The living unit called ``name``.

        Priority: units alive at the start of the year; then a unit renamed
        *to* this name by a log entry dated this year (the rename came first);
        then a child minted earlier this year (a transient). The second value
        is ``"ok"``, ``"coarse-mismatch"`` or the reason for failure.
        """
        found, mismatch = alive.candidates(name, coarse)
        if len(found) == 1:
            return found[0], ("coarse-mismatch" if mismatch else "ok")
        if len(found) > 1:
            return None, f"ambiguous among {found}"
        if norm(name) in pending:
            return pending[norm(name)], "ok"
        key = (norm(name), norm(coarse) if (use_coarse and coarse is not None) else None)
        if key in same_year:
            return same_year[key], "ok"
        return None, "no living unit by that name"

    years = sorted({r["event_year"] for r in rows} | {r["event_year"] for r in ncl_rows})
    for year in years:
        same_year: dict[tuple[str, str | None], str] = {}
        children: dict[str, tuple[str, str | None]] = {}
        parents_ceasing: set[str] = set()
        renames: list[tuple[str, str]] = []
        recoarse: list[tuple[str, str | None]] = []
        coarse_renames: list[tuple[str, str]] = []
        pending: dict[str, str] = {}  # normalised new name -> uid, log renames dated this year
        deferred_ncl: list[dict] = []

        # Name-change log rows dated this year come first: an event dated the
        # same year may refer to the unit by its new name (India's Amroha,
        # renamed and split in 2012). Rows for this level fill their unit_id
        # and rename the unit; rows for the coarse level rename the coarse
        # unit under which our units are filed. A row whose unit does not
        # exist yet (created this year) is retried after the events.
        for rec in ncl_rows:
            if rec["event_year"] != year:
                continue
            old, new = rec["old_name"], rec["new_name"]
            uid = rec["unit_id"]
            lvl_raw = rec.get("level")
            lvl: int | None = None if _is_blank(lvl_raw) else int(str(lvl_raw).strip())
            is_coarse_row = (
                (my_level is not None and lvl is not None and lvl == my_level - 1)
                or (uid is not None and _number_of(uid, prefix) is None and lvl is None)
            )
            if is_coarse_row:
                if old is not None and new is not None:
                    coarse_renames.append((old, new))
                continue
            if my_level is not None and lvl is not None and lvl != my_level:
                continue  # some other admin level's rename; not ours to fill
            if old is None or new is None:
                warnings.append(f"name-change row {rec['_label']} ({year}): old or new name blank")
                continue
            if uid is None:
                cands, _ = alive.candidates(old, None)
                if len(cands) != 1:
                    deferred_ncl.append(rec)
                    continue
                uid = cands[0]
                rec["unit_id"] = uid
                n_resolved += 1
            if uid in alive:
                renames.append((uid, new))
                pending[norm(new)] = uid
            else:
                deferred_ncl.append(rec)

        # A child id already present on any row of a territorial event anchors
        # every row of that child this year, whichever row carries it.
        for pos, r in enumerate(rows):
            if r["event_year"] != year or r["event_type"] not in TERRITORIAL_EVENT_TYPES:
                continue
            cid, cname = r["child_id"], r.get("child_name")
            if cid is None or cname is None:
                continue
            ccoarse = r.get("child_coarse_name") if has_coarse_cols else None
            key = (norm(cname), norm(ccoarse) if (use_coarse and ccoarse is not None) else None)
            if key in same_year and same_year[key] != cid:
                warnings.append(
                    f"row {index[pos]} ({year} {r['event_type']}): child {cname!r} already "
                    f"has id {same_year[key]} this year; row's child_id {cid} replaced"
                )
            else:
                same_year.setdefault(key, cid)

        for pos, r in enumerate(rows):
            if r["event_year"] != year:
                continue
            label = index[pos]
            etype = r["event_type"]
            pname, cname = r.get("parent_name"), r.get("child_name")
            pcoarse = r.get("parent_coarse_name") if has_coarse_cols else None
            ccoarse = r.get("child_coarse_name") if has_coarse_cols else None
            if pname is None:
                errors.append(f"row {label} ({year} {etype}): parent_name is blank")
                continue

            # Parent side.
            resolved, why = resolve_parent(pname, pcoarse, pending, same_year)
            pid = r["parent_id"]
            if pid is None:
                if resolved is None:
                    hint = alive.nearest(pname, pcoarse)
                    errors.append(
                        f"row {label} ({year} {etype}): parent {pname!r}"
                        + (f" in {pcoarse!r}" if pcoarse else "")
                        + f" — {why}"
                        + (f"; nearest living names: {hint}" if hint else "")
                    )
                    continue
                pid = resolved
                n_resolved += 1
                if why == "coarse-mismatch":
                    warnings.append(
                        f"row {label} ({year} {etype}): parent {pname!r} resolved to {pid} "
                        f"by name; its coarse unit is {alive.coarse_of.get(pid)!r}, the row "
                        f"says {pcoarse!r}"
                    )
            else:
                if resolved is not None and resolved != pid:
                    warnings.append(
                        f"row {label} ({year} {etype}): parent_id {pid} kept, but the name "
                        f"{pname!r} resolves to {resolved}"
                    )
                if pid not in alive and pid not in children:
                    warnings.append(
                        f"row {label} ({year} {etype}): parent_id {pid} is neither in the "
                        f"baseline nor created by an earlier event; trusted as given"
                    )
                    alive.add(pid, pname, pcoarse)
            r["parent_id"] = pid

            # Child side.
            if cname is None:
                errors.append(f"row {label} ({year} {etype}): child_name is blank")
                continue
            if etype in TERRITORIAL_EVENT_TYPES:
                key = (norm(cname), norm(ccoarse) if (use_coarse and ccoarse is not None) else None)
                cid = r["child_id"]
                if key in same_year:
                    cid = same_year[key]
                elif cid is None:
                    cid = mint()
                    minted.append({
                        "unit_id": cid, "name": cname, "coarse_name": ccoarse,
                        "event_year": year, "event_type": etype, "row": label,
                    })
                    same_year[key] = cid
                else:
                    same_year[key] = cid
                r["child_id"] = cid
                parents_ceasing.add(pid)
                children[cid] = (cname, ccoarse)
            else:  # NameChange / Coarse: the unit keeps its id.
                cid = r["child_id"]
                if cid is None:
                    cid = pid
                    n_resolved += 1
                elif cid != pid:
                    warnings.append(
                        f"row {label} ({year} {etype}): child_id {cid} differs from parent_id "
                        f"{pid}; the schema expects them equal for {etype}. Kept as given."
                    )
                r["child_id"] = cid
                if etype == "NameChange":
                    renames.append((pid, cname))
                else:
                    recoarse.append((pid, ccoarse))

            # Coarse id columns: resolved from the baseline / coarse lineage.
            if coarse_resolver is not None:
                for side, at_year in (("parent", year), ("child", year + 1)):
                    col, nm = f"{side}_coarse_id", r.get(f"{side}_coarse_name")
                    if r.get(col) is not None or nm is None:
                        continue
                    found, cands = coarse_resolver.resolve(nm, at_year)
                    if found is None:
                        errors.append(
                            f"row {label} ({year} {etype}): {side} coarse unit {nm!r} "
                            + ("is ambiguous among " + str(cands) if cands else
                               "has no id in the baseline"
                               + ("" if coarse_resolver.graph is not None
                                  else " (pass coarse_lineage= if it is created by an event)"))
                        )
                    else:
                        r[col] = found
                        n_resolved += 1

        # Log rows that found no living unit before the events: the unit may
        # have been created this year.
        for rec in deferred_ncl:
            old, new, uid = rec["old_name"], rec["new_name"], rec["unit_id"]
            if uid is None:
                living, _ = alive.candidates(str(old), None)
                cands = living if len(living) > 1 else sorted(
                    cid for cid, (nm, _) in children.items() if norm(nm) == norm(str(old))
                )
                if len(cands) != 1:
                    warnings.append(
                        f"name-change row {rec['_label']} ({year}): {old!r} "
                        + ("is ambiguous among " + str(cands)
                           + "; fill unit_id on that row" if cands else
                           "matches no living unit; unit_id left blank")
                    )
                    continue
                uid = cands[0]
                rec["unit_id"] = uid
                n_resolved += 1
            if uid in alive or uid in children:
                renames.append((uid, str(new)))
            else:
                warnings.append(
                    f"name-change row {rec['_label']} ({year}): unit_id {uid} is not alive "
                    f"in {year}; rename not applied"
                )

        # End of year: apply the year's changes at once, as build_snapshot does.
        for uid in parents_ceasing:
            alive.remove(uid)
        for cid, (nm, co) in children.items():
            alive.add(cid, nm, co)
        for uid, new_name in renames:
            alive.rename(uid, new_name)
        for uid, co in recoarse:
            alive.recoarse(uid, co)
        for old, new in coarse_renames:
            alive.rename_coarse(old, new)
            if coarse_resolver is not None:
                coarse_resolver.rename(old, new)

    if errors:
        raise IdAssignmentError(
            f"{len(errors)} row(s) could not be assigned an id:\n  " + "\n  ".join(errors)
        )

    out = pd.DataFrame(rows, columns=list(rt.columns), index=index)
    out["event_year"] = out["event_year"].astype(int)
    validate_relationship_table(out)
    graph = LineageGraph.from_dataframe(out, validate=False)
    issues = validate_lineage(graph)

    ncl_out: pd.DataFrame | None = None
    if ncl is not None:
        ncl_out = ncl.copy()
        ncl_out["unit_id"] = [rec["unit_id"] for rec in ncl_rows]

    minted_df = pd.DataFrame(
        minted, columns=["unit_id", "name", "coarse_name", "event_year", "event_type", "row"]
    )
    return IdAssignment(
        lineage=out,
        baseline=base,
        name_change_log=ncl_out,
        minted=minted_df,
        id_prefix=prefix,
        id_width=width,
        mode=mode,
        n_resolved=n_resolved,
        warnings=warnings,
        issues=issues,
    )
