"""Compare two :class:`Lineage` objects on everything a user can read off them.

Used by the tests that prove a lineage loaded one way (from a folder as
published, or from keyword arguments) is the same as the lineage loaded
another way. ``lineage_differences`` returns a list of human-readable
differences rather than asserting, so a test can also show that a planted
change *is* reported — a comparison that cannot fail proves nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import openpyxl
import pandas as pd

FACTS = (
    "country_code",
    "admin_level",
    "max_level",
    "validity_start_year",
    "coverage_end_year",
    "min_year",
    "notes",
)


def _frame_diff(label: str, x: pd.DataFrame | None, y: pd.DataFrame | None) -> list[str]:
    if x is None or y is None:
        return [] if x is None and y is None else [f"{label}: one side is None"]
    try:
        pd.testing.assert_frame_equal(
            x.reset_index(drop=True), y.reset_index(drop=True), check_dtype=True
        )
    except AssertionError as exc:
        return [f"{label}: {str(exc).splitlines()[0]}"]
    return []


def _issues(issues: Iterable) -> list[tuple]:
    return [(i.severity, i.category, i.message, tuple(i.ids or ())) for i in issues]


def _file_content(p: Path) -> object:
    if p.suffix == ".xlsx":
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        try:
            return {ws.title: list(ws.iter_rows(values_only=True)) for ws in wb.worksheets}
        finally:
            wb.close()
    return p.read_bytes()


def _tree_diff(label: str, a: Path, b: Path) -> list[str]:
    fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    out = [f"{label}: only on one side: {sorted(map(str, fa ^ fb))}"] if fa != fb else []
    out += [
        f"{label}: {rel} differs"
        for rel in sorted(fa & fb)
        if _file_content(a / rel) != _file_content(b / rel)
    ]
    return out


def lineage_differences(
    a,
    b,
    *,
    years: Iterable[int] = (),
    names: Iterable[str] = (),
    export_dir: Path | None = None,
    export_years: list[int] | None = None,
    admin0: str = "Country",
) -> list[str]:
    """Everything that differs between lineages ``a`` and ``b``.

    Compared: the facts in :data:`FACTS`; the relationship table, baseline
    and name-change log as loaded; the working graph's events and, for a
    two-level lineage, the coarse graph's; ``validation_issues`` and
    ``validate_levels()``; the snapshot in each of ``years``; the default
    normalizer on each of ``names``; and, when ``export_dir`` is given, every
    file ``export_fews`` writes for ``export_years`` (workbooks by cell value).
    """
    diffs: list[str] = []
    for fact in FACTS:
        va, vb = getattr(a, fact), getattr(b, fact)
        if va != vb:
            diffs.append(f"{fact}: {va!r} != {vb!r}")

    diffs += _frame_diff("relationship_table", a.relationship_table, b.relationship_table)
    diffs += _frame_diff("baseline", a.baseline, b.baseline)
    diffs += _frame_diff("name_change_log", a.name_change_log, b.name_change_log)
    diffs += _frame_diff("lineage.events", a.lineage.events, b.lineage.events)
    if a.admin_level >= 2 and b.admin_level >= 2:
        try:
            ga, gb = a.graph(level=a.admin_level - 1), b.graph(level=b.admin_level - 1)
        except ValueError as exc:
            diffs.append(f"coarse graph: {exc}")
        else:
            diffs += _frame_diff("coarse graph events", ga.events, gb.events)
        if _issues(a.validate_levels()) != _issues(b.validate_levels()):
            diffs.append("validate_levels() differs")

    if _issues(a.validation_issues) != _issues(b.validation_issues):
        diffs.append(
            f"validation_issues differ ({len(a.validation_issues)} vs "
            f"{len(b.validation_issues)})"
        )

    for y in years:
        diffs += _frame_diff(f"snapshot({y})", a.snapshot(year=y), b.snapshot(year=y))

    na, nb = a.default_normalizer, b.default_normalizer
    bad = [n for n in names if na(n) != nb(n)]
    if bad:
        diffs.append(f"default_normalizer differs on {len(bad)} name(s), e.g. {bad[:3]!r}")

    if export_dir is not None:
        da, db = Path(export_dir) / "a", Path(export_dir) / "b"
        a.export_fews(da, years=export_years, admin0=admin0)
        b.export_fews(db, years=export_years, admin0=admin0)
        diffs += _tree_diff("export_fews", da, db)
    return diffs
