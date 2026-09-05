"""I/O helpers for StableBound inputs.

Each ``read_*`` function loads a file (CSV or XLSX, picked by extension),
lowercases column names, validates against the canonical schema in
``schemas.py``, and returns a clean DataFrame. ``SchemaError`` is raised
on any column/row issue with a descriptive message.

The lowercase-then-project pattern keeps the package agnostic to whether
the input file uses `EVENT_YEAR` (legacy India), `event_year` (canonical),
or any case variation. Users still need to know what the columns mean,
just not their exact casing.

Also exports ``merge_name_changes`` for callers who track name changes in
a separate file and want to merge them into the RT before passing to
``LineageGraph``.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .schemas import (
    BASELINE_OPTIONAL_COLUMNS,
    BASELINE_REQUIRED_COLUMNS,
    NCL_REQUIRED_COLUMNS,
    RT_OPTIONAL_COLUMNS,
    RT_REQUIRED_COLUMNS,
    SchemaError,
    STATS_REQUIRED_COLUMNS,
    validate_baseline,
    validate_name_change_log,
    validate_relationship_table,
    validate_stats,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip whitespace from column names.

    The canonical schemas in ``schemas.py`` use lowercase names
    (``event_year``, ``unit_id``, etc.). Many real-world files use
    ALL_CAPS (legacy India RT: ``EVENT_YEAR``) or Mixed_Case. Rather
    than burden users with a column-rename step before every load, we
    normalize at the I/O boundary. ``str(c).strip()`` handles the
    occasional Excel quirk where a column name has a stray trailing
    space.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def read_relationship_table(path: Path | str) -> pd.DataFrame:
    """Load a canonical relationship table (CSV or XLSX).

    Pipeline:
        1. Pick reader by file extension.
        2. Lowercase column names.
        3. Validate (raises SchemaError on bad columns / event_types).
        4. Project to canonical-only columns and reset the index.

    The projection step (``df[keep]``) drops any extra columns the input
    file might have had — keeps downstream code from accidentally
    relying on non-canonical fields.
    """
    path = Path(path)
    # Extension-based reader pick. ``.xls`` is included for old-school
    # Excel files; pandas handles both via openpyxl/xlrd.
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _normalize_columns(df)
    validate_relationship_table(df)
    # Project to canonical columns only. Dropping extras here means the
    # rest of the package can assume a clean column set without further
    # validation.
    keep = [c for c in (*RT_REQUIRED_COLUMNS, *RT_OPTIONAL_COLUMNS) if c in df.columns]
    return df[keep].reset_index(drop=True)


def read_stats(path: Path | str, base_year: int) -> pd.DataFrame:
    """Load a long-form stats CSV. Validates schema and rejects pre-base-year rows.

    The ``base_year`` parameter flows through to ``validate_stats``,
    which enforces the ``min_year == base_year`` invariant: any row with
    ``year < base_year`` is a SchemaError. This is intentional —
    silently dropping out-of-range rows would mask data issues.

    No XLSX path: stats are typically large and CSV is the canonical
    interchange format. If a researcher has stats in XLSX they convert
    once.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    validate_stats(df, base_year=base_year)
    return df[list(STATS_REQUIRED_COLUMNS)].reset_index(drop=True)


def read_shapefile(path: Path | str, id_column: str) -> gpd.GeoDataFrame:
    """Load a shapefile / GeoJSON / etc. Verifies the id_column is present.

    Unlike the other readers, this one does NOT lowercase columns —
    shapefile column names often carry exact case from the source data
    provider (GADM uses ``GID_1``, ``NAME_1`` etc.) and the user
    explicitly names the ID column in Config. Lowercasing would change
    the contract.
    """
    path = Path(path)
    gdf = gpd.read_file(path)
    # The package downstream needs to look up unit_ids by this column.
    # Fail early with a useful message if the column doesn't exist.
    if id_column not in gdf.columns:
        raise SchemaError(
            f"id_column {id_column!r} not present in shapefile columns: "
            f"{list(gdf.columns)}"
        )
    return gdf


def read_name_change_log(path: Path | str) -> pd.DataFrame:
    """Load a name-change log (CSV or XLSX).

    A name-change log is a separate file — distinct from the RT — that
    lists official renames. Some FEWS NET data ships these as a sidecar
    rather than embedding them in the RT. Pass to ``merge_name_changes``
    to fold them into the RT before constructing a LineageGraph.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _normalize_columns(df)
    validate_name_change_log(df)
    # Keep `level` when the file carries it. India's log covers both admin
    # levels and the FEWS deliverable has to split them; projecting it away
    # forced callers to re-read the raw file just to recover the column.
    cols = list(NCL_REQUIRED_COLUMNS)
    if "level" in df.columns:
        cols.append("level")
    return df[cols].reset_index(drop=True)


def read_baseline(path: Path | str, base_year: int | None = None) -> pd.DataFrame:
    """Load a baseline-snapshot CSV/XLSX.

    Columns are lowercased and validated. If a ``year`` column is
    present AND ``base_year`` is provided, rows are filtered to that
    year — convenience for researchers who keep multi-year snapshots
    (1991, 1992, …, 2024) in a single CSV (the legacy India case has
    21,311 rows across all years; we want just the 467 from 1991).

    If only some baseline rows are needed, prefer this filtering over
    pre-slicing the file: keeps the source file as the single source of
    truth and avoids forking baselines per analysis.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _normalize_columns(df)
    # Year-filter BEFORE validation: a multi-year file might have
    # duplicate unit_ids (one row per year per unit), which fails the
    # validator's uniqueness check. Filtering first avoids the false
    # positive.
    if "year" in df.columns and base_year is not None:
        df = df[df["year"] == base_year].copy()
    validate_baseline(df)
    keep = [c for c in (*BASELINE_REQUIRED_COLUMNS, *BASELINE_OPTIONAL_COLUMNS) if c in df.columns]
    return df[keep].reset_index(drop=True)


def merge_name_changes(rt_df: pd.DataFrame, ncl_df: pd.DataFrame) -> pd.DataFrame:
    """Merge a separate name-change log into a relationship table.

    Conversion rule: each NCL row becomes one ``NameChange`` row in the
    RT with ``parent_id == child_id`` (the canonical NameChange
    semantics — same unit, different name). ``old_name`` becomes the
    parent_name (pre-rename), ``new_name`` becomes the child_name
    (post-rename).

    The merged frame is re-sorted by (event_year, event_type) with a
    stable sort so the original RT row order is preserved within each
    group — important for any downstream code that relies on iteration
    order matching the source.
    """
    validate_name_change_log(ncl_df)
    # Build the new NameChange rows directly. Note: we don't fill in
    # the optional coarse_id / coarse_name columns — name changes
    # don't usually have parent admin context.
    new_rows = pd.DataFrame(
        {
            "event_year": ncl_df["event_year"].astype(int),
            "event_type": "NameChange",
            "parent_id": ncl_df["unit_id"],     # NameChange: parent_id == child_id
            "parent_name": ncl_df["old_name"],
            "child_id": ncl_df["unit_id"],
            "child_name": ncl_df["new_name"],
        }
    )
    merged = pd.concat([rt_df, new_rows], ignore_index=True, sort=False)
    # ``kind="stable"`` preserves the original within-group ordering —
    # important if any caller is ordering-sensitive.
    return merged.sort_values(["event_year", "event_type"], kind="stable").reset_index(drop=True)
