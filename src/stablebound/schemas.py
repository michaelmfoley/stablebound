"""Canonical column names and validators for StableBound inputs.

The package commits to a single canonical schema for each input.
Researchers reshape their inputs once before passing them in; the
pipeline does no column-name guessing, fuzzy matching, or type coercion
beyond what's needed to read CSV/XLSX. The strictness is intentional —
trying to be flexible at the input layer is where every multi-country
data pipeline gets bogged down.

Five schemas are defined:

- Relationship table (RT): one row per boundary event. Required and
  optional column lists; valid event_type values.
- Stats (long-form): one row per (unit_id, year, season, variable, value)
  observation. Strict — additive variables only.
- Name change log: one row per official rename. Merged into the RT at
  load time as NameChange events.
- Baseline snapshot: one row per unit alive at the base year.

Each schema has a corresponding ``validate_*(df, ...)`` function that
raises ``SchemaError`` with a row/column-specific message on failure.
"""

from __future__ import annotations

import pandas as pd

# --- Relationship table --------------------------------------------------
#
# One row per boundary event. The 6 required columns are the structural
# minimum; the 4 optional ones (coarse_*) carry parent admin context like
# state names, used by upstream matching scripts for homonym disambiguation
# (e.g., "Hamirpur, Himachal Pradesh" vs "Hamirpur, Uttar Pradesh").

RT_REQUIRED_COLUMNS = (
    "event_year",
    "event_type",
    "parent_id",
    "parent_name",
    "child_id",
    "child_name",
)

RT_OPTIONAL_COLUMNS = (
    "parent_coarse_id",
    "parent_coarse_name",
    "child_coarse_id",
    "child_coarse_name",
)

# Five recognized event types. Three are territorial; two are not.
#
#   Split        — 1 parent → ≥1 children. Parent ceases.
#   Merge        — ≥1 parents → 1 child. Parents cease.
#   Redistribute — territory exchange. Parent and child IDs both new.
#   NameChange   — same unit, new name. parent_id == child_id.
#   Coarse       — parent admin level reassigned. parent_id == child_id.
#
# Coarse was added for India (247 such rows). Algorithmically it's
# treated identically to NameChange — it doesn't move territory, so
# snapshot/group construction skips it.
VALID_EVENT_TYPES = frozenset({"Split", "Merge", "Redistribute", "NameChange", "Coarse"})

# --- Stats table ---------------------------------------------------------
#
# Long-form. Every observation is one row. Variables must be additive
# (extensive) — yields and other intensives are recomputed
# post-aggregation by ``derive_intensive``.

STATS_REQUIRED_COLUMNS = (
    "unit_id",
    "year",
    "season",
    "variable",
    "value",
)

# --- Name change log -----------------------------------------------------
#
# Optional sidecar file for renames. Folded into the RT as NameChange rows
# at load time via ``io.merge_name_changes``.

NCL_REQUIRED_COLUMNS = (
    "event_year",
    "unit_id",
    "old_name",
    "new_name",
)

# --- Baseline snapshot ---------------------------------------------------
#
# Lists every unit alive at the base year (or earliest year covered by
# the relationship table). Required when the RT does not cover every
# unit — e.g., units that have existed unchanged through the full RT
# timeline have no events and would otherwise be invisible to the
# package. India: 136 of 467 1991-baseline districts never appear in the
# RT at all; without the baseline file they'd be missing from every
# snapshot.
#
# Pairs naturally with FEWS NET / HarvestStat-style data, which
# typically distributes a relationship table alongside an initial-year
# snapshot.

BASELINE_REQUIRED_COLUMNS = (
    "unit_id",
    "name",
)

BASELINE_OPTIONAL_COLUMNS = (
    "year",         # if present, used to filter rows to the base year
    "coarse_id",
    "coarse_name",
)


class SchemaError(ValueError):
    """Raised when an input frame does not conform to the canonical schema.

    Subclasses ``ValueError`` so callers that catch broad value errors
    still see these. The message always identifies the specific column
    or row that triggered the failure.
    """


def validate_relationship_table(df: pd.DataFrame) -> None:
    """Raise SchemaError if the relationship table is malformed.

    Four checks:
        1. Required columns present.
        2. Every event_type value is in VALID_EVENT_TYPES.
        3. No NaN event_year values.
        4. event_year is integer-coercible (we tolerate float columns
           where every value is a whole number — pandas can give you
           that when reading XLSX with mixed types).
    """
    # Check 1: required columns. Listing the missing set in the message
    # makes the fix obvious to the researcher.
    missing = [c for c in RT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"relationship table is missing required columns: {missing}. "
            f"Required: {list(RT_REQUIRED_COLUMNS)}."
        )

    # Check 2: event_type values. Set difference catches typos
    # (``'Splt'``, ``'Merger'``) and any out-of-spec types
    # (``'Coarse Redistribution'`` was a legacy variant we don't accept).
    bad_types = set(df["event_type"].dropna().unique()) - VALID_EVENT_TYPES
    if bad_types:
        raise SchemaError(
            f"relationship table contains unknown event_type values: {sorted(bad_types)}. "
            f"Valid values: {sorted(VALID_EVENT_TYPES)}."
        )

    # Check 3: NaN event_year. A NaN year would silently break the
    # chronological walk in build_snapshot.
    if df["event_year"].isna().any():
        raise SchemaError("relationship table has rows with missing event_year.")

    # Check 4: integer event_year. Excel sometimes returns ints as floats
    # (``2003.0``). We tolerate that case (every value is a whole number)
    # but reject genuinely fractional years.
    if not pd.api.types.is_integer_dtype(df["event_year"]):
        if not (df["event_year"] == df["event_year"].astype(int)).all():
            raise SchemaError("relationship table event_year must be integer.")


def validate_stats(df: pd.DataFrame, base_year: int) -> None:
    """Raise SchemaError if the stats frame is malformed.

    Six checks:
        1. Required columns present.
        2. No NaN year values.
        3. No rows with year < base_year (the ``min_year == base_year``
           invariant).
        4. No NaN unit_id.
        5. No NaN variable.
        6. No NaN season values. Use an explicit sentinel like
           ``"Annual"`` for non-seasonal data — the previous lenient
           behavior silently grouped NaN-season rows into their own
           bucket, which left the user unsure whether the rows were
           annual data, missing-tag rows, or a schema bug.

    Note: ``value`` IS allowed to have NaN rows — null value
    legitimately indicates "not reported" and the aggregator handles
    it via ``min_count=1`` on the sum.
    """
    # Check 1: schema.
    missing = [c for c in STATS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"stats table is missing required columns: {missing}. "
            f"Required (long-form): {list(STATS_REQUIRED_COLUMNS)}."
        )

    # Check 2: year nulls.
    if df["year"].isna().any():
        raise SchemaError("stats table has rows with missing year.")

    # Check 3: pre-base-year rows. We INTENTIONALLY raise rather than
    # silently filter — the user might be passing data they expect to
    # see in the output, and silently dropping it would surprise.
    too_early = df["year"] < base_year
    if too_early.any():
        n = int(too_early.sum())
        raise SchemaError(
            f"stats table contains {n} rows with year < base_year ({base_year}). "
            "Stable boundaries are only valid forward from the base year."
        )

    # Checks 4-5: required-key nulls. unit_id is the join key against
    # the remap; variable identifies what the value represents.
    if df["unit_id"].isna().any():
        raise SchemaError("stats table has rows with missing unit_id.")
    if df["variable"].isna().any():
        raise SchemaError("stats table has rows with missing variable.")

    # Check 6: season nulls. Strict — force the user to be explicit so
    # downstream consumers don't have to interpret NaN seasons.
    if df["season"].isna().any():
        n = int(df["season"].isna().sum())
        raise SchemaError(
            f"stats table has {n} row(s) with missing season. Use an "
            "explicit sentinel ('Annual', 'Y', etc.) for non-seasonal "
            "data — NaN seasons are rejected so downstream consumers "
            "don't have to guess whether the rows are annual, "
            "untagged, or a schema bug."
        )


def validate_name_change_log(df: pd.DataFrame) -> None:
    """Raise SchemaError if the name-change log is malformed.

    Light validator — just checks required columns. We don't validate
    row-level content because the merge step (``merge_name_changes``)
    will fail loudly on any malformed row data.
    """
    missing = [c for c in NCL_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"name change log is missing required columns: {missing}. "
            f"Required: {list(NCL_REQUIRED_COLUMNS)}."
        )


def validate_baseline(df: pd.DataFrame) -> None:
    """Raise SchemaError if the baseline snapshot is malformed.

    Three checks:
        1. Required columns present.
        2. No NaN unit_id (the join key).
        3. No duplicate unit_id values — every unit appears at most once
           per baseline. (For multi-year baselines, the year-filter in
           ``read_baseline`` runs BEFORE this check, so the validator
           sees only the rows for the target year.)
    """
    missing = [c for c in BASELINE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"baseline snapshot is missing required columns: {missing}. "
            f"Required: {list(BASELINE_REQUIRED_COLUMNS)}."
        )
    if df["unit_id"].isna().any():
        raise SchemaError("baseline snapshot has rows with missing unit_id.")

    # Duplicate detection — a baseline with duplicate IDs would silently
    # double-count units in the snapshot. Sample 5 in the message so the
    # researcher knows where to look.
    dup = df["unit_id"][df["unit_id"].duplicated()]
    if len(dup) > 0:
        raise SchemaError(
            f"baseline snapshot has {len(dup)} duplicate unit_id values; "
            f"sample: {sorted(set(dup))[:5]}"
        )
