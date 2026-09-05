"""Reshape India inputs into the package's canonical schemas.

Produces three files in `_prepared/`:
    - baseline.csv      (canonical baseline schema)
    - stats_long.csv    (canonical long-form stats schema)
    - (the IDed modern shapefile is produced by `prepare_shapefile.py`)

Run after `prepare_shapefile.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stablebound.data import BUNDLED_COUNTRIES  # noqa: E402
from stablebound.schemas import validate_baseline, validate_stats  # noqa: E402

import os

# Lineage inputs come from the PACKAGE BUNDLE (generated from canonical by
# `bundle_data.py`), so this reads exactly what the shipped wheel reads.
_IN = BUNDLED_COUNTRIES["IN"]
LINEAGE_PATH = _IN.lineage_path
BASELINE_PATH = _IN.baseline_path

# Still external: 376K rows, too large to bundle.
CROPS_DATA = Path(os.environ.get("STABLEBOUND_DATA_ROOT", "STABLEBOUND_DATA_ROOT-unset")) / "data/india"
STATS_PATH = CROPS_DATA / "statistics/upag_des_full/ag_stats_combined_fnid.csv"
OUT_DIR = Path(__file__).resolve().parent / "_prepared"

BASE_YEAR = int(os.environ.get("STABLEBOUND_BASE_YEAR", "1997"))
# 2025 is the inferred shapefile vintage
# (lineage events run through 2024; modern shapefile is 2025 snapshot).
MAX_YEAR = int(os.environ.get("STABLEBOUND_MAX_YEAR", "2025"))


def prepare_lineage() -> Path:
    """Read the India lineage XLSX and write a cleaned canonical CSV.

    Drops the 3 duplicate-row data errors flagged by ``validate_lineage``
    so downstream package code can load the lineage without
    ``LineageDataError``. The source XLSX is left untouched. To remove the
    duplicates from the source, edit ``ADM2_LINEAGE_NEW_IDS_complete.xlsx``
    directly — see the validation report for exact row numbers.
    """
    df = pd.read_excel(LINEAGE_PATH)
    # Normalise case BEFORE deduping. This script used to read the canonical
    # workbook, whose headers are UPPERCASE; it now reads the bundled copy,
    # whose headers are lowercase. Deduping on the old names raised
    # `KeyError: Index(['EVENT_YEAR', 'CHILD_ID', ...])` — and went unnoticed
    # because nothing re-ran this step after the repoint. Lowercasing first
    # makes it insensitive to which of the two it is handed.
    df.columns = [str(c).lower() for c in df.columns]
    before = len(df)
    df = df.drop_duplicates(
        subset=["event_year", "event_type", "parent_id", "child_id"],
        keep="first",
    )
    after = len(df)
    out = OUT_DIR / "lineage.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out} ({after} rows; dropped {before - after} duplicates)")
    return out


def prepare_baseline() -> Path:
    df = pd.read_csv(BASELINE_PATH).rename(
        columns={
            "YEAR": "year",
            "ADM1_ID": "coarse_id",
            "ADM1_NAME": "coarse_name",
            "ADM2_ID": "unit_id",
            "ADM2_NAME": "name",
        }
    )
    validate_baseline(df)
    out = OUT_DIR / "baseline.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out} ({len(df)} rows)")
    return out


def prepare_stats() -> Path:
    print(f"Loading stats: {STATS_PATH.name}")
    raw = pd.read_csv(
        STATS_PATH,
        usecols=[
            "ADM2_ID", "Year", "Season", "Source crop",
            "Area Planted: ha", "Quantity Produced: MT",
        ],
        dtype={"ADM2_ID": str},
    )
    raw = raw.rename(columns={"Source crop": "Crop"})
    print(f"  loaded {len(raw)} raw rows")

    # Drop rows with no ID, no Crop, or sentinel/control IDs like
    # "__FILTER__" (Delhi_Total aggregator rows in the source CSV).
    raw = raw[raw["ADM2_ID"].notna() & raw["Crop"].notna()]
    sentinel_mask = raw["ADM2_ID"].astype(str).str.match(r"^__.*__$")
    if sentinel_mask.any():
        print(f"  dropping {int(sentinel_mask.sum())} sentinel-ID rows "
              f"({sorted(raw[sentinel_mask]['ADM2_ID'].unique())})")
        raw = raw[~sentinel_mask]
    raw = raw[(raw["Year"] >= BASE_YEAR) & (raw["Year"] <= MAX_YEAR)]
    print(f"  after filter to ID+Crop present, {BASE_YEAR}-{MAX_YEAR}: {len(raw)} rows")

    # Pivot to long-form: one row per (unit_id, year, season, variable, value).
    # Each (year, season, crop) source row becomes 2 long-form rows: area + production.
    crop_norm = raw["Crop"].astype(str).str.lower().str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True)
    area_var = (crop_norm + "_area_ha")
    prod_var = (crop_norm + "_production_mt")

    area = pd.DataFrame({
        "unit_id":  raw["ADM2_ID"].astype(str),
        "year":     raw["Year"].astype(int),
        "season":   raw["Season"].astype(str),
        "variable": area_var,
        "value":    pd.to_numeric(raw["Area Planted: ha"], errors="coerce"),
    })
    prod = pd.DataFrame({
        "unit_id":  raw["ADM2_ID"].astype(str),
        "year":     raw["Year"].astype(int),
        "season":   raw["Season"].astype(str),
        "variable": prod_var,
        "value":    pd.to_numeric(raw["Quantity Produced: MT"], errors="coerce"),
    })
    long = pd.concat([area, prod], ignore_index=True)
    long = long.dropna(subset=["value"])

    # Whole Year vs sub-season de-duplication.
    # DESAGRI sometimes files an annual rollup ("Whole Year") AND
    # sub-season rows (Kharif/Rabi/Summer/Autumn/Winter) for the same
    # (unit_id, year, variable). The June 2026 correctness audit
    # showed 650 of 768 dual-tagged
    # stable cells (85%) had Whole Year != sum(sub-seasons) — meaning
    # the two are NOT consistent rollups but distinct (often
    # conflicting) observations. Modern's Total Year derivation also
    # double-counts Whole Year + sub-seasons in 543 cells.
    #
    # Convention: when a unit reports BOTH Whole Year AND sub-seasons
    # for the same (unit_id, year, variable), trust the sub-season
    # rows (more granular, more auditable) and drop the Whole Year row.
    # Whole Year as the SOLE season tag (perennials, year-round crops)
    # is preserved.
    SUB_SEASONS = {"Kharif", "Rabi", "Summer", "Autumn", "Winter"}
    n_before = len(long)
    has_subseason = (
        long[long["season"].isin(SUB_SEASONS)]
        .drop_duplicates(["unit_id", "year", "variable"])
        [["unit_id", "year", "variable"]]
        .assign(_has_sub=True)
    )
    long = long.merge(
        has_subseason, on=["unit_id", "year", "variable"], how="left"
    )
    drop_mask = (long["season"] == "Whole Year") & long["_has_sub"].fillna(False)
    n_dropped = int(drop_mask.sum())
    long = long.loc[~drop_mask].drop(columns=["_has_sub"]).reset_index(drop=True)
    print(
        f"  dropped {n_dropped:,} 'Whole Year' rows where sub-season data "
        f"also present for same (unit, year, variable); "
        f"{n_before - n_dropped:,} rows remain"
    )

    # Same-key dedup (added 2026-06-05 after U2 matcher fix).
    # U2's ancestor walk for ambiguous cases (e.g. multiple modern TN
    # districts → 00371 Chennai for 1997) can put two source rows on
    # the same (unit_id, year, season, variable) key. Variables are
    # all extensive (area_ha, production_mt), so summing is the right
    # semantic — they're both inputs to the same stable cell.
    # Stable aggregation absorbs duplicates correctly (sum across group);
    # this dedup just eliminates the schema warning and prevents any
    # downstream consumer from double-processing.
    key_cols = ["unit_id", "year", "season", "variable"]
    n_before_dedup = len(long)
    long = (long.groupby(key_cols, as_index=False, sort=False)
                  .agg({"value": "sum"}))
    n_after_dedup = len(long)
    n_collapsed = n_before_dedup - n_after_dedup
    if n_collapsed:
        print(
            f"  collapsed {n_collapsed:,} duplicate-key rows by summing values "
            f"(same (unit_id, year, season, variable) from U2 ancestor remaps); "
            f"{n_after_dedup:,} unique-key rows remain"
        )

    validate_stats(long, base_year=BASE_YEAR)
    out = OUT_DIR / "stats_long.csv"
    long.to_csv(out, index=False)
    print(f"  Wrote {out} ({len(long)} long-form rows)")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Preparing lineage (deduplicated)…")
    prepare_lineage()
    print("Preparing baseline…")
    prepare_baseline()
    print("Preparing stats…")
    prepare_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
