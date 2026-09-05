"""Regenerate the bundled India data files from the canonical sources.

The package ships copies of India's lineage, baseline and name-change log.
Until now those copies were made by hand, which is how two of them drifted
into different schemas and how a bad name-change-log row had to be fixed
twice. This script is the link: canonical is authored, bundled is generated.

    python tools/india/bundle_data.py            # regenerate + report
    python tools/india/bundle_data.py --check    # verify only, exit 1 on drift

``--check`` is the CI-facing mode: it regenerates into a temp dir and compares,
so a canonical edit that was never propagated fails loudly instead of silently
shipping stale data.

Canonical sources (authored by hand, outside this repo):

    $STABLEBOUND_DATA_ROOT/data/india/boundaries/relationship_table/
        lineage_files/ADM2_LINEAGE_NEW_IDS_complete.xlsx
        lineage_files/ADM1_LINEAGE.xlsx
        lineage_files/NAME_CHANGE_LOG.xlsx
        admin_snapshots/admin_snapshot_1991.csv

Bundled outputs (generated, shipped in the wheel):

    src/stablebound/data/IN/
        lineage.xlsx            ADM2 events, canonical lowercase schema
        lineage_adm1.xlsx       ADM1 events — needed for the FEWS export's
                                admin1 relationship rows
        baseline.csv            1991 snapshot, canonical schema
        name_change_log.xlsx    all levels, with LEVEL retained
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PKG = REPO / "src" / "stablebound" / "data" / "IN"

# Resolve through tools/india/paths.py rather than reading the environment
# directly. Both files key off $STABLEBOUND_DATA_ROOT, but they used to mean
# different things by it: paths.py documents it as the Crops root, while this
# script defaulted to Crops/data/india and appended from there. Setting the
# variable exactly as paths.py documents therefore broke this script -- and
# the old regression harness ran it as its FIRST step, so it failed on the one
# configuration its own documentation told you to use.
sys.path.insert(0, str(HERE))
from paths import resolve  # noqa: E402

ROOT = resolve("data/india")
RT = ROOT / "boundaries" / "relationship_table"

LINEAGE_COLS = ["event_year", "event_type", "parent_id", "parent_name",
                "child_id", "child_name", "parent_coarse_id",
                "parent_coarse_name", "child_coarse_id", "child_coarse_name"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _lineage(src: Path, out: Path) -> None:
    """Canonical UPPERCASE lineage -> bundled lowercase canonical schema."""
    df = pd.read_excel(src)
    df.columns = [c.lower() for c in df.columns]
    # The canonical ADM2 file has carried duplicate rows before (three, fixed
    # 2026-06); dedupe on the event identity so a re-introduced duplicate
    # can't reach the shipped copy.
    before = len(df)
    df = df.drop_duplicates(
        subset=["event_year", "event_type", "parent_id", "child_id"], keep="first"
    )
    if len(df) != before:
        print(f"    note: dropped {before - len(df)} duplicate event row(s)")
    df = df[[c for c in LINEAGE_COLS if c in df.columns]]
    df.to_excel(out, index=False)


def _baseline(src: Path, out: Path) -> None:
    """Canonical admin snapshot -> bundled baseline schema, 1991 only."""
    df = pd.read_csv(src, dtype=str)
    df = df.rename(columns={
        "YEAR": "year", "ADM1_ID": "coarse_id", "ADM1_NAME": "coarse_name",
        "ADM2_ID": "unit_id", "ADM2_NAME": "name",
    })
    df = df[df["year"] == "1991"]
    df[["unit_id", "name", "year", "coarse_id", "coarse_name"]].to_csv(out, index=False)


def _ncl(src: Path, out: Path) -> None:
    """Canonical NCL -> bundled schema, KEEPING level.

    ``level`` is retained (it was dropped in the old hand-made copy) so the
    FEWS deliverable can split the log per admin level without inferring it
    from the id prefix. Nothing filters on it yet — the ADM2 product still
    merges all rows, as it always has.
    """
    df = pd.read_excel(src)
    df = df.rename(columns={
        "CHANGE_YEAR": "event_year", "LEVEL": "level", "UNIT_ID": "unit_id",
        "OLD_OFFICIAL_NAME": "old_name", "NEW_OFFICIAL_NAME": "new_name",
    })
    df[["event_year", "level", "unit_id", "old_name", "new_name"]].to_excel(
        out, index=False
    )


JOBS = [
    ("lineage.xlsx", RT / "lineage_files/ADM2_LINEAGE_NEW_IDS_complete.xlsx", _lineage),
    ("lineage_adm1.xlsx", RT / "lineage_files/ADM1_LINEAGE.xlsx", _lineage),
    ("baseline.csv", RT / "admin_snapshots/admin_snapshot_1991.csv", _baseline),
    ("name_change_log.xlsx", RT / "lineage_files/NAME_CHANGE_LOG.xlsx", _ncl),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify bundled files match canonical; exit 1 on drift")
    args = ap.parse_args()

    missing = [str(src) for _, src, _ in JOBS if not src.exists()]
    if missing:
        print("canonical sources not found (set STABLEBOUND_DATA_ROOT):")
        for m in missing:
            print(f"  {m}")
        return 2

    dest_dir = Path(tempfile.mkdtemp()) if args.check else PKG
    dest_dir.mkdir(parents=True, exist_ok=True)
    drift = []
    for name, src, fn in JOBS:
        out = dest_dir / name
        fn(src, out)
        bundled = PKG / name
        if args.check:
            # Excel writes are not byte-stable (timestamps), so compare
            # parsed content rather than bytes.
            same = bundled.exists() and _content_equal(out, bundled)
            status = "ok" if same else "DRIFT"
            if not same:
                drift.append(name)
        else:
            status = f"sha {_sha(out)}"
        print(f"  {name:<24} <- {src.name:<38} {status}")

    if args.check:
        shutil.rmtree(dest_dir)
        if drift:
            print(f"\n{len(drift)} bundled file(s) differ from canonical: {drift}")
            print("Run without --check to regenerate.")
            return 1
        print("\nall bundled files match canonical.")
    else:
        print(f"\nregenerated {len(JOBS)} file(s) into {PKG}")
    return 0


def _content_equal(a: Path, b: Path) -> bool:
    read = pd.read_csv if a.suffix == ".csv" else pd.read_excel
    try:
        da, db = read(a, dtype=str), read(b, dtype=str)
    except Exception:
        return False
    if list(da.columns) != list(db.columns) or len(da) != len(db):
        return False
    return da.fillna("").equals(db.fillna(""))


if __name__ == "__main__":
    sys.exit(main())
