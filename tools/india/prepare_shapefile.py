"""One-shot script: produce an India modern shapefile with canonical IDs.

Reads:
    - The Geolocet 2023/2024 shapefile (no IDs, just names)
    - The bundled India lineage (via ``Lineage("IN")``)
    - Country-specific name overrides (MANUAL_OVERRIDES) from `india_overrides.py`

Writes:
    - `_prepared/India_modern_with_ids.geojson` — same geometry as input,
      plus a `unit_id` column with the canonical `IN.ADM2.XXXXX` IDs.
    - `_prepared/match_log.csv` — per-feature: proposal CSV from the
      package's matcher (review-ready).
    - `_prepared/match_review.txt` — human-readable summary.

Algorithm comes from ``stablebound.match.propose_shapefile_mapping``;
this script is just the I/O wrapper around the bundled India data plus one
India-specific override dict (MANUAL_OVERRIDES in ``india_overrides.py``).
Homonyms are resolved from the shapefile's own ``state`` column rather than by
a hand-maintained list. The India normalizer adds suffix stripping
("district", "dist", "Dr.", "Sri", "Shri") on top of the package's
default.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stablebound import Lineage  # noqa: E402
from stablebound.match import (  # noqa: E402
    normalize_name,
    propose_shapefile_mapping,
)

from india_overrides import MANUAL_OVERRIDES  # noqa: E402

# Resolve external data through tools/india/paths.py so a fresh checkout only
# has to set STABLEBOUND_DATA_ROOT (see tools/india/sources.toml).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import resolve  # noqa: E402

# --- Inputs --------------------------------------------------------------

CROPS_DATA = resolve("data/india")
# The trimmed, state-carrying file built by prepare_geolocet.py -- not the raw
# 160-column Geolocet. The state column is what lets the matcher separate the
# two Bilaspurs; without it they collapse onto one id and one polygon dissolves
# into a stable group 1,100 km away (stablebound-dev#4).
SHAPEFILE_PATH = (
    CROPS_DATA
    / "boundaries/modified_shapefiles/India_districts_2024/India_districts_2024.shp"
)
#: The column carrying the ADM1 name, used to disambiguate homonyms.
COARSE_COLUMN = "state"

MODERN_YEAR = 2024  # filename says 2023 but the data is actually 2024
FUZZY_THRESHOLD = 0.80
SKETCHY_FUZZY_THRESHOLD = 0.90

OUT_DIR = Path(__file__).resolve().parent / "_prepared"


def india_normalizer(name: str) -> str:
    """India-specific normalizer.

    Strips administrative suffixes ("district", "dist.", "dist") and
    common honorifics (Dr., Sri, Shri) before applying the package's
    default normalization. Captures the country-specific spelling
    quirks that the generic matcher wouldn't otherwise resolve.
    """
    s = str(name).lower().strip()
    s = s.replace(" district", "").replace(" dist.", "").replace(" dist", "")
    for pat in (r"\bdr\.\s*", r"\bsri\s+", r"\bshri\s+"):
        s = re.sub(pat, "", s)
    return normalize_name(s)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading shapefile: {SHAPEFILE_PATH.name}")
    # prepare_geolocet.py writes a .cpg declaring UTF-8, so the encoding is no
    # longer a guess. The explicit argument stays as a belt-and-braces: without
    # it the raw file's readers fell back to Latin-1 and mangled Mahe,
    # Tseminyu, Zunheboto and Chumoukedima badly enough that two lost their
    # ids entirely.
    gdf = gpd.read_file(SHAPEFILE_PATH, encoding="utf-8")
    print(f"  {len(gdf)} features")

    print("Loading bundled India lineage…")
    ln = Lineage("IN")
    print(f"  {len(ln.relationship_table)} RT rows, baseline {len(ln.baseline)} units")

    print(f"Matching features against the year={MODERN_YEAR} snapshot…")
    proposal = propose_shapefile_mapping(
        gdf,
        ln.lineage,
        name_column="name",
        year=MODERN_YEAR,
        baseline=ln.baseline,
        name_change_log=ln.name_change_log,
        manual_overrides=MANUAL_OVERRIDES,
        coarse_column=COARSE_COLUMN,
        normalizer=india_normalizer,
        fuzzy_threshold=FUZZY_THRESHOLD,
        sketchy_threshold=SKETCHY_FUZZY_THRESHOLD,
    )
    method_counts = proposal.proposals["method"].value_counts().to_dict()
    print(f"  matched: {dict(method_counts)}")

    # Attach unit_ids by index (preserves homonym disambiguation that
    # name-based joins would lose). The proposal is sorted by
    # source_idx, which matches gdf's row order.
    gdf_out = gdf[["name", "geometry"]].copy()
    gdf_out["unit_id"] = proposal.proposals["proposed_unit_id"].replace("", None).tolist()
    out_geo = OUT_DIR / "India_modern_with_ids.geojson"
    gdf_out.to_file(out_geo, driver="GeoJSON")
    print(f"Wrote {out_geo}")

    # Match log + human-readable review.
    proposal.to_csv(OUT_DIR / "match_log.csv")
    proposal.write_review(OUT_DIR / "match_review.txt")
    print(f"Wrote {OUT_DIR / 'match_log.csv'}")
    print(f"Wrote {OUT_DIR / 'match_review.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
