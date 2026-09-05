"""Build the publishable India district shapefile: trimmed, with a state column.

WHY THIS EXISTS

The raw Geolocet file carries 160 columns -- name_ar, name_zh, census20_7 and
the rest of OpenStreetMap's tag sprawl -- in a 10 MB DBF, and has no usable
state column: addr_state has 2 non-null values across 786 features, is_in_stat
has 4, census2010 covers 62.

That missing state column is stablebound-dev#4. Two districts called Bilaspur,
one in Himachal Pradesh and one in Chhattisgarh, cannot be told apart by name,
so both take the first id -- which does not merely mislabel a polygon, it
dissolves it into a stable group 1,100 km away. It is the whole of the
remaining 0.37% -> 0.111% of misplaced area.

The state cannot come from the lineage: the state is what we need in order to
choose the right lineage id in the first place. Geometry is the only
independent arbiter, which is why the overrides this replaces were hand-keyed
to centroid coordinates.

WHAT IT DOES

Joins each district to the official 2021 state boundaries and keeps four
columns. Two quirks of that state layer are handled here rather than
downstream:

  * Its names carry systematic diacritic corruption -- ">" is a long A and "|"
    a long I, so GUJAR>T is GUJARAT and CHAND|GARH is CHANDIGARH.
  * Four of its 40 features are DISPUTED (...) territories, not states.

The join uses representative_point(), not centroid: a centroid can fall outside
a concave district and land in the wrong state.

Three state names still need aligning to the lineage's vocabulary. That list
was five until the coarse-name lookup was made year-aware; Odisha and
Puducherry now resolve on their own.

Writes a .cpg declaring UTF-8, which fixes at source the encoding bug
prepare_shapefile.py currently works around -- the raw file ships without one,
so readers guess Latin-1 and four district names arrive as mojibake.

    python tools/india/prepare_geolocet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

# paths.py sits alongside this script in tools/india/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import resolve  # noqa: E402

RAW = resolve(
    "data/india/boundaries/modified_shapefiles/India_Geolocet_2023_simplified/"
    "India_admin_level_5_geolocet.shp"
)
STATES = resolve(
    "shapefiles/India/official_shapefiles/India_Official_Boundaries_2021/"
    "STATE_BOUNDARY.shp"
)
OUT_DIR = resolve("data/india/boundaries/modified_shapefiles/India_districts_2024")
OUT = OUT_DIR / "India_districts_2024.shp"

#: Long vowels in the official layer arrive as punctuation.
DIACRITIC_REPAIR = {">": "A", "|": "I"}

#: Shapefile spelling -> the lineage's vocabulary. Only three remain: the
#: "&" forms and Delhi's official name.
STATE_ALIASES = {
    "Andaman & Nicobar": "Andaman And Nicobar Islands",
    "Dadra & Nagar Haveli & Daman & Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Delhi": "NCT Of Delhi",
}

KEEP = ["name", "state", "geometry"]


def main() -> int:
    # encoding="utf-8" is required: the raw file has no .cpg, so the reader
    # falls back to Latin-1 and mangles Mahe, Tseminyu, Zunheboto and
    # Chumoukedima badly enough that two lose their ids entirely.
    districts = gpd.read_file(RAW, encoding="utf-8")
    n_in = len(districts)
    print(f"districts: {n_in} features from {RAW.name}")

    states = gpd.read_file(STATES)[["STATE", "geometry"]]
    cleaned = states["STATE"]
    for bad, good in DIACRITIC_REPAIR.items():
        cleaned = cleaned.str.replace(bad, good, regex=False)
    states["state"] = cleaned.str.title().replace(STATE_ALIASES)
    disputed = states["state"].str.startswith("Disputed")
    print(f"states:    {len(states)} features, dropping {int(disputed.sum())} DISPUTED")
    states = states[~disputed].to_crs(districts.crs)[["state", "geometry"]]

    # A representative point is guaranteed to lie inside its polygon; a
    # centroid is not, and a district shaped around a bay would be assigned by
    # a point in the sea.
    probes = districts[["geometry"]].copy()
    probes["geometry"] = districts.geometry.representative_point()
    joined = gpd.sjoin(probes, states, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    districts["state"] = joined["state"].to_numpy()

    unresolved = int(districts["state"].isna().sum())
    if unresolved:
        sample = districts.loc[districts["state"].isna(), "name"].head(5).tolist()
        raise SystemExit(
            f"{unresolved} district(s) fell in no state polygon: {sample}\n"
            "  A district with no state cannot disambiguate a homonym, which is "
            "the entire point of this file. Refusing to write a partial one."
        )
    if len(districts) != n_in:
        raise SystemExit(f"feature count changed: {n_in} -> {len(districts)} -- refusing")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    districts[KEEP].to_file(OUT, driver="ESRI Shapefile", encoding="utf-8")
    # to_file does not always emit one, and its absence is the original bug.
    (OUT.with_suffix(".cpg")).write_text("UTF-8\n", encoding="ascii")

    back = gpd.read_file(OUT)
    assert len(back) == n_in, f"wrote {len(back)}, expected {n_in}"
    assert back["state"].notna().all(), "null state survived the write"
    mb = sum(f.stat().st_size for f in OUT_DIR.glob("India_districts_2024.*")) / 1e6
    print(f"wrote {OUT}  ({len(back)} features, {len(KEEP)} columns, {mb:.1f} MB)")
    print(f"  states assigned: {back['state'].nunique()} distinct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
