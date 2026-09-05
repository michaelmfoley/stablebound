"""Generate the synthetic fixture registry.

Each fixture is a tiny country in exactly the shape a real user supplies:
``lineage.csv``, ``baseline.csv``, ``modern.geojson``, ``stats.csv``, plus an
``expected.json`` declaring what should come out. ``tests/conftest.py`` loads
them and ``tests/test_qualification.py`` runs one invariant battery across all
of them — so adding a hazard costs one dict entry, not a test module.

Why generated rather than hand-written: 12 fixtures x 5 files is 60 files whose
geometry has to tile exactly for conservation to hold. Generating them keeps the
arithmetic in one place and makes the registry regenerable, which is the same
rule the rest of the project's derived data follows.

Regenerate:  python tests/fixtures/synthetic/_build.py

Geometry convention: every unit is a box of height 1 somewhere on the x axis.
A split's children tile the parent exactly (parent [x, x+1] becomes [x, x+0.5]
and [x+0.5, x+1]), so dissolving any group reproduces the base-year footprint
and area conservation is checkable rather than approximate.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

LINEAGE_COLS = ["event_year", "event_type", "parent_id", "parent_name",
                "child_id", "child_name"]
COARSE_COLS = ["parent_coarse_id", "parent_coarse_name",
               "child_coarse_id", "child_coarse_name"]


def box(x0: float, x1: float) -> list:
    """A closed rectangle from x0 to x1, height 1. Ring order matches GeoJSON."""
    return [[[x1, 0.0], [x1, 1.0], [x0, 1.0], [x0, 0.0], [x1, 0.0]]]


# Each fixture: baseline units (id, name, x0, x1[, coarse]), events, modern
# units, stats rows, and the expectations the battery asserts against.
#
# `stats` rows are (unit_id, year, variable, value). Season is filled in as
# "Annual" — a NaN season is a hard error by design, and the fuzz suite covers
# that case rather than the registry.
FIXTURES: dict[str, dict] = {}


def fixture(name, *, doc, baseline, events, modern, stats, expected, coarse=False):
    FIXTURES[name] = dict(doc=doc, baseline=baseline, events=events,
                          modern=modern, stats=stats, expected=expected,
                          coarse=coarse)


# --- 1. no events at all -------------------------------------------------
fixture(
    "static",
    doc="Zero events. The degenerate case: every stable group is one unit, and "
        "any code path that assumes at least one event should still work.",
    baseline=[("S.001", "Solo", 0, 1), ("S.002", "Duo", 1, 2)],
    events=[],
    modern=[("S.001", "Solo", 0, 1), ("S.002", "Duo", 1, 2)],
    stats=[("S.001", y, "area_ha", 10.0) for y in (2010, 2011)]
          + [("S.002", y, "area_ha", 20.0) for y in (2010, 2011)],
    expected={"snapshots": {"2010": 2, "2011": 2}, "n_stable_groups": 2,
              "conserves": True},
)

# --- 2. one split --------------------------------------------------------
fixture(
    "simple_split",
    doc="One parent becomes two children in 2014. The base case for grouping: "
        "the two children must land in one stable group with the parent.",
    baseline=[("P.001", "Parent", 0, 1), ("P.002", "Other", 1, 2)],
    events=[(2014, "Split", "P.001", "Parent", "P.011", "North"),
            (2014, "Split", "P.001", "Parent", "P.012", "South")],
    modern=[("P.011", "North", 0, 0.5), ("P.012", "South", 0.5, 1),
            ("P.002", "Other", 1, 2)],
    stats=[("P.001", 2010, "area_ha", 100.0), ("P.001", 2014, "area_ha", 100.0),
           ("P.011", 2015, "area_ha", 60.0), ("P.012", 2015, "area_ha", 40.0),
           ("P.002", 2010, "area_ha", 50.0), ("P.002", 2015, "area_ha", 50.0)],
    expected={"snapshots": {"2010": 2, "2014": 2, "2015": 3},
              "n_stable_groups": 2, "conserves": True},
)

# --- 3. three-deep cascade ----------------------------------------------
fixture(
    "cascade",
    doc="A splits into B and C (2012); C splits again into D and E (2016). "
        "Everything must collapse into ONE stable group — a grouping that only "
        "walks one generation gets two.",
    baseline=[("C.001", "Aaa", 0, 1)],
    events=[(2012, "Split", "C.001", "Aaa", "C.011", "Bbb"),
            (2012, "Split", "C.001", "Aaa", "C.012", "Ccc"),
            (2016, "Split", "C.012", "Ccc", "C.021", "Ddd"),
            (2016, "Split", "C.012", "Ccc", "C.022", "Eee")],
    modern=[("C.011", "Bbb", 0, 0.5), ("C.021", "Ddd", 0.5, 0.75),
            ("C.022", "Eee", 0.75, 1)],
    stats=[("C.001", 2010, "area_ha", 80.0), ("C.011", 2013, "area_ha", 40.0),
           ("C.012", 2013, "area_ha", 40.0), ("C.021", 2017, "area_ha", 20.0),
           ("C.022", 2017, "area_ha", 20.0)],
    expected={"snapshots": {"2010": 1, "2012": 1, "2013": 2, "2016": 2, "2017": 3},
              "n_stable_groups": 1, "conserves": True},
)

# --- 4. merge where not every parent reports -----------------------------
fixture(
    "multi_parent_merge",
    doc="Three parents merge into one child in 2015, but only two ever report. "
        "Completeness must show the third as missing rather than silently "
        "dividing by the number that happened to report.",
    baseline=[("M.001", "One", 0, 1), ("M.002", "Two", 1, 2), ("M.003", "Three", 2, 3)],
    events=[(2015, "Merge", "M.001", "One", "M.011", "Big"),
            (2015, "Merge", "M.002", "Two", "M.011", "Big"),
            (2015, "Merge", "M.003", "Three", "M.011", "Big")],
    modern=[("M.011", "Big", 0, 3)],
    stats=[("M.001", 2010, "area_ha", 10.0), ("M.002", 2010, "area_ha", 20.0),
           ("M.001", 2014, "area_ha", 10.0), ("M.002", 2014, "area_ha", 20.0),
           ("M.011", 2016, "area_ha", 60.0)],
    expected={"snapshots": {"2010": 3, "2015": 3, "2016": 1},
              "n_stable_groups": 1, "conserves": True,
              "incomplete_year": 2010, "missing_contains": "M.003"},
)

# --- 5. redistribute: the cross-group union ------------------------------
fixture(
    "redistribute",
    doc="R.001 and R.002 exchange territory in 2013. Two properties at once. "
        "(a) Grouping: the units were independent and the event forces their "
        "groups to union — get it wrong and you get two groups whose dissolved "
        "geometry overlaps. (b) Id convention: a Redistribute retires every "
        "unit it touches and issues new ids even where the NAME continues, "
        "because the id denotes a territorial extent and that extent changed. "
        "Real data does exactly this — India's Chennai IN.ADM2.00371 becomes "
        "00966 and Vietnam's Dong Nai VN.ADM1.00008 becomes 00066, both keeping "
        "their names. A fixture that let the giver keep its id would encode a "
        "convention no real lineage uses.",
    baseline=[("R.001", "Giver", 0, 1), ("R.002", "Taker", 1, 2), ("R.003", "Idle", 2, 3)],
    events=[(2013, "Redistribute", "R.001", "Giver", "R.011", "Giver"),
            (2013, "Redistribute", "R.001", "Giver", "R.012", "Taker"),
            (2013, "Redistribute", "R.002", "Taker", "R.012", "Taker"),
            (2013, "Redistribute", "R.002", "Taker", "R.011", "Giver")],
    modern=[("R.011", "Giver", 0, 0.7), ("R.012", "Taker", 0.7, 2), ("R.003", "Idle", 2, 3)],
    stats=[("R.001", 2010, "area_ha", 100.0), ("R.002", 2010, "area_ha", 100.0),
           ("R.011", 2014, "area_ha", 70.0), ("R.012", 2014, "area_ha", 130.0),
           ("R.003", 2010, "area_ha", 5.0)],
    expected={"snapshots": {"2010": 3, "2013": 3, "2014": 3}, "n_stable_groups": 2,
              "conserves": True},
)

# --- 6. rename carried in the lineage ------------------------------------
fixture(
    "rename_in_lineage",
    doc="A NameChange encoded as a self-referencing row. The unit keeps its id, "
        "so this must NOT create a new stable group, and name lookups must "
        "return the year-accurate name.",
    baseline=[("N.001", "Oldname", 0, 1), ("N.002", "Steady", 1, 2)],
    events=[(2016, "NameChange", "N.001", "Oldname", "N.001", "Newname")],
    modern=[("N.001", "Newname", 0, 1), ("N.002", "Steady", 1, 2)],
    stats=[("N.001", 2010, "area_ha", 10.0), ("N.001", 2018, "area_ha", 12.0),
           ("N.002", 2010, "area_ha", 20.0)],
    expected={"snapshots": {"2010": 2, "2018": 2}, "n_stable_groups": 2,
              "conserves": True, "name_at": {"N.001": {"2010": "Oldname",
                                                       "2018": "Newname"}}},
)

# --- 7. reporting outside a unit's lifespan ------------------------------
fixture(
    "late_early",
    doc="The parent keeps reporting for two years after it ceased, and a child "
        "reports a year before it existed. Both must be flagged rather than "
        "silently attributed, and neither may be dropped.",
    baseline=[("L.001", "Ancestor", 0, 1)],
    events=[(2014, "Split", "L.001", "Ancestor", "L.011", "Left"),
            (2014, "Split", "L.001", "Ancestor", "L.012", "Right")],
    modern=[("L.011", "Left", 0, 0.5), ("L.012", "Right", 0.5, 1)],
    stats=[("L.001", 2010, "area_ha", 100.0),
           ("L.001", 2016, "area_ha", 90.0),   # late: ceased in 2014
           ("L.001", 2017, "area_ha", 80.0),   # late
           ("L.011", 2013, "area_ha", 10.0),   # early: not created until 2015
           ("L.011", 2016, "area_ha", 50.0), ("L.012", 2016, "area_ha", 40.0)],
    expected={"snapshots": {"2010": 1, "2014": 1, "2015": 2, "2016": 2}, "n_stable_groups": 1,
              "conserves": True, "expect_late_reporting": True},
)

# --- 8. homonyms distinguished only by their parent ----------------------
fixture(
    "homonym_coarse",
    doc="Two units genuinely share a name and are separable only by coarse "
        "attribution. Name-only matching cannot resolve them, so the matcher "
        "must either use the coarse column or refuse — never guess.",
    baseline=[("H.001", "Springfield", 0, 1, "H.A1.01", "North"),
              ("H.002", "Springfield", 1, 2, "H.A1.02", "South"),
              ("H.003", "Shelbyville", 2, 3, "H.A1.02", "South")],
    events=[],
    modern=[("H.001", "Springfield", 0, 1), ("H.002", "Springfield", 1, 2),
            ("H.003", "Shelbyville", 2, 3)],
    stats=[("H.001", 2010, "area_ha", 10.0), ("H.002", 2010, "area_ha", 20.0),
           ("H.003", 2010, "area_ha", 30.0)],
    expected={"snapshots": {"2010": 3}, "n_stable_groups": 3, "conserves": True,
              "duplicate_names": ["Springfield"]},
    coarse=True,
)

# --- 9. non-ASCII names --------------------------------------------------
fixture(
    "diacritics",
    doc="Names differing only by diacritics and spacing. Measures whether "
        "normalisation collapses them correctly without merging genuinely "
        "distinct units.",
    baseline=[("D.001", "Đắk Lắk", 0, 1), ("D.002", "Kon Tum", 1, 2),
              ("D.003", "Thừa Thiên Huế", 2, 3)],
    events=[],
    modern=[("D.001", "Dak Lak", 0, 1), ("D.002", "KonTum", 1, 2),
            ("D.003", "Thua Thien Hue", 2, 3)],
    stats=[("D.001", 2010, "area_ha", 10.0), ("D.002", 2010, "area_ha", 20.0),
           ("D.003", 2010, "area_ha", 30.0)],
    expected={"snapshots": {"2010": 3}, "n_stable_groups": 3, "conserves": True,
              "match_by_name": {"Dak Lak": "D.001", "KonTum": "D.002",
                                "Thua Thien Hue": "D.003"}},
)

# --- 10. a unit with no geometry ----------------------------------------
fixture(
    "missing_geometry",
    doc="A unit present in the lineage and the statistics but absent from the "
        "shapefile. It must be reported, not silently dropped — a dropped unit "
        "removes its production from every downstream total.",
    baseline=[("G.001", "Mapped", 0, 1), ("G.002", "Unmapped", 1, 2)],
    events=[],
    modern=[("G.001", "Mapped", 0, 1)],
    stats=[("G.001", 2010, "area_ha", 10.0), ("G.002", 2010, "area_ha", 20.0)],
    expected={"snapshots": {"2010": 2}, "unmapped_units": ["G.002"]},
)

# --- 11. two-level country ----------------------------------------------
fixture(
    "admin2_coarse",
    doc="Districts carrying coarse (admin1) attribution, including a district "
        "that changes parent. Exercises the cross-level path the FEWS "
        "deliverable depends on.",
    baseline=[("A.001", "Dist1", 0, 1, "A.A1.01", "StateA"),
              ("A.002", "Dist2", 1, 2, "A.A1.01", "StateA"),
              ("A.003", "Dist3", 2, 3, "A.A1.02", "StateB")],
    events=[(2015, "Split", "A.001", "Dist1", "A.011", "Dist1", "A.A1.01",
             "StateA", "A.A1.01", "StateA"),
            (2015, "Split", "A.001", "Dist1", "A.012", "Dist1b", "A.A1.01",
             "StateA", "A.A1.02", "StateB")],
    modern=[("A.011", "Dist1", 0, 0.5), ("A.012", "Dist1b", 0.5, 1),
            ("A.002", "Dist2", 1, 2), ("A.003", "Dist3", 2, 3)],
    stats=[("A.001", 2010, "area_ha", 10.0), ("A.002", 2010, "area_ha", 20.0),
           ("A.003", 2010, "area_ha", 30.0), ("A.011", 2016, "area_ha", 6.0),
           ("A.012", 2016, "area_ha", 4.0)],
    expected={"snapshots": {"2010": 3, "2015": 3, "2016": 4}, "n_stable_groups": 3,
              "conserves": True},
    coarse=True,
)


# --- 12. the shape a FEWS relationship table describes -------------------
_XX_2010 = (("XX.ADM1.00001", 100.0), ("XX.ADM1.00002", 80.0),
            ("XX.ADM1.00003", 90.0), ("XX.ADM1.00004", 20.0))
_XX_2016 = (("XX.ADM1.00005", 60.0), ("XX.ADM1.00006", 40.0),
            ("XX.ADM1.00002", 80.0), ("XX.ADM1.00003", 90.0), ("XX.ADM1.00004", 20.0))
_XX_2026 = (("XX.ADM1.00005", 60.0), ("XX.ADM1.00006", 40.0),
            ("XX.ADM1.00008", 100.0), ("XX.ADM1.00003", 90.0))
fixture(
    "legacy_admin1",
    doc="One admin1 country in the shape a FEWS relationship table describes: a "
        "base vintage, a split, a rename carried as a same-id successor, and a "
        "merge that makes the unit count fall. Its legacy-dialect twin is "
        "tests/fixtures/rt_convert/relationshiptable_XX.csv; each is the "
        "other's oracle, and neither is generated from the other. The ids are "
        "the ones rt_convert allocates from the twin (00007 is skipped because "
        "the rename collapses onto 00003), so the golden test can compare the "
        "converter's output against these files directly.",
    baseline=[("XX.ADM1.00001", "Alpha", 0, 1), ("XX.ADM1.00002", "Bravo", 1, 2),
              ("XX.ADM1.00004", "Delta", 2, 3), ("XX.ADM1.00003", "Charlie", 3, 4)],
    events=[(2015, "Split", "XX.ADM1.00001", "Alpha", "XX.ADM1.00005", "Alpha North"),
            (2015, "Split", "XX.ADM1.00001", "Alpha", "XX.ADM1.00006", "Alpha South"),
            (2020, "NameChange", "XX.ADM1.00003", "Charlie", "XX.ADM1.00003", "Charlton"),
            (2025, "Merge", "XX.ADM1.00002", "Bravo", "XX.ADM1.00008", "BravoDelta"),
            (2025, "Merge", "XX.ADM1.00004", "Delta", "XX.ADM1.00008", "BravoDelta")],
    modern=[("XX.ADM1.00005", "Alpha North", 0, 0.5), ("XX.ADM1.00006", "Alpha South", 0.5, 1),
            ("XX.ADM1.00008", "BravoDelta", 1, 3), ("XX.ADM1.00003", "Charlton", 3, 4)],
    stats=[(u, y, "area_ha", v) for y in (2010, 2012) for u, v in _XX_2010]
          + [(u, y, "area_ha", v) for y in (2016, 2018, 2021) for u, v in _XX_2016]
          + [(u, 2026, "area_ha", v) for u, v in _XX_2026],
    expected={"snapshots": {"2010": 4, "2015": 4, "2016": 5, "2020": 5, "2021": 5,
                            "2025": 5, "2026": 4},
              "n_stable_groups": 3, "conserves": True,
              "name_at": {"XX.ADM1.00003": {"2010": "Charlie", "2021": "Charlton"}}},
)


def write(name: str, spec: dict) -> None:
    d = HERE / name
    d.mkdir(parents=True, exist_ok=True)
    has_coarse = spec["coarse"]

    # baseline.csv
    cols = ["unit_id", "name", "year"] + (["coarse_id", "coarse_name"] if has_coarse else [])
    lines = [",".join(cols)]
    for row in spec["baseline"]:
        uid, nm, _x0, _x1 = row[:4]
        vals = [uid, nm, "2010"] + (list(row[4:6]) if has_coarse else [])
        lines.append(",".join(vals))
    (d / "baseline.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # lineage.csv
    cols = LINEAGE_COLS + (COARSE_COLS if has_coarse else [])
    lines = [",".join(cols)]
    for ev in spec["events"]:
        lines.append(",".join(str(v) for v in ev))
    (d / "lineage.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # modern.geojson
    feats = [{"type": "Feature",
              "properties": {"unit_id": uid, "unit_name": nm},
              "geometry": {"type": "Polygon", "coordinates": box(x0, x1)}}
             for uid, nm, x0, x1 in spec["modern"]]
    (d / "modern.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False,
                   indent=1) + "\n", encoding="utf-8")

    # stats.csv
    lines = ["unit_id,year,season,variable,value"]
    for uid, yr, var, val in spec["stats"]:
        lines.append(f"{uid},{yr},Annual,{var},{val}")
    (d / "stats.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # expected.json — the fixture's own declaration of what should happen.
    exp = dict(spec["expected"])
    exp["_doc"] = spec["doc"]
    exp["_has_coarse"] = has_coarse
    (d / "expected.json").write_text(
        json.dumps(exp, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")


def main() -> None:
    for name, spec in FIXTURES.items():
        write(name, spec)
    print(f"wrote {len(FIXTURES)} fixtures into {HERE}")
    for n in FIXTURES:
        print(f"  {n}")


if __name__ == "__main__":
    main()
