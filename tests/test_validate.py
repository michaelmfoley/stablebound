"""Lineage validator tests."""

from __future__ import annotations

import pandas as pd

from stablebound.lineage import LineageGraph
from stablebound.validate import format_issues, validate_lineage


def _rt(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name"]
    return pd.DataFrame(rows, columns=cols)


def test_clean_lineage_passes_with_only_info_findings():
    df = _rt([
        (2003, "Split", "A", "A", "A1", "A"),
        (2003, "Split", "A", "A", "A2", "B"),
    ])
    issues = validate_lineage(LineageGraph.from_dataframe(df))
    sev = {i.severity for i in issues}
    assert "error" not in sev
    assert "warning" not in sev


def test_duplicate_row_is_an_error():
    df = _rt([
        (2003, "Split", "A", "A", "A1", "A"),
        (2003, "Split", "A", "A", "A2", "B"),
        (2003, "Split", "A", "A", "A2", "B"),  # duplicate
    ])
    # validate=False because the auto-validator would raise on the duplicate.
    issues = validate_lineage(LineageGraph.from_dataframe(df, validate=False))
    errors = [i for i in issues if i.severity == "error"]
    assert any(i.category == "duplicate_row" for i in errors)
    dup = next(i for i in errors if i.category == "duplicate_row")
    assert "A → A2" in dup.message
    assert "appears 2 times" in dup.message
    assert "Fix" in dup.detail


def test_self_referential_split_is_an_error():
    df = _rt([
        (2010, "Split", "X", "X", "X", "X"),
    ])
    issues = validate_lineage(LineageGraph.from_dataframe(df, validate=False))
    errors = [i for i in issues if i.severity == "error"]
    assert any(i.category == "self_referential_territorial" for i in errors)


def test_auto_validate_raises_on_duplicate_row():
    import pytest
    from stablebound.validate import LineageDataError
    df = _rt([
        (2003, "Split", "A", "A", "A1", "A"),
        (2003, "Split", "A", "A", "A1", "A"),  # duplicate
    ])
    with pytest.raises(LineageDataError) as exc:
        LineageGraph.from_dataframe(df)  # default validate=True
    # The exception message must be researcher-actionable.
    assert "duplicate_row" in str(exc.value)
    assert "Fix" in str(exc.value)


def test_auto_validate_does_not_raise_on_warnings_or_info():
    # Multi-parent split is INFO; Split-looks-like-rename is WARNING.
    # Neither should block construction.
    df = _rt([
        (2010, "Split", "RR_OLD", "Rangareddi", "RR_NEW", "Rangareddi"),
    ])
    LineageGraph.from_dataframe(df)  # should not raise


def test_namechange_distinct_ids_is_a_warning():
    df = _rt([
        (2003, "Split", "A", "A", "A1", "A"),
        (2003, "Split", "A", "A", "A2", "B"),
        (2010, "NameChange", "A1", "A", "A1_NEW", "A_renamed"),
    ])
    issues = validate_lineage(LineageGraph.from_dataframe(df))
    warnings = [i for i in issues if i.severity == "warning"]
    assert any(i.category == "namechange_distinct_ids" for i in warnings)


def test_multi_parent_split_is_info():
    # Bokaro: parents Bhojpur and Munger → child 00468 (1991, two Split rows)
    df = _rt([
        (1991, "Split", "Bhojpur", "Bhojpur", "Bokaro", "Bokaro"),
        (1991, "Split", "Munger", "Munger", "Bokaro", "Bokaro"),
    ])
    issues = validate_lineage(LineageGraph.from_dataframe(df))
    infos = [i for i in issues if i.category == "multi_parent_split"]
    assert len(infos) == 1
    assert "2 parents" in infos[0].message
    assert "Redistribute" in infos[0].detail


def test_split_that_looks_like_rename_warns():
    # Single-child Split where parent and child names match — likely a rename.
    df = _rt([
        (2016, "Split", "RR_OLD", "Rangareddi", "RR_NEW", "Rangareddi"),
    ])
    issues = validate_lineage(LineageGraph.from_dataframe(df))
    warnings = [i for i in issues if i.severity == "warning"]
    assert any(i.category == "split_looks_like_rename" for i in warnings)


def test_format_issues_groups_by_severity():
    df = _rt([
        (2003, "Split", "A", "A", "A1", "A"),
        (2003, "Split", "A", "A", "A1", "A"),  # duplicate
    ])
    issues = validate_lineage(LineageGraph.from_dataframe(df, validate=False))
    out = format_issues(issues)
    assert "Errors (must fix)" in out


# --- Stats × lineage consistency validator -----------------------------


def _stats(rows):
    return pd.DataFrame(
        rows, columns=["unit_id", "year", "season", "variable", "value"]
    )


def test_stats_validator_flags_malformed_unit_ids():
    """Sentinel '__FILTER__' style unit_ids should error."""
    from stablebound.validate import validate_stats_lineage_consistency

    rt = _rt([(2003, "Split", "A", "A", "A1", "A1"), (2003, "Split", "A", "A", "A2", "A2")])
    g = LineageGraph.from_dataframe(rt)
    stats = _stats([
        ("A1", 2005, "annual", "area", 100.0),
        ("__FILTER__", 2005, "annual", "area", 999.0),
        ("", 2005, "annual", "area", 999.0),
    ])
    issues = validate_stats_lineage_consistency(stats, g, modern_unit_ids={"A1", "A2"})
    errors = [i for i in issues if i.severity == "error"]
    assert any(i.category == "stats_unit_id_malformed" for i in errors)


def test_stats_validator_flags_unknown_unit_ids():
    """unit_ids not in lineage AND not in modern → warning."""
    from stablebound.validate import validate_stats_lineage_consistency

    rt = _rt([(2003, "Split", "A", "A", "A1", "A1"), (2003, "Split", "A", "A", "A2", "A2")])
    g = LineageGraph.from_dataframe(rt)
    stats = _stats([
        ("A1", 2005, "annual", "area", 100.0),
        ("UNKNOWN_DISTRICT", 2005, "annual", "area", 50.0),
    ])
    issues = validate_stats_lineage_consistency(stats, g, modern_unit_ids={"A1", "A2"})
    warnings = [i for i in issues if i.severity == "warning"]
    unknown = [i for i in warnings if i.category == "stats_unit_id_unknown"]
    assert unknown
    assert "UNKNOWN_DISTRICT" in unknown[0].ids


def test_stats_validator_flags_post_cease_reporting():
    """unit_id reporting after its terminal territorial event → warning."""
    from stablebound.validate import validate_stats_lineage_consistency

    rt = _rt([
        (2003, "Split", "A", "A", "A1", "A1"),
        (2003, "Split", "A", "A", "A2", "A2"),
    ])
    g = LineageGraph.from_dataframe(rt)
    # A ceased in 2003 but still has reports in 2005, 2010 — this is the
    # Giridih (JH) bug pattern.
    stats = _stats([
        ("A", 2005, "annual", "area", 100.0),
        ("A", 2010, "annual", "area", 110.0),
        ("A1", 2005, "annual", "area", 60.0),
        ("A2", 2005, "annual", "area", 40.0),
    ])
    issues = validate_stats_lineage_consistency(stats, g, modern_unit_ids={"A1", "A2"})
    warnings = [i for i in issues if i.severity == "warning"]
    post_cease = [i for i in warnings if i.category == "stats_post_cease_reporting"]
    assert post_cease
    assert "A" in post_cease[0].ids
    # Detail should mention the cease year and stale-row count
    assert "2003" in post_cease[0].detail
    assert "2 rows" in post_cease[0].detail


def test_stats_validator_clean_input_produces_no_findings():
    """Well-formed stats data passes with zero findings."""
    from stablebound.validate import validate_stats_lineage_consistency

    rt = _rt([(2003, "Split", "A", "A", "A1", "A1"), (2003, "Split", "A", "A", "A2", "A2")])
    g = LineageGraph.from_dataframe(rt)
    stats = _stats([
        ("A", 2001, "annual", "area", 100.0),
        ("A1", 2005, "annual", "area", 60.0),
        ("A2", 2005, "annual", "area", 40.0),
    ])
    issues = validate_stats_lineage_consistency(stats, g, modern_unit_ids={"A1", "A2"})
    assert issues == []


def test_stats_validator_works_without_modern_unit_ids():
    """Modern shapefile is optional; without it, only the lineage-only checks fire."""
    from stablebound.validate import validate_stats_lineage_consistency

    rt = _rt([(2003, "Split", "A", "A", "A1", "A1"), (2003, "Split", "A", "A", "A2", "A2")])
    g = LineageGraph.from_dataframe(rt)
    stats = _stats([
        ("A1", 2005, "annual", "area", 100.0),
        ("OUTSIDE", 2005, "annual", "area", 5.0),
    ])
    issues = validate_stats_lineage_consistency(stats, g)  # no modern_unit_ids
    unknown = [i for i in issues if i.category == "stats_unit_id_unknown"]
    assert unknown
    assert "OUTSIDE" in unknown[0].ids


# --- Spatial contiguity validator ---------------------------------------

def _grid_gdf(cells):
    """Build a tiny GeoDataFrame from a {unit_id: (col, row)} mapping.

    Each cell is a 1×1 square at integer (col, row) so adjacent cells
    share a full edge — `touches` returns True for orthogonal neighbors,
    False for diagonals and disconnected cells.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    rows = []
    for unit_id, (col, row) in cells.items():
        poly = Polygon([
            (col, row), (col + 1, row),
            (col + 1, row + 1), (col, row + 1),
        ])
        rows.append({"unit_id": unit_id, "geometry": poly})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def test_contiguity_clean_groups_produce_no_issues():
    from stablebound.validate import validate_stable_group_contiguity

    # Two adjacent cells in the same stable group.
    gdf = _grid_gdf({"A": (0, 0), "B": (1, 0), "C": (5, 5)})
    remap = {"A": "S1", "B": "S1", "C": "S2"}
    issues = validate_stable_group_contiguity(remap, gdf, "unit_id")
    assert issues == []


def test_contiguity_disconnected_group_warns():
    from stablebound.validate import validate_stable_group_contiguity

    # A and B are far apart but assigned the same stable group — homonym bug.
    gdf = _grid_gdf({"A": (0, 0), "B": (10, 10)})
    remap = {"A": "S1", "B": "S1"}
    issues = validate_stable_group_contiguity(remap, gdf, "unit_id")
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].category == "stable_group_disconnected"
    assert "S1" in issues[0].message
    assert set(issues[0].ids) == {"A", "B"}


def test_contiguity_singleton_groups_ignored():
    from stablebound.validate import validate_stable_group_contiguity

    # Lone members are trivially "connected" — must not warn.
    gdf = _grid_gdf({"A": (0, 0), "B": (5, 5)})
    remap = {"A": "S1", "B": "S2"}
    issues = validate_stable_group_contiguity(remap, gdf, "unit_id")
    assert issues == []


def test_contiguity_chain_via_shared_borders_passes():
    from stablebound.validate import validate_stable_group_contiguity

    # A–B–C in a horizontal chain, all in S1. C is not directly adjacent
    # to A but is reachable through B — BFS should find it.
    gdf = _grid_gdf({"A": (0, 0), "B": (1, 0), "C": (2, 0)})
    remap = {"A": "S1", "B": "S1", "C": "S1"}
    issues = validate_stable_group_contiguity(remap, gdf, "unit_id")
    assert issues == []


def test_contiguity_diagonal_only_neighbors_are_disconnected():
    from stablebound.validate import validate_stable_group_contiguity

    # Diagonal cells touch only at a corner. shapely's `touches` returns
    # True (boundaries share a point), so these COUNT as neighbors. To
    # get a disconnect we need cells that don't share any boundary.
    gdf = _grid_gdf({"A": (0, 0), "B": (3, 3)})
    remap = {"A": "S1", "B": "S1"}
    issues = validate_stable_group_contiguity(remap, gdf, "unit_id")
    assert len(issues) == 1
    assert issues[0].category == "stable_group_disconnected"


def test_contiguity_remap_entries_not_in_modern_are_ignored():
    from stablebound.validate import validate_stable_group_contiguity

    # GHOST is in the remap (e.g., a deceased historical unit) but has no
    # modern shapefile feature — should be skipped, not crash.
    gdf = _grid_gdf({"A": (0, 0), "B": (1, 0)})
    remap = {"A": "S1", "B": "S1", "GHOST": "S1"}
    issues = validate_stable_group_contiguity(remap, gdf, "unit_id")
    assert issues == []


# --- NameChange unit/name mismatch ---------------------------------------
#
# merge_name_changes folds a name-change log into the graph as NameChange
# events, so a mistyped unit_id there renames an unrelated district in every
# name lookup — including the FEWS admin-definition workbooks. India shipped
# two such rows undetected. These pin the check that finds them.


def _graph_with_namechange(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name",
            "child_id", "child_name"]
    return LineageGraph.from_dataframe(pd.DataFrame(rows, columns=cols))


def test_namechange_pointing_at_wrong_unit_is_flagged():
    # A.1 is only ever called "Alpha"; the rename claims it was "Beta"->"Bravo".
    # A.2 is the unit actually called Beta, so it is the likely correct id.
    graph = _graph_with_namechange([
        (2000, "Split", "A.0", "Root", "A.1", "Alpha"),
        (2000, "Split", "A.0", "Root", "A.2", "Beta"),
        (2010, "NameChange", "A.1", "Beta", "A.1", "Bravo"),
    ])
    issues = [i for i in validate_lineage(graph)
              if i.category == "namechange_unit_name_mismatch"]
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].ids[0] == "A.1"
    # The correction is in the message, not left to the reader.
    assert "A.2" in issues[0].ids
    assert "likely correct unit_id" in issues[0].detail


def test_namechange_on_the_right_unit_is_not_flagged():
    graph = _graph_with_namechange([
        (2000, "Split", "A.0", "Root", "A.1", "Alpha"),
        (2010, "NameChange", "A.1", "Alpha", "A.1", "Alpha Renamed"),
    ])
    assert not [i for i in validate_lineage(graph)
                if i.category == "namechange_unit_name_mismatch"]


def test_namechange_matching_the_post_rename_name_is_not_flagged():
    # Corroboration from either side counts: a later event refers to the unit
    # by its NEW name, which is just as good as the old one.
    graph = _graph_with_namechange([
        (2000, "Split", "A.0", "Root", "A.1", "Alpha"),
        (2010, "NameChange", "A.1", "Alpha", "A.1", "Bravo"),
        (2015, "Split", "A.1", "Bravo", "A.3", "Charlie"),
    ])
    assert not [i for i in validate_lineage(graph)
                if i.category == "namechange_unit_name_mismatch"]


def test_namechange_as_the_only_mention_of_a_unit_is_not_guessed_at():
    # Nothing corroborates or contradicts it — stay quiet rather than guess.
    graph = _graph_with_namechange([
        (2000, "Split", "A.0", "Root", "A.1", "Alpha"),
        (2010, "NameChange", "Z.9", "Zulu", "Z.9", "Zulu Renamed"),
    ])
    assert not [i for i in validate_lineage(graph)
                if i.category == "namechange_unit_name_mismatch"]


def test_namechange_mismatch_tolerates_cosmetic_spelling():
    # "Alpha-North" vs "Alpha North" is the same unit, not a mismatch.
    graph = _graph_with_namechange([
        (2000, "Split", "A.0", "Root", "A.1", "Alpha-North"),
        (2010, "NameChange", "A.1", "Alpha North", "A.1", "Alpha N"),
    ])
    assert not [i for i in validate_lineage(graph)
                if i.category == "namechange_unit_name_mismatch"]


# --- Shapefile / lineage cross-checks ------------------------------------
#
# Exampleland's Alpha (E.001) is split into E.011 + E.012 in 2015, so E.001 is
# a real id that is alive at 2014 and retired at 2019. That is the shape of
# every wrong-id defect worth catching: not a malformed string, but a
# well-formed id for the wrong year.

_SHP_RT = [
    (2015, "Split", "E.001", "Alpha", "E.011", "Alpha North"),
    (2015, "Split", "E.001", "Alpha", "E.012", "Alpha South"),
]
_SHP_BASELINE = pd.DataFrame(
    {"unit_id": ["E.001", "E.002"], "name": ["Alpha", "Bravo"], "year": [2010, 2010]}
)


def _shp_graph():
    return LineageGraph.from_dataframe(_rt(_SHP_RT))


def _shp_check(ids, year):
    from stablebound.validate import validate_shapefile_lineage_consistency

    return validate_shapefile_lineage_consistency(
        ids, _shp_graph(), year, baseline=_SHP_BASELINE
    )


def _categories(issues):
    return {i.category for i in issues}


def test_shapefile_checks_are_silent_on_a_correct_map():
    """The 2019 map, with every live district and nothing else."""
    issues = _shp_check(["E.002", "E.011", "E.012"], 2019)
    assert issues == [], _categories(issues)


def test_shapefile_flags_an_id_retired_by_that_vintage():
    """E.001 was replaced in 2015; a 2019 map must not still carry it.

    Both halves of the swap have to be reported: the polygon holding the dead
    id, and the live district left with no polygon. Reporting only the first
    tells you something is wrong but not what should have been there.
    """
    issues = _shp_check(["E.002", "E.001", "E.012"], 2019)
    assert _categories(issues) == {
        "shapefile_retired_id",
        "shapefile_unit_without_polygon",
    }
    retired = next(i for i in issues if i.category == "shapefile_retired_id")
    missing = next(i for i in issues if i.category == "shapefile_unit_without_polygon")
    assert retired.ids == ["E.001"]
    assert missing.ids == ["E.011"], "the district that should hold the polygon"
    assert retired.severity == "warning", "a historical map fails this legitimately"


def test_shapefile_accepts_the_same_id_at_a_year_it_was_alive():
    """The identical id set is correct at 2014 and wrong at 2019.

    This is what separates the check from a static id whitelist: nothing about
    E.001 is malformed, and only the year makes it an error.
    """
    assert _shp_check(["E.001", "E.002"], 2014) == []
    assert _categories(_shp_check(["E.001", "E.002"], 2019)) == {
        "shapefile_retired_id",
        "shapefile_unit_without_polygon",
    }


def test_shapefile_flags_an_id_the_lineage_has_never_heard_of():
    issues = _shp_check(["E.002", "E.011", "E.012", "E.999"], 2019)
    unknown = next(i for i in issues if i.category == "shapefile_unknown_id")
    assert unknown.severity == "error"
    assert unknown.ids == ["E.999"]
    # An id the lineage has never seen cannot ALSO be reported as retired --
    # that would count one mistake twice.
    assert "shapefile_retired_id" not in _categories(issues)


def test_shapefile_flags_one_id_on_two_polygons():
    """Two features, one id -- the homonym collapse, in miniature."""
    issues = _shp_check(["E.002", "E.011", "E.011"], 2019)
    dup = next(i for i in issues if i.category == "shapefile_duplicate_id")
    assert dup.ids == ["E.011"]
    assert "x2" in dup.message


def test_duplicate_check_survives_a_deduplicating_caller():
    """Regression: the duplicate check must be given the column, not a set.

    attach_shapefile builds a SET of ids for infer_year, and passing that set
    to the validator silently disables this one check -- deduplicating is
    exactly what hides a duplicate. The first wiring did that, and the check
    reported clean on a file already known to contain the defect.
    """
    ids = ["E.002", "E.011", "E.011"]
    assert _categories(_shp_check(ids, 2019)) >= {"shapefile_duplicate_id"}
    assert "shapefile_duplicate_id" not in _categories(_shp_check(set(ids), 2019)), (
        "a set cannot carry a duplicate -- if this ever fails, the test below "
        "is no longer proving anything"
    )


def test_shapefile_sentinels_are_not_reported_as_unknown_ids():
    """UNMATCHED_* already surfaces via unmatched_features; not twice."""
    issues = _shp_check(["E.002", "E.011", "E.012", "UNMATCHED_7"], 2019)
    assert "shapefile_unknown_id" not in _categories(issues)
