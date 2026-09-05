"""Tests for stablebound.match — the human-in-the-loop name matcher."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import warnings

import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from stablebound.lineage import LineageGraph
from stablebound.match import (
    attach_shapefile_ids,
    attach_stats_ids,
    normalize_name,
    propose_shapefile_mapping,
    propose_stats_mapping,
    read_mapping,
)

EXAMPLELAND = Path(__file__).resolve().parents[1] / "examples" / "exampleland"


# --- Test fixtures -------------------------------------------------------


@pytest.fixture()
def exampleland_graph() -> LineageGraph:
    rt = pd.read_csv(EXAMPLELAND / "relationship_table.csv")
    return LineageGraph.from_dataframe(rt)


@pytest.fixture()
def exampleland_baseline() -> pd.DataFrame:
    """Synthesize a 2010 baseline from the RT (no separate file in this fixture)."""
    return pd.DataFrame(
        {
            "unit_id": ["E.001", "E.002", "E.003", "E.004", "E.005"],
            "name": ["Alpha", "Bravo", "Charlie", "Delta", "Echo"],
        }
    )


def _make_shapefile_gdf(
    names: list[str], states: list[str] | None = None
) -> gpd.GeoDataFrame:
    """Build a tiny GeoDataFrame keyed by name only (no unit_id).

    If ``states`` is provided, adds a parallel 'state' column for
    testing coarse-hierarchy disambiguation.
    """
    geom = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i, _ in enumerate(names)
    ]
    data = {"name": names, "geometry": geom}
    if states is not None:
        data["state"] = states
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# --- Normalization -------------------------------------------------------


def test_normalize_lowercases_and_strips_whitespace():
    assert normalize_name("  Alpha  North  ") == "alpha north"


def test_normalize_removes_diacritics():
    # NFKD decomposition then drop combining codepoints. "São Paulo" → "sao paulo".
    assert normalize_name("São Paulo") == "sao paulo"


def test_normalize_strips_punctuation_and_dashes():
    assert normalize_name("Alpha-North!") == "alpha north"
    assert normalize_name("foo_bar/baz") == "foo bar baz"


def test_normalize_collapses_whitespace():
    assert normalize_name("Alpha   North") == "alpha north"


# --- Shapefile proposals -------------------------------------------------


def test_propose_shapefile_exact_match(exampleland_graph, exampleland_baseline):
    # Use names matching the post-2018 snapshot exactly. Year 2019 sees the
    # split (Alpha North/South), the rename (Charlie Renamed), and the merge
    # (DeltaEcho).
    gdf = _make_shapefile_gdf(
        ["Alpha North", "Alpha South", "Bravo", "Charlie Renamed", "DeltaEcho"]
    )
    proposal = propose_shapefile_mapping(
        gdf,
        exampleland_graph,
        name_column="name",
        year=2019,
        baseline=exampleland_baseline,
    )
    assert (proposal.proposals["method"] == "exact").all()
    assert proposal.proposals["proposed_unit_id"].tolist() == [
        "E.011",
        "E.012",
        "E.002",
        "E.003",
        "E.013",
    ]


def test_propose_shapefile_fuzzy_match(exampleland_graph, exampleland_baseline):
    # "Charlie Reanmed" — single-character transposition. Above the default
    # 0.80 threshold, so should fuzzy-match Charlie Renamed.
    gdf = _make_shapefile_gdf(["Charlie Reanmed"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    row = proposal.proposals.iloc[0]
    assert row["method"] == "fuzzy"
    assert row["proposed_unit_id"] == "E.003"
    assert row["score"] >= 0.80


def test_propose_shapefile_unmatched(exampleland_graph, exampleland_baseline):
    gdf = _make_shapefile_gdf(["Zzzzzzz", "Alpha North"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    methods = proposal.proposals.sort_values("source_idx")["method"].tolist()
    assert methods == ["unmatched", "exact"]
    unmatched = proposal.unmatched
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["source_name"] == "Zzzzzzz"


def test_propose_shapefile_manual_override(exampleland_graph, exampleland_baseline):
    # "AlphaN" with no obvious fuzzy match. Manual override maps it.
    gdf = _make_shapefile_gdf(["AlphaN"])
    proposal = propose_shapefile_mapping(
        gdf,
        exampleland_graph,
        name_column="name",
        year=2019,
        baseline=exampleland_baseline,
        manual_overrides={"alphan": "Alpha North"},
    )
    row = proposal.proposals.iloc[0]
    assert row["method"] == "manual"
    assert row["proposed_unit_id"] == "E.011"


def test_propose_shapefile_homonym_override(exampleland_graph, exampleland_baseline):
    # "Alpha North" would normally exact-match E.011. Homonym override
    # forces it to E.012 (Alpha South), proving the override takes
    # priority over even exact match.
    gdf = _make_shapefile_gdf(["Alpha North"])
    proposal = propose_shapefile_mapping(
        gdf,
        exampleland_graph,
        name_column="name",
        year=2019,
        baseline=exampleland_baseline,
        homonym_overrides={0: ("E.012", "Alpha South")},
    )
    row = proposal.proposals.iloc[0]
    assert row["method"] == "homonym"
    assert row["proposed_unit_id"] == "E.012"


def test_propose_shapefile_year_governs_snapshot(exampleland_graph, exampleland_baseline):
    # At year=2010 (pre-2014-split), the snapshot contains Alpha but not
    # Alpha North or Alpha South. Asking for them at 2010 must NOT produce
    # E.011/E.012 — those unit_ids don't exist yet.
    gdf = _make_shapefile_gdf(["Alpha North", "Alpha"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2010, baseline=exampleland_baseline,
    )
    by_name = {r["source_name"]: r for _, r in proposal.proposals.iterrows()}
    assert by_name["Alpha"]["method"] == "exact"
    assert by_name["Alpha"]["proposed_unit_id"] == "E.001"
    # E.011 doesn't exist in the 2010 snapshot, so we must NOT propose it.
    assert by_name["Alpha North"]["proposed_unit_id"] != "E.011"


def test_propose_shapefile_post_split_year_sees_children(
    exampleland_graph, exampleland_baseline
):
    # At year=2019 (post-2014-split), Alpha North and Alpha South are the
    # canonical units. Both exact-match.
    gdf = _make_shapefile_gdf(["Alpha North", "Alpha South"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    by_name = {r["source_name"]: r for _, r in proposal.proposals.iterrows()}
    assert by_name["Alpha North"]["proposed_unit_id"] == "E.011"
    assert by_name["Alpha South"]["proposed_unit_id"] == "E.012"


def test_propose_shapefile_missing_name_column_raises(exampleland_graph):
    gdf = _make_shapefile_gdf(["Alpha"])
    with pytest.raises(KeyError, match="name_column"):
        propose_shapefile_mapping(gdf, exampleland_graph, name_column="not_there")


# --- Stats proposals -----------------------------------------------------


def test_propose_stats_unique_names_only(exampleland_graph, exampleland_baseline):
    # 20 rows but only 3 unique names — output should have 3 rows.
    stats = pd.DataFrame(
        {
            "district": ["Alpha"] * 5 + ["Bravo"] * 10 + ["Charlie"] * 5,
            "year": list(range(2010, 2015)) * 4,
            "variable": ["rice"] * 20,
            "value": [1.0] * 20,
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        baseline=exampleland_baseline,
    )
    assert len(proposal.proposals) == 3
    by_name = {r["source_name"]: r for _, r in proposal.proposals.iterrows()}
    assert by_name["Alpha"]["proposed_unit_id"] == "E.001"
    assert by_name["Bravo"]["proposed_unit_id"] == "E.002"
    # Charlie renames to "Charlie Renamed" in 2016, but the union lookup
    # spans the whole stats year range (here 2010-2014). "Charlie" should
    # still resolve to E.003 via the pre-2016 snapshot.
    assert by_name["Charlie"]["proposed_unit_id"] == "E.003"


def test_propose_stats_handles_pre_and_post_rename(exampleland_graph, exampleland_baseline):
    # Stats include both "Charlie" (pre-2016) and "Charlie Renamed"
    # (post-2016). Both should resolve to E.003.
    stats = pd.DataFrame(
        {
            "district": ["Charlie", "Charlie Renamed"],
            "year": [2014, 2017],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        baseline=exampleland_baseline,
    )
    assert set(proposal.proposals["proposed_unit_id"]) == {"E.003"}


# --- Year-aware stats matching (opt-in) ---------------------------------


def test_propose_stats_year_aware_resolves_per_year(
    exampleland_graph, exampleland_baseline
):
    # Alpha splits into Alpha North/South in 2014 (parent E.001 ends).
    # "Alpha North" reported in 2019 is the post-split child E.011; the
    # same name reported in 2012 (before the split existed) must walk up
    # to the pre-split parent E.001. The union path would collapse both
    # to E.011 — year-aware keeps them distinct.
    stats = pd.DataFrame(
        {
            "district": ["Alpha North", "Alpha North"],
            "year": [2019, 2012],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    by_year = {
        int(r["source_year"]): r for _, r in proposal.proposals.iterrows()
    }
    assert by_year[2019]["proposed_unit_id"] == "E.011"      # post-split child
    assert by_year[2012]["proposed_unit_id"] == "E.001"      # pre-split parent
    assert "ancestor-walk" in by_year[2012]["notes"]         # remap is audited
    assert by_year[2019]["notes"] == ""                      # in-year, no remap
    # The remap to an OLDER unit must be surfaced loudly, not silent.
    assert by_year[2012]["method"].endswith("+ancestor_walk")
    assert by_year[2019]["method"] == "exact"                # untouched
    rem = proposal.remapped
    assert list(rem["proposed_unit_id"]) == ["E.001"]
    assert (rem["source_year"].astype(int) == 2012).all()
    assert "VERIFY" in proposal.summary()


def test_propose_stats_year_aware_merge_ancestor_is_deterministic(
    exampleland_graph, exampleland_baseline
):
    # DeltaEcho (E.013) is the 2018 merge of Delta (E.004) + Echo (E.005).
    # A pre-merge (2012) "DeltaEcho" walks to a parent; with both parents
    # alive from baseline the earliest-created + id tie-break picks E.004.
    stats = pd.DataFrame({"district": ["DeltaEcho"], "year": [2012], "value": [1.0]})
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    assert proposal.proposals.iloc[0]["proposed_unit_id"] == "E.004"


def test_propose_stats_year_aware_requires_year_column(
    exampleland_graph, exampleland_baseline
):
    stats = pd.DataFrame({"district": ["Alpha"], "value": [1.0]})
    with pytest.raises(ValueError, match="requires a year_column"):
        propose_stats_mapping(
            stats,
            exampleland_graph,
            name_column="district",
            year_aware=True,
            baseline=exampleland_baseline,
        )


def test_attach_stats_ids_year_aware_join(exampleland_graph, exampleland_baseline):
    # The (name, year) join attaches different ids to the same name across
    # years; a plain name-only join could not.
    stats = pd.DataFrame(
        {
            "district": ["Alpha North", "Alpha North", "Bravo"],
            "year": [2019, 2012, 2015],
            "value": [1.0, 2.0, 3.0],
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    out = attach_stats_ids(
        stats, proposal.proposals, name_column="district", year_column="year"
    )
    got = {
        (r["district"], int(r["year"])): r["unit_id"] for _, r in out.iterrows()
    }
    assert got[("Alpha North", 2019)] == "E.011"
    assert got[("Alpha North", 2012)] == "E.001"
    assert got[("Bravo", 2015)] == "E.002"


def test_summary_counts_include_ancestor_walk_rows(
    exampleland_graph, exampleland_baseline
):
    # The by-method block used to iterate the five bare method names, so
    # "exact+ancestor_walk" rows were dropped from it and the counts no
    # longer summed to the total.
    stats = pd.DataFrame(
        {
            "district": ["Alpha North", "Alpha North"],
            "year": [2019, 2012],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    assert len(proposal.remapped) == 1          # one row carries the suffix
    text = proposal.summary()
    total = len(proposal.proposals)
    counted = sum(
        int(line.split()[-1])
        for line in text.splitlines()
        if line.startswith("  ") and line.split() and line.split()[-1].isdigit()
        and line.split()[0] in {"homonym", "exact", "manual", "fuzzy", "unmatched"}
    )
    assert counted == total


def test_remapped_survives_a_reviewer_clearing_notes(
    exampleland_graph, exampleland_baseline
):
    # `.remapped` keys on `method`, so blanking the free-text notes column
    # in Excel must not lose the flag.
    stats = pd.DataFrame(
        {"district": ["Alpha North"], "year": [2012], "value": [1.0]}
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    assert len(proposal.remapped) == 1
    proposal.proposals["notes"] = ""
    assert len(proposal.remapped) == 1


# --- No-silent-misattribution guard --------------------------------------
#
# A year-aware (or coarse-keyed) proposal has several rows per name. Joining
# it on name alone used to build dict(zip(source_name, proposed_unit_id)),
# which silently kept whichever row came last — attaching every year of a
# district's data to one arbitrary year's unit_id, with no warning. These
# tests pin that this is now an error.


def test_attach_stats_ids_year_keyed_mapping_joined_name_only_raises(
    exampleland_graph, exampleland_baseline
):
    stats = pd.DataFrame(
        {
            "district": ["Alpha North", "Alpha North"],
            "year": [2019, 2012],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    # "Alpha North" resolves to E.011 in 2019 and E.001 in 2012, so a
    # name-only join cannot be correct for both.
    with pytest.raises(ValueError) as exc:
        attach_stats_ids(stats, proposal.proposals, name_column="district")
    msg = str(exc.value)
    assert "more than one" in msg
    # The error has to say what to do about it, not just that it failed.
    assert "year_column" in msg
    assert "Alpha North" in msg


def test_attach_stats_ids_coarse_keyed_mapping_joined_without_coarse_raises(
    homonym_graph
):
    stats = pd.DataFrame(
        {
            "district": ["Hamirpur", "Hamirpur"],
            "state": ["Himachal Pradesh", "Uttar Pradesh"],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats, homonym_graph, name_column="district", coarse_column="state",
    )
    with pytest.raises(ValueError) as exc:
        attach_stats_ids(stats, proposal.proposals, name_column="district")
    assert "coarse_column" in str(exc.value)


def test_proposal_carries_source_coarse_so_homonyms_are_distinguishable(
    homonym_graph
):
    # Without a source_coarse column the two rows below are byte-identical
    # in the review CSV — a human could not tell which Hamirpur is which.
    stats = pd.DataFrame(
        {
            "district": ["Hamirpur", "Hamirpur"],
            "state": ["Himachal Pradesh", "Uttar Pradesh"],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats, homonym_graph, name_column="district", coarse_column="state",
    )
    assert "source_coarse" in proposal.proposals.columns
    got = {
        (r["source_name"], r["source_coarse"]): r["proposed_unit_id"]
        for _, r in proposal.proposals.iterrows()
    }
    assert got[("Hamirpur", "Himachal Pradesh")] == "IN.ADM2.HP_HAM"
    assert got[("Hamirpur", "Uttar Pradesh")] == "IN.ADM2.UP_HAM"


def test_attach_stats_ids_coarse_join_disambiguates_homonyms(homonym_graph):
    stats = pd.DataFrame(
        {
            "district": ["Hamirpur", "Hamirpur", "Hamirpur"],
            "state": ["Himachal Pradesh", "Uttar Pradesh", "Himachal Pradesh"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    proposal = propose_stats_mapping(
        stats, homonym_graph, name_column="district", coarse_column="state",
    )
    out = attach_stats_ids(
        stats,
        proposal.proposals,
        name_column="district",
        coarse_column="state",
    )
    assert list(out["unit_id"]) == [
        "IN.ADM2.HP_HAM",
        "IN.ADM2.UP_HAM",
        "IN.ADM2.HP_HAM",
    ]


def test_source_coarse_survives_the_review_csv_round_trip(
    tmp_path, homonym_graph
):
    # The reviewer edits the CSV in Excel; the key column must come back.
    stats = pd.DataFrame(
        {
            "district": ["Hamirpur", "Hamirpur"],
            "state": ["Himachal Pradesh", "Uttar Pradesh"],
            "value": [1.0, 2.0],
        }
    )
    proposal = propose_stats_mapping(
        stats, homonym_graph, name_column="district", coarse_column="state",
    )
    path = tmp_path / "mapping.csv"
    proposal.to_csv(path)
    out = attach_stats_ids(
        stats, path, name_column="district", coarse_column="state",
    )
    assert list(out["unit_id"]) == ["IN.ADM2.HP_HAM", "IN.ADM2.UP_HAM"]


# --- MatchProposal artifacts --------------------------------------------


def test_match_proposal_to_csv_and_read_back(tmp_path, exampleland_graph, exampleland_baseline):
    gdf = _make_shapefile_gdf(["Alpha North", "Bravo"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    csv_path = tmp_path / "mapping.csv"
    proposal.to_csv(csv_path)
    assert csv_path.exists()

    loaded = read_mapping(csv_path)
    assert {"source_name", "proposed_unit_id"} <= set(loaded.columns)
    assert len(loaded) == 2


def test_match_proposal_summary_lists_unmatched(exampleland_graph, exampleland_baseline):
    gdf = _make_shapefile_gdf(["Zzzzzzz"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    text = proposal.summary()
    assert "Unmatched rows: 1" in text
    assert "Zzzzzzz" in text


def test_read_mapping_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["Foo"]}).to_csv(csv_path, index=False)
    with pytest.raises(KeyError, match="missing required columns"):
        read_mapping(csv_path)


# --- Attach helpers ------------------------------------------------------


def test_attach_shapefile_ids_round_trip(tmp_path, exampleland_graph, exampleland_baseline):
    gdf = _make_shapefile_gdf(["Alpha North", "Bravo"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    csv_path = tmp_path / "mapping.csv"
    proposal.to_csv(csv_path)

    attached = attach_shapefile_ids(gdf, csv_path, name_column="name")
    assert "unit_id" in attached.columns
    assert attached["unit_id"].tolist() == ["E.011", "E.002"]


def test_attach_shapefile_ids_homonym_safe(tmp_path, exampleland_graph, exampleland_baseline):
    # Two features with the same name but different unit_ids (via
    # homonym overrides). The CSV round-trip must preserve them
    # separately — the previous name-based join collapsed both to the
    # same unit_id.
    gdf = _make_shapefile_gdf(["Alpha North", "Alpha North"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019,
        baseline=exampleland_baseline,
        homonym_overrides={
            0: ("E.011", "Alpha North"),
            1: ("E.012", "Alpha South"),
        },
    )
    csv_path = tmp_path / "mapping.csv"
    proposal.to_csv(csv_path)
    attached = attach_shapefile_ids(gdf, csv_path, name_column="name")
    # Distinct IDs survive even with identical names.
    assert attached["unit_id"].tolist() == ["E.011", "E.012"]


def test_attach_shapefile_ids_name_fallback_warns_on_duplicates(tmp_path):
    # Hand-built mapping without source_idx and with duplicate names:
    # name-based fallback is lossy; user gets a warning.
    import warnings as _w
    gdf = _make_shapefile_gdf(["Alpha", "Alpha"])
    mapping = pd.DataFrame({
        "source_name": ["Alpha", "Alpha"],
        "proposed_unit_id": ["X1", "X2"],
    })
    csv_path = tmp_path / "hand_built.csv"
    mapping.to_csv(csv_path, index=False)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        attach_shapefile_ids(gdf, csv_path, name_column="name")
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("duplicate source_name" in m for m in msgs)


def test_attach_shapefile_ids_handles_unmatched(tmp_path, exampleland_graph, exampleland_baseline):
    # An unmatched row leaves proposed_unit_id blank; attach should
    # surface NA — the rest of the pipeline rejects NA unit_ids loudly.
    gdf = _make_shapefile_gdf(["Zzzzzzz", "Bravo"])
    proposal = propose_shapefile_mapping(
        gdf, exampleland_graph, name_column="name", year=2019, baseline=exampleland_baseline,
    )
    csv_path = tmp_path / "mapping.csv"
    proposal.to_csv(csv_path)

    attached = attach_shapefile_ids(gdf, csv_path, name_column="name")
    assert pd.isna(attached.loc[0, "unit_id"])
    assert attached.loc[1, "unit_id"] == "E.002"


def test_attach_stats_ids(tmp_path, exampleland_graph, exampleland_baseline):
    stats = pd.DataFrame(
        {
            "district": ["Alpha", "Alpha", "Bravo"],
            "year": [2010, 2011, 2010],
            "value": [1.0, 2.0, 3.0],
        }
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        baseline=exampleland_baseline,
    )
    csv_path = tmp_path / "stats_mapping.csv"
    proposal.to_csv(csv_path)

    attached = attach_stats_ids(stats, csv_path, name_column="district")
    assert attached["unit_id"].tolist() == ["E.001", "E.001", "E.002"]


# --- Bundled data smoke test --------------------------------------------


@pytest.fixture()
def homonym_graph() -> LineageGraph:
    """Synthetic RT with two distinct units sharing the name 'Hamirpur'.

    Models the India case: Hamirpur (Himachal Pradesh) and Hamirpur
    (Uttar Pradesh). Each unit gets a creation event so it shows up
    in the snapshot, and parent_coarse_name carries the state.
    """
    rt = pd.DataFrame({
        "event_year": [2010, 2010],
        "event_type": ["Split", "Split"],
        "parent_id": ["IN.ADM1.HP_ROOT", "IN.ADM1.UP_ROOT"],
        "parent_name": ["HP State", "UP State"],
        "parent_coarse_id": ["IN.ADM1.HP", "IN.ADM1.UP"],
        "parent_coarse_name": ["Himachal Pradesh", "Uttar Pradesh"],
        "child_id": ["IN.ADM2.HP_HAM", "IN.ADM2.UP_HAM"],
        "child_name": ["Hamirpur", "Hamirpur"],
        "child_coarse_id": ["IN.ADM1.HP", "IN.ADM1.UP"],
        "child_coarse_name": ["Himachal Pradesh", "Uttar Pradesh"],
    })
    return LineageGraph.from_dataframe(rt)


def test_propose_shapefile_coarse_disambiguates_homonyms(homonym_graph):
    # Two shapefile features both named "Hamirpur" but in different states.
    # Without coarse_column, the matcher falls back to first-write-wins
    # (both rows pick the same unit). With coarse_column='state' they
    # resolve to their respective state-specific unit_ids.
    gdf = _make_shapefile_gdf(
        ["Hamirpur", "Hamirpur"],
        states=["Himachal Pradesh", "Uttar Pradesh"],
    )
    proposal = propose_shapefile_mapping(
        gdf, homonym_graph, name_column="name",
        coarse_column="state", year=2011,
    )
    by_state = {
        r["source_idx"]: r["proposed_unit_id"]
        for _, r in proposal.proposals.iterrows()
    }
    # idx 0 is HP, idx 1 is UP.
    assert by_state[0] == "IN.ADM2.HP_HAM"
    assert by_state[1] == "IN.ADM2.UP_HAM"


def test_propose_shapefile_without_coarse_collapses_homonyms(homonym_graph):
    # Same fixture as above without coarse_column. Both features get
    # the same first-write-wins unit_id — the legacy behavior, which
    # is preserved for backward compatibility.
    gdf = _make_shapefile_gdf(["Hamirpur", "Hamirpur"])
    proposal = propose_shapefile_mapping(
        gdf, homonym_graph, name_column="name", year=2011,
    )
    ids = set(proposal.proposals["proposed_unit_id"])
    assert len(ids) == 1, f"expected first-write-wins fallback, got {ids}"


def test_propose_shapefile_coarse_column_missing_raises(homonym_graph):
    gdf = _make_shapefile_gdf(["Hamirpur"])
    with pytest.raises(KeyError, match="coarse_column"):
        propose_shapefile_mapping(
            gdf, homonym_graph, name_column="name",
            coarse_column="state",  # not in gdf
        )


def test_propose_stats_with_coarse_column(homonym_graph):
    # Stats with two rows of "Hamirpur" but different state values:
    # coarse_column should produce one proposal per (name, state) pair.
    stats = pd.DataFrame({
        "district": ["Hamirpur", "Hamirpur", "Hamirpur"],
        "state": ["Himachal Pradesh", "Uttar Pradesh", "Himachal Pradesh"],
        "year": [2011, 2011, 2012],
        "variable": ["rice"] * 3,
        "value": [1.0, 2.0, 3.0],
    })
    proposal = propose_stats_mapping(
        stats, homonym_graph, name_column="district",
        coarse_column="state", year_column="year",
    )
    # 2 unique (name, state) pairs → 2 proposals.
    assert len(proposal.proposals) == 2
    by_state = {
        r["best_candidate"] or "?": r["proposed_unit_id"]
        for _, r in proposal.proposals.iterrows()
    }
    # The proposal rows should reflect the two distinct unit_ids.
    assert set(proposal.proposals["proposed_unit_id"]) == {
        "IN.ADM2.HP_HAM", "IN.ADM2.UP_HAM",
    }


def test_bundled_india_has_normalizer():
    # The India bundle ships a domain-specific normalizer. Without it,
    # the README quickstart hits 92% unmatched on Geolocet-style names.
    from stablebound import BUNDLED_COUNTRIES, normalize_name
    in_normalizer = BUNDLED_COUNTRIES["IN"].normalizer
    assert in_normalizer is not None
    # The India normalizer strips ' District' suffixes.
    assert in_normalizer("Hamirpur District") == "hamirpur"
    # The default normalizer doesn't.
    assert normalize_name("Hamirpur District") == "hamirpur district"


def test_a_country_without_a_normalizer_uses_the_package_default(exampleland):
    # Exampleland is not bundled and declares no normalizer, so the package
    # default applies unchanged: no suffix stripping.
    assert exampleland.default_normalizer("Hamirpur District") == "hamirpur district"


def test_lineage_propose_uses_bundled_normalizer():
    # Calling Lineage('IN').propose_shapefile_mapping without an
    # explicit normalizer should apply the India bundled one.
    from stablebound import Lineage
    ln = Lineage("IN")
    # `default_normalizer` exposes what would be used.
    assert ln.default_normalizer("Hamirpur District") == "hamirpur"
    # An explicit user normalizer overrides the bundled one.
    def custom(s):
        return s.upper()

    assert ln._resolve_normalizer(custom) is custom


def test_bundled_india_loads():
    from stablebound import BUNDLED_COUNTRIES
    from stablebound.io import read_relationship_table

    entry = BUNDLED_COUNTRIES["IN"]
    assert entry.lineage_path.exists(), f"missing bundled file: {entry.lineage_path}"
    assert entry.baseline_path.exists()
    assert entry.name_change_log_path is not None and entry.name_change_log_path.exists()

    rt = read_relationship_table(entry.lineage_path)
    # India has hundreds of events. Just check that we got a non-empty
    # frame with the canonical schema.
    assert len(rt) > 100
    assert {"event_year", "event_type", "parent_id", "child_id"} <= set(rt.columns)


# --- Q1: name-based year inference --------------------------------------


@pytest.fixture()
def year_inference_graph() -> LineageGraph:
    """RT with one Split in 2005: {Alpha} → {Alpha-N, Alpha-S}.

    Pre-2005 snapshot has 1 unit ("Alpha"); post-2005 has 2 units
    ("Alpha-N", "Alpha-S"). Year inference should distinguish them by
    name set.
    """
    rt = pd.DataFrame({
        "event_year": [2005, 2005],
        "event_type": ["Split", "Split"],
        "parent_id": ["XX.001", "XX.001"],
        "parent_name": ["Alpha", "Alpha"],
        "child_id": ["XX.002", "XX.003"],
        "child_name": ["Alpha-N", "Alpha-S"],
    })
    return LineageGraph.from_dataframe(rt)


def test_year_inference_picks_pre_event_year_for_pre_event_names(year_inference_graph):
    """Shapefile with one feature 'Alpha' → infer year ≤ 2005."""
    gdf = _make_shapefile_gdf(["Alpha"])
    proposal = propose_shapefile_mapping(
        gdf, year_inference_graph, name_column="name",
        year_range=range(2000, 2011),
    )
    assert proposal.inferred_year is not None
    assert proposal.inferred_year <= 2005, (
        f"expected pre-event year, got {proposal.inferred_year}"
    )
    # Sanity: the inferred-year snapshot is the one used for matching.
    assert proposal.proposals.iloc[0]["proposed_unit_id"] == "XX.001"


def test_year_inference_picks_post_event_year_for_post_event_names(year_inference_graph):
    gdf = _make_shapefile_gdf(["Alpha-N", "Alpha-S"])
    proposal = propose_shapefile_mapping(
        gdf, year_inference_graph, name_column="name",
        year_range=range(2000, 2011),
    )
    assert proposal.inferred_year is not None
    assert proposal.inferred_year > 2005, (
        f"expected post-event year, got {proposal.inferred_year}"
    )


def test_year_inference_tolerates_typos_via_fuzzy(year_inference_graph):
    """A typo in the shapefile name still resolves to the right era."""
    # "Alpha-Nrth" (typo on 'North' → 'Nrth') should still fuzzy-match.
    gdf = _make_shapefile_gdf(["Alpha-Nrth", "Alpha-S"])
    proposal = propose_shapefile_mapping(
        gdf, year_inference_graph, name_column="name",
        year_range=range(2000, 2011),
    )
    assert proposal.inferred_year is not None
    assert proposal.inferred_year > 2005


def test_year_inference_earliest_wins_ties(year_inference_graph):
    """When multiple years tie on mismatch, earliest year is picked."""
    # Pre-event years (2000..2004) all have the same name set {Alpha}.
    # A shapefile with just "Alpha" ties across all of them → 2000 wins.
    gdf = _make_shapefile_gdf(["Alpha"])
    proposal = propose_shapefile_mapping(
        gdf, year_inference_graph, name_column="name",
        year_range=range(2000, 2005),  # all pre-event
    )
    assert proposal.inferred_year == 2000


def test_explicit_year_bypasses_inference(year_inference_graph):
    """Passing year=N skips inference even when year_range is given."""
    gdf = _make_shapefile_gdf(["Alpha"])
    proposal = propose_shapefile_mapping(
        gdf, year_inference_graph, name_column="name",
        year=2010, year_range=range(2000, 2011),
    )
    assert proposal.inferred_year is None
    assert proposal.year_inference_mismatches is None


def test_no_year_range_no_inference_fallback_modern(year_inference_graph):
    """Without year_range, falls back to graph.max_event_year + 1."""
    gdf = _make_shapefile_gdf(["Alpha-N"])
    proposal = propose_shapefile_mapping(
        gdf, year_inference_graph, name_column="name",
    )
    assert proposal.inferred_year is None
    # Match was against year=2006 (max_event_year=2005 + 1).
    assert proposal.proposals.iloc[0]["proposed_unit_id"] == "XX.002"


def test_lineage_propose_uses_inference_by_default():
    """Lineage.propose_shapefile_mapping passes self.years automatically."""
    from stablebound import Lineage
    from tests.conftest import get_synthetic
    # A custom country whose modern names pin a year: the cascade fixture's
    # latest snapshot exists only from 2017 on.
    c = get_synthetic("cascade")
    ln = Lineage("CA", relationship_table_path=c.lineage_path, baseline_path=c.baseline_path)
    snap = ln.snapshot()
    gdf = _make_shapefile_gdf(snap["unit_name"].tolist())
    proposal = ln.propose_shapefile_mapping(gdf, name_column="name")
    assert proposal.inferred_year is not None, (
        "Lineage wrapper should auto-pass year_range and trigger inference."
    )


# --- Q3: homonym warning ------------------------------------------------


def test_homonym_warning_fires_without_coarse_column(homonym_graph):
    gdf = _make_shapefile_gdf(["Hamirpur", "Hamirpur"])
    with pytest.warns(UserWarning, match="homonym"):
        proposal = propose_shapefile_mapping(
            gdf, homonym_graph, name_column="name", year=2011,
        )
    # Notes should mention homonym + the alternative unit_ids.
    notes = proposal.proposals["notes"].iloc[0]
    assert "homonym" in notes
    assert "IN.ADM2.HP_HAM" in notes or "IN.ADM2.UP_HAM" in notes
    assert "coarse_column" in notes


def test_no_homonym_warning_when_coarse_resolves(homonym_graph):
    import warnings as _w
    gdf = _make_shapefile_gdf(
        ["Hamirpur", "Hamirpur"],
        states=["Himachal Pradesh", "Uttar Pradesh"],
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        propose_shapefile_mapping(
            gdf, homonym_graph, name_column="name",
            coarse_column="state", year=2011,
        )
    assert not any("homonym" in str(w.message) for w in caught), (
        f"unexpected homonym warning: {[str(w.message) for w in caught]}"
    )


def test_no_homonym_warning_when_homonym_overrides_used(homonym_graph):
    import warnings as _w
    gdf = _make_shapefile_gdf(["Hamirpur", "Hamirpur"])
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        propose_shapefile_mapping(
            gdf, homonym_graph, name_column="name", year=2011,
            homonym_overrides={
                0: ("IN.ADM2.HP_HAM", "Hamirpur"),
                1: ("IN.ADM2.UP_HAM", "Hamirpur"),
            },
        )
    assert not any("homonym" in str(w.message) for w in caught)


def test_homonym_warning_fires_for_stats_too(homonym_graph):
    stats = pd.DataFrame({
        "district": ["Hamirpur"],
        "year": [2011],
        "variable": ["rice"],
        "value": [1.0],
    })
    with pytest.warns(UserWarning, match="homonym"):
        propose_stats_mapping(
            stats, homonym_graph, name_column="district", year_column="year",
        )


# --- Ancestor-walk status ------------------------------------------------
#
# The walk used to have three silent-return paths. The one that mattered:
# when no year-compatible ancestor exists it kept the picked id, attaching
# data to a unit that did not exist that year with nothing flagged. These
# tests pin all four outcomes.


def _year_aware(stats, graph, baseline):
    return propose_stats_mapping(
        stats,
        graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=baseline,
    )


def test_ancestor_status_not_needed_when_pick_is_alive(
    exampleland_graph, exampleland_baseline
):
    stats = pd.DataFrame({"district": ["Bravo"], "year": [2015], "value": [1.0]})
    row = _year_aware(stats, exampleland_graph, exampleland_baseline).proposals.iloc[0]
    assert row["proposed_unit_id"] == "E.002"
    assert row["ancestor_status"] == "not-needed"


def test_ancestor_status_ok_for_single_ancestor(
    exampleland_graph, exampleland_baseline
):
    # "Alpha North" (E.011) is created by the 2014 split; reported in 2012 it
    # must walk to its single pre-split parent E.001.
    stats = pd.DataFrame({"district": ["Alpha North"], "year": [2012], "value": [1.0]})
    row = _year_aware(stats, exampleland_graph, exampleland_baseline).proposals.iloc[0]
    assert row["proposed_unit_id"] == "E.001"
    assert row["ancestor_status"] == "ok"


def test_ancestor_status_ambiguous_for_a_merge_child(
    exampleland_graph, exampleland_baseline
):
    # DeltaEcho (E.013) is the 2018 merge of Delta + Echo. Pre-merge, BOTH
    # parents are year-compatible, so the pick is a deterministic choice
    # among several — the reviewer should see that it was a choice.
    stats = pd.DataFrame({"district": ["DeltaEcho"], "year": [2012], "value": [1.0]})
    proposal = _year_aware(stats, exampleland_graph, exampleland_baseline)
    row = proposal.proposals.iloc[0]
    assert row["ancestor_status"] == "ambiguous"
    assert row["proposed_unit_id"] in {"E.004", "E.005"}
    assert "year-compatible ancestors" in row["notes"]
    assert len(proposal.ancestor_issues) == 1


def test_no_ancestor_row_is_dropped_not_silently_kept(
    exampleland_graph, exampleland_baseline
):
    # "Alpha" (E.001) ceases at the 2014 split. Reported in 2020 it is not
    # alive, and being an initial unit it has no ancestors either — so there
    # is no correct id. The row must become unmatched rather than keep E.001.
    stats = pd.DataFrame({"district": ["Alpha"], "year": [2020], "value": [1.0]})
    proposal = _year_aware(stats, exampleland_graph, exampleland_baseline)
    row = proposal.proposals.iloc[0]

    assert row["ancestor_status"] == "no-ancestor"
    assert row["method"] == "unmatched"
    assert row["proposed_unit_id"] == ""
    # Nothing is lost: the rejected candidate is still there for the reviewer.
    assert row["best_candidate"] == "Alpha"
    assert "no year-compatible ancestor" in row["notes"]
    assert len(proposal.ancestor_issues) == 1
    assert len(proposal.unmatched) == 1


def test_ancestor_issues_surface_in_summary(
    exampleland_graph, exampleland_baseline
):
    stats = pd.DataFrame(
        {"district": ["Alpha", "DeltaEcho"], "year": [2020, 2012], "value": [1.0, 2.0]}
    )
    text = _year_aware(stats, exampleland_graph, exampleland_baseline).summary()
    assert "Ancestor-walk issues" in text
    assert "1 dropped" in text
    assert "1 ambiguous" in text


def test_ancestor_status_column_present_for_non_year_aware_paths(
    exampleland_graph, exampleland_baseline
):
    # The column exists everywhere so the review CSV has one stable shape.
    stats = pd.DataFrame({"district": ["Bravo"], "year": [2015], "value": [1.0]})
    p = propose_stats_mapping(
        stats, exampleland_graph, name_column="district", baseline=exampleland_baseline
    )
    assert (p.proposals["ancestor_status"] == "not-needed").all()


def test_stats_years_beyond_the_lineage_window_still_match(
    exampleland_graph, exampleland_baseline
):
    """A unit is presumed to persist past the last recorded event.

    Statistics routinely outrun their relationship table — Philippines
    reports to 2024 from a lineage ending in 2013. The year-aware alive-index
    used to span only the lineage's own window, so every row in a later year
    read as "not alive"; the ancestor walk then found nothing (no ancestor is
    alive in a year the index doesn't cover either) and the no-ancestor policy
    dropped the row. On Philippines that was 753 of 1,215 keys lost.

    Absence of evidence of change is not evidence of death.
    """
    # Exampleland's last event is 2018; ask about 2030.
    stats = pd.DataFrame(
        {"district": ["Bravo", "Bravo"], "year": [2015, 2030], "value": [1.0, 2.0]}
    )
    proposal = propose_stats_mapping(
        stats,
        exampleland_graph,
        name_column="district",
        year_column="year",
        year_aware=True,
        baseline=exampleland_baseline,
    )
    by_year = {int(r["source_year"]): r for _, r in proposal.proposals.iterrows()}
    assert by_year[2015]["proposed_unit_id"] == "E.002"
    assert by_year[2030]["proposed_unit_id"] == "E.002", (
        "a unit with no later events must still match beyond the lineage window"
    )
    assert by_year[2030]["ancestor_status"] == "not-needed"
    assert proposal.ancestor_issues.empty


def test_a_manual_override_naming_a_unit_id_warns_instead_of_doing_nothing():
    """The value is a lineage NAME; an id silently matched nothing.

    Found in the Philippines dry run: `{"mountain province": "PH.ADM1.00047"}`
    left the match count unchanged and emitted no warning, so there was no way
    to tell the override had been read at all. A natural first guess must not
    fail silently.
    """
    import warnings as _w

    import geopandas as gpd
    from shapely.geometry import Polygon

    from stablebound.lineage import LineageGraph
    from stablebound.match import propose_shapefile_mapping

    graph = LineageGraph.from_dataframe(
        pd.DataFrame(
            [(2014, "Split", "Z.001", "Alpha", "Z.011", "Mountain")],
            columns=["event_year", "event_type", "parent_id", "parent_name",
                     "child_id", "child_name"],
        )
    )
    gdf = gpd.GeoDataFrame(
        {"NAME": ["Mountain Province"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326",
    )

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        propose_shapefile_mapping(
            gdf, graph, name_column="NAME", year=2015,
            manual_overrides={"mountain province": "Z.011"},   # an id: wrong
        )
    msgs = [str(c.message) for c in caught if "manual_overrides" in str(c.message)]
    assert len(msgs) == 1, msgs
    assert "Z.011" in msgs[0]
    assert "not a unit_id" in msgs[0]

    # The documented form works and emits no such warning.
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        prop = propose_shapefile_mapping(
            gdf, graph, name_column="NAME", year=2015,
            manual_overrides={"mountain province": "Mountain"},
        )
    assert not [c for c in caught if "manual_overrides" in str(c.message)]
    assert prop.proposals.iloc[0]["proposed_unit_id"] == "Z.011"


def test_a_rename_in_the_lineage_aliases_like_one_in_a_name_change_log():
    """The two encodings of the same fact must behave the same.

    A rename can be recorded either as a NameChange row in the lineage (the
    canonical schema) or in a separate name-change log. Before this, only the
    log produced backward aliases: matching 'Oldname' against a post-rename
    year succeeded with a log and failed without one, penalising exactly the
    countries that follow the canonical schema. VN/TH/BD/KR come out of
    rt_convert with NameChange events and no log at all.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    from stablebound.lineage import LineageGraph
    from stablebound.match import propose_shapefile_mapping

    graph = LineageGraph.from_dataframe(
        pd.DataFrame(
            [(2016, "NameChange", "X.001", "Oldname", "X.001", "Newname")],
            columns=["event_year", "event_type", "parent_id", "parent_name",
                     "child_id", "child_name"],
        )
    )
    baseline = pd.DataFrame(
        {"unit_id": ["X.001"], "name": ["Oldname"], "year": [2010]}
    )
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    def match(name, year):
        gdf = gpd.GeoDataFrame({"NAME": [name]}, geometry=[poly], crs="EPSG:4326")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prop = propose_shapefile_mapping(
                gdf, graph, name_column="NAME", year=year, baseline=baseline
            )
        return prop.proposals.iloc[0]["proposed_unit_id"]

    # Both names resolve in both directions: statistics carry historical names,
    # shapefiles (GAUL) carry current ones, and either can be matched against
    # either era.
    assert match("Oldname", 2012) == "X.001", "pre-rename name, pre-rename year"
    assert match("Newname", 2020) == "X.001", "post-rename name, post-rename year"
    assert match("Oldname", 2020) == "X.001", "historical name, modern year"
    assert match("Newname", 2012) == "X.001", "modern name, historical year"


def test_graph_aliasing_does_not_invent_matches_for_unrelated_names():
    """The alias must not be a licence to match anything."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    from stablebound.lineage import LineageGraph
    from stablebound.match import propose_shapefile_mapping

    graph = LineageGraph.from_dataframe(
        pd.DataFrame(
            [(2016, "NameChange", "X.001", "Oldname", "X.001", "Newname")],
            columns=["event_year", "event_type", "parent_id", "parent_name",
                     "child_id", "child_name"],
        )
    )
    baseline = pd.DataFrame(
        {"unit_id": ["X.001"], "name": ["Oldname"], "year": [2010]}
    )
    gdf = gpd.GeoDataFrame(
        {"NAME": ["Somewhere Else Entirely"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop = propose_shapefile_mapping(
            gdf, graph, name_column="NAME", year=2020, baseline=baseline
        )
    assert prop.proposals.iloc[0]["method"] == "unmatched"


def test_india_normalizer_strips_the_desagri_state_suffix():
    """DESAGRI writes "Villupuram (TN)"; without stripping, nothing matches.

    Measured on the real statistics: the documented path
    (`Lineage("IN").propose_stats_mapping`) resolved 13,503 of 17,049
    combinations (79%) while the bespoke India matcher resolved 16,468, and
    almost the entire gap was this suffix rather than any algorithmic
    difference. With it stripped the generic matcher reaches 16,616 (97.5%).
    See qualification/compare_matchers.py.
    """
    from stablebound.data import BUNDLED_COUNTRIES

    norm = BUNDLED_COUNTRIES["IN"].normalizer
    assert norm("Villupuram (TN)") == "villupuram"
    assert norm("Beed (MH)") == "beed"
    assert norm("Saran (BR)") == "saran"

    # Narrow on purpose: only 2-3 capitals in trailing parens, so a genuine
    # parenthetical in a unit name survives.
    assert norm("Bilaspur") == "bilaspur"
    assert norm("North And Middle Andaman") == "north and middle andaman"
    assert "kanpur" in norm("Kanpur (Rural)"), norm("Kanpur (Rural)")


def test_attach_shapefile_ids_uses_coarse_to_split_homonyms():
    """Two districts with one name must not collapse onto one id.

    This is stablebound-dev#4 in miniature. India has two Bilaspurs, one in
    Himachal Pradesh and one in Chhattisgarh. Joined on name alone, both take
    the first id -- which does not merely mislabel a polygon, it dissolves it
    into a stable group 1,100 km away, so the shipped geometry is wrong rather
    than just its label.

    The positional join already handles this; the name-based fallback, used for
    hand-built mappings, did not.
    """
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Bilaspur", "Bilaspur"],
            "state": ["Himachal Pradesh", "Chhattisgarh"],
            "geometry": [Point(76.7, 31.3).buffer(0.1), Point(82.1, 22.1).buffer(0.1)],
        },
        crs="EPSG:4326",
    )
    mapping = pd.DataFrame(
        {
            "source_name": ["Bilaspur", "Bilaspur"],
            "source_coarse": ["Himachal Pradesh", "Chhattisgarh"],
            "proposed_unit_id": ["IN.ADM2.00144", "IN.ADM2.00994"],
        }
    )

    # Without the state column the join collapses them, and says so.
    with pytest.warns(UserWarning, match="coarse_column"):
        collapsed = attach_shapefile_ids(gdf, mapping, name_column="name")
    assert collapsed["unit_id"].nunique() == 1, "expected the homonym collapse"

    # With it, each polygon keeps its own id.
    out = attach_shapefile_ids(gdf, mapping, name_column="name", coarse_column="state")
    assert list(out["unit_id"]) == ["IN.ADM2.00144", "IN.ADM2.00994"]


def test_snapshot_lookup_carries_the_states_of_its_own_year():
    """A lookup built for year Y must speak Y's state vocabulary.

    The baseline records each district's state as of the start of the record.
    Reading it in preference to the lineage means a lookup built for 2024 hands
    back 1991 names -- India's returned "Orissa" for three districts and
    "Pondicherry U.T." for four, decades after both states were renamed. A
    modern shapefile says Odisha and Puducherry, so those are precisely the
    districts a state-keyed join drops on the floor.

    Both routes into the coarse name are checked, because they fail
    separately: a district the lineage knows about, and one that appears
    nowhere but the baseline.
    """
    from stablebound.match import _build_snapshot_lookup

    df = pd.DataFrame(
        [
            [2011, "NameChange", "S1", "Oldland", "S1", "Newland", None, None, None, None],
            [2000, "Split", "P", "P", "HASEVENT", "Hasevent", "S1", "Oldland", "S1", "Oldland"],
        ],
        columns=[
            "event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name",
            "parent_coarse_id", "parent_coarse_name", "child_coarse_id", "child_coarse_name",
        ],
    )
    graph = LineageGraph.from_dataframe(df)
    # Both districts are filed under the state's ORIGINAL name, which is what a
    # baseline frozen at the start of the record actually looks like.
    baseline = pd.DataFrame(
        {
            "unit_id": ["HASEVENT", "NOEVENT"],
            "name": ["Hasevent", "Noevent"],
            "coarse_id": ["S1", "S1"],
            "coarse_name": ["Oldland", "Oldland"],
        }
    )

    lookup = _build_snapshot_lookup(graph, 2024, baseline=baseline)
    states = {entry[0]: entry[2] for bucket in lookup.values() for entry in bucket}

    assert states["HASEVENT"] == "Newland", "the lineage's year-aware state lost to the baseline"
    assert states["NOEVENT"] == "Newland", (
        "a district present only in the baseline never heard about the rename"
    )
    assert "Oldland" not in set(states.values())
