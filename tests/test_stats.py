"""Long-form aggregation + derive_intensive."""

from __future__ import annotations

import pandas as pd

from stablebound.lineage import LineageGraph
from stablebound.stats import aggregate, derive_intensive


def _stats(rows):
    cols = ["unit_id", "year", "season", "variable", "value"]
    return pd.DataFrame(rows, columns=cols)


def _empty_graph() -> LineageGraph:
    """Graph with no events; every unit_id present in remap is treated as
    alive at every year (via the additional_units flow in build_snapshot).
    Use for tests where the snapshot semantics aren't being exercised.
    """
    return LineageGraph.from_dataframe(
        pd.DataFrame(columns=[
            "event_year", "event_type",
            "parent_id", "parent_name", "child_id", "child_name",
        ])
    )


def _graph(rows) -> LineageGraph:
    """Build a graph from a list of (year, type, p_id, p_name, c_id, c_name) tuples."""
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name"]
    return LineageGraph.from_dataframe(pd.DataFrame(rows, columns=cols))


def test_aggregate_sums_constituents_into_stable_polygon():
    stats = _stats(
        [
            ("A1", 2018, "Annual", "area_ha", 60.0),
            ("A2", 2018, "Annual", "area_ha", 40.0),
            ("A1", 2018, "Annual", "production_mt", 120.0),
            ("A2", 2018, "Annual", "production_mt", 80.0),
        ]
    )
    remap = {"A1": "A", "A2": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020)
    assert len(out) == 2
    by_var = {row["variable"]: row for _, row in out.iterrows()}
    assert by_var["area_ha"]["value"] == 100.0
    assert by_var["area_ha"]["n_constituents"] == 2
    assert by_var["area_ha"]["constituent_ids"] == "A1,A2"
    assert by_var["production_mt"]["value"] == 200.0


def test_aggregate_handles_early_reporting_implicitly_via_remap():
    # A unit reports in 2010, but the modern shapefile + remap already include
    # its stable_id. The aggregation routes it correctly with no special case.
    stats = _stats(
        [
            ("FUTURE_UNIT", 2010, "Annual", "area_ha", 50.0),
        ]
    )
    remap = {"FUTURE_UNIT": "PARENT"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020)
    assert len(out) == 1
    assert out.iloc[0]["stable_id"] == "PARENT"
    assert out.iloc[0]["value"] == 50.0
    assert out.iloc[0]["late_reporting"] == False  # noqa: E712


def test_aggregate_flags_late_reporting_for_units_not_in_remap():
    # Stats include a unit_id absent from the remap (a unit reporting after
    # dissolution, or simply unmatched). We surface it via late_reporting=True
    # rather than silently dropping it.
    stats = _stats(
        [
            ("A1", 2018, "Annual", "area_ha", 60.0),
            ("DEAD_UNIT", 2018, "Annual", "area_ha", 7.0),
        ]
    )
    remap = {"A1": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020)
    by_stable = {row["stable_id"]: row for _, row in out.iterrows()}
    assert by_stable["A"]["late_reporting"] == False  # noqa: E712
    assert by_stable["DEAD_UNIT"]["late_reporting"] == True  # noqa: E712
    assert by_stable["DEAD_UNIT"]["value"] == 7.0


def test_aggregate_rejects_pre_base_year_rows():
    import pytest

    stats = _stats(
        [
            ("A1", 2005, "Annual", "area_ha", 50.0),
        ]
    )
    with pytest.raises(Exception):
        aggregate(stats, remap={"A1": "A"}, graph=_empty_graph(), base_year=2010, max_year=2020)


def test_aggregate_groups_by_year_season_variable():
    stats = _stats(
        [
            ("A1", 2018, "Kharif", "area_ha", 60.0),
            ("A2", 2018, "Kharif", "area_ha", 40.0),
            ("A1", 2018, "Rabi", "area_ha", 30.0),
            ("A2", 2018, "Rabi", "area_ha", 20.0),
            ("A1", 2019, "Kharif", "area_ha", 65.0),
            ("A2", 2019, "Kharif", "area_ha", 45.0),
        ]
    )
    remap = {"A1": "A", "A2": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020)
    assert len(out) == 3
    rows = out.set_index(["year", "season"])["value"].to_dict()
    assert rows[(2018, "Kharif")] == 100.0
    assert rows[(2018, "Rabi")] == 50.0
    assert rows[(2019, "Kharif")] == 110.0


def test_derive_intensive_recomputes_yield_from_aggregated_extensives():
    agg = pd.DataFrame(
        [
            {
                "year": 2018, "season": "Annual", "variable": "production_mt",
                "stable_id": "A", "value": 200.0, "n_constituents": 2,
                "constituent_ids": "A1,A2", "late_reporting": False,
            },
            {
                "year": 2018, "season": "Annual", "variable": "area_ha",
                "stable_id": "A", "value": 100.0, "n_constituents": 2,
                "constituent_ids": "A1,A2", "late_reporting": False,
            },
        ]
    )
    out = derive_intensive(agg, intensive_pairs={"yield_mt_ha": ("production_mt", "area_ha")})
    yield_rows = out[out["variable"] == "yield_mt_ha"]
    assert len(yield_rows) == 1
    assert yield_rows.iloc[0]["value"] == 2.0  # 200 / 100


def test_derive_intensive_handles_zero_denominator():
    agg = pd.DataFrame(
        [
            {
                "year": 2018, "season": "Annual", "variable": "production_mt",
                "stable_id": "A", "value": 100.0, "n_constituents": 1,
                "constituent_ids": "A", "late_reporting": False,
            },
            {
                "year": 2018, "season": "Annual", "variable": "area_ha",
                "stable_id": "A", "value": 0.0, "n_constituents": 1,
                "constituent_ids": "A", "late_reporting": False,
            },
        ]
    )
    out = derive_intensive(agg, intensive_pairs={"yield_mt_ha": ("production_mt", "area_ha")})
    yield_rows = out[out["variable"] == "yield_mt_ha"]
    assert len(yield_rows) == 1
    assert pd.isna(yield_rows.iloc[0]["value"])


def test_late_reporting_post_dissolution_is_flagged():
    """A → B,C in 2010. A reports in 2012 (post-dissolution + past grace) → flag."""
    graph = _graph([
        (2010, "Split", "A", "A", "B", "B"),
        (2010, "Split", "A", "A", "C", "C"),
    ])
    stats = _stats([("A", 2012, "Annual", "area_ha", 50.0)])
    remap = {"A": "A", "B": "B", "C": "C"}   # post-split base; A is singleton
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = aggregate(stats, remap, graph, base_year=2010, max_year=2015)
    a_row = out[(out["stable_id"] == "A") & (out["year"] == 2012)]
    assert len(a_row) == 1
    assert bool(a_row.iloc[0]["late_reporting"]) is True
    # Warning surfaced.
    assert any("late-reporting" in str(w.message) for w in caught)


def test_early_reporting_pre_creation_is_flagged():
    """B is created from A in 2015. B reports in 2010 → flagged early."""
    graph = _graph([
        (2015, "Split", "A", "A", "B", "B"),
        (2015, "Split", "A", "A", "C", "C"),
    ])
    stats = _stats([("B", 2010, "Annual", "area_ha", 30.0)])
    remap = {"A": "A", "B": "A", "C": "A"}
    out = aggregate(stats, remap, graph, base_year=2010, max_year=2018)
    b_row = out[out["year"] == 2010]
    assert len(b_row) == 1
    assert bool(b_row.iloc[0]["late_reporting"]) is True
    assert b_row.iloc[0]["stable_id"] == "B"   # surfaced under own id, not re-routed


def test_event_year_grace_does_not_flag():
    """A → B,C in 2014. B reports in the event year 2014 → grace, not flagged."""
    graph = _graph([
        (2014, "Split", "A", "A", "B", "B"),
        (2014, "Split", "A", "A", "C", "C"),
    ])
    stats = _stats([
        ("B", 2014, "Annual", "area_ha", 60.0),   # event-year report → grace
        ("A", 2014, "Annual", "area_ha", 100.0),  # event-year report → grace (parent)
    ])
    remap = {"A": "A", "B": "A", "C": "A"}
    out = aggregate(stats, remap, graph, base_year=2010, max_year=2018)
    assert (out["late_reporting"] == False).all()  # noqa: E712


def test_completeness_columns_for_partial_reporting():
    """3-member group, only 2 report → n_in_group=3, complete=False, missing names the absent one."""
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 60.0),
        ("A2", 2018, "Annual", "area_ha", 40.0),
        # A3 doesn't report
    ])
    remap = {"A1": "A", "A2": "A", "A3": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_constituents"] == 2
    assert row["n_in_group"] == 3
    assert row["complete"] == False  # noqa: E712
    assert row["missing_unit_ids"] == "A3"


def test_completeness_columns_for_full_reporting():
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 60.0),
        ("A2", 2018, "Annual", "area_ha", 40.0),
    ])
    remap = {"A1": "A", "A2": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020)
    row = out.iloc[0]
    assert row["n_constituents"] == 2
    assert row["n_in_group"] == 2
    assert row["complete"] == True  # noqa: E712
    assert row["missing_unit_ids"] == ""


def test_completeness_columns_NA_for_late_reporting_rows():
    """Late-reporting rows get NA completeness (concept doesn't apply)."""
    graph = _graph([
        (2010, "Split", "A", "A", "B", "B"),
        (2010, "Split", "A", "A", "C", "C"),
    ])
    stats = _stats([("A", 2013, "Annual", "area_ha", 50.0)])
    remap = {"A": "A", "B": "B", "C": "C"}
    out = aggregate(stats, remap, graph, base_year=2010, max_year=2015)
    row = out.iloc[0]
    assert bool(row["late_reporting"]) is True
    assert pd.isna(row["n_in_group"])
    assert pd.isna(row["complete"])
    assert row["missing_unit_ids"] == ""


def test_nan_season_raises_schema_error():
    """Strict: NaN seasons raise SchemaError with directive to use a sentinel."""
    import pytest
    from stablebound.schemas import SchemaError
    stats = pd.DataFrame({
        "unit_id": ["A"],
        "year": [2010],
        "season": [None],
        "variable": ["area_ha"],
        "value": [100.0],
    })
    with pytest.raises(SchemaError, match=r"missing season.*Annual"):
        aggregate(stats, remap={"A": "A"}, graph=_empty_graph(),
                  base_year=2010, max_year=2020)


def test_derive_intensive_drops_existing_intensive_rows_before_recomputing():
    # If the input already has rows for the intensive variable (perhaps
    # mistakenly summed by aggregate), they get replaced — never averaged.
    agg = pd.DataFrame(
        [
            {
                "year": 2018, "season": "Annual", "variable": "yield_mt_ha",
                "stable_id": "A", "value": 99.0, "n_constituents": 2,
                "constituent_ids": "A1,A2", "late_reporting": False,
            },
            {
                "year": 2018, "season": "Annual", "variable": "production_mt",
                "stable_id": "A", "value": 200.0, "n_constituents": 2,
                "constituent_ids": "A1,A2", "late_reporting": False,
            },
            {
                "year": 2018, "season": "Annual", "variable": "area_ha",
                "stable_id": "A", "value": 100.0, "n_constituents": 2,
                "constituent_ids": "A1,A2", "late_reporting": False,
            },
        ]
    )
    out = derive_intensive(agg, intensive_pairs={"yield_mt_ha": ("production_mt", "area_ha")})
    yield_rows = out[out["variable"] == "yield_mt_ha"]
    assert len(yield_rows) == 1
    assert yield_rows.iloc[0]["value"] == 2.0


# --- Completeness denominator: strict snapshot, not the grace window -----
#
# `late_reporting` uses a one-year grace window so the normal parent->child
# handoff at an event year isn't flagged. Completeness must NOT: a parent
# ceasing during year T and its children first existing at T+1 are not both
# members of year T. Reproduces the Gujarat 1997 case found on real India
# data, where two parents split jointly into a shared child.


def _joint_split_graph() -> LineageGraph:
    """Two parents split during 2014, jointly creating a shared child.

    P1 -> {C1, SHARED}, P2 -> {C2, SHARED}. The shared child forces both
    parents' ancestor groups into one stable group (Union-Find), mirroring
    Banas Kantha + Mahesana -> Patan.
    """
    return _graph([
        (2014, "Split", "P1", "Parent One", "C1", "Child One"),
        (2014, "Split", "P1", "Parent One", "SHARED", "Shared Child"),
        (2014, "Split", "P2", "Parent Two", "C2", "Child Two"),
        (2014, "Split", "P2", "Parent Two", "SHARED", "Shared Child"),
    ])


def _joint_split_remap() -> dict[str, str]:
    return {u: "P1" for u in ("P1", "P2", "C1", "C2", "SHARED")}


def test_event_year_completeness_counts_only_units_alive_that_year():
    graph = _joint_split_graph()
    # Both parents report in the event year. The children do not exist yet
    # (event_year=2014 => children first appear in the 2014+1 snapshot).
    stats = _stats([
        ("P1", 2014, "Annual", "area_ha", 10.0),
        ("P2", 2014, "Annual", "area_ha", 20.0),
    ])
    out = aggregate(stats, _joint_split_remap(), graph, base_year=2010, max_year=2018)
    row = out[(out["year"] == 2014) & (out["variable"] == "area_ha")].iloc[0]

    assert row["n_constituents"] == 2
    # Strict snapshot(2014) = {P1, P2}. The grace window would have added
    # C1, C2 and SHARED and reported them as missing 2014 data.
    assert row["n_in_group"] == 2
    assert bool(row["complete"]) is True
    assert row["missing_unit_ids"] == ""


def test_post_event_year_completeness_counts_the_children():
    graph = _joint_split_graph()
    stats = _stats([
        ("C1", 2015, "Annual", "area_ha", 10.0),
        ("C2", 2015, "Annual", "area_ha", 20.0),
    ])
    out = aggregate(stats, _joint_split_remap(), graph, base_year=2010, max_year=2018)
    row = out[(out["year"] == 2015) & (out["variable"] == "area_ha")].iloc[0]

    # snapshot(2015) = {C1, C2, SHARED}; SHARED genuinely didn't report.
    assert row["n_in_group"] == 3
    assert bool(row["complete"]) is False
    assert row["missing_unit_ids"] == "SHARED"


def test_complete_never_disagrees_with_missing_unit_ids():
    # `complete` is a subset test, not a cardinality comparison. An early
    # reporter (alive only from 2015, reporting in 2014) makes the counts
    # match while the sets differ; the old `len(reporters) == len(alive)`
    # would have called this complete on a coincidence.
    graph = _joint_split_graph()
    stats = _stats([
        ("P1", 2014, "Annual", "area_ha", 10.0),
        ("C1", 2014, "Annual", "area_ha", 5.0),   # early reporter
    ])
    out = aggregate(stats, _joint_split_remap(), graph, base_year=2010, max_year=2018)
    live = out[~out["late_reporting"].astype(bool)]
    for _, row in live.iterrows():
        assert bool(row["complete"]) == (row["missing_unit_ids"] == "")


# --- Intensive completeness ----------------------------------------------
#
# derive_intensive used to hardcode every completeness column to NA/"", so
# yield — usually the headline variable — carried no completeness at all.
# A ratio is only as trustworthy as the scarcer of its two inputs.


def _two_unit_graph() -> LineageGraph:
    return _graph([(2014, "Split", "P", "Parent", "C1", "Child One"),
                   (2014, "Split", "P", "Parent", "C2", "Child Two")])


def test_intensive_carries_completeness_from_both_inputs():
    graph = _two_unit_graph()
    remap = {u: "P" for u in ("P", "C1", "C2")}
    # 2015: C1 and C2 are both alive. Both report area; only C1 reports
    # production. The yield is therefore based on C1 alone.
    stats = _stats([
        ("C1", 2015, "Annual", "area_ha", 10.0),
        ("C2", 2015, "Annual", "area_ha", 30.0),
        ("C1", 2015, "Annual", "production_mt", 20.0),
    ])
    agg = aggregate(stats, remap, graph, base_year=2010, max_year=2018)
    out = derive_intensive(agg, {"yield": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield"].iloc[0]

    # Contributors are the INTERSECTION: only C1 reported both.
    assert y["constituent_ids"] == "C1"
    assert y["n_constituents"] == 1
    assert y["n_in_group"] == 2
    # Missing is the UNION of what each side lacked: production is missing C2.
    assert y["missing_unit_ids"] == "C2"
    assert bool(y["complete"]) is False
    assert y["completeness"] == 0.5


def test_intensive_is_complete_when_both_inputs_are():
    graph = _two_unit_graph()
    remap = {u: "P" for u in ("P", "C1", "C2")}
    stats = _stats([
        ("C1", 2015, "Annual", "area_ha", 10.0),
        ("C2", 2015, "Annual", "area_ha", 30.0),
        ("C1", 2015, "Annual", "production_mt", 20.0),
        ("C2", 2015, "Annual", "production_mt", 60.0),
    ])
    agg = aggregate(stats, remap, graph, base_year=2010, max_year=2018)
    out = derive_intensive(agg, {"yield": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield"].iloc[0]
    assert bool(y["complete"]) is True
    assert y["missing_unit_ids"] == ""
    assert y["completeness"] == 1.0
    assert y["value"] == 2.0                       # 80 / 40


def test_completeness_ratio_on_extensive_rows():
    graph = _two_unit_graph()
    remap = {u: "P" for u in ("P", "C1", "C2")}
    stats = _stats([("C1", 2015, "Annual", "area_ha", 10.0)])
    agg = aggregate(stats, remap, graph, base_year=2010, max_year=2018)
    row = agg.iloc[0]
    assert row["n_constituents"] == 1
    assert row["n_in_group"] == 2
    assert row["completeness"] == 0.5


def test_completeness_is_na_not_a_crash_for_late_rows():
    # Late rows have n_in_group = NA; `pd.NA and x` raises, so the ratio has
    # to null-check before testing truthiness.
    graph = _two_unit_graph()
    remap = {u: "P" for u in ("P", "C1", "C2")}
    stats = _stats([("P", 2017, "Annual", "area_ha", 5.0)])   # P ceased in 2014
    agg = aggregate(stats, remap, graph, base_year=2010, max_year=2018)
    late = agg[agg["late_reporting"].astype(bool)]
    assert len(late) == 1
    assert pd.isna(late.iloc[0]["completeness"])


def test_derive_intensive_without_completeness_columns_degrades_to_na():
    # The modern product's frame has no completeness columns and round-trips
    # through derive_intensive. It must not raise.
    frame = pd.DataFrame({
        "year": [2015, 2015], "season": ["Annual", "Annual"],
        "variable": ["production_mt", "area_ha"], "stable_id": ["A", "A"],
        "value": [20.0, 10.0], "late_reporting": [False, False],
    })
    out = derive_intensive(frame, {"yield": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield"].iloc[0]
    assert y["value"] == 2.0
    assert pd.isna(y["completeness"])


def test_completeness_never_exceeds_one_with_early_reporters():
    """Coverage of the expected set, not a raw reporter count.

    An early reporter (alive only from 2015, reporting in 2014) is counted in
    n_constituents but is not in the strict snapshot(2014), so
    n_constituents / n_in_group would exceed 1.0. Completeness measures how
    much of the expected set reported, so it stays in [0, 1] and agrees with
    `complete` by construction.
    """
    graph = _joint_split_graph()
    stats = _stats([
        ("P1", 2014, "Annual", "area_ha", 10.0),
        ("P2", 2014, "Annual", "area_ha", 20.0),
        ("C1", 2014, "Annual", "area_ha", 5.0),   # early reporter
    ])
    out = aggregate(stats, _joint_split_remap(), graph, base_year=2010, max_year=2018)
    live = out[~out["late_reporting"].astype(bool)]
    row = live.iloc[0]
    assert row["n_constituents"] == 3          # three units reported
    assert row["n_in_group"] == 2              # but only two were expected
    assert row["completeness"] == 1.0          # and both of them did report
    assert bool(row["complete"]) is True
    for _, r in live.iterrows():
        assert 0.0 <= float(r["completeness"]) <= 1.0


# --- Intensives computed inside aggregate(): units reporting both ----------
#
# v0.1.4. A ratio built from the full sums is biased whenever a unit reports
# one input and not the other. aggregate(intensive=...) therefore builds each
# ratio from its own aggregation of only the units that reported both, and the
# completeness columns of the derived row describe exactly that set.


def test_aggregate_intensive_uses_only_units_reporting_both():
    """A1: area 100, production 200. A2: production 800 only.

    Yield must be 2.0 (A1 alone), not 10.0 (all production over A1's area).
    """
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 100.0),
        ("A1", 2018, "Annual", "production_mt", 200.0),
        ("A2", 2018, "Annual", "production_mt", 800.0),
    ])
    remap = {"A1": "A", "A2": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020,
                    intensive={"yield_mt_ha": ("production_mt", "area_ha")})
    by_var = out.set_index("variable")
    # Extensive rows keep every reporter.
    assert by_var.loc["area_ha", "value"] == 100.0
    assert by_var.loc["production_mt", "value"] == 1000.0
    y = by_var.loc["yield_mt_ha"]
    assert y["value"] == 2.0
    assert y["n_constituents"] == 1
    assert y["constituent_ids"] == "A1"
    assert y["n_in_group"] == 2
    assert y["missing_unit_ids"] == "A2"
    assert bool(y["complete"]) is False
    assert y["completeness"] == 0.5


def test_aggregate_intensive_equals_post_hoc_ratio_when_all_report_both():
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 60.0),
        ("A2", 2018, "Annual", "area_ha", 40.0),
        ("A1", 2018, "Annual", "production_mt", 120.0),
        ("A2", 2018, "Annual", "production_mt", 100.0),
    ])
    remap = {"A1": "A", "A2": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020,
                    intensive={"yield_mt_ha": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield_mt_ha"].iloc[0]
    assert abs(y["value"] - 2.2) < 1e-12
    assert y["constituent_ids"] == "A1,A2"
    assert bool(y["complete"]) is True
    assert y["completeness"] == 1.0


def test_aggregate_intensive_nan_input_does_not_count_as_reporting():
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 100.0),
        ("A1", 2018, "Annual", "production_mt", 200.0),
        ("A2", 2018, "Annual", "area_ha", 50.0),
        ("A2", 2018, "Annual", "production_mt", float("nan")),
    ])
    remap = {"A1": "A", "A2": "A"}
    out = aggregate(stats, remap, _empty_graph(), base_year=2010, max_year=2020,
                    intensive={"yield_mt_ha": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield_mt_ha"].iloc[0]
    assert y["value"] == 2.0
    assert y["constituent_ids"] == "A1"
    assert y["missing_unit_ids"] == "A2"


def test_aggregate_drops_input_rows_carrying_an_intensive_name():
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 100.0),
        ("A1", 2018, "Annual", "production_mt", 200.0),
        ("A1", 2018, "Annual", "yield_mt_ha", 99.0),   # a yield the source shipped
    ])
    out = aggregate(stats, {"A1": "A"}, _empty_graph(), base_year=2010, max_year=2020,
                    intensive={"yield_mt_ha": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield_mt_ha"]
    assert len(y) == 1
    assert y.iloc[0]["value"] == 2.0


def test_aggregate_intensive_absent_when_no_unit_reports_both():
    stats = _stats([
        ("A1", 2018, "Annual", "area_ha", 100.0),
        ("A2", 2018, "Annual", "production_mt", 800.0),
    ])
    out = aggregate(stats, {"A1": "A", "A2": "A"}, _empty_graph(), base_year=2010,
                    max_year=2020, intensive={"yield_mt_ha": ("production_mt", "area_ha")})
    assert "yield_mt_ha" not in set(out["variable"])
    assert len(out) == 2


def test_aggregate_intensive_late_rows_pair_only_with_late_rows():
    """A ceased in 2010; its 2013 area and production are both late.

    They pair with each other under stable_id=A, late_reporting=True — never
    with the live rows of the group A once belonged to.
    """
    graph = _graph([(2010, "Split", "A", "A", "B", "B"),
                    (2010, "Split", "A", "A", "C", "C")])
    stats = _stats([
        ("A", 2013, "Annual", "area_ha", 50.0),
        ("A", 2013, "Annual", "production_mt", 100.0),
        ("B", 2013, "Annual", "area_ha", 10.0),
        ("B", 2013, "Annual", "production_mt", 40.0),
    ])
    remap = {"A": "A", "B": "B", "C": "B"}
    out = aggregate(stats, remap, graph, base_year=2010, max_year=2015,
                    intensive={"yield_mt_ha": ("production_mt", "area_ha")})
    y = out[out["variable"] == "yield_mt_ha"].set_index("stable_id")
    assert bool(y.loc["A", "late_reporting"]) is True
    assert y.loc["A", "value"] == 2.0
    assert bool(y.loc["B", "late_reporting"]) is False
    assert y.loc["B", "value"] == 4.0
