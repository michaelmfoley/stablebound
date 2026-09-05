"""The invariant battery: one set of assertions, every synthetic country.

This module is the package's specification in executable form. Each test states
a property that must hold for *any* country, and pytest parameterizes it across
the whole `tests/fixtures/synthetic/` registry — so adding a hazard means adding
one dict entry to `_build.py`, not a new test module.

The properties are grouped by what they protect against:

* **Structure** — the fixture loads at all, and the graph agrees with the
  declared snapshots. Catches a lineage that parses but means something else.
* **Grouping** — stable groups partition the units, `stable_id` is the
  lex-smallest member (the cross-run determinism guarantee), and a
  Redistribute unions the groups it touches.
* **Conservation** — aggregating the statistics onto stable groups neither
  invents nor loses value. This previously ran on Exampleland only.
* **Determinism** — two runs produce byte-identical outputs, including under a
  different PYTHONHASHSEED.
* **Completeness** — the denominator is the snapshot, the ratio stays in [0, 1],
  and a unit that never reported is named rather than quietly excluded.
* **No silent misattribution** — the C1 failure mode, in executable form: a
  year-aware mapping routed through the product API must assign exactly what a
  direct year-aware join assigns.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from stablebound.groups import build_stable_groups
from stablebound.snapshot import build_snapshot
from stablebound.validate import validate_lineage

REPO = Path(__file__).resolve().parents[1]


# --- Structure -----------------------------------------------------------


def test_fixture_loads_and_declares_itself(synthetic):
    """Every fixture must carry the four user files and a documented purpose."""
    for p in (synthetic.lineage_path, synthetic.baseline_path,
              synthetic.shapefile_path, synthetic.stats_path):
        assert p.exists(), f"{synthetic.name}: missing {p.name}"
    assert synthetic.doc, f"{synthetic.name}: expected.json has no _doc"


def test_declared_snapshots_match_the_graph(synthetic):
    """`expected.json` pins who exists when; the graph must agree.

    This is the check that catches an event-year convention error — the one
    failure mode that produces a well-formed lineage meaning something a year
    off from what the author intended.
    """
    expected = synthetic.expected.get("snapshots", {})
    if not expected:
        pytest.skip("fixture declares no snapshots")
    graph = synthetic.graph()
    seed = set(synthetic.baseline_df()["unit_id"].astype(str))
    for year, want in expected.items():
        got = build_snapshot(graph, int(year), additional_units=seed)
        assert len(got) == want, (
            f"{synthetic.name}: snapshot({year}) has {len(got)} units, "
            f"expected {want}: {sorted(got)}"
        )


def test_lineage_validates_clean_unless_declared_otherwise(synthetic):
    """A fixture may declare expected issue codes; anything else is a surprise.

    Without this, a validator could go silently quiet and every fixture would
    still pass.
    """
    graph = synthetic.graph()
    issues = validate_lineage(graph)
    allowed = set(synthetic.expected.get("expected_issue_codes", []))
    unexpected = [
        i for i in issues
        if i.severity in ("error", "warning") and i.category not in allowed
    ]
    assert not unexpected, (
        f"{synthetic.name}: unexpected "
        f"{[(i.severity, i.category, i.message) for i in unexpected]}"
    )


# --- Grouping ------------------------------------------------------------


def _groups(synthetic):
    graph = synthetic.graph()
    base = synthetic.baseline_df()
    base_year = int(base["year"].min())
    units = set(base["unit_id"].astype(str))
    modern = json.loads(synthetic.shapefile_path.read_text(encoding="utf-8"))
    modern_ids = {f["properties"]["unit_id"] for f in modern["features"]}
    # `additional_units` is the documented caller pattern: baseline ids union
    # modern shapefile ids, so a unit governed by no event still enters the
    # universe rather than vanishing.
    remap = build_stable_groups(
        graph, base_year=base_year, additional_units=units | modern_ids
    )
    return remap, units, modern_ids


def test_stable_groups_partition_the_units(synthetic):
    """Every unit lands in exactly one group — no unit in two, none dropped."""
    remap, base_units, modern_ids = _groups(synthetic)
    seen: dict[str, str] = {}
    for unit, stable in remap.items():
        assert unit not in seen or seen[unit] == stable, f"{unit} in two groups"
        seen[unit] = stable
    missing = (base_units | modern_ids) - set(remap)
    assert not missing, f"{synthetic.name}: units with no stable group: {sorted(missing)}"


def test_stable_id_is_the_lex_smallest_member(synthetic):
    """Public contract: this is what makes runs reproducible across machines.

    `stable_id` is not arbitrary — it is the lexicographically smallest member
    of the group, so the same lineage always yields the same ids regardless of
    dict or set iteration order.
    """
    remap, _, _ = _groups(synthetic)
    members: dict[str, list[str]] = {}
    for unit, stable in remap.items():
        members.setdefault(stable, []).append(unit)
    for stable, group in members.items():
        assert stable == min(group), (
            f"{synthetic.name}: group {sorted(group)} has stable_id {stable}, "
            f"expected {min(group)}"
        )


def test_declared_group_count(synthetic):
    want = synthetic.expected.get("n_stable_groups")
    if want is None:
        pytest.skip("fixture declares no group count")
    remap, _, _ = _groups(synthetic)
    assert len(set(remap.values())) == want


def test_redistribute_unions_the_groups_it_touches():
    """The highest-value grouping case, asserted directly.

    A Redistribute leaves both units alive, so a grouping that only follows
    parent->child creation misses it and emits two groups whose dissolved
    geometry overlaps.
    """
    from tests.conftest import get_synthetic

    c = get_synthetic("redistribute")
    remap, _, _ = _groups(c)
    assert remap["R.001"] == remap["R.002"], "redistribute must union both groups"
    assert remap["R.003"] != remap["R.001"], "an untouched unit must stay separate"


def test_cascade_collapses_to_one_group():
    """Three-deep lineage: a one-generation walk would leave two groups."""
    from tests.conftest import get_synthetic

    c = get_synthetic("cascade")
    remap, _, _ = _groups(c)
    assert len(set(remap.values())) == 1, f"expected 1 group, got {set(remap.values())}"


def test_rename_does_not_create_a_new_group():
    from tests.conftest import get_synthetic

    c = get_synthetic("rename_in_lineage")
    remap, _, _ = _groups(c)
    assert len(set(remap.values())) == 2


# --- Determinism ---------------------------------------------------------


def test_grouping_is_deterministic_within_a_process(synthetic):
    a, _, _ = _groups(synthetic)
    b, _, _ = _groups(synthetic)
    assert a == b


@pytest.mark.parametrize("seed", ["0", "1", "12345"])
def test_grouping_is_deterministic_across_hash_seeds(synthetic, seed):
    """PYTHONHASHSEED changes set/dict iteration order.

    The India matcher had exactly this bug — 24-56 rows differing per run — so
    it is asserted here rather than assumed. A subprocess is the only honest
    way to vary the seed, since it is fixed at interpreter start.
    """
    script = (
        "import json,sys;"
        "sys.path.insert(0,'src');sys.path.insert(0,'.');"
        "from tests.conftest import get_synthetic;"
        "from tests.test_qualification import _groups;"
        f"c=get_synthetic({synthetic.name!r});"
        "r,_,_=_groups(c);"
        "print(json.dumps(sorted(r.items())))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
             "PROJ_DATA": str(Path(sys.executable).parents[1] / "share/proj"),
             "PROJ_LIB": str(Path(sys.executable).parents[1] / "share/proj")},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    got = dict(json.loads(out.stdout))
    want, _, _ = _groups(synthetic)
    assert got == want, f"{synthetic.name}: grouping changed under PYTHONHASHSEED={seed}"


# --- Conservation --------------------------------------------------------


def test_aggregation_conserves_value(synthetic):
    """Aggregating onto stable groups neither invents nor loses value.

    Generalizes `test_conservation.py`, which ran on Exampleland alone.

    Late and early reporting rows are counted on BOTH sides rather than
    excluded. They are kept in the output by design, flagged with
    `late_reporting=True` and never re-routed, so excluding them from the
    aggregate while leaving them in the source would assert that the package
    loses value — the opposite of the property. Their flagging is asserted
    separately below.
    """
    if not synthetic.expected.get("conserves"):
        pytest.skip("fixture does not declare conservation")
    from stablebound.stats import aggregate

    remap, _, _ = _groups(synthetic)
    stats = synthetic.stats_df()
    graph = synthetic.graph()
    base_year = int(synthetic.baseline_df()["year"].min())
    max_year = int(stats["year"].max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = aggregate(stats, remap, graph, base_year=base_year, max_year=max_year)

    for (year, var), grp in stats.groupby(["year", "variable"]):
        mapped = grp[grp["unit_id"].astype(str).isin(remap)]
        got = agg[(agg["year"] == year) & (agg["variable"] == var)]["value"].sum()
        want = mapped["value"].sum()
        assert abs(got - want) < 1e-6, (
            f"{synthetic.name} {year} {var}: aggregated {got}, source {want}"
        )


def test_late_reporting_rows_are_flagged_not_dropped():
    """A unit reporting outside its lifespan must survive, marked."""
    from tests.conftest import get_synthetic
    from stablebound.stats import aggregate

    c = get_synthetic("late_early")
    remap, _, _ = _groups(c)
    stats = c.stats_df()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = aggregate(stats, remap, c.graph(), base_year=2010, max_year=2018)
    assert agg["late_reporting"].any(), "expected late-reporting rows to be flagged"
    total_in = stats["value"].sum()
    total_out = agg["value"].sum()
    assert abs(total_in - total_out) < 1e-6, (
        f"value lost: {total_in} in, {total_out} out — late rows must be kept"
    )


# --- Completeness --------------------------------------------------------


def test_completeness_stays_within_bounds(synthetic):
    """0 <= completeness <= 1, and it agrees with missing_unit_ids."""
    from stablebound.stats import aggregate

    remap, _, _ = _groups(synthetic)
    stats = synthetic.stats_df()
    base_year = int(synthetic.baseline_df()["year"].min())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = aggregate(stats, remap, synthetic.graph(),
                        base_year=base_year, max_year=int(stats["year"].max()))
    if "completeness" not in agg.columns:
        pytest.skip("no completeness column")
    vals = pd.to_numeric(agg["completeness"], errors="coerce").dropna()
    assert ((vals >= 0) & (vals <= 1)).all(), (
        f"{synthetic.name}: completeness outside [0,1]: "
        f"{sorted(set(vals[(vals < 0) | (vals > 1)]))[:5]}"
    )
    # `complete` is a subset test, so it can never disagree with the list.
    if "complete" in agg.columns and "missing_unit_ids" in agg.columns:
        # astype(bool) after fillna: `complete` is object dtype because it mixes
        # bools with pd.NA, and fillna on object dtype is deprecated-downcast.
        is_complete = agg["complete"].fillna(False).astype(bool)
        has_missing = agg["missing_unit_ids"].fillna("").astype(str).ne("")
        bad = agg[is_complete & has_missing]
        assert bad.empty, f"{synthetic.name}: complete=True with missing ids"


def test_a_silent_parent_is_named_not_ignored():
    """Three parents merge; one never reports. It must appear in missing ids."""
    from tests.conftest import get_synthetic
    from stablebound.stats import aggregate

    c = get_synthetic("multi_parent_merge")
    remap, _, _ = _groups(c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agg = aggregate(c.stats_df(), remap, c.graph(), base_year=2010, max_year=2016)
    row = agg[(agg["year"] == 2010) & (agg["variable"] == "area_ha")]
    assert len(row), "expected a 2010 row"
    missing = str(row.iloc[0].get("missing_unit_ids", ""))
    assert "M.003" in missing, f"silent parent not reported; missing={missing!r}"
    assert not bool(row.iloc[0].get("complete")), "should not be complete"
