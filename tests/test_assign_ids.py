"""`assign_unit_ids`: minting canonical ids for a lineage that only has names.

Two kinds of oracle. The synthetic registry supplies structure: blank every
id, re-mint, and the graph must still say who exists when and how many
stable groups there are — the same expectations `test_qualification` pins
for the authored ids. The bundled India files supply realism: a rename and a
split in the same year, districts moving into a state created that year,
two living districts that share a name, a name-change log for both levels.
Blanking the bundled ids and recovering them is the strongest check this
helper can get without a second country.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound import (
    BUNDLED_COUNTRIES,
    IdAssignmentError,
    Lineage,
    assign_unit_ids,
)
from stablebound.groups import build_stable_groups
from stablebound.lineage import LineageGraph
from stablebound.snapshot import build_snapshot
from stablebound.validate import validate_lineage
from tests.conftest import get_synthetic

ID_COLS = ("parent_id", "child_id", "parent_coarse_id", "child_coarse_id")


def _blank_ids(df: pd.DataFrame, cols=ID_COLS) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = None
    return out


def _as_str(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(str).reset_index(drop=True)


# --- The battery: every synthetic country, ids re-minted from names --------


def test_rebuilt_ids_reproduce_the_declared_snapshots_and_groups(synthetic):
    """Blank every id, re-mint, and the graph must agree with expected.json."""
    base = synthetic.baseline_df()
    res = assign_unit_ids(_blank_ids(synthetic.lineage_df()), base, mode="rebuild")
    graph = LineageGraph.from_dataframe(res.lineage, validate=False)
    seed = set(base["unit_id"].astype(str))
    for year, want in synthetic.expected.get("snapshots", {}).items():
        got = build_snapshot(graph, int(year), additional_units=seed)
        assert len(got) == want, f"{synthetic.name}: snapshot({year}) = {sorted(got)}"
    want_groups = synthetic.expected.get("n_stable_groups")
    if want_groups is not None:
        remap = build_stable_groups(graph, base_year=int(base["year"].min()),
                                    additional_units=seed)
        assert len(set(remap.values())) == want_groups
    allowed = set(synthetic.expected.get("expected_issue_codes", []))
    unexpected = [i for i in validate_lineage(graph)
                  if i.severity in ("error", "warning") and i.category not in allowed]
    assert not unexpected, [(i.severity, i.category, i.message) for i in unexpected]
    assert not res.warnings, res.warnings


def test_fill_mode_is_idempotent_on_an_authored_lineage(synthetic):
    """Nothing blank means nothing minted and nothing changed."""
    rt = synthetic.lineage_df()
    res = assign_unit_ids(rt, synthetic.baseline_df())
    assert res.minted.empty
    assert _as_str(res.lineage).equals(_as_str(rt))


def test_two_runs_agree(synthetic):
    base = synthetic.baseline_df()
    blank = _blank_ids(synthetic.lineage_df())
    a = assign_unit_ids(blank, base, mode="rebuild").lineage
    b = assign_unit_ids(blank, base, mode="rebuild").lineage
    assert _as_str(a).equals(_as_str(b))


# --- Fill mode: existing ids are kept, only blanks are minted ---------------


def test_fill_mints_only_the_blank_cell_and_numbers_after_the_highest_id():
    syn = get_synthetic("legacy_admin1")
    rt, base = syn.lineage_df(), syn.baseline_df()
    edited = rt.copy()
    edited.loc[0, "child_id"] = None  # Alpha North, authored as XX.ADM1.00005
    res = assign_unit_ids(edited, base)
    assert res.id_prefix == "XX.ADM1." and res.id_width == 5
    # The highest id in play is the merge child 00008, so the blank gets 00009.
    assert res.lineage.loc[0, "child_id"] == "XX.ADM1.00009"
    assert res.minted["unit_id"].tolist() == ["XX.ADM1.00009"]
    rest = res.lineage.drop(index=0)
    assert _as_str(rest).equals(_as_str(rt.drop(index=0)))


def test_inserting_a_new_event_moves_nothing_else():
    """The motivating case: a newly discovered event is appended with names only."""
    syn = get_synthetic("legacy_admin1")
    rt, base = syn.lineage_df(), syn.baseline_df()
    new = pd.DataFrame([
        {"event_year": 2030, "event_type": "Split", "parent_id": None,
         "parent_name": "Charlton", "child_id": None, "child_name": "Charlton North"},
        {"event_year": 2030, "event_type": "Split", "parent_id": None,
         "parent_name": "Charlton", "child_id": None, "child_name": "Charlton South"},
    ])
    res = assign_unit_ids(pd.concat([rt, new], ignore_index=True), base)
    assert _as_str(res.lineage.iloc[: len(rt)]).equals(_as_str(rt))
    tail = res.lineage.iloc[len(rt):]
    # Charlton is XX.ADM1.00003 (renamed from Charlie in 2020): resolved by its current name.
    assert tail["parent_id"].tolist() == ["XX.ADM1.00003", "XX.ADM1.00003"]
    assert tail["child_id"].tolist() == ["XX.ADM1.00009", "XX.ADM1.00010"]
    assert res.n_resolved == 2


def test_multi_parent_child_shares_one_id():
    syn = get_synthetic("multi_parent_merge")
    res = assign_unit_ids(_blank_ids(syn.lineage_df()), syn.baseline_df())
    assert res.lineage["child_id"].nunique() == 1
    assert len(res.minted) == 1


def test_a_prefilled_child_id_anchors_its_sibling_rows():
    syn = get_synthetic("multi_parent_merge")
    rt = _blank_ids(syn.lineage_df())
    rt.loc[1, "child_id"] = "M.099"
    res = assign_unit_ids(rt, syn.baseline_df())
    assert res.lineage["child_id"].tolist() == ["M.099"] * 3
    assert res.minted.empty


def test_two_different_prefilled_ids_in_one_group_are_reported():
    syn = get_synthetic("multi_parent_merge")
    rt = _blank_ids(syn.lineage_df())
    rt.loc[0, "child_id"] = "M.098"
    rt.loc[2, "child_id"] = "M.099"
    res = assign_unit_ids(rt, syn.baseline_df())
    assert res.lineage["child_id"].tolist() == ["M.098"] * 3
    assert any("already has id" in w for w in res.warnings)


def test_rebuild_discards_non_baseline_ids_and_renumbers_densely():
    syn = get_synthetic("legacy_admin1")
    res = assign_unit_ids(syn.lineage_df(), syn.baseline_df(), mode="rebuild")
    # The fixture skips 00007 on purpose; a rebuild does not.
    assert res.minted["unit_id"].tolist() == [
        "XX.ADM1.00005", "XX.ADM1.00006", "XX.ADM1.00007"
    ]
    assert res.lineage["parent_id"].tolist()[:2] == ["XX.ADM1.00001"] * 2


# --- Names that need a coarse unit to be told apart --------------------------


def _springfield_split(with_coarse: bool) -> pd.DataFrame:
    row = {"event_year": 2015, "event_type": "Split", "parent_id": None,
           "parent_name": "Springfield", "child_id": None}
    rows = [dict(row, child_name="Springfield East"), dict(row, child_name="Springfield West")]
    if with_coarse:
        for r in rows:
            r.update(parent_coarse_id=None, parent_coarse_name="South",
                     child_coarse_id=None, child_coarse_name="South")
    return pd.DataFrame(rows)


def test_homonyms_resolve_through_the_coarse_name():
    base = get_synthetic("homonym_coarse").baseline_df()
    res = assign_unit_ids(_springfield_split(with_coarse=True), base)
    assert res.lineage["parent_id"].tolist() == ["H.002", "H.002"]
    assert res.lineage["parent_coarse_id"].tolist() == ["H.A1.02", "H.A1.02"]
    assert res.lineage["child_id"].tolist() == ["H.004", "H.005"]


def test_homonyms_without_a_coarse_column_are_refused_by_name():
    base = get_synthetic("homonym_coarse").baseline_df()
    with pytest.raises(IdAssignmentError, match="ambiguous"):
        assign_unit_ids(_springfield_split(with_coarse=False), base)


def test_a_coarse_name_that_only_disambiguates_is_not_a_veto():
    """One living unit with the name resolves even if the row's coarse name is
    stale (the coarse level may have been renamed); a warning says so."""
    syn = get_synthetic("admin2_coarse")
    rt = _blank_ids(syn.lineage_df(), cols=("parent_id", "child_id"))
    rt["parent_coarse_name"] = "State A (old spelling)"
    res = assign_unit_ids(rt, syn.baseline_df())
    assert res.lineage["parent_id"].tolist() == ["A.001", "A.001"]
    assert any("coarse unit" in w for w in res.warnings)


def test_unresolvable_parent_names_are_all_reported_with_hints():
    syn = get_synthetic("simple_split")
    rt = _blank_ids(syn.lineage_df())
    rt["parent_name"] = "Parnt"
    with pytest.raises(IdAssignmentError) as exc:
        assign_unit_ids(rt, syn.baseline_df())
    msg = str(exc.value)
    assert "2 row(s)" in msg
    assert "row 0" in msg and "row 1" in msg
    assert "Parent (P.001)" in msg


# --- Id format --------------------------------------------------------------


def test_format_is_inferred_from_the_baseline():
    syn = get_synthetic("simple_split")
    res = assign_unit_ids(_blank_ids(syn.lineage_df()), syn.baseline_df())
    assert (res.id_prefix, res.id_width) == ("P.", 3)
    assert res.lineage["child_id"].tolist() == ["P.003", "P.004"]


def test_india_style_ids_infer_prefix_and_five_digits():
    base = pd.DataFrame({"unit_id": ["IN.ADM2.00001", "IN.ADM2.00002"],
                         "name": ["Alpha", "Beta"], "year": [1991, 1991]})
    rt = pd.DataFrame([{"event_year": 2000, "event_type": "Split", "parent_id": None,
                        "parent_name": "Alpha", "child_id": None, "child_name": "Alpha North"}])
    res = assign_unit_ids(rt, base)
    assert (res.id_prefix, res.id_width) == ("IN.ADM2.", 5)
    assert res.lineage.loc[0, "child_id"] == "IN.ADM2.00003"


def test_mixed_prefixes_need_an_explicit_format():
    base = pd.DataFrame({"unit_id": ["A.001", "B.001"], "name": ["Alpha", "Beta"]})
    rt = pd.DataFrame([{"event_year": 2000, "event_type": "Split", "parent_id": None,
                        "parent_name": "Alpha", "child_id": None, "child_name": "Alpha North"}])
    with pytest.raises(IdAssignmentError, match="cannot infer"):
        assign_unit_ids(rt, base)
    res = assign_unit_ids(rt, base, id_prefix="A.", id_width=3)
    assert res.lineage.loc[0, "child_id"] == "A.002"


def test_bad_mode_is_refused():
    syn = get_synthetic("simple_split")
    with pytest.raises(IdAssignmentError, match="mode"):
        assign_unit_ids(syn.lineage_df(), syn.baseline_df(), mode="redo")


# --- The bundled India files as a real-data oracle ----------------------------


@pytest.fixture(scope="module")
def india():
    e = BUNDLED_COUNTRIES["IN"]
    return {
        "rt": pd.read_excel(e.lineage_path),
        "base": pd.read_csv(e.baseline_path),
        "ncl": pd.read_excel(e.name_change_log_path),
        "adm1": pd.read_excel(e.coarse_lineage_path),
    }


def test_india_coarse_ids_are_recovered_through_the_adm1_lineage(india):
    """Districts filed under Bihar/Jharkhand/... in 2000 must get the ids of
    the states as they stood after that year's split, which only the ADM1
    lineage knows."""
    rt = india["rt"].copy()
    mask = (rt["event_year"] == 2000) & (rt["event_type"] == "Coarse")
    rt.loc[mask, ["parent_coarse_id", "child_coarse_id"]] = None
    res = assign_unit_ids(rt, india["base"], name_change_log=india["ncl"],
                          coarse_lineage=india["adm1"])
    got = res.lineage.loc[mask, ["parent_coarse_id", "child_coarse_id"]]
    want = india["rt"].loc[mask, ["parent_coarse_id", "child_coarse_id"]]
    assert _as_str(got).equals(_as_str(want))
    assert res.minted.empty and not res.warnings


def test_india_coarse_ids_created_by_an_event_need_the_coarse_lineage(india):
    rt = india["rt"].copy()
    mask = (rt["event_year"] == 2000) & (rt["event_type"] == "Coarse")
    rt.loc[mask, ["parent_coarse_id", "child_coarse_id"]] = None
    with pytest.raises(IdAssignmentError, match="coarse_lineage"):
        assign_unit_ids(rt, india["base"], name_change_log=india["ncl"])


def test_india_full_rebuild_reproduces_the_bundled_history(india):
    """Every unit id blanked and re-minted from names: the same districts must
    exist in every year and the same stable groups must come out."""
    rt = india["rt"].copy()
    rt[["parent_id", "child_id"]] = None
    res = assign_unit_ids(rt, india["base"], name_change_log=india["ncl"],
                          coarse_lineage=india["adm1"], mode="rebuild")
    graph = LineageGraph.from_dataframe(res.lineage, validate=False)
    seed = set(india["base"]["unit_id"].astype(str))
    bundled = Lineage("IN")
    for year in (1991, 2000, 2012, 2025):
        assert len(build_snapshot(graph, year, additional_units=seed)) == len(bundled.snapshot(year))
    mine = build_stable_groups(graph, base_year=1991, additional_units=seed)
    theirs = build_stable_groups(bundled.lineage, base_year=1991, additional_units=seed)
    assert len(set(mine.values())) == len(set(theirs.values()))
    assert not [i for i in res.issues if i.severity == "error"]
    # Two Karnataka/Chhattisgarh districts share the name Bijapur; the log row
    # renaming one of them carries no coarse column, so it is left for a human.
    assert len(res.warnings) == 1 and "Bijapur" in res.warnings[0]


def test_india_same_year_rename_then_split_resolves_by_the_new_name(india):
    """Jyotiba Phule Nagar became Amroha in 2012 and split as Amroha in 2012."""
    rt = india["rt"].copy()
    mask = (rt["event_year"] == 2012) & (rt["parent_name"] == "Amroha")
    rt.loc[mask, "parent_id"] = None
    res = assign_unit_ids(rt, india["base"], name_change_log=india["ncl"])
    assert set(res.lineage.loc[mask, "parent_id"]) == {"IN.ADM2.00659"}
    assert not res.warnings


def test_india_name_change_log_unit_id_is_filled_from_the_old_name(india):
    ncl = india["ncl"].copy()
    mask = (ncl["event_year"] == 1994) & (ncl["old_name"] == "Bhabua")
    ncl.loc[mask, "unit_id"] = None
    res = assign_unit_ids(india["rt"], india["base"], name_change_log=ncl)
    assert res.name_change_log.loc[mask, "unit_id"].tolist() == ["IN.ADM2.00483"]


# --- Report and files ---------------------------------------------------------


def test_report_and_write(tmp_path):
    syn = get_synthetic("legacy_admin1")
    res = assign_unit_ids(_blank_ids(syn.lineage_df()), syn.baseline_df(), mode="rebuild")
    text = res.report()
    assert "ids minted      : 3" in text and "XX.ADM1.00005" in text
    written = res.write(tmp_path)
    assert set(written) == {"lineage", "baseline", "report"}
    back = pd.read_csv(written["lineage"])
    assert _as_str(back).equals(_as_str(res.lineage))
    assert (tmp_path / "id_assignment_report.txt").read_text(encoding="utf-8") == text
