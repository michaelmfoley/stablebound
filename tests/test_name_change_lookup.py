"""Regression for github issue #11: a unit created by an event and later renamed
(same id, via the name-change log) must carry its PRE-rename name in every
year before the rename, whichever naming convention the creation row uses."""

import pandas as pd
import pytest

from stablebound import LineageGraph
from stablebound.fnid import build_admin1_only_code_map
from stablebound.io import merge_name_changes
from stablebound.lineage import name_of_unit_in_year
from stablebound.unit_defs import build_admin1_defs_table

COARSE = {"parent_coarse_id": "XX.ADM0.00001", "parent_coarse_name": "Land",
          "child_coarse_id": "XX.ADM0.00001", "child_coarse_name": "Land"}


def _graph(creation_name: str) -> tuple[LineageGraph, pd.DataFrame]:
    baseline = pd.DataFrame({"unit_id": ["XX.ADM1.00001"], "name": ["Alpha"], "year": [1980],
                             "coarse_id": ["XX.ADM0.00001"], "coarse_name": ["Land"]})
    events = pd.DataFrame([
        {"event_year": 1990, "event_type": "Split", "parent_id": "XX.ADM1.00001", "parent_name": "Alpha",
         "child_id": "XX.ADM1.00002", "child_name": "Alpha", **COARSE},
        {"event_year": 1990, "event_type": "Split", "parent_id": "XX.ADM1.00001", "parent_name": "Alpha",
         "child_id": "XX.ADM1.00003", "child_name": creation_name, **COARSE},
    ])
    ncl = pd.DataFrame([{"event_year": 2000, "level": 1, "unit_id": "XX.ADM1.00003",
                         "old_name": "Beta", "new_name": "Gamma"}])
    graph = LineageGraph.from_dataframe(merge_name_changes(events, ncl), validate=False)
    return graph, baseline


@pytest.mark.parametrize("creation_name", ["Beta", "Gamma"], ids=["per-event-name", "modern-name"])
def test_pre_rename_years_carry_old_name(creation_name):
    graph, baseline = _graph(creation_name)
    assert name_of_unit_in_year(graph, "XX.ADM1.00003", 1995) == "Beta"
    assert name_of_unit_in_year(graph, "XX.ADM1.00003", 2000) == "Beta"   # rename during 2000
    assert name_of_unit_in_year(graph, "XX.ADM1.00003", 2001) == "Gamma"
    cm = build_admin1_only_code_map(graph, baseline, iso="XX")
    names = {y: dict(zip(*[build_admin1_defs_table(graph, baseline, cm, iso="XX", admin0="Land", year=y)[c]
                          for c in ("FNID", "admin1")])) for y in (1995, 2000, 2005)}
    assert names[1995]["XX1995A103"] == "Beta"
    assert names[2000]["XX2000A103"] == "Beta"
    assert names[2005]["XX2005A103"] == "Gamma"


def test_chained_renames_pick_the_next_one():
    graph, baseline = _graph("Delta")
    ncl = pd.DataFrame([
        {"event_year": 2000, "level": 1, "unit_id": "XX.ADM1.00003", "old_name": "Beta", "new_name": "Gamma"},
        {"event_year": 2010, "level": 1, "unit_id": "XX.ADM1.00003", "old_name": "Gamma", "new_name": "Delta"},
    ])
    ev = graph.events[graph.events["event_type"] != "NameChange"]
    g2 = LineageGraph.from_dataframe(merge_name_changes(ev, ncl), validate=False)
    assert name_of_unit_in_year(g2, "XX.ADM1.00003", 1995) == "Beta"
    assert name_of_unit_in_year(g2, "XX.ADM1.00003", 2005) == "Gamma"
    assert name_of_unit_in_year(g2, "XX.ADM1.00003", 2015) == "Delta"
