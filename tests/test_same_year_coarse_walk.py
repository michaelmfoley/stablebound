"""A Coarse re-attribution in the same year a unit is created must win.

Regression for the Africa ADM2 rollout (SLE 2017 Karene, BFA 1996): the
attribution walk applied a year's rows in alphabetical event_type order, so
"Coarse" ran before "Split" and the Split's inherited admin1 overwrote the
same-year Coarse move. Territorial rows now go first; Coarse/NameChange rows
then apply only to units alive after them.
"""
import pandas as pd

from stablebound import LineageGraph
from stablebound.unit_defs import _attribution_for_year, build_admin1_attribution_table


def _graph(rows):
    cols = ["event_year", "event_type", "parent_id", "parent_name", "child_id", "child_name",
            "parent_coarse_id", "parent_coarse_name", "child_coarse_id", "child_coarse_name"]
    return LineageGraph(pd.DataFrame(rows, columns=cols))


def test_same_year_split_child_then_coarse_move():
    rows = [
        # Coarse row listed FIRST in the file (alphabetically first too)
        (2017, "Coarse", "D.02", "Karene", "D.02", "Karene", "S.N", "Northern", "S.NW", "North Western"),
        (2017, "Split", "D.01", "Bombali", "D.01r", "Bombali", "S.N", "Northern", "S.N", "Northern"),
        (2017, "Split", "D.01", "Bombali", "D.02", "Karene", "S.N", "Northern", "S.N", "Northern"),
    ]
    base = pd.DataFrame({"unit_id": ["D.01"], "name": ["Bombali"], "year": [1980],
                         "coarse_id": ["S.N"], "coarse_name": ["Northern"]})
    att = build_admin1_attribution_table(_graph(rows), base)
    after = _attribution_for_year(att, 2018)
    assert after["D.02"][0] == "S.NW", after
    assert after["D.01r"][0] == "S.N"
    # the consumed parent has no post-2017 row (callers intersect with the snapshot)
    assert "D.01" not in set(att[att["year"] == 2018]["unit_id"])


def test_same_year_coarse_on_consumed_parent_is_a_noop():
    rows = [
        (2005, "Coarse", "D.01", "Old", "D.01", "Old", "P.1", "Pref", "P.2", "Province"),
        (2005, "Merge", "D.01", "Old", "D.09", "New", "P.1", "Pref", "P.2", "Province"),
    ]
    base = pd.DataFrame({"unit_id": ["D.01"], "name": ["Old"], "year": [2002],
                         "coarse_id": ["P.1"], "coarse_name": ["Pref"]})
    att = build_admin1_attribution_table(_graph(rows), base)
    after = _attribution_for_year(att, 2006)
    assert after["D.09"][0] == "P.2"
    assert "D.01" not in set(att[att["year"] == 2006]["unit_id"])
