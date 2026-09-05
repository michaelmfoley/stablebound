"""Tests for ``stablebound.india.matcher`` — the DESAGRI stats matcher.

This module produced India's published match assignments and had a
``PYTHONHASHSEED`` non-determinism bug (24-56 rows differing between runs)
that was fixed by hand in 2026-07 with no regression guard. The determinism
test below is that guard: it re-runs the matcher in fresh subprocesses under
different hash seeds and requires byte-identical output.

The rest of the tests pin the behaviours the India pipeline depends on:
the state-name alias table, the ``"Name (XX)"`` suffix parser, the
``__FILTER__`` sentinel for aggregate rows, and the U2 ancestor walk.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from stablebound.india.matcher import (
    STATS_DIST_COL,
    _build_parents_map,
    STATS_STATE_COL,
    STATS_YEAR_COL,
    match_stats_to_lineage,
    normalize_district,
    normalize_state,
    parse_stats_district,
)

REPO = Path(__file__).resolve().parents[1]


# --- Fixtures ------------------------------------------------------------
#
# The India matcher consumes the *legacy* frames (uppercase columns), not
# LineageGraph — see the module docstring. These builders mirror that shape.


def _snapshots() -> pd.DataFrame:
    """Per-year admin snapshot: (YEAR, ADM1_*, ADM2_*).

    Alpha splits into Alpha + Beta during 2005, so Beta first appears in
    the 2006 snapshot. Gamma is a stable district in another state.
    """
    rows = []
    for year in range(2000, 2006):
        rows += [
            (year, "IN.ADM1.01", "Testland", "IN.ADM2.001", "Alpha"),
            (year, "IN.ADM1.02", "Otherland", "IN.ADM2.003", "Gamma"),
        ]
    for year in range(2006, 2011):
        rows += [
            (year, "IN.ADM1.01", "Testland", "IN.ADM2.001", "Alpha"),
            (year, "IN.ADM1.01", "Testland", "IN.ADM2.002", "Beta"),
            (year, "IN.ADM1.02", "Otherland", "IN.ADM2.003", "Gamma"),
        ]
    return pd.DataFrame(
        rows, columns=["YEAR", "ADM1_ID", "ADM1_NAME", "ADM2_ID", "ADM2_NAME"]
    )


def _lineage() -> pd.DataFrame:
    return pd.DataFrame(
        [(2005, "Split", "IN.ADM2.001", "Alpha", "IN.ADM2.002", "Beta")],
        columns=[
            "EVENT_YEAR", "EVENT_TYPE",
            "PARENT_ID", "PARENT_NAME", "CHILD_ID", "CHILD_NAME",
        ],
    )


def _name_log() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["CHANGE_YEAR", "LEVEL", "UNIT_ID",
                 "OLD_OFFICIAL_NAME", "NEW_OFFICIAL_NAME"]
    )


def _stats(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[STATS_STATE_COL, STATS_DIST_COL, STATS_YEAR_COL])


# --- Normalization helpers ----------------------------------------------


def test_parse_stats_district_strips_state_abbreviation():
    assert parse_stats_district("Alpha (TL)") == ("Alpha", "TL")
    assert parse_stats_district("Alpha") == ("Alpha", None)


def test_normalize_district_strips_suffixes_and_honorifics():
    assert normalize_district("Alpha District") == "alpha"
    assert normalize_district("Dr. Alpha") == "alpha"
    assert normalize_district("Alpha-Beta") == "alpha beta"


def test_normalize_state_applies_alias_table():
    assert normalize_state("Orissa") == "Odisha"
    assert normalize_state("Uttaranchal") == "Uttarakhand"
    assert normalize_state("Testland") == "Testland"   # pass-through


# --- Matching behaviour --------------------------------------------------


def test_exact_year_match():
    stats = _stats([("Testland", "Alpha", 2003)])
    match_map, unmatched, _, _ = match_stats_to_lineage(
        stats, _snapshots(), _name_log(), lineage=_lineage()
    )
    assert not unmatched
    assert match_map[("Testland", "Alpha", 2003)][0] == "IN.ADM2.001"


def test_post_split_child_resolves_in_its_own_year():
    stats = _stats([("Testland", "Beta", 2008)])
    match_map, _, _, _ = match_stats_to_lineage(
        stats, _snapshots(), _name_log(), lineage=_lineage()
    )
    assert match_map[("Testland", "Beta", 2008)][0] == "IN.ADM2.002"


def test_pre_creation_name_walks_to_the_year_compatible_ancestor():
    # "Beta" reported in 2003 predates Beta's existence; the U2 walk must
    # route it to the pre-split parent rather than attach it to a unit
    # that did not exist.
    stats = _stats([("Testland", "Beta", 2003)])
    match_map, unmatched, _, diag = match_stats_to_lineage(
        stats, _snapshots(), _name_log(), lineage=_lineage()
    )
    key = ("Testland", "Beta", 2003)
    if key in match_map:
        adm2_id, method = match_map[key]
        assert adm2_id == "IN.ADM2.001"          # the parent
        assert "ancestor" in method              # and it is audited
    else:
        # The other acceptable outcome is an explicit no-ancestor drop —
        # what must never happen is silently keeping the post-split id.
        assert unmatched or diag


def test_delhi_aggregate_rows_hit_the_filter_sentinel():
    stats = _stats([("Delhi", "Delhi_Total", 2003)])
    match_map, _, _, _ = match_stats_to_lineage(
        stats, _snapshots(), _name_log(), lineage=_lineage()
    )
    assert match_map[("Delhi", "Delhi_Total", 2003)] == ("__FILTER__", "filter")


def test_unknown_district_is_reported_unmatched_not_guessed():
    stats = _stats([("Testland", "Nowhere", 2003)])
    match_map, unmatched, _, _ = match_stats_to_lineage(
        stats, _snapshots(), _name_log(), lineage=_lineage()
    )
    assert ("Testland", "Nowhere", 2003) not in match_map
    assert len(unmatched) == 1


# --- Determinism guard ---------------------------------------------------

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo!r} + "/src")
    sys.path.insert(0, {repo!r} + "/tests")
    from test_india_matcher import _snapshots, _lineage, _name_log, _stats
    from stablebound.india.matcher import match_stats_to_lineage

    # Several same-normalized names across states and years, so any
    # set/dict iteration order inside the matcher has a chance to show.
    rows = []
    for year in (2003, 2007, 2009):
        for state, dist in (
            ("Testland", "Alpha"), ("Testland", "Beta"),
            ("Otherland", "Gamma"), ("Testland", "Alpha District"),
            ("Testland", "Dr. Alpha"), ("Otherland", "Gamma (OL)"),
        ):
            rows.append((state, dist, year))
    match_map, unmatched, match_log, diag = match_stats_to_lineage(
        _stats(rows), _snapshots(), _name_log(), lineage=_lineage()
    )
    print(json.dumps({{
        "match_map": sorted((list(k), list(v)) for k, v in match_map.items()),
        "unmatched": sorted(str(u) for u in unmatched),
        "log": sorted(json.dumps(r, sort_keys=True, default=str) for r in match_log),
        "diag": sorted(json.dumps(r, sort_keys=True, default=str) for r in diag),
    }}, sort_keys=True))
    """
)


def _run_with_hashseed(seed: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT.format(repo=str(REPO))],
        capture_output=True,
        text=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    return json.loads(out.stdout)


def test_matcher_output_is_hash_seed_independent():
    """The 2026-07 non-determinism fix, pinned.

    Python randomizes str hashing per process, so set/dict iteration order
    varies between runs unless the code sorts explicitly. This previously
    moved 24-56 India rows between runs. Fresh subprocesses are required —
    PYTHONHASHSEED is read once at interpreter start.
    """
    a = _run_with_hashseed("0")
    b = _run_with_hashseed("12345")
    c = _run_with_hashseed("98765")
    assert a == b == c


def test_build_parents_map_raises_on_lowercase_columns():
    """The wrong column case must fail, not return an empty index.

    Series.get returns None for a missing key rather than raising, so handed a
    frame with the package's canonical lowercase columns -- which is what
    LineageGraph.events and read_relationship_table both produce -- every row
    failed the notna check and this returned {} with no error.

    That is worse than a crash. The ancestor walk then finds no candidates for
    any row, match_stats_to_lineage tags them no_ancestor, and the rows are
    dropped. A silently empty index does not fail; it quietly discards data.
    """
    rows = [
        {"PARENT_ID": "A", "CHILD_ID": "A1"},
        {"PARENT_ID": "A", "CHILD_ID": "A2"},
        {"PARENT_ID": "B", "CHILD_ID": "B"},  # self-loop, must be excluded
    ]
    upper = pd.DataFrame(rows)
    assert _build_parents_map(upper) == {"A1": {"A"}, "A2": {"A"}}

    lower = upper.rename(columns={"PARENT_ID": "parent_id", "CHILD_ID": "child_id"})
    with pytest.raises(KeyError) as exc:
        _build_parents_map(lower)
    # The message must say what to do, not merely that a key is absent.
    assert "canonical lowercase schema" in str(exc.value)
