"""The 11-column legacy dialect, and the round trip it makes possible.

Before this writer existed the package could not test its own FEWS export at
all: ``read_legacy_relationship_table`` could not read
``write_relationship_table``'s output, the consumer is external, and so the
deliverable's correctness rested on inspection. Writing the dialect FEWS
distributes closes the loop — export, re-import, and assert the lineage means
the same thing.

What "the same thing" means is deliberately narrow, and the limits are pinned
as tests rather than left to be discovered:

* **Ids are not preserved.** ``rt_convert._assign_ids`` allocates dense ids on
  import, so a round trip is snapshot-preserving, not id-preserving.
* **Coarse events do not survive as events.** They record a change of upper-level
  attribution, not of territory, and the dialect has no temporal row for that —
  they reappear as a changed hierarchical parent.
* **Split and Redistribute are reclassified.** The dialect does not label them;
  the reader infers from arity, so a child with several parents comes back as
  Redistribute regardless of how it was authored.

The assertion that survives all three, and the one that matters, is that the
reconstruction agrees about **which units exist in each year and what they are
called**.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound import Lineage
from stablebound.fews_export import (
    LEGACY_RELATIONSHIP_COLUMNS,
    RelationshipLevelInput,
    build_legacy_relationship_table,
)
from stablebound.lineage import LineageGraph
from stablebound.rt_convert import (
    convert_relationship_table_to_lineage,
    read_legacy_relationship_table,
)
from stablebound.snapshot import build_snapshot

FIXTURE = "tests/fixtures/rt_convert/relationshiptable_XX.csv"


def _synthetic_country() -> Lineage:
    """The canonical half of the fixture pair, as a custom (unregistered) country."""
    from tests.conftest import get_synthetic

    c = get_synthetic("legacy_admin1")
    return Lineage("XX", relationship_table_path=c.lineage_path, baseline_path=c.baseline_path)


def _names_by_year(events: pd.DataFrame, baseline: pd.DataFrame, years) -> dict:
    """{year: sorted names in force}, applying only renames that have happened."""
    graph = LineageGraph.from_dataframe(events, validate=False)
    seed = set(baseline["unit_id"])
    out = {}
    for y in years:
        snap = build_snapshot(graph, y, additional_units=seed)
        names = dict(zip(baseline["unit_id"], baseline["name"]))
        for row in events.itertuples():
            if row.event_year < y:
                names[row.child_id] = row.child_name
        out[y] = sorted(str(names.get(u, u)) for u in snap)
    return out


@pytest.fixture(scope="module")
def synthetic_legacy():
    """The synthetic country written in the legacy dialect, with the tail year the rule needs."""
    from stablebound.fnid import build_admin1_only_code_map
    from stablebound.unit_defs import build_admin1_defs_table

    ln = _synthetic_country()
    years = list(range(ln.min_year, ln.lineage.max_event_year + 2))
    cm = build_admin1_only_code_map(ln.lineage, ln.baseline, iso="XX")
    defs = {
        y: build_admin1_defs_table(
            ln.lineage, ln.baseline, cm, iso="XX", admin0="Exampleland", year=y
        )
        for y in years
    }
    table = build_legacy_relationship_table(
        iso="XX",
        admin0="Exampleland",
        years=years,
        level_inputs=[RelationshipLevelInput(1, ln.lineage, cm, defs)],
    )
    return ln, table


# --- Shape matches what FEWS actually distributes ------------------------


def test_columns_are_the_legacy_dialect(synthetic_legacy):
    _, table = synthetic_legacy
    assert list(table.columns) == LEGACY_RELATIONSHIP_COLUMNS
    real = pd.read_csv(FIXTURE)
    assert list(table.columns) == list(real.columns)


def test_vintages_and_membership_match_the_hand_authored_file(synthetic_legacy):
    """Reproduce the hand-authored legacy file's structure from the canonical twin.

    ``relationshiptable_XX.csv`` has vintages 2010/2015/2020/2025 holding
    4/5/5/4 units (the count falls at the merge). Reproducing that from the
    canonical lineage confirms both the vintage convention (V lists what is in
    force after year V, i.e. ``snapshot(V+1)``) and the base-vintage label. The
    oracle is a file we wrote rather than one FEWS distributed — it is still
    independent of the exporter, but the claim "our output has the shape of a
    real FEWS file" rests on the convention having been established against
    Korea's distributed table during development, not on this test.
    """
    _, table = synthetic_legacy
    real = pd.read_csv(FIXTURE)

    def per_vintage(df):
        h = df[df["category"] == "hierarchical"]
        return h.groupby(h["fnid_from_unit"].str[2:6]).size().to_dict()

    assert per_vintage(table) == per_vintage(real)
    assert per_vintage(table) == {"2010": 4, "2015": 5, "2020": 5, "2025": 4}


def test_successor_and_hierarchical_row_counts_match_the_hand_authored_file(synthetic_legacy):
    """Row counts by kind agree, with one deliberate dialect difference.

    The dialect allows a rename to be written two ways: an explicit
    ``name change`` row, or a 1-to-1 ``successor`` row whose names differ. The
    writer emits the first; the hand-authored file uses the second so the
    reader's other branch stays exercised. The two forms are therefore counted
    together as continuations.
    """
    _, table = synthetic_legacy
    real = pd.read_csv(FIXTURE)
    continuation = {"successor", "name change"}

    def count(df, cat, rels):
        return len(df[(df["category"] == cat) & (df["relationship_type"].isin(rels))])

    for cat, rels in (("hierarchical", {"admin1_0"}), ("temporal", continuation),
                      ("temporal", {"split"}), ("temporal", {"merge"})):
        mine, theirs = count(table, cat, rels), count(real, cat, rels)
        assert mine == theirs, f"{cat}/{sorted(rels)}: {mine} vs {theirs}"
    assert count(table, "temporal", {"name change"}) == 1
    assert count(real, "temporal", {"name change"}) == 0


def test_surrogate_ids_are_unique_per_vintage_and_unit(synthetic_legacy):
    """rt_convert groups on these to tell a 1→2 split from two successors.

    Reusing an id across vintages would silently merge unrelated events into a
    bogus split, so this is a correctness constraint, not tidiness.
    """
    _, table = synthetic_legacy
    h = table[table["category"] == "hierarchical"]
    assert h["from_unit"].is_unique
    pairs = set(zip(h["from_unit"], h["fnid_from_unit"]))
    assert len(pairs) == len(set(h["from_unit"]))


# --- The round trip -------------------------------------------------------


def test_round_trips_every_event(synthetic_legacy):
    ln, table = synthetic_legacy
    lineage, baseline = convert_relationship_table_to_lineage(
        read_legacy_relationship_table_from(table), country="XX", admin_level=1
    )

    def sig(ev):
        return sorted(
            (int(r.event_year), str(r.event_type).lower(), str(r.parent_name),
             str(r.child_name))
            for r in ev.itertuples()
        )

    assert sig(lineage) == sig(ln.lineage.events)
    assert int(baseline["year"].iloc[0]) == ln.min_year, "base vintage label drifted"


def test_india_round_trip_preserves_who_exists_and_what_they_are_called():
    """The semantic assertion, on the hardest bundled country.

    India is two-level, 1,017 events, and exercises every event type. Ids and
    event-type labels do not survive (see the module docstring); the name set
    per year must.
    """
    import tempfile

    ln = Lineage("IN")
    years = range(1997, 2026)
    with tempfile.TemporaryDirectory() as td:
        res = ln.export_fews(td, years=years, admin0="India", legacy_relationship=True)
        raw = read_legacy_relationship_table(res["legacy_relationship"][0])
    lineage, baseline = convert_relationship_table_to_lineage(
        raw, country="IN", admin_level=2
    )

    got = _names_by_year(lineage, baseline, years)
    want = _names_by_year(ln.lineage.events, ln.baseline, years)
    differing = [y for y in years if got[y] != want[y]]
    assert differing == [], f"name set differs in {len(differing)} year(s): {differing[:5]}"


def test_export_fews_writes_the_legacy_file_only_when_asked(tmp_path):
    """The upload set stays exactly three files."""
    ln = Lineage("IN")
    plain = ln.export_fews(tmp_path / "a", years=range(2015, 2018), admin0="India")
    assert "legacy_relationship" not in plain
    assert not list((tmp_path / "a").glob("relationshiptable_*.csv"))

    withit = ln.export_fews(
        tmp_path / "b", years=range(2015, 2018), admin0="India",
        legacy_relationship=True,
    )
    assert withit["legacy_relationship"][0].name == "relationshiptable_IN.csv"
    # ...and the upload file is untouched by its presence.
    assert (
        pd.read_csv(plain["relationship"][0])
        .equals(pd.read_csv(withit["relationship"][0]))
    )


# --- The vintage rule refuses rather than dropping an event ---------------


def test_an_event_past_the_window_is_announced_not_silently_dropped():
    """Vintage V reads its content from year V+1, so the last year has none.

    A last-year event vanished from the output while this was being written —
    the file stayed well-formed and simply lost an event. Narrowing the window
    is legitimate (a caller may want 2015-2018 of a country with events through
    2025), so this warns rather than raises, but it must say what it dropped
    and which year would recover it. The synthetic country's 2025 merge is the
    event that falls off a one-year-short window.
    """
    from stablebound.fnid import build_admin1_only_code_map
    from stablebound.unit_defs import build_admin1_defs_table

    ln = _synthetic_country()
    short = list(range(ln.min_year, ln.lineage.max_event_year + 1))  # one year short
    cm = build_admin1_only_code_map(ln.lineage, ln.baseline, iso="XX")
    defs = {
        y: build_admin1_defs_table(
            ln.lineage, ln.baseline, cm, iso="XX", admin0="Exampleland", year=y
        )
        for y in short
    }
    with pytest.warns(UserWarning, match="outside the emittable vintage range"):
        table = build_legacy_relationship_table(
            iso="XX", admin0="Exampleland", years=short,
            level_inputs=[RelationshipLevelInput(1, ln.lineage, cm, defs)],
        )
    assert "2025" not in {s[2:6] for s in table["fnid_from_unit"]}


def test_build_deliverables_extends_the_window_so_the_default_path_is_clean(tmp_path):
    """The auto-extension exists so a normal export never trips the warning."""
    import warnings as _w

    ln = _synthetic_country()
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        res = ln.export_fews(
            tmp_path, years=range(ln.min_year, ln.lineage.max_event_year + 1),
            admin0="Exampleland", legacy_relationship=True,
        )
    table = pd.read_csv(res["legacy_relationship"][0])
    vintages = {s[2:6] for s in table["fnid_from_unit"]}
    assert "2025" in vintages, "the last event's vintage must survive the default path"


def test_level2_without_attribution_is_refused():
    ln = Lineage("IN")
    from stablebound.fnid import build_admin2_code_map

    cm = build_admin2_code_map(ln.lineage, ln.baseline, iso="IN")
    with pytest.raises(ValueError, match="attribution"):
        build_legacy_relationship_table(
            iso="IN", admin0="India", years=[2015, 2016],
            level_inputs=[RelationshipLevelInput(2, ln.lineage, cm, {})],
        )


def read_legacy_relationship_table_from(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the reader's derived columns to an in-memory table."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rt.csv"
        table.to_csv(p, index=False)
        return read_legacy_relationship_table(p)
