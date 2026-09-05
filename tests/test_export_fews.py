"""Multi-level Lineage and the export_fews facade.

Producing the FEWS deliverable used to take ~80 lines of per-country glue:
build a graph per admin level by hand, split the name-change log per level,
rename the baseline's columns, then call ``build_deliverables`` with a dozen
keyword arguments. ``Lineage.export_fews`` does that from the registry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound import BUNDLED_COUNTRIES, Lineage


def test_admin_level_is_recorded_not_inferred():
    """Recorded in the registry — id prefixes only happen to encode it."""
    assert Lineage("IN").admin_level == 2
    for cc, entry in BUNDLED_COUNTRIES.items():
        assert entry.admin_level in (1, 2), cc


def test_graph_defaults_to_the_working_level():
    ln = Lineage("IN")
    assert ln.graph() is ln.lineage
    assert ln.graph(level=2) is ln.lineage


def test_coarse_level_graph_is_separate_and_small():
    """India's states split too; their events live in their own id space."""
    ln = Lineage("IN")
    adm1 = ln.graph(level=1)
    assert adm1 is not ln.lineage
    assert len(adm1.events) < len(ln.lineage.events)
    ids = set(adm1.events["parent_id"]) | set(adm1.events["child_id"])
    assert all(i.startswith("IN.ADM1.") for i in ids)


def test_coarse_graph_gets_only_its_own_renames():
    """A state's rename must not be injected into the district graph."""
    ln = Lineage("IN")
    ncl = ln.name_change_log
    assert "level" in ncl.columns, "bundled IN log should carry level"
    adm1 = ln.graph(level=1)
    nc = adm1.events[adm1.events["event_type"] == "NameChange"]
    assert len(nc) == int((ncl["level"].astype(int) == 1).sum())
    assert all(str(u).startswith("IN.ADM1.") for u in nc["parent_id"])


def test_single_level_country_has_no_coarse_graph(exampleland):
    with pytest.raises(ValueError, match="no level below"):
        exampleland.graph(level=0)


def test_asking_for_an_unknown_level_says_what_exists():
    ln = Lineage("IN")
    with pytest.raises(ValueError, match="asked for 5"):
        ln.graph(level=5)


def test_export_fews_writes_the_three_deliverables(tmp_path):
    ln = Lineage("IN")
    res = ln.export_fews(tmp_path, years=range(2015, 2018), admin0="India")

    assert len(res["admin_definitions"]) == 3
    rel = res["relationship"][0]
    assert rel.name == "IN_GeographicUnitRelationship.csv"
    assert rel.exists()
    # No stats passed -> no AgStats workbook, rather than an empty one.
    assert "agstats" not in res

    df = pd.read_csv(rel)
    assert len(df) > 0
    assert set(df["relationship_type"]) <= {
        "successor", "split", "merge", "redistribute", "name change",
    }
    for p in res["admin_definitions"]:
        sheets = pd.ExcelFile(p).sheet_names
        assert any("admin1" in s for s in sheets)
        assert any("admin2" in s for s in sheets)


def test_export_fews_defaults_to_the_covered_span(tmp_path):
    """The default window is what the source describes, not an arbitrary range."""
    ln = Lineage("IN")
    res = ln.export_fews(tmp_path, admin0="India")
    years = sorted(
        int(p.stem.rsplit("_", 1)[1]) for p in res["admin_definitions"]
    )
    assert min(years) == ln.min_year
    assert max(years) == ln.coverage_end_year


def test_single_level_country_exports_admin1_only(exampleland, tmp_path):
    res = exampleland.export_fews(tmp_path, years=range(2010, 2012), admin0="Exampleland")
    for p in res["admin_definitions"]:
        sheets = pd.ExcelFile(p).sheet_names
        assert any("admin1" in s for s in sheets)
        assert not any("admin2" in s for s in sheets)


def test_stableboundary_delegates(tmp_path):
    """A user holding a product shouldn't have to reach back to the lineage."""
    from stablebound import StableBoundary

    ln = Lineage("IN")
    sb = StableBoundary(ln, output_dir=tmp_path / "out")
    res = sb.export_fews(tmp_path / "fews", years=range(2015, 2017), admin0="India")
    assert res["relationship"][0].exists()


def test_lazy_state_is_initialised_on_both_construction_paths():
    """from_legacy_rt builds via cls.__new__, so it never runs __init__.

    Every lazily-populated attribute must still exist, or the object raises
    AttributeError only on whichever path happens to touch the missing one.
    """
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parent / "fixtures/rt_convert/relationshiptable_XX.csv"
    )
    direct = Lineage("IN")
    converted = Lineage.from_legacy_rt(fixture, country="XX", admin_level=1)
    lazy = [a for a in vars(direct) if a.startswith("_")]
    missing = [a for a in lazy if not hasattr(converted, a)]
    assert not missing, f"from_legacy_rt left these unset: {missing}"


def test_admin2_without_an_upper_layer_is_refused_clearly(bundle_synthetic, tmp_path):
    """A district-level lineage with no division structure at all.

    Its baseline has no ``coarse_id`` and its events carry no coarse columns,
    so there is nothing to build the admin1 tab from. Emitting its districts
    as "admin1" would misdescribe the hierarchy in a file handed to FEWS, so
    this must refuse — and say what is missing, rather than surfacing a bare
    ``KeyError: 'ORIGIN_ADMIN1_ID'`` from deep in the code map builder. The
    synthetic ``simple_split`` country is registered as admin level 2 to get
    into that state; ``admin_level`` is only reachable through the registry.
    """
    bundle_synthetic("simple_split", code="ZQ", admin_level=2)
    ln = Lineage("ZQ")
    assert ln.admin_level == 2
    assert "coarse_id" not in ln.baseline.columns
    with pytest.raises(ValueError, match="no upper-level attribution"):
        ln.export_fews(tmp_path, years=range(2010, 2012), admin0="Splitland")
