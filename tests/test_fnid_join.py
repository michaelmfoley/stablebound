"""Attaching unit_ids to stats that already carry FNIDs.

FEWS-sourced statistics generally ship an FNID column, and the converter
builds ``{fnid: unit_id}`` internally anyway — it used to discard it. Joining
on ids skips name matching entirely, which is the most expensive part of
onboarding a country (India needed ~150 hand-curated aliases).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stablebound import Lineage, parse_fnid, validate_fnid

XX_RT = Path(__file__).resolve().parent / "fixtures/rt_convert/relationshiptable_XX.csv"


@pytest.fixture()
def xx() -> Lineage:
    return Lineage.from_legacy_rt(XX_RT, country="XX", admin_level=1)


def test_fnid_map_is_exposed_for_converted_lineages(xx):
    assert xx.fnid_map, "converted lineage should carry a FNID map"
    for fnid, uid in list(xx.fnid_map.items())[:5]:
        assert validate_fnid(fnid, iso="XX", level=1)
        assert uid.startswith("XX.ADM1.")


def test_canonical_lineage_has_no_fnid_map():
    """India is hand-authored — there are no FNIDs to join on."""
    ln = Lineage("IN")
    assert ln.fnid_map is None
    with pytest.raises(RuntimeError, match="no FNID map"):
        ln.attach_stats_by_fnid(pd.DataFrame({"FNID": ["IN2015A20102"]}))


def test_attach_by_fnid_resolves_exactly(xx):
    fnids = list(xx.fnid_map)[:3]
    stats = pd.DataFrame({
        "FNID": fnids,
        "year": [2000, 2001, 2002],
        "value": [1.0, 2.0, 3.0],
    })
    out = xx.attach_stats_by_fnid(stats)
    assert list(out["unit_id"]) == [xx.fnid_map[f] for f in fnids]
    # The input frame is not mutated.
    assert "unit_id" not in stats.columns


def test_unmapped_fnids_warn_and_keep_null_rather_than_vanish(xx):
    stats = pd.DataFrame({
        "FNID": [next(iter(xx.fnid_map)), "XX9999A199"],
        "value": [1.0, 2.0],
    })
    with pytest.warns(UserWarning, match="absent from the XX lineage"):
        out = xx.attach_stats_by_fnid(stats)
    # Two rows in, two rows out — a dropped row is a silent data loss.
    assert len(out) == 2
    assert out["unit_id"].isna().sum() == 1


def test_missing_fnid_column_names_what_it_looked_for(xx):
    with pytest.raises(KeyError, match="fnid_column"):
        xx.attach_stats_by_fnid(pd.DataFrame({"code": ["x"]}))


def test_custom_column_names_are_honoured(xx):
    fnid = next(iter(xx.fnid_map))
    out = xx.attach_stats_by_fnid(
        pd.DataFrame({"geo": [fnid]}), fnid_column="geo", id_column="adm1_id"
    )
    assert out["adm1_id"].iloc[0] == xx.fnid_map[fnid]


# --- FNID parsing --------------------------------------------------------


def test_parse_handles_every_real_world_shape():
    assert parse_fnid("IN2015A20102") == ("IN", 2015, "A", 2, "0102")
    assert parse_fnid("VN1991A101") == ("VN", 1991, "A", 1, "01")
    # Sri Lanka keys its statistics on crop-region ids, not admin ids.
    assert parse_fnid("LK1978R20101").unit_type == "R"


def test_parts_expose_ss_and_dd():
    p = parse_fnid("IN2015A20102")
    assert (p.ss, p.dd) == ("01", "02")
    assert parse_fnid("VN1991A101").dd == ""


@pytest.mark.parametrize("bad", ["", "IN", "IN2015", "1N2015A20102", "INxxxxA20102"])
def test_malformed_fnids_raise_rather_than_slice(bad):
    with pytest.raises(ValueError):
        parse_fnid(bad)


def test_wrong_code_width_is_rejected():
    with pytest.raises(ValueError, match="4-char code"):
        validate_fnid("IN2015A2010")


def test_level_three_is_refused_not_guessed():
    """The only known ADM3 ids carry a 4-char code, contradicting SS+DD+EE."""
    with pytest.raises(NotImplementedError, match="admin level 3"):
        validate_fnid("CD1997A30910")


def test_expected_level_and_iso_are_checked():
    with pytest.raises(ValueError, match="expected 1"):
        validate_fnid("IN2015A20102", level=1)
    with pytest.raises(ValueError, match="expected VN"):
        validate_fnid("IN2015A20102", iso="VN")


def test_a_blank_fnid_is_not_reported_as_a_vintage_mismatch():
    """A missing FNID and an unknown FNID are different problems.

    `astype(str)` turns NaN into the string "nan", which then fails the map
    lookup and looks exactly like an FNID the lineage is missing. The
    Philippines dry run surfaced this: 34,263 rows with no FNID at all were
    reported as "carrying an FNID absent from the lineage (sample: ['nan'])",
    which points a reader at a vintage mismatch that does not exist. Every one
    of that country's 85,047 real FNIDs resolved.
    """
    import warnings as _w

    ln = Lineage.from_legacy_rt(XX_RT, country="XX", admin_level=1)
    known = sorted(ln.fnid_map)[0]
    stats = pd.DataFrame({
        "FNID": [known, None, float("nan"), "", "XX9999A999"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = ln.attach_stats_by_fnid(stats, fnid_column="FNID")
    msgs = [str(c.message) for c in caught]

    blank = [m for m in msgs if "no FNID at all" in m]
    unknown = [m for m in msgs if "absent from the" in m]
    assert len(blank) == 1, msgs
    assert "3 of 5" in blank[0], blank[0]
    assert "not a lineage problem" in blank[0]

    assert len(unknown) == 1, msgs
    assert "XX9999A999" in unknown[0]
    # ...and the blank rows must not be quoted as if they were FNIDs.
    assert "nan" not in unknown[0].lower().replace("lineage", "")

    assert out["unit_id"].notna().sum() == 1
