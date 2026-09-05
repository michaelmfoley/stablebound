"""Schema validators."""

from __future__ import annotations

import pandas as pd
import pytest

from stablebound import SchemaError
from stablebound.schemas import (
    validate_name_change_log,
    validate_relationship_table,
    validate_stats,
)


def _good_rt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_year": [2003, 2003],
            "event_type": ["Split", "Split"],
            "parent_id": ["IN.ADM2.00001", "IN.ADM2.00001"],
            "parent_name": ["A", "A"],
            "child_id": ["IN.ADM2.00010", "IN.ADM2.00011"],
            "child_name": ["A", "B"],
        }
    )


def test_relationship_table_accepts_canonical_schema():
    validate_relationship_table(_good_rt())


def test_relationship_table_rejects_missing_columns():
    df = _good_rt().drop(columns=["child_name"])
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_relationship_table(df)


def test_relationship_table_rejects_unknown_event_type():
    df = _good_rt()
    df.loc[0, "event_type"] = "Whatever"
    with pytest.raises(SchemaError, match="unknown event_type"):
        validate_relationship_table(df)


def _good_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["IN.ADM2.00001", "IN.ADM2.00010"],
            "year": [2018, 2019],
            "season": ["Kharif", "Kharif"],
            "variable": ["rice_area_ha", "rice_area_ha"],
            "value": [100.0, 95.0],
        }
    )


def test_stats_accepts_canonical_schema():
    validate_stats(_good_stats(), base_year=2003)


def test_stats_rejects_pre_base_year_rows():
    df = _good_stats()
    df.loc[0, "year"] = 1999
    with pytest.raises(SchemaError, match="year < base_year"):
        validate_stats(df, base_year=2003)


def test_stats_rejects_missing_required_columns():
    df = _good_stats().drop(columns=["variable"])
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_stats(df, base_year=2003)


def test_name_change_log_accepts_canonical_schema():
    df = pd.DataFrame(
        {
            "event_year": [2016],
            "unit_id": ["IN.ADM2.00013"],
            "old_name": ["District K"],
            "new_name": ["District H"],
        }
    )
    validate_name_change_log(df)


def test_version_is_declared_once_in_effect():
    """`__version__` and pyproject's version must agree.

    The version is written in two places and nothing kept them in step. Bump
    one and the package ships announcing the other -- which the reproduction
    environment cannot catch, because it pins by git tag and never reads the
    version string at all. A release would then be unidentifiable from inside
    the package.
    """
    import re
    from pathlib import Path

    import stablebound

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    # Regex rather than tomllib: tomllib is 3.11+, and this file must work on
    # the 3.10 the project targets.
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert m, "no version found in pyproject.toml"
    assert stablebound.__version__ == m.group(1), (
        f"stablebound.__version__ is {stablebound.__version__!r} but "
        f"pyproject.toml says {m.group(1)!r}"
    )

    # CITATION.cff is the third declaration site and the one that actually
    # drifted: it sat at 0.1.0 through two releases, because nothing reads it
    # on any code path. It is what a citing paper quotes, so a stale version
    # here misattributes results to a release that did not produce them.
    citation = (Path(__file__).resolve().parents[1] / "CITATION.cff").read_text()
    c = re.search(r"^version:\s*(\S+)", citation, re.MULTILINE)
    assert c, "no version found in CITATION.cff"
    assert stablebound.__version__ == c.group(1), (
        f"stablebound.__version__ is {stablebound.__version__!r} but "
        f"CITATION.cff says {c.group(1)!r}"
    )
