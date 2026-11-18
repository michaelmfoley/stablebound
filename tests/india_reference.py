"""India as it is published: headers, column mapping and name rules as data.

India's relationship tables are maintained in the stablebound-india
repository and published in its Zenodo deposit with their own headers
(``ADM2_ID``, ``CHANGE_YEAR``, ``OLD_OFFICIAL_NAME`` ...), not the package's
canonical ones. ``write_india_folder`` rebuilds that published form from the
bundled copy — the bundle was converted from those files by renaming columns
only, so renaming them back gives the published headers with the same cells —
and writes a ``stablebound.json`` beside them. The tests then load the folder
with ``Lineage.from_folder`` and require it to equal ``Lineage("IN")``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stablebound import BUNDLED_COUNTRIES

#: India's name-tidying rules, written as data. They restate
#: ``stablebound.data._india_normalizer`` step for step.
INDIA_NAME_RULES: dict[str, list[str]] = {
    "strip_before_lowercase": [r"\s*\(([A-Z]{2,3})\)\s*$"],
    "remove_substrings": [" district", " dist.", " dist"],
    "remove_patterns": [r"\bdr\.\s*", r"\bsri\s+", r"\bshri\s+"],
}

#: Published header -> canonical name, for the two tables whose headers differ
#: by more than case. The lineage tables need no entry (``EVENT_YEAR`` is
#: ``event_year`` once lowercased).
INDIA_COLUMNS: dict[str, dict[str, str]] = {
    "baseline": {
        "YEAR": "year",
        "ADM1_ID": "coarse_id",
        "ADM1_NAME": "coarse_name",
        "ADM2_ID": "unit_id",
        "ADM2_NAME": "name",
    },
    "name_change_log": {
        "CHANGE_YEAR": "event_year",
        "LEVEL": "level",
        "UNIT_ID": "unit_id",
        "OLD_OFFICIAL_NAME": "old_name",
        "NEW_OFFICIAL_NAME": "new_name",
    },
}

#: The deposit's file names (stablebound-india package_deposit.py).
INDIA_FILES = {
    "relationship_table": "lineage_adm2.xlsx",
    "coarse_lineage": "lineage_adm1.xlsx",
    "baseline": "baseline_1991.csv",
    "name_change_log": "name_changes.xlsx",
}

#: Names that exercise every rule: state suffixes, each "district" variant,
#: each honorific, and things the rules must leave alone.
AWKWARD_NAMES = (
    "Villupuram (TN)",
    "Beed (MH)",
    "Saran (BR)",
    "Hamirpur District",
    "Bijapur Dist.",
    "Bijapur Dist",
    "Dr. B.R. Ambedkar Konaseema",
    "Sri Potti Sriramulu Nellore",
    "Shri Ganganagar",
    "Sri Ganganagar",
    "North 24 Parganas",
    "Purba Bardhaman (East)",
    "Y.S.R.",
    "Ri-Bhoi",
    "Anantnag (Islamabad)",
    "  Leh (Ladakh) ",
    "Kanniyakumari",
    "Kamrup Metropolitan",
    "Seraikela-Kharsawan",
)


def india_names() -> list[str]:
    """Every distinct name in the bundled India tables, plus the awkward set."""
    b = BUNDLED_COUNTRIES["IN"]
    names: set[str] = set(AWKWARD_NAMES)
    for path in (b.lineage_path, b.coarse_lineage_path):
        df = pd.read_excel(path, dtype=str)
        for col in ("parent_name", "child_name", "parent_coarse_name", "child_coarse_name"):
            names.update(df[col].dropna())
    base = pd.read_csv(b.baseline_path, dtype=str)
    names.update(base["name"].dropna())
    names.update(base["coarse_name"].dropna())
    ncl = pd.read_excel(b.name_change_log_path, dtype=str)
    names.update(ncl["old_name"].dropna())
    names.update(ncl["new_name"].dropna())
    return sorted(names)


def india_description(**override) -> dict:
    """The ``stablebound.json`` content for India's published folder."""
    b = BUNDLED_COUNTRIES["IN"]
    desc = {
        "_about": "India as published by stablebound-india; rebuilt for the tests.",
        "format": "stablebound-country/1",
        "country_code": "IN",
        "admin_level": b.admin_level,
        "validity_start_year": b.validity_start_year,
        "coverage_end_year": b.coverage_end_year,
        "notes": b.notes,
        "files": dict(INDIA_FILES),
        "columns": {t: dict(m) for t, m in INDIA_COLUMNS.items()},
        "name_rules": {k: list(v) for k, v in INDIA_NAME_RULES.items()},
    }
    desc.update(override)
    return desc


def write_india_folder(dest: Path, **override) -> Path:
    """Write India's four tables under their published headers, plus the JSON.

    Returns ``dest``. Keyword arguments replace top-level keys of the
    description (e.g. ``admin_level=1``), for tests that plant a fault.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    b = BUNDLED_COUNTRIES["IN"]

    for src, name in ((b.lineage_path, INDIA_FILES["relationship_table"]),
                      (b.coarse_lineage_path, INDIA_FILES["coarse_lineage"])):
        df = pd.read_excel(src)
        df.columns = [c.upper() for c in df.columns]
        df.to_excel(dest / name, index=False)

    base = pd.read_csv(b.baseline_path)
    to_published = {v: k for k, v in INDIA_COLUMNS["baseline"].items()}
    base = base.rename(columns=to_published)
    base["Alias"] = ""  # the published baseline carries it; the package ignores it
    base[["YEAR", "ADM1_ID", "ADM1_NAME", "ADM2_ID", "ADM2_NAME", "Alias"]].to_csv(
        dest / INDIA_FILES["baseline"], index=False
    )

    ncl = pd.read_excel(b.name_change_log_path)
    ncl = ncl.rename(columns={v: k for k, v in INDIA_COLUMNS["name_change_log"].items()})
    ncl[["CHANGE_YEAR", "LEVEL", "UNIT_ID", "OLD_OFFICIAL_NAME", "NEW_OFFICIAL_NAME"]].to_excel(
        dest / INDIA_FILES["name_change_log"], index=False
    )

    (dest / "stablebound.json").write_text(
        json.dumps(india_description(**override), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest
