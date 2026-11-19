"""India as it is published: headers, column mapping and name rules as data.

India's relationship tables are maintained in the stablebound-india
repository and published in its Zenodo deposit under their own file names
and headers (``ADM2_ID``, ``CHANGE_YEAR``, ``OLD_OFFICIAL_NAME`` ...), with a
``stablebound.json`` beside them. ``tests/fixtures/india_as_shipped`` is a
frozen copy of exactly that folder (see its PROVENANCE.md). This module holds
what the tests need to say about it: the rules and mapping its description
must carry, the names it contains, and a way to copy it with a planted fault.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tests.conftest import INDIA_FIXTURE, india_tables

#: India's name-tidying rules, written as data. They restate the India
#: normalizer the package carried until 0.2.0, step for step; the fixture's
#: stablebound.json must carry exactly these (test_country_folder checks).
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
        "ADM1_ID": "coarse_id",
        "ADM1_NAME": "coarse_name",
        "ADM2_ID": "unit_id",
        "ADM2_NAME": "name",
    },
    "name_change_log": {
        "CHANGE_YEAR": "event_year",
        "OLD_OFFICIAL_NAME": "old_name",
        "NEW_OFFICIAL_NAME": "new_name",
    },
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

#: India's numbers recorded 18 Nov 2026 from ``Lineage("IN")`` — the copy then
#: bundled in the package (stablebound 0.1.5) — before any of the 0.2.0 work
#: (run B0 of the unbundle plan). They come from that run, not from the code
#: under test, so a wrong mapping, fact or rule shows up as a changed number.
B0_ANCHORS = {
    "admin_level": 2,
    "validity_start_year": 1991,
    "coverage_end_year": 2025,
    "min_year": 1991,
    "n_events": 1019,
    "n_baseline_rows": 467,
    "n_name_change_rows": 50,
    "units_by_year": {1991: 467, 1997: 523, 2000: 592, 2001: 592, 2011: 641, 2014: 674,
                      2019: 723, 2022: 745, 2024: 785, 2025: 778},
    "states_1991": 32,
    "validation_findings": 45,
    "normalized": {
        "Bijapur Dist.": "bijapur",
        "Dr. B.R. Ambedkar Konaseema": "br ambedkar konaseema",
        "Hamirpur District": "hamirpur",
        "Shri Ganganagar": "ganganagar",
        "Sri Potti Sriramulu Nellore": "potti sriramulu nellore",
        "Villupuram (TN)": "villupuram",
    },
}


def india_names() -> list[str]:
    """Every distinct name in India's four tables, plus the awkward set."""
    t = india_tables()
    names: set[str] = set(AWKWARD_NAMES)
    for key in ("rt", "adm1"):
        for col in ("parent_name", "child_name", "parent_coarse_name", "child_coarse_name"):
            names.update(t[key][col].dropna().astype(str))
    names.update(t["base"]["name"].dropna().astype(str))
    names.update(t["base"]["coarse_name"].dropna().astype(str))
    names.update(t["ncl"]["old_name"].dropna().astype(str))
    names.update(t["ncl"]["new_name"].dropna().astype(str))
    return sorted(names)


def anchors_of(ln) -> dict:
    """The same numbers as :data:`B0_ANCHORS`, read off a ``Lineage``."""
    from stablebound.snapshot import build_snapshot
    from stablebound.validate import coarse_universe

    adm1 = ln.graph(level=1)
    universe = coarse_universe(ln.lineage, adm1, baseline=ln.baseline)
    return {
        "admin_level": ln.admin_level,
        "validity_start_year": ln.validity_start_year,
        "coverage_end_year": ln.coverage_end_year,
        "min_year": ln.min_year,
        "n_events": len(ln.lineage.events),
        "n_baseline_rows": len(ln.baseline),
        "n_name_change_rows": len(ln.name_change_log),
        "units_by_year": {y: len(ln.snapshot(year=y)) for y in B0_ANCHORS["units_by_year"]},
        "states_1991": len(set(build_snapshot(adm1, 1991, additional_units=universe))),
        "validation_findings": len(ln.validation_issues),
        "normalized": {n: ln.default_normalizer(n) for n in B0_ANCHORS["normalized"]},
    }


def copy_india_folder(dest: Path, **changes) -> Path:
    """Copy the frozen fixture to ``dest``, replacing top-level JSON keys.

    A value of ``None`` deletes the key. For tests that plant a fault; the
    fixture itself is never modified.
    """
    dest = Path(dest)
    shutil.copytree(INDIA_FIXTURE, dest)
    p = dest / "stablebound.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest
