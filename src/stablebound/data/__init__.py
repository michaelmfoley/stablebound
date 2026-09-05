"""Bundled country data registry.

StableBound ships the canonical relationship table, baseline and name-change
log for India, the country described in the StableBound data paper. Use
``Lineage("IN")`` to load it without specifying paths. Custom-country users
pass paths into ``Lineage(...)`` directly.

India is the only bundled country. Draft tables for other countries were
carried in earlier development versions and removed from the public release
(0.1.5); adding a country means adding a validated entry here and widening
the package-data globs in ``pyproject.toml``.

Each :class:`BundledCountry` may also ship a country-specific
``normalizer`` — used by the matcher (:mod:`stablebound.match`) when
the user doesn't pass their own. India's bundled normalizer strips
common administrative suffixes ("district", "dist.") and honorifics
("Dr.", "Sri", "Shri") that the package-default normalizer leaves
alone.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from ..match import normalize_name as _default_normalize_name


@dataclass(frozen=True)
class BundledCountry:
    """Pointers to the bundled files for a single country.

    All paths are absolute (resolved from package resources at import
    time). ``name_change_log_path`` may be ``None`` if the country
    doesn't track renames in a separate file (no bundled country does today).

    ``validity_start_year`` is the earliest year for which the
    bundled lineage + baseline supports a valid stable-boundary
    product. Earlier years lack the baseline coverage needed for
    correct attribution.

    ``admin_level`` is the level of ``lineage_path`` — the level the products
    operate on. It is recorded rather than inferred from id prefixes, which
    only happen to encode it for the countries bundled today.

    ``coverage_end_year`` is the last year the SOURCE affirms the unit set —
    not the last event year. The case that motivated the field was a draft
    Bangladesh table whose final event was 1986 while its FEWS vintages ran
    to 2015, all showing the same 64 districts: positive evidence of no change
    through 2015. That table is no longer in this repository; the behaviour is
    pinned in ``tests/test_coverage.py`` on a synthetic country. Beyond this
    year the lineage is silent rather than confirmed-stable, which is a
    different claim: a product built past it may simply be missing later
    reorganisations.

    ``coarse_lineage_path`` points at the parent admin level's events when
    the country maintains them separately (India: ADM1). ``None`` otherwise.

    ``normalizer`` is the recommended name-normalizer for this
    country's data, layered on top of the package default. ``None``
    means the package default :func:`stablebound.match.normalize_name`
    is fine as-is.
    """

    country_code: str
    lineage_path: Path
    baseline_path: Path
    name_change_log_path: Path | None
    validity_start_year: int
    notes: str = ""
    # Events for the level ABOVE ``lineage_path``'s level, in that level's own
    # id space. India's states split too (Uttarakhand 2000, Telangana 2014),
    # and the FEWS relationship table needs those rows alongside the district
    # ones. ``None`` for single-level countries, where the export uses the
    # country's own graph for both levels.
    coarse_lineage_path: Path | None = None
    coverage_end_year: int | None = None
    admin_level: int = 1
    normalizer: Callable[[str], str] | None = field(default=None, compare=False)


def _data_path(*parts: str) -> Path:
    """Return an absolute filesystem path for a bundled data file.

    Bundled files live inside the package data directory and ship with the
    wheel via ``[tool.setuptools.package-data]``.
    """
    # ``resources.files`` returns a Traversable; coerce to a real Path
    # so the existing io loaders (which call ``Path.exists()`` etc.)
    # work unchanged.
    return Path(str(resources.files(__name__).joinpath(*parts)))


# --- Country-specific normalizers ---------------------------------------


_IN_SUFFIXES = (" district", " dist.", " dist")
_IN_HONORIFICS = (r"\bdr\.\s*", r"\bsri\s+", r"\bshri\s+")

#: DESAGRI writes district names with the state abbreviation appended —
#: "Villupuram (TN)", "Beed (MH)", "Saran (BR)". Matched narrowly (2-3 capitals
#: in trailing parentheses) so a genuine parenthetical in a unit name is left
#: alone. Anchored to the end for the same reason.
_IN_STATE_SUFFIX = re.compile(r"\s*\(([A-Z]{2,3})\)\s*$")


def _india_normalizer(s: str) -> str:
    """India-specific normalizer.

    Layered on top of :func:`stablebound.match.normalize_name`:
    strips the DESAGRI state-abbreviation suffix, administrative suffixes
    (variants of "District") and honorifics ("Dr.", "Sri", "Shri") that show up
    in administrative name lists but aren't structurally part of the name.
    These rules were ported from the legacy India production matcher
    (``tools/india/prepare_shapefile.py``).

    The state-abbreviation strip was added 2026-07-27 after measuring the
    generic matcher against ``stablebound.india.matcher`` on the real DESAGRI
    statistics (``qualification/compare_matchers.py``). Without it the
    normalizer produced ``"villupuram tn"``, which can never match
    ``"villupuram"``, and the documented path
    (``Lineage("IN").propose_stats_mapping``) matched 13,503 of 17,049
    combinations — 79%. With it, 16,616 — 97.5%, slightly ahead of the bespoke
    India matcher's 16,468, and agreeing with it on 99.86% of the rows both
    resolve.

    That measurement is also why India's extra match passes were NOT ported
    into the generic ladder: nearly the whole gap was this one source-format
    quirk, not an algorithmic difference.
    """
    # Drop the state abbreviation the source tacks on, turning a name like
    # "Villupuram (TN)" back into "Villupuram". This has to happen before the
    # name is lowercased below, because it recognises the abbreviation by its
    # capital letters. Moving it later breaks nothing loudly — it just stops
    # working, and the match rate quietly falls by about a fifth.
    s = _IN_STATE_SUFFIX.sub("", str(s).strip())
    s = s.lower().strip()
    # Strip the word "district" and honorifics like "Dr." or "Sri", which
    # appear in official name lists but aren't part of the name.
    for suffix in _IN_SUFFIXES:
        s = s.replace(suffix, "")
    for pat in _IN_HONORIFICS:
        s = re.sub(pat, "", s)
    # Finish with the tidying every country gets, so India only has to
    # describe what makes it different.
    return _default_normalize_name(s)


BUNDLED_COUNTRIES: dict[str, BundledCountry] = {
    "IN": BundledCountry(
        country_code="IN",
        lineage_path=_data_path("IN", "lineage.xlsx"),
        baseline_path=_data_path("IN", "baseline.csv"),
        name_change_log_path=_data_path("IN", "name_change_log.xlsx"),
        coarse_lineage_path=_data_path("IN", "lineage_adm1.xlsx"),
        coverage_end_year=2025,  # Hand-authored through 2025.
        admin_level=2,
        validity_start_year=1991,
        notes=(
            "ADM2 (district-level) lineage. Baseline is the 1991 admin "
            "snapshot — products are only valid for base_year >= 1991."
        ),
        normalizer=_india_normalizer,
    ),

}


__all__ = ["BundledCountry", "BUNDLED_COUNTRIES"]
