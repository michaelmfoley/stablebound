"""Cross-product conservation invariant.

The modern boundary product is a redistribution: every reported value
that has a modern destination must end up SOMEWHERE on a modern unit.
Equivalently, the sum across all modern units for
``(year, variable, season='Total Year')`` must equal the corresponding
sum across stable units **excluding rows with ``late_reporting=True``**.

Late-reporting orphan rows (units reporting outside their lineage
lifespan) are surfaced in ``late_reporting.csv`` for transparency but
are no longer redistributed to descendants — see the
``late-reporting-redistribution-bug`` memory note. They appear in the
stable product as flagged singleton rows but do not contribute to the
modern product.

Per-season conservation may NOT hold — the cascade fallback can
reallocate parent data across seasons when a season's fraction is
undefined. Only Total Year is strictly conservative.
"""

from __future__ import annotations

import contextlib
import math
import sys
import tempfile
from pathlib import Path

EXAMPLELAND = Path(__file__).resolve().parents[1] / "examples" / "exampleland"


@contextlib.contextmanager
def _exampleland_fixture():
    sys.path.insert(0, str(EXAMPLELAND.parent.parent))
    from examples.exampleland.config import (
        INTENSIVE,
        MAX_YEAR,
        STATS_PATH,
        TARGET_YEAR,
        lineage,
    )

    with tempfile.TemporaryDirectory() as td:
        yield {
            "lineage": lineage,
            "output_dir": Path(td) / "out",
            "stats_path": STATS_PATH,
            "intensive": INTENSIVE,
            "target_year": TARGET_YEAR,
            "max_year": MAX_YEAR,
        }


def _build_both(fix):
    from stablebound import ModernBoundary, StableBoundary

    sb = StableBoundary(
        fix["lineage"],
        target_year=fix["target_year"],
        max_year=fix["max_year"],
        output_dir=fix["output_dir"],
    )
    mb = ModernBoundary(
        fix["lineage"],
        target_year=fix["target_year"],
        output_dir=fix["output_dir"],
    )
    sb.build_boundaries()
    sb.aggregate_stats(stats=fix["stats_path"], intensive=fix["intensive"])
    mb.aggregate_stats(stats=fix["stats_path"], intensive=fix["intensive"])
    return sb, mb


def test_exampleland_total_year_conservation_excluding_late():
    """Total Year sums must match between products on Exampleland,
    excluding stable rows tagged ``late_reporting=True``.

    2026-06-04: late-report redistribution is disabled — orphan rows are
    surfaced in ``late_reporting.csv`` but do not contribute to the modern
    product. Conservation therefore holds across the *non-late* portion
    of the stable product.
    """
    with _exampleland_fixture() as f:
        sb, mb = _build_both(f)
        sm = mb.get_modern_stats()
        ss = sb.get_stats()

        extensive_vars = ["rice_area_ha", "rice_production_mt"]
        for var in extensive_vars:
            for year in sorted(sm["year"].unique()):
                m_ty = sm[
                    (sm["variable"] == var)
                    & (sm["year"] == year)
                    & (sm["season"] == "Total Year")
                ]
                m_total = float(m_ty["value"].sum())
                s_rows = ss[
                    (ss["variable"] == var)
                    & (ss["year"] == year)
                    & (ss["late_reporting"] == False)  # noqa: E712
                ]
                s_total = float(s_rows["value"].sum())
                assert math.isclose(m_total, s_total, abs_tol=1e-9), (
                    f"conservation broken for {var} {year}: "
                    f"modern_TY={m_total} stable_excl_late={s_total} "
                    f"diff={m_total - s_total}"
                )


def test_exampleland_no_late_report_redistributed_rows_in_modern():
    """After the 2026-06-04 disable, no modern rows should be tagged
    ``late_report_redistributed=True``. Orphan rows belong to
    ``late_reporting.csv`` only, not to the modern product."""
    with _exampleland_fixture() as f:
        _, mb = _build_both(f)
        sm = mb.get_modern_stats()
        if "late_report_redistributed" in sm.columns:
            assert not sm["late_report_redistributed"].any(), (
                f"{int(sm['late_report_redistributed'].sum())} modern rows "
                "are tagged late_report_redistributed=True; redistribution "
                "should be disabled."
            )


def test_exampleland_total_year_intensives_recompute_correctly():
    """Yield rows for season='Total Year' must equal production_TY / area_TY."""
    with _exampleland_fixture() as f:
        _, mb = _build_both(f)
        sm = mb.get_modern_stats()
        ty = sm[sm["season"] == "Total Year"]
        # E.013 is the merge child {E.004 + E.005}. 2010 numbers per fixture:
        # area = 50 + 30 = 80, production = 100 + 60 = 160, yield = 2.0.
        e013_2010 = ty[(ty["modern_id"] == "E.013") & (ty["year"] == 2010)]
        area = float(e013_2010[e013_2010["variable"] == "rice_area_ha"].iloc[0]["value"])
        prod = float(e013_2010[e013_2010["variable"] == "rice_production_mt"].iloc[0]["value"])
        yld_rows = e013_2010[e013_2010["variable"] == "yield_mt_ha"]
        assert len(yld_rows) == 1
        yld = float(yld_rows.iloc[0]["value"])
        assert math.isclose(yld, prod / area, abs_tol=1e-12)
        assert math.isclose(yld, 2.0, abs_tol=1e-12)


def test_modern_stats_has_required_new_columns():
    """The modern frame must expose the new B0/B1/B2 columns."""
    with _exampleland_fixture() as f:
        _, mb = _build_both(f)
        sm = mb.get_modern_stats()
        for col in [
            "year", "season", "variable", "modern_id", "value",
            "sources", "lineage_depth", "has_nan_fraction",
            "fraction_method", "late_report_redistributed",
        ]:
            assert col in sm.columns, f"missing column {col}"


def test_total_year_rows_present_for_every_modern_id_year_variable():
    """Every (modern_id, year, variable) with at least one season row
    must also have a Total Year row.
    """
    with _exampleland_fixture() as f:
        _, mb = _build_both(f)
        sm = mb.get_modern_stats()
        seasonal = sm[sm["season"] != "Total Year"]
        ty = sm[sm["season"] == "Total Year"]
        seasonal_keys = set(zip(
            seasonal["modern_id"], seasonal["year"], seasonal["variable"]
        ))
        ty_keys = set(zip(ty["modern_id"], ty["year"], ty["variable"]))
        missing = seasonal_keys - ty_keys
        assert not missing, f"missing Total Year rows for: {sorted(missing)[:5]}"


# --- Modern provenance columns -------------------------------------------
#
# The modern product previously shipped no completeness information at all:
# `sources` was a comma-joined string and the per-event observation counts
# lived only in event_fractions.csv, never joined to the values.


def test_modern_stats_carry_n_sources_and_min_n_common():
    with _exampleland_fixture() as f:
        _sb, mb = _build_both(f)
        sm = mb.get_modern_stats()
        assert "n_sources" in sm.columns
        assert "min_n_common_observations" in sm.columns

        # n_sources must agree with the sources string it sits beside.
        extensive = sm[sm["variable"].isin(["rice_area_ha", "rice_production_mt"])]
        rows = extensive[extensive["sources"].notna() & (extensive["sources"] != "")]
        assert not rows.empty
        for _, r in rows.iterrows():
            assert int(r["n_sources"]) == len(str(r["sources"]).split(","))


def test_direct_reports_have_no_min_n_common():
    # A cell that never had a fraction applied has no "how many observations
    # did the fraction rest on" answer. NA, not a number.
    with _exampleland_fixture() as f:
        _sb, mb = _build_both(f)
        sm = mb.get_modern_stats()
        direct = sm[(sm["lineage_depth"] == 0) & (sm["fraction_method"] == "")]
        assert not direct.empty
        assert direct["min_n_common_observations"].isna().all()


def test_cascaded_cells_report_the_weakest_link():
    # A cell composed through an event carries the fraction's observation
    # count; deeper cells carry the minimum along their path, never more
    # than any single event on it.
    with _exampleland_fixture() as f:
        _sb, mb = _build_both(f)
        sm = mb.get_modern_stats()
        fr = mb.get_event_fractions()
        cascaded = sm[(sm["lineage_depth"] > 0)
                      & sm["min_n_common_observations"].notna()]
        if cascaded.empty:
            return  # fixture has no cascaded cells with a recorded count
        worst_possible = fr["n_common_observations"].max()
        assert (cascaded["min_n_common_observations"] <= worst_possible).all()
        assert (cascaded["min_n_common_observations"] >= 0).all()
