"""Coverage metadata: how current is a lineage, and does the analysis window
respect that?

``coverage_end_year`` answers a question the lineage alone cannot: the last
year the SOURCE affirms the unit set. It is not the last event year. The case
that motivated the field was a district table whose final event was 1986 while
the publisher's vintages ran to 2015, all showing the same units — so 1986 is
"nothing changed", while 2016 onward is "we do not know".

That distinction drives two behaviours pinned here: the default analysis window
extends to the coverage end, and asking past it warns instead of silently
implying the boundaries are attested.

India cannot be the subject — its last event year and its coverage end are both
2025 — so the synthetic ``legacy_admin1`` country (last event 2025) is loaded
with a stated coverage end of 2030.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import geopandas as gpd

from stablebound import Lineage, StableBoundary
from tests.conftest import get_synthetic

SYN = get_synthetic("legacy_admin1")
NAME_COL = "unit_name"
LEGACY_RT = Path(__file__).resolve().parent / "fixtures/rt_convert/relationshiptable_XX.csv"


def _covered(synthetic_with_facts) -> Lineage:
    """The synthetic country, with a stated coverage end past its last event."""
    return synthetic_with_facts("legacy_admin1", code="ZY", coverage_end_year=2030)


def _attach(ln: Lineage) -> None:
    gdf = gpd.read_file(SYN.shapefile_path)
    prop = ln.propose_shapefile_mapping(gdf, name_column=NAME_COL)
    ln.attach_shapefile(gdf, mapping=prop.proposals, name_column=NAME_COL)


def test_india_as_shipped_declares_coverage(india):
    """India's stablebound.json states its coverage, and it is not before validity."""
    assert india.coverage_end_year == 2025
    assert india.coverage_end_year >= india.validity_start_year


def test_coverage_is_not_merely_the_last_event_year(synthetic_with_facts):
    """The field exists because these two years can differ."""
    ln = _covered(synthetic_with_facts)
    assert ln.lineage.max_event_year == 2025
    assert ln.coverage_end_year == 2030


def test_custom_country_has_no_coverage_claim(tmp_path):
    """The package cannot know how current a user-supplied lineage is."""
    rt = tmp_path / "rt.csv"
    rt.write_text(
        "event_year,event_type,parent_id,parent_name,child_id,child_name\n"
        "2000,Split,A,Alpha,B,Beta\n"
    )
    ln = Lineage("ZZ", relationship_table_path=rt)
    assert ln.coverage_end_year is None


def test_default_window_extends_to_coverage_end(synthetic_with_facts):
    """The modern unit set first appears in snapshot(2026); year inference ties
    across the unchanged years after that and takes the earliest, so the
    inferred vintage is 2026 for a country covered through 2030. Without the
    coverage floor the build would stop at the last event.
    """
    ln = _covered(synthetic_with_facts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _attach(ln)
        assert ln.shapefile_year < ln.coverage_end_year, "premise of this test"
        with tempfile.TemporaryDirectory() as td:
            sb = StableBoundary(ln, output_dir=Path(td))
            sb.build_boundaries()
            years = sorted(
                int(p.stem.split("_")[1]) for p in Path(td).glob("stable_*.geojson")
            )
    assert max(years) == ln.coverage_end_year
    assert len(years) > 3


def test_asking_past_coverage_warns_but_still_builds(synthetic_with_facts):
    ln = _covered(synthetic_with_facts)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _attach(ln)
        with tempfile.TemporaryDirectory() as td:
            sb = StableBoundary(ln, max_year=2035, output_dir=Path(td))
            sb.build_boundaries()
            report = (Path(td) / "shapefile_vintage_report.txt").read_text()
            built = sorted(Path(td).glob("stable_*.geojson"))
        msgs = [str(w.message) for w in caught if "coverage end" in str(w.message)]

    assert len(msgs) == 1, f"expected one coverage warning, got {len(msgs)}"
    assert "2030" in msgs[0] and "2035" in msgs[0]
    # A warning, not a refusal — the user may know better than the source.
    assert built
    # And it is recorded on disk, not only in a console the user may not see.
    assert "exceeds coverage end" in report


def test_no_warning_inside_coverage(synthetic_with_facts):
    ln = _covered(synthetic_with_facts)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _attach(ln)
        with tempfile.TemporaryDirectory() as td:
            StableBoundary(ln, max_year=2028, output_dir=Path(td)).build_boundaries()
        msgs = [str(w.message) for w in caught if "coverage end" in str(w.message)]
    assert not msgs


def test_from_legacy_rt_derives_coverage_from_its_vintages():
    """A converted RT knows its own last vintage."""
    ln = Lineage.from_legacy_rt(LEGACY_RT, country="XX", admin_level=1)
    # The fixture publishes vintages 2010/2015/2020/2025.
    assert ln.coverage_end_year == 2025
