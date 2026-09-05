"""One-command smoke test for the Exampleland bundled fixture.

Runs both the stable and modern boundary pipelines end-to-end against
the synthetic 5-district dataset, writing outputs to a temporary
directory and printing a short summary. Useful as:

    python examples/exampleland/run_pipeline.py

to confirm the package is installed and working, or as the smallest
possible end-to-end example for someone new to the codebase.

The lineage object and constants come from ``examples/exampleland/
config.py``. The shapefile already has canonical unit_ids attached;
the matcher isn't exercised here. See section 5 of ``examples/README.md``
for the matching workflow on a shapefile with imperfect names, and
``tools/india/prepare_shapefile.py`` for the real India run (which needs
the source data tree).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure the repo root is on sys.path so `examples.exampleland.config`
# is importable when this script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.exampleland.config import (  # noqa: E402
    INTENSIVE,
    MAX_YEAR,
    STATS_PATH,
    TARGET_YEAR,
    lineage,
)
from stablebound import ModernBoundary, StableBoundary  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "exampleland_smoke"

        print("== Stable product ==")
        sb = StableBoundary(
            lineage,
            target_year=TARGET_YEAR,
            max_year=MAX_YEAR,
            output_dir=out,
        )
        sb.build_boundaries()
        sb.aggregate_stats(stats=STATS_PATH, intensive=INTENSIVE)
        stable_summary = sb.summary()
        print(f"  n_modern_units:           {stable_summary['n_modern_units']}")
        print(f"  n_stable_groups_at_base:  {stable_summary['n_stable_groups_at_base']}")
        print(f"  year_range:               {stable_summary['year_range']}")
        print(f"  long-form stats rows:     {len(sb.get_stats())}")

        print()
        print("== Modern product ==")
        mb = ModernBoundary(
            lineage,
            target_year=TARGET_YEAR,
            output_dir=out,
        )
        mb.aggregate_stats(stats=STATS_PATH, intensive=INTENSIVE)
        modern_summary = mb.summary()
        print(f"  n_modern_units:        {modern_summary['n_modern_units']}")
        print(f"  n_events_processed:    {modern_summary['n_events_processed']}")
        print(f"  n_late_reporting_rows: {modern_summary['n_late_reporting_rows']}")
        print(f"  long-form modern rows: {len(mb.get_modern_stats())}")

        print()
        print(f"Outputs were written under {out} (temporary; cleaned up at exit).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
