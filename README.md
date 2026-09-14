# StableBound

A reproducible framework for harmonizing subnational data across administrative
boundary changes.

> **Status: v0.1.5.** API has stabilized around the
> `Lineage` / `StableBoundary` / `ModernBoundary` triad. India is the
> bundled, validated country and the one this release describes; the
> method itself is country-agnostic, and any country can be loaded from
> its own relationship table (see [Inputs](#inputs)).

## Overview

Administrative boundaries shift constantly through splits, merges,
redistributions, or renaming of territory. When subnational statistics (crop production,
demographics, etc.) are reported on top of these shifting boundaries,
simple name- or ID-matching across years misattributes data wherever a
boundary changed.

StableBound takes:

1. A **canonical relationship table** describing every boundary event,
2. A **baseline snapshot** listing units alive at the earliest covered year,
3. A **modern shapefile** of the country (with names — IDs get matched
   in-package), and
4. (Optional) A **statistics table**,

and produces two complementary products:

- **Stable boundaries**: for any chosen target year, dissolve the
  modern shapefile into a set of stable groups that preserve additive
  sums across the entire timeframe from that target year onwards. Outputs a per-year
  shapefile plus a long-form aggregated statistics table on consistent
  geography.
- **Modern boundaries**: rescale historical statistics onto today's
  shapefile, with audit columns showing which fraction each cell came
  from and which events were composed.

In other words, StableBound takes care of unifying spatial and tabular information
with historical administrative unit evolution to produce accurate and harmonized
datasets.

For India the relationship table, baseline and name-change log are bundled;
`Lineage("IN")` is all a user needs. StableBound also includes functionality
to provide your own relationship table and shapefile for any country.

## Install

Directly from GitHub, pinned to a release tag:

```bash
pip install "git+https://github.com/michaelmfoley/stablebound.git@v0.1.5"
```

A wheel and sdist for each release will also be attached to the Zenodo deposit of
the India dataset (see the `stablebound-india` repository).

For development, from a local clone:

```bash
pip install -e .
```

Python ≥ 3.10. Depends on `pandas`, `numpy`, `geopandas`, `shapely`,
`openpyxl`.

## Quickstart

This runs immediately after install:

```python
from stablebound import Lineage

ln = Lineage("IN")      # the bundled, validated lineage (India)

ln.lineage              # parsed LineageGraph
ln.snapshot(year=2014)  # DataFrame of canonical (unit_id, unit_name) pairs
ln.years                # range(1991, 2027): the years the lineage can answer for
print(ln.validation_report())  # 8 automatic data-quality checks
```

For a guided end-to-end tour (lineage → stable boundaries → aggregated
stats → modern product) on a tiny synthetic country, open
[`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) or run
the one-command smoke test (both require a clone — see
[Run the bundled example](#run-the-bundled-example)).

### With your own shapefile and stats

Boundary products need a modern shapefile, and stats aggregation needs a 
long-form statistics table:

```python
from stablebound import Lineage, StableBoundary

ln = Lineage("IN")

# Attach a modern shapefile. The matcher fuzzy-matches feature names
# against the lineage; pass coarse_column when your shapefile has a
# state/region column to auto-disambiguate homonyms (e.g., the two
# "Hamirpur" districts in India). Review the proposal CSV before
# attaching.
proposal = ln.propose_shapefile_mapping(
    "my_modern_shapefile.shp",
    name_column="DISTRICT_NAME",
    coarse_column="STATE",                    # optional but recommended
)
proposal.to_csv("mapping.csv")                # review/edit in Excel
ln.attach_shapefile(
    "my_modern_shapefile.shp", mapping="mapping.csv", name_column="DISTRICT_NAME",
)

# Stable product: per-year dissolved shapefiles + aggregated stats.
sb = StableBoundary(ln, target_year=1997)
sb.build_boundaries()
sb.aggregate_stats(
    stats="my_stats.csv",
    extensive=["rice_area_ha", "rice_production_mt"],
    # Yield is recomputed per stable polygon from the units that reported
    # BOTH production and area in that year and season -- never summed,
    # never averaged, never biased by a partial report.
    intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
)
sb.get_stats(variable="yield_mt_ha", year=2018)
```

## Inputs

### Bundled relationship tables

`Lineage("IN")` loads the bundled India data from `src/stablebound/data/IN/`
(see `stablebound.BUNDLED_COUNTRIES["IN"].notes` for coverage and provenance).
India is the only bundled country for now (others to come). To bring your own, supply paths:

```python
ln = Lineage(
    "PT",
    relationship_table_path="my_country/relationship_table.csv",
    baseline_path="my_country/baseline.csv",
    name_change_log_path="my_country/name_changes.xlsx",  # optional
)
```

If your relationship table is written in names and only the baseline has
ids, `assign_unit_ids(lineage, baseline)` mints the missing ids by replaying
the events (see `docs/USAGE.md`, "If your lineage has names but no ids").

### Relationship table schema (canonical)

Required columns (lowercased):

| Column | Type | Description |
|---|---|---|
| `event_year` | int | Year the event takes effect. |
| `event_type` | str | One of `Split`, `Merge`, `Redistribute`, `NameChange`, `Coarse`. |
| `parent_id` | str | ID of the parent unit (pre-event). |
| `parent_name` | str | Human-readable name. |
| `child_id` | str | ID of the child unit (post-event). |
| `child_name` | str | Human-readable name. |

Optional `parent_coarse_id` / `parent_coarse_name` / `child_coarse_id`
/ `child_coarse_name` carry the parent admin level (e.g., state name
for an ADM2 lineage).

### Baseline snapshot schema

| Column | Type | Description |
|---|---|---|
| `unit_id` | str | Canonical ID. |
| `name` | str | Human-readable name. |
| `year` | int (optional) | If present, multi-year baselines are filtered to the relevant year. |
| `coarse_id` / `coarse_name` (optional) | str | Parent admin level. |

### Statistics table schema (long-form)

| Column | Type | Description |
|---|---|---|
| `unit_id` | str | Canonical ID matching the relationship table. |
| `year` | int | Observation year (must be ≥ `target_year`). |
| `season` | str (required) | Season label. Use an explicit sentinel like `"Annual"` for non-seasonal data — NaN/null seasons raise `SchemaError`. |
| `variable` | str | Variable identifier (e.g., `rice_area_ha`). |
| `value` | float | Additive value. |

If your file uses different column names, pass `stats_columns={user: canonical}`
to `aggregate_stats(...)`. If your `unit_id` column is missing entirely,
use `Lineage.propose_stats_mapping(...)` to match names against the
lineage's canonical IDs first.

## Run the bundled example

A one-command smoke test exercises both products on the synthetic
Exampleland fixture:

```bash
python examples/exampleland/run_pipeline.py
```

It writes outputs to a temporary directory and prints summary counts;
useful for confirming a fresh install works end-to-end.

**Expected warnings on the example run.** Exampleland's synthetic stats
deliberately include rows that exercise the late-reporting diagnostics —
you'll see `UserWarning` lines about late reports and partial post-event
windows on a clean run. These are the package surfacing its diagnostics
on the example data, not failures. The `modern/late_reporting.csv` output
lists the rows the warnings reference.

## Documentation

- [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) — hands-on tutorial: the full workflow on a tiny synthetic country.
- [`docs/USAGE.md`](docs/USAGE.md) — practical end-to-end guide: inputs, configuration, pipeline calls, outputs, troubleshooting.
- [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) — code walkthrough in pipeline order; the most detailed tour of what each call actually does.
- [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md) — guided tour through the codebase, intended to be read alongside the source.
- [`docs/methodology.md`](docs/methodology.md) — stable-product algorithm reference, paired to the paper.
- [`docs/methodology_modern.md`](docs/methodology_modern.md) — modern-product algorithm reference.
- [`tools/india/README.md`](tools/india/README.md) — how the bundled India data and the shipped shapefile ids are regenerated from the canonical sources.
- [`examples/README.md`](examples/README.md) — worked examples with real output, on the synthetic countries that ship with the tests.

## Known Issues

Current as of v0.1.5. These are deliberate, documented limitations rather
than surprises — several are referenced by name from error messages the
package raises.

- **Reconciliation is not available.** `reconcile.py` implements the
  paper's Algorithm 4, but `aggregate_stats(reconcile_mode=...)` accepts
  only `"off"`; every other mode raises `NotImplementedError`. On India,
  `merge` mode deleted legitimate post-event child rows in the large
  majority of flagged cases, so the diagnostic was disabled pending a
  rework. Treat the module as experimental.
- **Late-report redistribution is disabled in the modern product.** Rows
  from units reporting outside their lineage lifespan are surfaced in
  `modern/late_reporting.csv` for inspection but are not redistributed to
  descendants, so they do not contribute to `stats_modern.csv`.
- **FEWS export supports admin levels 1 and 2 only.** The relationship-table
  importer accepts deeper hierarchies (an ADM3 table converts, but is
  untested), but the FNID code map and the admin-definition workbooks assume
  at most two levels. FEWS "crop region" units (`R`-type FNIDs) are not
  representable either — `build_fnid` always emits the `A` admin type.
- **In-lineage renames are not aliased backward when matching stats.** A
  rename modelled as a `NameChange` event is not used to resolve a unit
  reported under its pre-rename name in a later year. Pass a separate
  `name_change_log` for countries where this matters, and review the
  matcher's fuzzy and remapped bands.
- **The boundary cache does not detect changed inputs.** `build_boundaries`
  reuses a cached dissolve keyed on country code and target year only. If
  you edit a relationship table in place — the normal loop when preparing a
  new country — pass `refresh=True` or clear `output_dir`.
- **The India data generators need an external data tree.** Scripts under
  `tools/india/` resolve the canonical source files through
  `STABLEBOUND_DATA_ROOT`, which is not part of this repository, so they do
  not run on a clean checkout. The test suite (`pytest`) is self-contained;
  two of its tests self-skip for the same reason, and the breakpoint tests
  skip unless the optional `Rbeast` dependency is installed
  (`pip install "stablebound[analysis]"`).
- **The modern product has not been re-audited on India since the 0.1.3
  lineage fixes.** It is validated on the synthetic fixtures and by the
  conservation tests. A June 2026 audit of the India modern product found
  cascade over-allocation that traced to the upstream id mapping the 0.1.3
  fixes addressed; that audit has not been rerun. Treat India modern-product
  outputs as provisional until it is.

## License

MIT — see [`LICENSE`](LICENSE).
