# StableBound

A reproducible framework for harmonizing subnational data across administrative
boundary changes.

> **Status: v0.1.5.** API has stabilized around the
> `Lineage` / `StableBoundary` / `ModernBoundary` triad. The package is
> the method only and ships no country's data: any country loads from its
> own relationship tables, as published, with `Lineage.from_folder` (see
> [Inputs](#inputs)). India — the validated country this work was built
> on — is published that way with the StableBound-India dataset.

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

A country's relationship table, baseline and name-change log are kept by
whoever maintains them — a data repository, a deposit, a collaborator's
folder — in their own file names and headers. A small `stablebound.json`
beside them says which file is which and states the facts the tables cannot
(admin level, years covered, how names are tidied), so
`Lineage.from_folder(folder)` loads them exactly as shipped.

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

Point it at a folder of relationship tables with a `stablebound.json`:

```python
from stablebound import Lineage

# India's district tables as published with the StableBound-India dataset
# (Zenodo, doi:10.5281/zenodo.21653283): the deposit's intermediates/ folder.
ln = Lineage.from_folder("stablebound_india/intermediates")

ln.lineage              # parsed LineageGraph
ln.snapshot(year=2014)  # DataFrame of canonical (unit_id, unit_name) pairs
ln.years                # range(1991, 2027): the years the lineage can answer for
print(ln.validation_report())  # 8 automatic data-quality checks
```

In a clone, `Lineage.from_folder("examples/exampleland")` loads a five-unit
synthetic country the same way.

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

ln = Lineage.from_folder("stablebound_india/intermediates")

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

### Loading relationship tables

StableBound ships no country's tables. Each country's are published by
whoever maintains them; India's district tables (1991 baseline, 1991–2025)
are published with the StableBound-India dataset. A folder holding the
tables and a `stablebound.json` loads as it is:

```python
ln = Lineage.from_folder("my_country/")
```

`stablebound.json` names the files, maps any header that differs from the
canonical columns below by more than case (India's baseline says `ADM2_ID`
for `unit_id`), and states what the tables cannot: `admin_level`, the
`coarse_lineage` of the level above, `validity_start_year`,
`coverage_end_year`, notes, and the country's name-tidying rules as data.
The format, with India's file as the example, is in
[`docs/USAGE.md`](docs/USAGE.md) ("Loading a folder as it is shipped");
[`examples/exampleland/stablebound.json`](examples/exampleland/stablebound.json)
is the minimal case. The same can be passed as keywords:

```python
ln = Lineage(
    "PT",
    relationship_table_path="my_country/relationship_table.csv",
    baseline_path="my_country/baseline.csv",
    name_change_log_path="my_country/name_changes.xlsx",  # optional
    admin_level=2,                                        # default 1
    coarse_lineage_path="my_country/relationship_table_adm1.csv",
)
```

Without them a district-level country is read as state-level (admin level
1), with no coverage claim and the package's default name tidying.

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
- **The breakpoint tests need an optional dependency.** The test suite
  (`pytest`) is self-contained — its real-country tests read a frozen copy of
  India's published tables in `tests/fixtures/india_as_shipped/` — but the
  breakpoint tests skip unless `Rbeast` is installed
  (`pip install "stablebound[analysis]"`).
- **The modern product has not been re-audited on India since the 0.1.3
  lineage fixes.** It is validated on the synthetic fixtures and by the
  conservation tests. A June 2026 audit of the India modern product found
  cascade over-allocation that traced to the upstream id mapping the 0.1.3
  fixes addressed; that audit has not been rerun. Treat India modern-product
  outputs as provisional until it is.

## License

MIT — see [`LICENSE`](LICENSE).
