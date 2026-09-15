# StableBound — Usage Notes

A practical, end-to-end guide. Pairs with [`methodology.md`](methodology.md)
and [`methodology_modern.md`](methodology_modern.md) (the algorithm references).

## Install

From GitHub, pinned to a release tag:

```bash
pip install "git+https://github.com/michaelmfoley/stablebound.git@v0.1.5"
```

or for development, from a local clone:

```bash
pip install -e .
```

Python ≥ 3.10. Runtime dependencies: `pandas`, `numpy`, `geopandas`,
`shapely`, `openpyxl`. Dev extras (`pytest`, `jupyter`) are in
`pip install -e .[dev]`.

## The four (or five) inputs

| Input | Required? | Format | What it is |
|---|---|---|---|
| Relationship table | always | CSV or XLSX | One row per boundary event. Canonical schema (see below). |
| Modern shapefile | always | any geopandas-readable | Today's geometry. Must have a column whose values are the canonical unit IDs the relationship table uses. |
| Baseline snapshot | usually | CSV or XLSX | Lists every unit alive at the base year. Required when the RT doesn't include a creation event for every starting unit (typical of FEWS / HarvestStat data — many units have no events). |
| Long-form statistics | optional | CSV | Required only if you want stats aggregation. Long-form: `unit_id, year, season, variable, value`. |
| Name change log | optional | CSV or XLSX | A separate file of unit renames. Merged into the relationship table at load time via `io.merge_name_changes`. |

Name matching is human-in-the-loop: `propose_shapefile_mapping` /
`propose_stats_mapping` generate a reviewable mapping proposal (with fuzzy
matching and homonym warnings), the researcher reviews/edits it as a CSV,
and `attach_shapefile` / `attach_stats_ids` apply it. The pipeline itself
then runs on canonical IDs only. See `tools/india/prepare_shapefile.py`
for a worked matching script.

### Relationship table schema

Required columns (lowercased):

| Column | Type | Description |
|---|---|---|
| `event_year` | int | Year the event takes effect. |
| `event_type` | str | One of `Split`, `Merge`, `Redistribute`, `NameChange`, `Coarse`. |
| `parent_id` | str | ID of the parent unit (pre-event). |
| `parent_name` | str | Human-readable name. |
| `child_id` | str | ID of the child unit (post-event). |
| `child_name` | str | Human-readable name. |

Optional: `parent_coarse_id`, `parent_coarse_name`, `child_coarse_id`,
`child_coarse_name` (e.g. state info for ADM2 lineage; used for homonym
disambiguation in upstream matching, not by the package itself).

**Event-type semantics:**

- `Split`: 1 parent → ≥1 children. Multiple rows for one event share `parent_id`.
- `Merge`: ≥1 parents → 1 child. Multiple rows for one event share `child_id`.
- `Redistribute`: ≥1 parents ↔ ≥1 children with territorial exchange.
- `NameChange`: same unit, new name. `parent_id == child_id`. Non-territorial.
- `Coarse`: same unit, parent admin level changed (e.g. moved between states). `parent_id == child_id`. Non-territorial.

When `event_type` is `Split` but the same `child_id` has multiple distinct
`parent_id` values across rows, the multi-parent child forces those parents'
ancestor groups to merge in `build_stable_groups` (via Union-Find walking the
shared descendant). Structurally indistinguishable from a Redistribute even
if the row is tagged `Split`. The validator flags this as an `info`-level
finding (`multi_parent_split`) suggesting you retag for clarity.

### Statistics table schema

Long-form, with canonical IDs already attached:

| Column | Type | Description |
|---|---|---|
| `unit_id` | str | Matches the RT. |
| `year` | int | Must be ≥ `target_year` passed to `StableBoundary` / `ModernBoundary`. |
| `season` | str (required) | E.g. `Kharif`, `Rabi`, `Annual`. Required and non-null: use an explicit sentinel like `"Annual"` for non-seasonal data. NaN/null seasons raise `SchemaError` so downstream consumers never have to guess whether a missing season tag means "annual data," "untagged," or "schema bug." |
| `variable` | str | E.g. `rice_area_ha`. |
| `value` | float | Must be additive (extensive). Yield etc. is recomputed by `aggregate_stats(intensive=...)` from the units that reported both inputs. |

### Baseline snapshot schema

| Column | Type | Description |
|---|---|---|
| `unit_id` | str | Required. |
| `name` | str | Required. |
| `year` | int | Optional. If present and `base_year` is provided, rows are filtered to `base_year`. |
| `coarse_id` / `coarse_name` | str | Optional. |

## Lineage construction

`Lineage` is the entry point. For the bundled country (India, `IN`), all
you need is the country code:

```python
from stablebound import Lineage

ln = Lineage("IN")
```

For custom countries, supply the three input paths:

```python
ln = Lineage(
    "PT",
    relationship_table_path="my_country/relationship_table.csv",
    baseline_path="my_country/baseline.csv",
    name_change_log_path="my_country/name_changes.xlsx",  # optional
)
```

### If your lineage has names but no ids

`Lineage` needs an id on every side of every event. When you author a
relationship table in names ("Bihar split into Bihar and Jharkhand in
2000") and only the baseline carries ids, let the package mint the rest:

```python
from stablebound import assign_unit_ids

res = assign_unit_ids("my_country/lineage_names_only.csv",
                      "my_country/baseline.csv",
                      name_change_log="my_country/name_changes.xlsx")  # optional
print(res.report())          # what was minted, what was resolved, any warnings
res.write("my_country/")     # lineage.csv, baseline.csv, [name_change_log.csv], report
```

The id format is read off the baseline (`IN.ADM2.00001` gives the prefix
`IN.ADM2.` and five digits). The events are replayed in order: a parent is
looked up by name (and coarse name, when present) among the units alive at
the start of its year; every child of a `Split` / `Merge` / `Redistribute`
gets the next unused number, one id per child name per year however many
parent rows feed it; `NameChange` and `Coarse` keep the parent's id. Renames
in the name-change log take part in the replay, so a later event may use the
new name.

By default (`mode="fill"`) ids already present are kept and only blank cells
are minted, so adding a newly discovered event later costs one id and moves
nothing else. `mode="rebuild"` discards every non-baseline id and re-mints
chronologically, which gives dense numbering at the cost of renumbering
everything after an insertion.

Coarse ids (`parent_coarse_id` / `child_coarse_id`) are resolved, never
minted: from the baseline's `coarse_id` / `coarse_name` columns, and, when
the coarse level changes too, from an already id'd coarse lineage passed as
`coarse_lineage=` (India: `lineage_adm1.xlsx`, so a district moving into
Jharkhand in 2000 is attached to Jharkhand's new id). For such a country run
the helper on the coarse level first, then on the fine level.

Two living units with one name (India has two Bijapurs) are told apart by
the coarse name: the coarse columns of a relationship-table row, or an
optional `coarse_name` column in the name-change log. Without one the
helper refuses rather than guesses (a log row is left blank with a
warning), and a name that matches nothing is reported with the nearest
living names. All table failures are collected into one
`IdAssignmentError` so a lineage with twenty typos is one round trip.

Before running a product you must attach a modern shapefile. The
package provides a human-in-the-loop matcher:

```python
proposal = ln.propose_shapefile_mapping(
    "modern.shp", name_column="DISTRICT_NAME",
)
proposal.to_csv("mapping.csv")        # review/edit in Excel
ln.attach_shapefile("modern.shp", mapping="mapping.csv",
                    name_column="DISTRICT_NAME")
```

If your shapefile already has a canonical `unit_id` column (i.e.,
you've matched IDs upstream), skip the proposal step:

```python
ln.attach_shapefile("modern_with_ids.geojson")
```

## The pipeline

```python
from stablebound import StableBoundary

sb = StableBoundary(
    ln,
    target_year=1997,             # earliest year the stable product covers
    max_year=2022,                # optional; defaults to inferred shapefile vintage
    output_dir="./_out",
)

# 1) Geometry layer (paper §3.2)
sb.build_boundaries()                    # writes per-year stable shapefiles + remap.json
gdf = sb.get_boundary(year=2018)         # GeoDataFrame for one snapshot year
year, mismatches = sb.infer_shapefile_year()   # paper Algorithm 2

# 2) Statistics layer (paper §3.3 + §3.4)
sb.aggregate_stats(
    stats="stats_long.csv",
    extensive=["rice_area_ha", "rice_production_mt"],   # variables to sum
    intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
)
df = sb.get_stats(variable="yield_mt_ha", year=2018)

# 3) Audit
sb.summary()                             # counts: events, stable groups, modern units, year range
sb.get_name_history()                    # (stable_id, year, unit_id, name) per constituent unit
```

Reusing a stats data dictionary across products or files:

```python
from stablebound import DataDictionary

dd = DataDictionary(
    stats_columns={"district_id": "unit_id", "yr": "year", "qty": "value"},
    extensive=["rice_area_ha", "rice_production_mt"],
    intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
    ignore=["price_inr"],
)
sb.aggregate_stats(stats="stats_long.csv", data_dict=dd)
```

## Auto-validation

Loading a relationship table runs `validate_lineage` automatically. Errors
raise `LineageDataError` with a formatted report; warnings and infos are
logged via the `stablebound.lineage` logger.

```python
import logging
logging.basicConfig(level=logging.INFO)  # see warnings/infos as they happen
```

The check categories are documented in
[`methodology.md` § Lineage validation](methodology.md). Errors block:
the package will not build stable groups on a lineage with duplicate rows
or self-referential territorial events. Warnings (`split_looks_like_rename`,
`namechange_distinct_ids`) are advisory — review them, decide whether the
encoding matches your intent.

To opt out (e.g., adversarial unit tests, batch loads where you've
validated separately):

```python
graph = LineageGraph.from_dataframe(rt, validate=False)
```

### Stats × lineage consistency check

`StableBoundary.aggregate_stats()` and `ModernBoundary.aggregate_stats()`
also run `validate_stats_lineage_consistency` automatically against
the stats DataFrame and lineage. It surfaces upstream FNID-assignment
bugs that would otherwise produce silently-wrong outputs. Three checks:

| Category | Severity | What it catches |
|---|---|---|
| `stats_unit_id_malformed` | error | Sentinel/control values like `__FILTER__`, blanks, NaN |
| `stats_unit_id_unknown` | warning | unit_ids that aren't in the lineage AND aren't in the modern shapefile |
| `stats_post_cease_reporting` | warning | unit_ids that report stats AFTER the unit ceased per a Split/Merge/Redistribute (the "Giridih (JH)" pattern — stats filed under the pre-1991 parent code that should have been remapped) |

To inspect findings manually:

```python
from stablebound.validate import (
    validate_stats_lineage_consistency, format_issues,
)

issues = validate_stats_lineage_consistency(stats_df, graph, modern_unit_ids)
print(format_issues(issues))
```

The post-cease check is the highest-leverage one for catching
upstream stats-prep bugs: once a unit ceased per the lineage, any
later stats under its old code are flagged as late reporting and left
under the ceased unit's id rather than routed to a successor, so a
stable group can silently lack data a successor should have carried.

### Stable-group spatial contiguity check

`StableBoundary.build_boundaries()` also runs
`validate_stable_group_contiguity` against the base-year remap and
modern shapefile. It surfaces stable groups whose modern members are
not all reachable through shared borders. Always warning-level — never
blocks the build.

| Category | Severity | What it catches |
|---|---|---|
| `stable_group_disconnected` | warning | A stable group whose modern members fall into 2+ spatially-disconnected components |

The most common cause is a homonym name in the relationship table
matched to a unit in another part of the country (e.g., "Hamirpur" in
both Himachal Pradesh and Uttar Pradesh). Other legitimate causes:
island/exclave geometry, and tiny gaps in the modern shapefile from
post-split topology imprecision. The warning detail lists each
disconnected component with sample unit_ids so you can quickly
classify which case applies.

To inspect findings manually:

```python
from stablebound.validate import (
    validate_stable_group_contiguity, format_issues,
)

issues = validate_stable_group_contiguity(remap, modern_gdf, id_column)
print(format_issues(issues))
```

## Outputs

`build_boundaries()` writes to the `output_dir` passed to `StableBoundary`:

| File | Description |
|---|---|
| `stable_<year>.geojson` (one per year) | Dissolved stable polygons. Columns: `stable_id`, `n_modern`, `source_ids`, `geometry`. |
| `remap.json` | Canonical `{unit_id → stable_id}` mapping for the base year. Loaded by `aggregate_stats`. |
| `snapshots.json` | `{year → [unit_id, …]}` of units active each year. |
| `shapefile_vintage_report.txt` | Inferred shapefile year + mismatch counts. |
| `name_history.csv` | `(stable_id, year, unit_id, name)` audit table. |
| `summary.json` | Counts. |

`aggregate_stats()` adds:

| File | Description |
|---|---|
| `stats_aggregated.csv` | Long-form aggregated stats with constituent IDs and `late_reporting` flag. Includes derived intensives. |

In-memory equivalents are returned by the `get_*` methods so notebook
users don't need to read from disk.

## Reconciliation (not available)

`aggregate_stats` accepts `reconcile_mode="off"` and nothing else — every
other value raises `NotImplementedError`, and `"off"` is the default, so
you can ignore the parameter entirely.

The paper's Algorithm 4 (drop test, sum-jump test, and the `merge` /
`subtract` repair modes) is implemented in `stablebound/reconcile.py` but
is not wired into either product. On India, `merge` mode removed
legitimate post-event child rows in the large majority of flagged cases —
series looked smoother because roughly half their post-event data was
gone — so the whole diagnostic was disabled pending a rework rather than
shipped with a known false-positive rate. `reconciliation_flags.csv` is
not produced.

Treat `reconcile.py` as experimental. See Known Issues in the README.

## Quickstart

The bundled `examples/exampleland/` dataset is a synthetic 5-district
country that exercises a clean Split (2014), a NameChange (2016), a
Merge (2018), plus stats rows that exercise the late-reporting and
partial-post-event-window diagnostics. The one-command smoke test:

```bash
python examples/exampleland/run_pipeline.py
```

Or, from the repository root, the same thing step by step:

```python
from examples.exampleland.config import lineage, INTENSIVE, STATS_PATH, TARGET_YEAR, MAX_YEAR
from stablebound import StableBoundary

sb = StableBoundary(lineage, target_year=TARGET_YEAR, max_year=MAX_YEAR, output_dir="_out")
sb.build_boundaries()
sb.aggregate_stats(stats=STATS_PATH, intensive=INTENSIVE)
print(sb.summary())
print(sb.get_stats(variable="yield_mt_ha", year=2018))
```

The last line prints four yield rows for 2018, one per stable group; the
`E.004` row has `complete=False` because its two 2010 parents had merged and
neither reported that year.

## The India application

The full-scale application — 786 modern districts, about 1,000 events over
1991–2025, the DES crop statistics — lives in the separate `stablebound-india`
repository, which pins this package. The scripts that regenerate the bundled
India data and the shipped shapefile ids from the canonical sources are under
`tools/india/` here (they need `STABLEBOUND_DATA_ROOT`).

## Modern boundary product

The modern boundary product is the second of the two products StableBound
ships. Where the stable product fixes geometry at the **base year** and
projects history forward (so each historical observation lands on its
base-year polygon), the modern product fixes geometry at **today's
shapefile** and rescales every historical observation backward onto
modern units. For each modern unit you get one long-form time series
covering `[base_year, max_year]` on the present-day extent.

### When to use which

- **Stable**: you want the finest temporally-consistent geometry that
  every observation in your window has a polygon for. Aggregation is
  upward (multiple modern units → fewer stable polygons) and exact
  for additive variables.
- **Modern**: you want today's familiar geometry and are willing to
  accept *estimated* fractions for splitting historical observations
  onto smaller modern units. Aggregation is downward (one historical
  unit → multiple modern children, allocated by fractions).

Both products read the same canonical inputs and can be run from the
same `Lineage`.

### Quickstart

```python
from stablebound import Lineage, ModernBoundary

ln = Lineage("IN")
ln.attach_shapefile("modern_with_ids.geojson")

mb = ModernBoundary(ln, target_year=1997, output_dir="./_out")
mb.aggregate_stats(
    stats="stats_long.csv",
    extensive=["rice_area_ha", "rice_production_mt"],
    intensive={"yield_mt_ha": ("rice_production_mt", "rice_area_ha")},
    modern_window_default=5,                   # post-event window in years
)

# Long-form time series filtered to one variable
df = mb.get_modern_stats(variable="rice_area_ha", year=2018)

# Audit table — what fraction was applied at each event?
fr = mb.get_event_fractions()

# Reports stranded on units with no modern destination
late = mb.get_late_reporting()

mb.summary()
```

### Algorithm (one-paragraph version)

For each modern unit, walk territorial events chronologically forward
from `base_year`. At each event, compute per-(variable, season)
fractions for each child by averaging the child's reported values over
a configurable post-event window (default 5 years). Pool the parents'
pre-event ledgers and distribute to children by these fractions. The
parent's pre-event data is then cleared from its ledger; only post-event
("late-reporting") data remains. Cascading lineages compose naturally —
each event multiplies through the inherited ledger, so deeper chains
get sequential fraction products without any explicit composition step.
After all events are processed, every modern unit's ledger is its
canonical per-modern-unit time series. Intensives (yield, etc.) are
recomputed at the end via `derive_intensive`.

See `docs/methodology_modern.md` for the full narrative.

### Configuration

The modern product adds two optional `aggregate_stats(...)` kwargs:

| Kwarg | Default | Description |
|---|---|---|
| `modern_window_default` | `5` | Default post-event window length (years) for fraction computation. |
| `modern_window` | `{}` | Per-variable override. Variables not in the dict use the default. |

```python
mb.aggregate_stats(
    stats="stats_long.csv",
    modern_window={"area_ha": 5, "production_mt": 3},
    modern_window_default=5,
)
```

### Outputs (under `output_dir / "modern/"`)

| File | Description |
|---|---|
| `stats_modern.csv` | Long-form: `year, season, variable, modern_id, value, sources, lineage_depth, has_nan_fraction, fraction_method, late_report_redistributed`. See column reference below. |
| `event_fractions.csv` | Audit table: one row per `(event, child, variable, season)`. Columns: `event_year, event_type, parent_ids, child_id, variable, season, window_used, n_common_observations, n_individual_observations, fraction, fraction_method, computed_at`. `n_common` is the per-event count of years (or cells, for total-year tier) used in the intersection; `n_individual` is this child's in-window count — sort by their difference to find children whose gap-filling would tighten the fraction estimate. |
| `late_reporting.csv` | Stats reported under a unit's ID after that unit's terminal event. These rows are **not** redistributed onto modern descendants and do not contribute to `stats_modern.csv` (see Known Issues in the README); this file is where they can be inspected. |
| `summary.json` | Run metadata + counts. |

#### `stats_modern.csv` column reference

| Column | Description |
|---|---|
| `year, season, variable, modern_id, value` | Long-form stats key + value. ``season='Total Year'`` rows are the sum across explicit seasons (see "Total Year" below). |
| `sources` | Comma-joined list of historical `unit_id`s that contributed to this cell. Provenance audit. |
| `lineage_depth` | Number of events composed into this cell. `0` = direct report, `1+` = inherited via fractions. |
| `has_nan_fraction` | True if the cell's attribution path passed through any NaN fraction (rare after the cascade fallback; mostly indicates a genuine all-children-zero case for that variable). |
| `fraction_method` | Worst tier of the cascade applied across this cell's lineage events. One of: `seasonal` (best — direct match), `total_year` (fallback A — child reported the variable in some other season but not this one), `area` (fallback B — child reported nothing for this variable; modern shapefile area used), `undefined` (all tiers failed; cell is NaN), or empty string (direct report; no fraction applied). |
| `late_report_redistributed` | Reserved. Always False in the current release, because late-report redistribution (Step D below) is disabled. |

### Cascade fraction fallback

For each `(event, child, variable, season)`, the fraction is chosen by
trying three tiers in order:

1. **Seasonal** — child's mean reported value for `(variable, season)` in
   the post-event window divided by the sum across siblings. Most
   accurate when the (var, season) combo has data on every child.

2. **Total-year fallback** — if (1) is NaN. Sum each child's means
   across ALL seasons for the variable, divide by sum across siblings.
   Captures the case where a child reports Kharif rice but not Rabi:
   their Rabi pre-event data gets attributed by their share of the
   variable's whole-year total.

3. **Modern-area fallback** — if (2) is NaN. Each child's share of
   modern shapefile area / sum of siblings' areas. No data needed —
   final tier to ensure no parent data is lost.

Only `'undefined'` (all three fail) produces a NaN cell, and only when a
child has zero reports anywhere AND no modern shapefile entry — extremely
rare.

### Late reports (Step D — redistribution disabled)

After the chronological event walk, parent units may still hold cells
with `year > terminal_event_year` — reports filed under their old code
after the unit was reorganized. These rows are written to
`late_reporting.csv` and are **not** redistributed onto the parent's
modern descendants, so they do not contribute to `stats_modern.csv`. A
redistribution pass exists in the code (`_redistribute_late_reports`) but
is not called: on India it double-counted real observations, because the
upstream id mapping files the same report under several canonical ids.
Until that is reworked, the honest output is the sidecar.

### "Total Year" derived rows

For every `(modern_id, year, variable)`, an extra row is appended with
`season='Total Year'` whose value is the NaN-aware sum across explicit
seasons (Kharif, Rabi, Summer, Whole Year, Autumn, Winter). For yields
and other declared intensives, `derive_intensive` recomputes from the
Total Year extensives — so `yield_mt_ha` for `season='Total Year'` is
`production_mt_TY / area_ha_TY`, never an average of seasonal yields.

**This is the season where conservation holds strictly**: for any
`(variable, year)`, sum across all modern units for `season='Total Year'`
equals sum across all stable units (excluding rows with no modern
destination). Per-season conservation may NOT hold — the cascade can
reallocate parent data across seasons when seasonal fractions are NaN.

### Season-year convention (sowing year)

The package stores values under their **sowing year**, matching India's
official agricultural-ministry standard:

- Kharif T → sown June–Sep T, harvested Oct–Dec T → year = T
- Rabi T → sown Oct–Dec T, harvested Mar–May T+1 → year = T (Rabi 2010
  spans Oct 2010 – Jan 2011)
- Summer T → sown Mar–May T, harvested Jun–Aug T → year = T
- Whole Year T → annual aggregate → year = T

The package does no year shifting; what's in your input stats is what
the package reports. If your raw data uses harvest-year attribution
(some other countries' conventions), shift it during preparation
before passing to `read_stats`.

### Tradeoffs and edge cases

- **Insufficient post-event data.** A child with only 1–4 reporting
  years (when window=5) still produces a fraction; a warning is logged
  with the partial year count. The seasonal tier still runs.
- **All-zero or all-NaN siblings for a (var, season).** Triggers the
  cascade — total-year and area tiers usually save the day.
- **Late-reporting units.** Surfaced in `late_reporting.csv`; not
  redistributed (Step D is disabled), so their values are absent from
  `stats_modern.csv`.
- **Multi-parent events (Redistribute and multi-parent Splits).**
  Parents are pooled before fraction-based distribution. The
  per-parent contribution to each child is intentionally lost — we
  don't attempt spatial-connectedness reasoning. This is a deliberate
  scope decision.
- **NameChange / Coarse events.** Non-territorial — the unit's data
  flows through unmodified. The audit table contains no rows for these
  events.

### Modern vs. stable: when results differ

For `season='Total Year'`, sums match exactly across products (modulo
floating-point rounding). For individual seasons, totals may differ
because the cascade fallback reallocates parent data across seasons —
e.g., a Rabi pre-event cell may be redistributed using the variable's
total-year fraction (which sums Kharif+Rabi+Summer reports) and end up
distributed across modern children in a slightly different
season-by-season pattern than they actually grew.

The fix when comparing products: always do cross-product sums on
`season='Total Year'`. Per-season aggregation is fine for understanding
spatial distribution but not for strict accounting against the source
totals.

## Performance notes

- `build_boundaries` for India, 1997–2022: about 15 s on a laptop, most
  of it the per-year GeoJSON writes rather than the grouping.
- `aggregate_stats` on India's ~750,000-row statistics table: on the order
  of ten seconds; it prints start and elapsed lines to stderr for inputs
  over 10,000 rows.
- Readability was preferred over performance throughout. Further work, if
  needed: parallelize per-year in `build_boundaries`.

## Cache policy

Both products cache their outputs under `output_dir` so a second run
on the same instance can skip work. The behavior is now deliberate:

**When outputs ARE reused (cache hit):**

- `StableBoundary.build_boundaries(refresh=False)` skips the build
  step IFF the on-disk `summary.json` matches the current run's
  `_schema`, `country_code`, and `target_year` (and `max_year` if it
  was passed explicitly). All three must match; any mismatch forces
  a fresh rebuild. Per-year geojson files and the canonical
  `remap.json` are loaded but not re-derived.
- The getter methods (`get_boundary`, `get_stats`,
  `get_modern_stats`, `get_event_fractions`, `get_late_reporting`,
  `get_name_history`, `summary`)
  lazy-load from disk on cold processes.

**When outputs are NOT reused:**

- `StableBoundary.aggregate_stats(...)` and
  `ModernBoundary.aggregate_stats(...)` always run the aggregation
  when called. A user calling with different stats or data
  dictionary gets the new results, not a stale cached
  frame. (Pre-v0.1.2 these short-circuited on prior runs; that was a
  silent-bug source.)
- `build_boundaries(refresh=True)` forces a rebuild regardless of
  on-disk state.

**When `refresh=True` is required:**

- You edited the relationship table, baseline, name change log,
  shapefile, or stats file externally between runs. The cache
  validator does not hash these inputs, so it won't notice the change.
- You suspect any other on-disk artifact has drifted from the
  current inputs.

**Recommendation:** use a fresh `output_dir` for each distinct
configuration, especially in parallel runs. The default
`./stablebound_out` is fine for interactive single-user work but is
NOT safe to share across runs with different inputs.

## When something goes wrong

Most issues come from input data, not the algorithm:

| Symptom | Most likely cause |
|---|---|
| `LineageDataError: duplicate_row` at load | Spreadsheet has duplicate rows; remove them. |
| `SchemaError: missing required columns` | Input file's columns don't match the canonical schema. Lowercase + rename. |
| `SchemaError: stats table contains rows with year < base_year` | Stats has pre-base-year data. Filter or raise the base year. |
| Many `late_reporting=True` rows in stats output | Stats has unit_ids absent from the remap. Either the IDs are wrong, the lineage is missing units, or the units genuinely report after dissolution (paper §3.4). |
| Many polygons missing from a year's stable shapefile | Modern shapefile is missing features for those stable groups, or they're singletons that didn't get matched upstream. Check `match_log.csv`. |
