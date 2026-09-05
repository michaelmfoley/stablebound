# StableBound — User Manual

A guided tour through the codebase, intended to be read alongside the
source. Companion docs:

| Doc | Purpose |
|---|---|
| [`USAGE.md`](USAGE.md) | Practical end-to-end how-to with code snippets. |
| [`methodology.md`](methodology.md) | Stable-product algorithm reference (per paper Algorithms 1–5). |
| [`methodology_modern.md`](methodology_modern.md) | Modern-product algorithm reference. |
| [`WALKTHROUGH.md`](WALKTHROUGH.md) | Module-by-module engineering tour in pipeline order. |

This manual ties code, decisions, and the other docs together so a reader
can navigate the package without getting lost. It does NOT duplicate the
algorithm walkthroughs in the methodology files — those are the canonical
references.

## 1. What the package does

StableBound harmonizes subnational time-series data across
administrative-boundary changes. Given:

- A **canonical relationship table** describing every territorial event
  (Splits, Merges, Redistributes) plus non-territorial labels
  (NameChange, Coarse).
- A **modern shapefile** of today's geometry.
- An optional **baseline snapshot** of units alive at the analysis-window
  start year.
- An optional **long-form statistics table** keyed on those unit IDs.

it produces TWO mutually-consistent outputs:

- **Stable boundary product**: a sequence of dissolved per-year shapefiles
  on a geometry that is consistent across the full analysis window, plus
  an aggregated long-form statistics table on those geometries.
- **Modern boundary product**: every historical observation rescaled
  backward onto today's modern shapefile via per-event area/production
  fractions.

Researchers pick whichever product matches their question. The two
products **conserve the same totals** at the Total Year × variable
level (this is a tested invariant — see `tests/test_conservation.py`).

## 2. Mental model: the two products

The two products are dual halves of one redistribution problem.

```
                  HISTORICAL DATA
                   /          \
                  /            \
            STABLE              MODERN
        (sums backward     (fractions forward
           in time)              in time)
            /                      \
   one polygon per                 one polygon per
   stable group at each year       modern unit, fixed
                                   geometry, all years

Both products' Total Year sums match exactly.
```

**Stable**: every modern unit that descends from base-year unit B
contributes its data to B's stable group. Sums are exact (additive
variables). Geometry coarsens at base year (fewer, larger polygons),
finer toward today (more, smaller polygons). Fits the question
"What was happening in this region of fixed geographic extent across
my analysis window?"

**Modern**: every historical observation is allocated to the modern
unit(s) covering its territory today, using per-event area/production
fractions with a three-tier cascade fallback. Geometry is fixed at
today's shapefile. Useful when the question is "What does this district,
defined by today's borders, look like in past years?"

## 3. Code organization

The package source tree (`src/stablebound/`):

| File | Purpose | Key functions / classes |
|---|---|---|
| `data/__init__.py` + `data/IN/` | The bundled India RT, baseline, ADM1 lineage and name change log, selected by ISO2 code. | `BUNDLED_COUNTRIES`, `BundledCountry` |
| `data_dict.py` | `DataDictionary` dataclass for declaring how a long-form stats file maps to canonical columns and which variables are extensive / intensive / ignored. | `DataDictionary` |
| `lineage_class.py` | The user-facing `Lineage` class. Wraps RT + baseline + name change log + attached shapefile, exposes lazy inspection (`lineage`, `snapshot`, `years`) and shapefile attachment with human-in-the-loop name matching. | `Lineage` |
| `match.py` | The country-agnostic name matcher (4-pass: homonym → exact → manual → fuzzy). | `propose_shapefile_mapping`, `propose_stats_mapping`, `MatchProposal`, `attach_shapefile_ids`, `attach_stats_ids`, `normalize_name` |
| `schemas.py` | Canonical column constants + schema validators that fail loudly on malformed inputs. | `validate_relationship_table`, `validate_stats`, `validate_baseline`, `validate_name_change_log`, `SchemaError` |
| `io.py` | All file readers. Lowercase-then-project pattern; never mutate inputs. | `read_relationship_table`, `read_stats`, `read_shapefile`, `read_baseline`, `read_name_change_log`, `merge_name_changes` |
| `lineage.py` | Parses the relationship table into a queryable graph. Auto-runs the validator. | `LineageGraph`, `name_history` |
| `validate.py` | Five validator suites: structural (lineage), cross-level (admin2 ↔ admin1 references), data-consistency (stats × lineage), spatial (stable-group contiguity), and shapefile ↔ lineage id consistency. Returns `LineageIssue` lists; errors raise `LineageDataError`. | `validate_lineage`, `validate_coarse_references`, `validate_stats_lineage_consistency`, `validate_stable_group_contiguity`, `validate_shapefile_lineage_consistency`, `format_issues` |
| `snapshot.py` | Paper Algorithm 1 + 2: which units are alive in year T, and which year does the modern shapefile match. | `build_snapshot`, `infer_year` |
| `groups.py` | Paper Algorithm 3: Union-Find construction of stable groups. The "no allocation assumptions" invariant lives here. | `build_stable_groups` |
| `dissolve.py` | Geometry-only step. Validates and dissolves modern-shapefile polygons by stable_id. | `dissolve` |
| `reconcile.py` | Paper Algorithm 4: drop test + sum-jump test. Implemented but not wired into either product (see Known Issues in the README). | `reconcile` |
| `stats.py` | Paper Algorithm 5: long-form aggregation + intensive recomputation. | `aggregate`, `derive_intensive` |
| `boundary.py` | User-facing orchestrator for the stable product. Owns disk caching, lazy state, the public API. | `StableBoundary` |
| `modern_algorithm.py` | The modern product's pure-function core. Three-tier cascade and Total Year aggregation (late-report redistribution is present but disabled). | `build_modern_ledger` |
| `modern.py` | User-facing orchestrator for the modern product. Mirrors `StableBoundary` without sharing a parent class. | `ModernBoundary` |

The split between `*_algorithm.py` (pure functions, easy to unit test)
and the orchestrator class (`StableBoundary` / `ModernBoundary`) is a
deliberate pattern — the orchestrator handles config, file I/O, and
caching; the algorithm core takes plain DataFrames + dicts and returns
plain DataFrames.

## 4. Load-bearing invariants and design decisions

Read these before you change anything. They were each made after
weighing alternatives, sometimes after shipping the wrong choice once.

### 4.1 Stable boundary: "no allocation assumptions, only sums of reported values"

A **multi-parent child** (Tirupati carved from Chittoor + SPS Nellore;
A, B, D all contributing to C in one event) cannot be cleanly attributed
to one parent's stable group without inventing a map. The only
assumption-free choice is to merge ALL ancestor base groups into ONE
stable group, accepting lost geographic resolution in exchange for
clean cross-year sums.

**Where**: `groups.py:build_stable_groups`.

**Don't propose** rules that disaggregate multi-parent children. That
includes the original "redistribute-safe rule" (skip them as singletons),
which was tried early in development and removed.

### 4.2 Event-year convention: `event_year=T` happens DURING year T

Parent is alive in the year-T snapshot; child first appears in the
year-(T+1) snapshot. Matches the FEWS NET / census admin-snapshot
convention exactly (verified against `admin_snapshot_<year>.csv` files
on India 1991–2025).

**Where**: `snapshot.py:build_snapshot` (filter `event_year < year`),
`lineage.py:adjacency_after` (filter `event_year >= year`),
`modern_algorithm.py:_pool_parents` (pool inclusion `year <= event_year`).

### 4.3 Modern boundary cascade is per-(event, var, season), NOT per-child

If one sibling has seasonal data (fraction = 1.0 because alone) and
another falls back to area (fraction = 0.5 of total area), the
combined distribution is 1.5× the pool — non-conservative.

The fix: pick ONE tier per event-and-(var, season) so all siblings
share the denominator and fractions sum to 1.

**Where**: `modern_algorithm.py:_compute_fractions` Pass 5. The comment
block there explains the conservation invariant in detail.

### 4.4 Modern Total Year MUST conserve

Per-season conservation between products may NOT hold (the cascade
fallback can reallocate parent data across seasons). Total Year (sum
across seasons) MUST hold strictly.

**Tested**: `tests/test_conservation.py::test_exampleland_total_year_conservation_strict`.

### 4.5 Singleton fallback for orphan modern features

Every modern shapefile feature gets a `stable_id`, even if it can't
be unioned with any base-year unit (it becomes its own singleton
group). This guarantees the dissolution at any year covers the entire
modern shapefile. Tradeoff: a post-base-year modern unit can show up
as its own polygon at pre-creation years (a unit created in 2012 appears
as a singleton in a 1986 dissolution).

**Where**: `groups.py:build_stable_groups` Step 5.

### 4.6 Yields are recomputed, never averaged

`aggregate(intensive=...)` (in `stats.py`) recomputes each ratio from a
separate aggregation of only the units that reported both inputs in that
cell, then divides. Summing intensives is mathematically wrong; averaging is
also wrong (loses production-weighting); dividing the full sums is biased
whenever a unit reported one input and not the other. Pass intensives to
`aggregate_stats(intensive=...)` as `{intensive_name: (num_var, den_var)}`
and the package handles them; the yield row's `constituent_ids` names the
units in the ratio.

### 4.7 Sowing-year season convention (no remap)

Rabi 2010 = sown Oct 2010, harvested May 2011, stored as `year=2010`.
Matches India's agricultural-ministry standard. The package does no
year-shifting. If your raw data uses harvest year, shift it before
calling `read_stats`.

### 4.8 Validator pattern: errors raise, warnings/infos log

`LineageGraph.from_dataframe` auto-runs `validate_lineage`.
`StableBoundary.aggregate_stats` and `ModernBoundary.aggregate_stats`
auto-run `validate_stats_lineage_consistency`. In both cases, errors
raise `LineageDataError` with the formatted report; warnings/infos log
a one-liner pointer. Researchers can `validate=False` or call the
validator manually.

When adding a new validator, follow this pattern.

## 5. End-to-end walkthrough

What happens when you write:

```python
from stablebound import Lineage, StableBoundary, ModernBoundary

ln = Lineage("IN")
ln.attach_shapefile("modern_with_ids.geojson")

sb = StableBoundary(ln, target_year=1997, max_year=2025, output_dir="./out")
sb.build_boundaries()
sb.aggregate_stats(
    stats="stats.csv",
    extensive=["production_mt", "area_ha"],
    intensive={"yield_mt_ha": ("production_mt", "area_ha")},
)

mb = ModernBoundary(ln, target_year=1997, output_dir="./out")
mb.aggregate_stats(
    stats="stats.csv",
    extensive=["production_mt", "area_ha"],
    intensive={"yield_mt_ha": ("production_mt", "area_ha")},
)
```

### 5.1 `Lineage("IN")` (instant)

The constructor validates inputs and stores paths but does not parse
the RT yet. Lazy properties (`lineage`, `baseline`, `name_change_log`,
`snapshot`) load on first access. For a bundled country, no path
arguments are needed; the package resolves them from
`BUNDLED_COUNTRIES`.

`lineage_class.py:60-160`.

### 5.2 `ln.attach_shapefile(...)`

Two paths through this method:

- **With mapping**: the shapefile is joined against a (possibly
  user-edited) mapping CSV via `attach_shapefile_ids`. The shapefile
  gains a canonical `unit_id` column.
- **Without mapping**: the shapefile is assumed to already have a
  `unit_id` column.

After attachment, `infer_year` runs over the validity window to set
`shapefile_year` — this becomes the default `max_year` for both
products.

`lineage_class.py:230-290`.

### 5.3 `sb = StableBoundary(ln, target_year=...)` (instant)

Just attribute assignment. The class uses **lazy state**:
`_modern_gdf`, `_all_remaps`, etc. are populated when the pipeline
runs. `target_year` defaults to `ln.min_year`; `max_year` defaults to
`ln.shapefile_year` at build time.

`boundary.py:60-105`.

### 5.4 `sb.build_boundaries()` (~15 s on India, 1997–2022)

Seven steps in `boundary.py:build_boundaries`:

1. **Disk-cache short-circuit**: if `remap.json` and `summary.json`
   already exist on disk and `refresh=False`, hydrate and return.
2. **Pull graph + shapefile** from the lineage. The shapefile must
   have been attached or this raises.
3. **max_year determination**: constructor override if set, else
   `lineage.shapefile_year` from the attachment step, else fall back
   to `graph.max_event_year + 1`.
4. **Assemble `additional_units`**: baseline ∪ modern shapefile IDs.
   Filtered to NOT-in-RT inside `build_snapshot`.
5. **Per-year stable groups + dissolutions**: for each year y in
   `[target_year, max_year]`:
   - `build_stable_groups(graph, y, additional_units)` → year-y remap
   - `dissolve(modern_gdf, remap, ...)` → polygons
   - Write `stable_<y>.geojson`
6. **Persist canonical artifacts**: target-year `remap.json`,
   `snapshots.json`, `name_history.csv`, `summary.json`,
   `shapefile_vintage_report.txt`.
7. **In-memory cache**: populate lazy attributes for subsequent
   `get_*()` calls.

### 5.5 `sb.aggregate_stats(stats=...)` (seconds on India)

Seven steps in `boundary.py:aggregate_stats`:

1. **Resolve data dictionary** from kwargs / `data_dict`. Individual
   kwargs win over `data_dict`.
2. Disk-cache short-circuit.
3. Run `build_boundaries` if not yet (lazy).
4. **Load + preprocess stats** via the shared `_load_stats` helper:
   read → apply `stats_columns` renames → attach `mapping` if
   provided → drop `ignore` variables → filter to `extensive` ∪
   intensive inputs → normalize + validate schema.
4b. **Stats × lineage validator** auto-runs. Errors raise; warnings log.
5. **Reconcile** (Algorithm 4): not run. `reconcile_mode` accepts only
   `"off"`; anything else raises `NotImplementedError`.
6. **Aggregate** (Algorithm 5): `aggregate(stats, remap, ...)` →
   long-form `(year, season, variable, stable_id, value, ...)`, with the
   intensive pass over matched constituents when `intensive` is non-empty.
7. Write `stats_aggregated.csv`; update the in-memory cache.

### 5.6 `mb.aggregate_stats()` (seconds on India)

`modern.py:aggregate_stats` orchestrates → calls
`modern_algorithm.py:build_modern_ledger`. The algorithm has these
phases (each is a labeled step in the function):

- **Step 0**: pre-pivot stats into nested-dict lookups for fast access.
  Two indexes: full `stats_lookup[unit_id][(y, s, var)]` and the
  per-`(unit_id, var)` index `stats_by_unit_var[(unit_id, var)] = list[(y, s, v)]`.
  The second one is critical for performance — without it the cascade
  computation is O(n_events × n_units × all-of-unit's-cells).
- **Step 1**: initialize ledger from raw stats.
- **Step 2**: chronological event walk. For each connected component
  in each year:
  - Build pool of parent pre-event cells (Step 2a).
  - Compute three-tier cascade fractions, picking one tier per
    (event, var, season) so siblings sum to 1 (Step 2b).
  - Distribute the pool to children scaled by their tier-chosen
    fraction (Step 2c).
  - Drop parents' pre-event cells; late-event cells stay for Step D
    (Step 2d).
- **Step D**: late reports. Cells at `year > terminal_event_year` on a
  ceased parent are written to `late_reporting.csv` and left there; the
  redistribution pass is present in the code but disabled (see
  `methodology_modern.md`).
- **Step 3**: assemble outputs into long-form DataFrames.
- **Step E**: append `season='Total Year'` rows. Vectorized via
  `pandas.groupby().agg()` — was the dominant cost (was ~280s on
  India before vectorization, now <1s).

All audit fields propagate through cell composition: `sources` (which
historical unit_ids contributed), `lineage_depth` (number of events
applied), `has_nan_fraction` (any NaN in the chain), `fraction_method`
(worst tier used: `seasonal` < `total_year` < `area` < `undefined`),
`late_report_redistributed`.

### 5.7 Outputs (under the product's `output_dir/`)

```
out/
├── stable_<year>.geojson              one per year in [base_year, max_year]
├── remap.json                         base-year unit_id → stable_id
├── snapshots.json                     year → list of active unit_ids
├── name_history.csv                   per-(stable_id, year, unit_id, name) audit
├── summary.json                       counts + run metadata
├── shapefile_vintage_report.txt       inferred year + per-year mismatch counts
├── stats_aggregated.csv               long-form on stable_id (after aggregate_stats)
└── modern/
    ├── stats_modern.csv               long-form on modern_id, includes Total Year rows
    ├── event_fractions.csv            per-(event, child, var, season) audit table
    ├── late_reporting.csv             late reports, not redistributed
    └── summary.json                   modern run metadata
```

Column references for the CSVs are in [`USAGE.md`](USAGE.md) §Outputs.

## 6. Validators

### 6.1 Lineage validator (`validate.py:validate_lineage`)

Runs on `LineageGraph.from_dataframe`. Eight checks:

| Category | Severity | What |
|---|---|---|
| `duplicate_row` | error | identical (event_year, event_type, parent_id, child_id) appearing twice |
| `self_referential_territorial` | error | Split/Merge/Redistribute with parent_id == child_id |
| `transient_unit` / `resurrected_unit` | error | a unit created and consumed in the same event year, or re-created after it ceased |
| `namechange_distinct_ids` | warning | NameChange with parent_id != child_id |
| `multi_parent_split` | info | a child appearing as Split-child of multiple parents (suggest retag as Redistribute) |
| `split_looks_like_rename` | warning | 1-to-1 Split with same parent_name and child_name |
| `leaf_count` | info | sanity check vs modern shapefile feature count |
| `namechange_unit_name_mismatch` | warning | a NameChange whose parent_name is not the name the unit carried at that point |

### 6.2 Stats × lineage cross-validator (`validate.py:validate_stats_lineage_consistency`)

Runs on `aggregate_stats` for both products. Three checks:

| Category | Severity | What |
|---|---|---|
| `stats_unit_id_malformed` | error | sentinel values like `__FILTER__`, blanks, NaN |
| `stats_unit_id_unknown` | warning | unit_id not in lineage AND not in modern shapefile |
| `stats_post_cease_reporting` | warning | unit_id reporting AFTER its terminal territorial event (the Giridih pattern: pre-1991 parent code still receiving reports through 2023) |

The post-cease check is the highest-leverage one for catching upstream
id-assignment bugs. On India it has surfaced stale-code unit ids whose
rows would otherwise sit under a ceased unit rather than the stable group
their successor belongs to.

### 6.3 Stable-group contiguity validator (`validate.py:validate_stable_group_contiguity`)

Runs on `build_boundaries` (step 7a-bis), against the base-year remap
and modern shapefile. One check:

| Category | Severity | What |
|---|---|---|
| `stable_group_disconnected` | warning | a stable group's modern members are not all reachable from each other through shared borders (BFS over a `predicate="touches"` neighbor graph) |

Common causes the validator surfaces:
- (a) **Homonym name-matching bug**: e.g., "Hamirpur" exists in both
  Himachal Pradesh and Uttar Pradesh; if the relationship table joins
  one to the wrong modern feature, the resulting stable group spans two
  parts of the country. **This is the bug class the validator was
  written to catch.**
- (b) **Island/exclave geometry**: e.g., Andaman Islands, Manipur's
  Jiribam exclave. Legitimate physical separation; not a bug.
- (c) **Topology imprecision**: the modern shapefile has tiny gaps
  between adjacent features (post-split shapefiles often do).
  Legitimate semantically; just a warning.

Always-warning: never blocks `build_boundaries`. On India the findings
are of types (b) and (c); the homonym case (a) is what the check exists
for and is absent from the shipped lineage.

## 7. Performance notes

Measured on a laptop, India 1997–2022:

| Step | India |
|---|---|
| `sb.build_boundaries` | ~15 s |
| `sb.aggregate_stats` | ~10 s on the ~750,000-row statistics table |
| `mb.aggregate_stats` | seconds |

Bottleneck history (for context when reading the code):

- **`_append_total_year_rows`**: the original Python-level `for group in
  groupby` loop was O(1.6M groups × 6 per-group ops in Python). Took
  280s on India. Vectorized to a single `groupby().agg()` call — now <1s.
- **Cascade fraction computation**: required pre-pivoting stats by
  `(unit_id, var)` so each `_window_means` / `_total_year_means` call
  iterates only the rows for that variable instead of all of a unit's
  reports. Implemented as `stats_by_unit_var` and `stats_vars_by_unit`
  in `build_modern_ledger` Step 0.
- **Validator's post-cease check**: was O(n_parents × n_stats_rows)
  filters; converted to a single `groupby('unit_id')` over the candidate
  subset.

## 8. Testing strategy

`tests/` contains about 600 tests across 30-odd files. The ones that
map most directly onto the algorithms:

| File | What it covers |
|---|---|
| `test_schema.py` | Schema validators raise on bad inputs. |
| `test_lineage.py` | `LineageGraph` parsing, adjacency views. |
| `test_validate.py` | All three validators (lineage + stats × lineage + stable-group contiguity). |
| `test_snapshot.py` | Algorithm 1 + 2 with hand-built fixtures. |
| `test_groups.py` | Algorithm 3 + the connected-component merge. |
| `test_dissolve.py` | Geometry dissolution. |
| `test_reconcile.py` | Drop / sum-jump tests in all three modes (the module is not wired into the products). |
| `test_stats.py` | Aggregation + intensive recomputation. |
| `test_modern_algorithm.py` | All edge cases for the cascade fraction logic. |
| `test_modern_boundary.py` | End-to-end on Exampleland. |
| `test_conservation.py` | Total Year strict equality between products. |
| `test_qualification.py` | One invariant battery run across every synthetic fixture under `tests/fixtures/synthetic/`. |
| `test_release_scope.py` | India is the only bundled country and the wheel carries exactly its data. |

Run: `pytest tests/` (about two minutes on a laptop).

## 9. Known limitations

Documented for honesty; see also Known Issues in the README.

- **India ~0.5–1.5% Total Year residual** on heavy-reorganization
  variables (rice, wheat) in the modern product. Mostly genuine
  `fraction_method='undefined'` cells where a child has no data anywhere
  AND no modern shapefile feature.
- **The modern product has not been re-audited on India** since the
  0.1.3 lineage fixes; treat its India outputs as provisional.
- **Late reports are not redistributed** in the modern product, and
  **reconciliation is not wired in**; both are described in the
  methodology documents.

## 10. Where to dig deeper

For specific concerns:

- **Adopting the package for a new country**: read `USAGE.md` §"Inputs"
  and §"Quickstart"; copy `tools/india/prepare_inputs.py` as a
  template.
- **The math behind a specific algorithm**: `methodology.md` for stable,
  `methodology_modern.md` for modern. Each section pairs to a paper
  algorithm.
- **Why a particular design decision**: §4 above, and the module
  docstrings, which record the alternatives that were tried.
- **Specific code paths**: each module's file-level docstring is the
  best entry point. They were written for this purpose.
