# StableBound Code Walkthrough

A condensed reference for the package, organized in pipeline order from package import through the two products. Sources for each section are the module file paths in `src/stablebound/`.

Source-of-truth for the algorithms is `docs/methodology.md` (stable) and `docs/methodology_modern.md` (modern). This document is the engineering tour.

---

## 1. Package surface (`__init__.py`)

Pure re-exports. Importing `stablebound` triggers no disk reads, validators, or computation. Public surface clusters into 7 groups:

- **Lineage primitives**: `Lineage`, `LineageGraph`, `DataDictionary`
- **Bundled data**: `BUNDLED_COUNTRIES`, `BundledCountry` (India, `IN`)
- **Name matching**: `MatchProposal`, `propose_shapefile_mapping`, `propose_stats_mapping`, `attach_shapefile_ids`, `attach_stats_ids`, `read_mapping`, `normalize_name`
- **Validation**: `validate_lineage`, `validate_stats_lineage_consistency`, `validate_stable_group_contiguity`, `LineageDataError`, `LineageIssue`, `SchemaError`, `format_issues`
- **FNID / unit_defs**: `assign_fnids`, `build_fnid`, `build_admin1_code_map`, `build_admin2_code_map`, `FNIDOverflowError`, `build_unit_defs_table`, `build_admin1_attribution_table`, `write_unit_defs_files`
- **Products**: `StableBoundary`, `ModernBoundary`
- **Utilities**: `convert_relationship_table_to_lineage`, `read_legacy_relationship_table`, `analyze_breakpoints`, `BreakpointResult`

No `Config` class — removed in the refactor. Per-call kwargs only.

---

## 2. Three construction paths for `Lineage`

| Path | Call | Source format |
|---|---|---|
| **Bundled** | `Lineage("IN")` | Canonical (event-only RT + baseline + optional NCL) bundled in `src/stablebound/data/IN/` |
| **Custom canonical** | `Lineage("XX", relationship_table_path=..., baseline_path=..., name_change_log_path=...)` | Same canonical schema, user-supplied paths |
| **FEWS legacy** | `Lineage.from_legacy_rt(path, country="XX", admin_level=N, name_change_log_path=...)` | FEWS hierarchical+temporal CSV (`category` + `relationship_type` columns) |

All three return functionally-identical `Lineage` objects. `from_legacy_rt` runs the conversion in-memory (no intermediate files); the baseline is derived from the earliest hierarchical snapshot.

### Construction is mostly lazy

- `__init__` does **zero disk reads** beyond `Path.exists()` checks.
- `from_legacy_rt` reads the FEWS RT + optional NCL to convert, but does NOT parse the graph or validate.
- `LineageGraph.from_dataframe` (parse + auto-validate) runs only on first access of `ln.lineage`.

So constructing 50 Lineages to enumerate metadata is essentially free; the cost is paid lazily when `.lineage` is touched.

### FEWS RT format quirks (handled by `rt_convert.py`)

Two row kinds via `category`:
- `hierarchical` — defines which units exist at each snapshot year. `relationship_type = "admin{N}_{parent_level}"` (e.g. `admin1_0`, `admin2_1`).
- `temporal` — transitions between snapshot years. `relationship_type ∈ {successor, split, merge, redistribute, "name change"}` (lowercase; "name change" has a space).

Snapshot year is embedded in the FNID, not a separate column (e.g. `XX2010A101` → 2010).

Two distinct rename encodings the converter handles:
- Explicit `relationship_type == "name change"` (rare; India has 1)
- 1-to-1 `successor` rows where `from_unit_name != to_unit_name` (dominant; India has 11)

FEWS RTs are often incomplete on renames; pass `name_change_log_path` to overlay a sidecar NCL.

---

## 3. `Lineage` class surface

Identity + metadata (cheap):
```
ln.country_code, ln.validity_start_year, ln.notes, ln.max_year
```

Lazy data properties (cached after first access):
```
ln.relationship_table     # canonical RT with NCL merged in as NameChange rows
ln.baseline               # baseline snapshot DataFrame (or None)
ln.name_change_log        # standalone NCL (or None)
ln.lineage                # parsed LineageGraph (auto-validates on first call; may raise LineageDataError)
ln.validation_issues      # list[LineageIssue] (warnings + infos only — errors already raised)
ln.validation_report()    # formatted text summary
```

Derived bounds:
```
ln.min_year, ln.years     # inclusive year range
```

Snapshot inspection:
```
ln.snapshot()             # DataFrame at the latest year (or shapefile_year)
ln.snapshot(year=2010)    # at a specific year
```

Name matching (human-in-the-loop):
```
ln.default_normalizer
ln.propose_shapefile_mapping(shapefile, name_column=, coarse_column=)  # returns MatchProposal; does NOT mutate
ln.attach_shapefile(shapefile, mapping=, name_column=, on_unmatched=)  # MUTATES (.shapefile, .shapefile_year, .unmatched_features)
ln.propose_stats_mapping(stats, name_column=, coarse_column=, year_column=)
ln.attach_stats_ids(stats, mapping=, name_column=)                     # does NOT mutate (returns DataFrame)
```

**Asymmetry**: shapefile attachment mutates the Lineage (products auto-read it); stats attachment does not (you pass the returned DataFrame to the product explicitly).

### Lazy-load cascade

```
ln.validation_issues  →  ln.lineage  →  ln.relationship_table  →  reads RT + NCL files, merges
ln.baseline           →  reads baseline_path (independent of RT)
ln.name_change_log    →  triggers ln.relationship_table (NCL loaded as side-effect)
```

`__repr__` reads cache fields directly without triggering loads — `print(ln)` is always cheap.

---

## 4. `LineageGraph.from_dataframe` (parse + validate)

`lineage.py:60-131`. Fires on first `.lineage` access. Four steps:

1. **Schema check** (`validate_relationship_table`) — 6 required cols + `event_type ∈ {Split, Merge, Redistribute, NameChange, Coarse}`. Raises `SchemaError`.
2. **Type coercion** — `event_year → int`.
3. **Column projection** — drop non-canonical columns.
4. **Data quality** (`validate_lineage`) — 8 checks. Errors raise `LineageDataError`; warnings/infos logged at INFO.

**LineageGraph** itself is a thin dataclass wrapping the events DataFrame. Computed views:
```
.events                          # the raw validated DataFrame
.territorial                     # Split | Merge | Redistribute only (NOT NameChange/Coarse)
.name_changes                    # NameChange only
.min_event_year, .max_event_year
.all_unit_ids(), .initial_units()
.adjacency_after(year)           # (parent→children, child→parents) dicts for territorial events with event_year >= year
.redistribute_children_after(year)
```

**`territorial` filter is load-bearing** — NameChange and Coarse are metadata-only (no territorial movement), so they're excluded from adjacency/snapshot construction.

### Event-year convention (CRITICAL)

`event_year = T` means the event happens **during** year T. Snapshot at start-of-T is **pre-event**; snapshot at start-of-(T+1) reflects the event. So `adjacency_after(T)` uses `>=`, but `snapshot(year=T)` uses events with `event_year < year` (strict).

---

## 5. `snapshot()` and `build_snapshot` (paper Algorithm 1)

```python
ln.snapshot(year=2010)   # DataFrame of (unit_id, unit_name, year) at year 2010
```

Calls `build_snapshot(graph, year, additional_units=baseline_unit_ids)` then attaches names.

### `build_snapshot` (`snapshot.py:24-108`)

Four phases:

1. **Initial active set**: `graph.initial_units()` — units that appear as territorial parents but never as territorial children.
2. **Augment with always-alive externals**: `additional_units - rt_units` (the critical trick). Only units that DON'T appear in any RT event are seeded; this lets baseline/modern IDs flow in without backdating units like Sejong (created 2012) into 1986.
3. **Filter events**: `territorial[event_year < year]`.
4. **Walk events year-by-year**: per year, atomically remove all parents and add all children.

**The `additional_units - rt_units` filter** is the single most important line. Over a hundred of India's 467 baseline districts have zero RT events; without this filter they'd be invisible.

### `infer_year(shapefile_units, graph, candidate_years)`

Returns `(best_year, mismatches_by_year)` — the year whose snapshot best matches a given unit set, by minimum symmetric difference. Earliest year wins ties.

---

## 6. Name matching: `propose_shapefile_mapping` + `attach_shapefile`

`match.py`. The user-facing flow is propose → review CSV → attach.

### `propose_shapefile_mapping` (the matcher)

```python
proposal = ln.propose_shapefile_mapping(
    shapefile,
    name_column="DISTRICT",
    coarse_column="STATE",   # optional, disambiguates homonyms
    year=None,               # None → infer (default); int → explicit
    year_range=None,         # auto-passed by Lineage wrapper as self.years
)
```

Pipeline:
1. Coerce input (path or GeoDataFrame)
2. **Year resolution** (in order of precedence):
   - `year=<int>` → use that
   - `year=None` + `year_range` provided → infer via `_infer_year_by_name` (exact + fuzzy at `fuzzy_threshold`, earliest-year tiebreak)
   - `year=None` + no `year_range` → fall back to `max_event_year + 1`
3. Build snapshot lookup: `{normalized_name: [(unit_id, original_name, coarse_name), ...]}` — values are LISTS to support homonyms.
4. Run 4-pass matcher (homonym → exact → manual → fuzzy → unmatched). Fuzzy is in a second loop after exact/manual/homonym claims so it can't steal cleanly-matchable units.

**Coarse disambiguation**: when a normalized-name bucket has multiple candidates, `_pick_by_coarse` compares the shapefile's coarse value to the lineage candidates' coarse names. If no `coarse_column` was passed (or no candidate matches), falls back to "pick first" — silently arbitrary.

**Homonym warning**: when `_pick_by_coarse` returns None despite multiple candidates, the row's `notes` is set and a `UserWarning` fires listing the affected rows + alternatives. User can resolve via `coarse_column=...` or `homonym_overrides={idx: (unit_id, name)}`.

### `MatchProposal` (the review artifact)

DataFrame `proposals` with columns:
```
source_idx, source_name, source_year (NA for shapefile),
proposed_unit_id, proposed_name,
method,            # homonym | exact | manual | fuzzy | unmatched
score,             # 1.0 for non-fuzzy, ratio for fuzzy, NaN for unmatched
best_candidate,    # near-miss when unmatched
notes              # homonym warnings + other per-row signals
```

Plus:
```
proposal.matched, proposal.unmatched, proposal.sketchy   # filtered views
proposal.summary()                                        # text report
proposal.to_csv("review.csv")                             # save for Excel review
proposal.inferred_year                                    # if year inference fired
proposal.year_inference_mismatches                        # {year: mismatch_count}
```

### `attach_shapefile`

```python
ln.attach_shapefile(
    shapefile,
    mapping="review.csv",        # or DataFrame, or None if id_column already on the shapefile
    name_column="DISTRICT",
    id_column="unit_id",
    on_unmatched="keep",         # "keep" | "drop" | "error"
)
```

- `"keep"` (default): unmatched features get sentinel IDs `UNMATCHED_<idx>` and flow through products as singleton stable groups.
- `"drop"`: silently filtered out; still recorded on `ln.unmatched_features` for audit.
- `"error"`: raise `ValueError` with offending sample.

After attachment: `ln.shapefile`, `ln.shapefile_year` (auto-inferred from real unit_ids via `infer_year`), `ln.unmatched_features`.

### Normalization layers

`_resolve_normalizer` picks:
1. Explicit `normalizer=` kwarg
2. `BUNDLED_COUNTRIES[ln.country_code].normalizer` (India: strips "district", "dist.", "Sri", "Shri", "Dr.")
3. Package default `normalize_name` (lowercase + NFKD + punctuation + whitespace)

---

## 7. `StableBoundary` (the first product)

`boundary.py`. Orchestrator for the paper's main product.

### Constructor

```python
sb = StableBoundary(
    lineage,                  # must have shapefile attached by build time
    target_year=None,         # default: lineage.min_year
    max_year=None,            # default: lineage.shapefile_year (then graph.max_event_year + 1)
    output_dir=None,          # default: ./stablebound_out
)
```

### Pipelines

```python
sb.build_boundaries(refresh=False)          # geometry only; writes stable_<year>.geojson per year
sb.aggregate_stats(stats, ...)              # stats; auto-calls build_boundaries if needed
```

### `build_boundaries` (geometry-only, no stats)

Per year in `[target_year, max_year]`:
1. `build_stable_groups(graph, year, additional_units)` → remap
2. `dissolve(modern_gdf, remap, "unit_id")` → polygons for that year
3. Write `stable_<year>.geojson`

Then writes: `remap.json` (canonical = target-year remap), `snapshots.json`, `name_history.csv`, `shapefile_vintage_report.txt`, `summary.json`, optional `unmatched_features.geojson`. Validates contiguity.

### `aggregate_stats`

```python
sb.aggregate_stats(
    stats,                          # path or DataFrame
    mapping=, name_column=,         # if stats need unit_id attached
    data_dict=,                     # shorthand for the next 4 kwargs
    stats_columns=, extensive=, intensive=, ignore=,
    reconcile_mode="off",           # the only accepted value; see §10
    refresh=False,
)
```

Steps: load stats → validate stats × lineage → `aggregate` (extensive pass, then the intensive pass over matched constituents) → write `stats_aggregated.csv`.

### Inspection

```python
sb.get_boundary(year)             # GeoDataFrame for one year (always reads from disk)
sb.get_stats(variable=, year=, years=, season=, stable_id=)
sb.get_reconciliation_flags()     # always empty: reconciliation is not wired in
sb.get_name_history()
sb.validation_report()            # combined: lineage + contiguity + stats × lineage
sb.summary()                      # counts dict
```

`get_stats` rejects unknown filter keys (no more silent typos).

### Cache layers

- **On-disk**: `summary.json`'s `_schema` (currently 3) + `country_code` + `target_year` must match the current run, else rebuild.
- **In-memory**: results stashed on the instance (`_modern_gdf`, `_all_remaps`, `_stats_agg`, etc.).
- `refresh=True` forces full re-run.

### Output files

```
output_dir/
├── stable_<year>.geojson         # one per year
├── remap.json                    # {unit_id: stable_id} — canonical = target_year remap
├── snapshots.json                # {year: [active unit_ids]}
├── name_history.csv              # stable_id × year × unit_id × name audit
├── shapefile_vintage_report.txt  # which max_year source won
├── unmatched_features.geojson    # (only if any)
├── summary.json                  # counts + schema version
└── stats_aggregated.csv          # (only after aggregate_stats)
```

---

## 8. `build_stable_groups` (paper Algorithm 3)

`groups.py`. Union-Find construction.

```python
remap = build_stable_groups(graph, base_year, additional_units)
# Returns: dict[unit_id → stable_id]
```

Steps:
1. Snapshot at base_year → "starting points"
2. Adjacency for events ≥ base_year (territorial only)
3. For each starting point, DFS forward → collect all descendants
4. Universe = base ∪ descendants ∪ additional_units
5. Union-Find: union each base with each descendant
6. stable_id = lex-smallest member of each group (deterministic across runs)

### Two load-bearing design choices

**Connected-component merge**: multi-parent children force their parents' groups to merge. If event A→C and B→C exist, A and B end up in one group via C. **Cost**: coarser geometry. **Benefit**: stats sum cleanly across the group at every year — zero allocation assumptions. This is the paper's whole contract.

**Singleton fallback**: every unit in `additional_units` ends up in the remap, even as an orphan group. Without this, modern features with no RT events would silently vanish from the dissolved output.

### Merge events

Same algorithm. `A + B → C` produces traces:
- `_trace_forward(A) = {A, C}`
- `_trace_forward(B) = {B, C}`
- Union-Find merges A and B via shared descendant C.
- Result: `{A, B, C}` are one stable group with `stable_id = min(A, B, C)`.

Stats from years before, during, after the merge all aggregate to the same stable_id. Conservation by construction.

---

## 9. `dissolve` (geometry merging)

`dissolve.py`. Short module, one function:

```python
gdf = dissolve(modern_gdf, remap, "unit_id")
# Returns: GeoDataFrame with stable_id, n_modern, source_ids, geometry
```

5 substantive lines: filter → map → drop NaN → repair → `groupby + unary_union`.

Notes:
- `unary_union(list)` over iterative pairwise — faster + more numerically stable.
- `make_valid` is opt-in per geometry (checked `is_valid` first) — expensive call.
- Features with no remap entry are silently dropped (rare with the singleton fallback).

---

## 10. `reconcile` (paper Algorithm 4)

`reconcile.py`. Conservation diagnostics on stats. **Implemented but not
wired into either product**: `aggregate_stats(reconcile_mode=...)` accepts
only `"off"`, and the module is reachable only by calling `reconcile`
directly. See Known Issues in the README for why.

```python
from stablebound.reconcile import reconcile
reconciled, modified_remap, flags = reconcile(
    stats_df, graph, remap,
    mode="flag",                     # "off" | "flag" | "merge" | "subtract"
    tau_drop=0.15, tau_sum=0.15,
    window=3,
)
```

### The two diagnostics

For each Split/Merge event × variable, over a 3-year pre/post window:

**Drop test** (eq. 3): `d_t < r_t - tau_drop` where `r_t = (Σ_j Y_{j,post}) / Y_{i,pre}` (expected drop) and `d_t = 1 - Y_{i,post}/Y_{i,pre}` (observed). Catches "parent didn't drop enough."

**Sum-jump test** (eq. 4): `(parent_post + Σ children_post) / parent_pre > 1 + tau_sum`. Catches "post-event sum exceeds pre-event baseline" = double-counting.

Symmetric handling for merges with parent/successor roles reversed. The "persisting parent" heuristic picks the highest-reporting parent among multiple.

### Modes (of the standalone function)

- **`off`**: skip diagnostic; return inputs unchanged + empty flags.
- **`flag`**: run diagnostics, never mutate. Surface findings.
- **`merge`**: union flagged stable polygons; drop redundant rows. Modifies remap. On India this deleted legitimate post-event child rows in most flagged cases, which is why the products do not call it.
- **`subtract`**: subtract double-counted rows from parent. Modifies stats. Falls back to merge if undefined.

### Redistribute events

Surfaced as `mode_applied="redistribute_unreconcilable"` for transparency — can't be reconciled from totals alone (paper §3.3).

### Performance note

`_build_window_lookup` pre-pivots stats once into `{(unit_id, variable): {year: value}}` so the per-event loop does O(window) lookups. The same trick is used in `aggregate`.

---

## 11. `aggregate` + `derive_intensive` (paper Algorithm 5)

`stats.py`.

### `aggregate`

```python
agg = aggregate(stats_df, remap, graph, base_year, max_year,
                intensive={"yield_mt_ha": ("production_mt", "area_ha")})
# Returns columns: year, season, variable, stable_id, value,
#                  n_constituents, n_in_group, completeness, complete,
#                  missing_unit_ids, constituent_ids, late_reporting
```

`intensive` (v0.1.4) adds one derived row per cell for each declared ratio,
computed from a second pass over only the rows whose unit reported both
inputs in that (unit, year, season); see "Intensive variables" in
`methodology.md`. Both passes go through the same routing helper.

Three routing cases:
- **Standard**: unit_id alive in snapshot(year) → route to `remap[unit_id]`.
- **Late reporting**: unit_id NOT in snapshot(year) AND NOT in snapshot(year+1) — flagged `late_reporting=True`, `stable_id=unit_id` (surfaced under own ID, not rerouted).
- **One-year forward grace**: child reporting in its creation year (or parent in its dissolution year) is NOT flagged — the event-year convention is ambiguous about that boundary.

### Completeness columns (snapshot-aware)

- `n_constituents` — distinct unit_ids that reported (current behavior)
- `n_in_group` — count of remap members alive at the row's year, using the
  strict `snapshot(year)` (NA for late_reporting rows). Note this is a
  *different* window from the one `late_reporting` uses: that check allows a
  one-year grace so the normal parent→child handoff at an event year isn't
  flagged, whereas completeness must not count a ceasing parent and its
  not-yet-existing children as members of the same year.
- `complete` — True iff every alive-at-year member reported, i.e.
  `missing_unit_ids == ""`. A subset test, not a count comparison: an early
  reporter can make the counts match while the sets differ.
- `missing_unit_ids` — alive-at-year members that didn't report

### Strict NaN-season validation

`validate_stats` raises `SchemaError` on missing season values. Use an explicit sentinel like `"Annual"` for non-seasonal data.

### `derive_intensive`

```python
agg = derive_intensive(agg, intensive_pairs={"yield_mt_ha": ("production_mt", "area_ha")})
```

Drops pre-existing rows for the intensive variable names, then per pair joins num/den on `(year, season, stable_id, late_reporting)` and emits `value = num / den.where(den != 0)`. Zero denominators → NaN (not Inf). Called by `aggregate` on the paired subset (so the ratio is over matched constituents) and by the modern product on its already-aggregated frame (full sums; no per-unit rows are available there).

### Two NaN details

- `min_count=1` on the sum: all-NaN groups → NaN (not 0). Mixed-NaN groups → sum of real values.
- `dropna=False` on the groupby: preserves NaN-season groups (no longer triggers thanks to strict validation, but kept for safety on other nullable columns).

### Late-reporting warning

After grouping, if any rows have `late_reporting=True`, emits a `UserWarning` listing count + first 5 samples + suggested actions.

### Progress feedback

Stderr line `[stablebound] aggregating N stats rows (...)` if N >= 10K. India runs print start + end-elapsed lines.

---

## 12. `ModernBoundary` (the second product)

`modern.py`. Orchestrator for the "history rescaled onto today's geometry" product.

### Conceptual difference vs StableBoundary

|  | Stable | Modern |
|---|---|---|
| **Geometry pinned to** | target year (past) | today's shapefile |
| **Output rows per** | stable_id × year | modern_id × year |
| **Allocation needed?** | No | **Yes** (post-event redistribution) |
| **Has `build_boundaries`?** | Yes | **No** — geometry IS the shapefile |

### API

```python
mb = ModernBoundary(lineage, target_year=None, max_year=None, output_dir=None)
mb.aggregate_stats(
    stats,
    mapping=, name_column=,
    data_dict=, stats_columns=, extensive=, intensive=, ignore=,
    modern_window={"rice_area_ha": 3},   # per-variable window override
    modern_window_default=5,             # default years
)
mb.get_modern_stats(...)
mb.get_event_fractions()
mb.get_late_reporting()
mb.validation_report()   # 2 sections (no contiguity check)
mb.summary()
```

Output subdirectory: `output_dir/modern/`.

### Output files (all under `output_dir/modern/`)

```
stats_modern.csv         # long-form: year, season, variable, modern_id, value,
                         #            sources, lineage_depth, has_nan_fraction,
                         #            fraction_method, late_report_redistributed
event_fractions.csv      # audit: per (event, child, variable, season) — the chosen fraction + method
late_reporting.csv       # reports stranded on units with no modern destination
summary.json             # counts + run metadata
unmatched_features.geojson  # if any
```

---

## 13. `build_modern_ledger` (paper Algorithm 6)

`modern_algorithm.py`. ~1100 lines, organized into 5 steps.

### The 5 steps

```
A — Initialize ledger from stats (each unit gets cells {(year, season, variable): value})
B — At each event, compute child fractions via 3-tier cascade
C — Pool parents, distribute pool to children by fraction
D — Late reports: written to late_reporting.csv (redistribution disabled)
E — Append Total Year derived rows
```

### Central data structure: ledger

```python
ledger: dict[unit_id, dict[(year, season, variable), cell]]

cell = {
    "value": float,
    "sources": set[str],         # which original unit_ids contributed
    "depth": int,                # how many events back the ancestor is
    "fraction_method": str,      # "seasonal" | "total_year" | "area" | "undefined"
    "has_nan_fraction": bool,
    "late_report_redistributed": bool,
}
```

### The 3-tier cascade (Step B)

Each tier picks ONE fraction method per `(var, season)` for the WHOLE event (all siblings share). This preserves `sum(fractions) = 1` per pool cell — conservation.

**Tier 1 — Seasonal (common years intersection)**: per `(var, season)`, intersect years all children reported in the window. Each child's mean over those common years; normalize across siblings. Apples-to-apples — not biased by one child's unique-year noise.

**Tier 2 — Total-year (common cells intersection)**: per season, intersect years across siblings; sum per-season means per child.

**Tier 3 — Modern area**: each child's share of the modern shapefile area.

**`undefined`**: all three fail. Cell value goes to NaN.

The tier chosen is recorded as `fraction_method`. Worst-method composition: a cell that's been through `seasonal` and then `area` events records `area` because that's worse.

### Per-event audit row (event_fractions.csv)

```
event_year, event_type, parent_ids, child_id, variable, season,
window_used, n_common_observations, n_individual_observations,
fraction, fraction_method
```

Sort by `n_individual - n_common` to find children whose gap-filling would tighten the fraction estimate.

### `_connected_components` (per-year event grouping)

Within a year, multiple events can be unrelated. Bipartite graph traversal groups parents + children into independent components. Each component is processed together — parents pooled, fractions computed jointly, distributed to all children.

A multi-row Split (one parent → N children) becomes one component. A multi-parent child (Tirupati-style or Merge) unites its parents into one component. Two unrelated splits in the same year stay separate.

### Step C: `_pool_parents` + `_distribute_pool_to_children`

```
For each parent in component:
    For each cell in parent's ledger with year <= event_year:
        sum value into pool[(year, season, variable)]
        union sources, OR flags, max depth, worst fraction_method

For each child:
    For each pool cell:
        inherited_value = pool_cell.value * fraction
        Add to child's ledger; compose method = worst(pool method, this event method)

Drop parents' pre-event cells (year <= event_year)
```

Late cells (`year > event_year`) stay on parent for Step D.

### Step D: late reports (redistribution disabled)

For each parent that ceased (has a terminal event), cells with `year > terminal_event_year` are written to `late_reporting.csv` and are not distributed to the children, so they do not appear in `stats_modern.csv`. The redistribution pass (`_redistribute_late_reports`, which would replay the terminal event's fractions and tag `late_report_redistributed=True`) is present but not called: on India it double-counted real observations that the upstream id mapping had filed under several canonical ids.

### Step E: Total Year rows

For each `(modern_id, year, variable)`, sum across explicit seasons → append a `season="Total Year"` row. **Strict conservation only holds at Total Year** (per-season conservation is approximate because the cascade can pick different tiers for different seasons).

Yields recomputed via `derive_intensive` over the Total Year subframe.

---

## 14. `validate.py` (data-quality conscience)

Twelve checks in three groups, plus two further suites (`validate_coarse_references` for admin2 ↔ admin1 attribution, run by `Lineage.validate_levels()` and before a FEWS export; `validate_shapefile_lineage_consistency`, run by `attach_shapefile`).

### Group 1: 8 lineage validators (auto-fire on `LineageGraph.from_dataframe`)

| Severity | Check | Catches |
|---|---|---|
| **error** | `duplicate_row` | Exact-duplicate `(year, type, parent_id, child_id)` |
| **error** | `self_referential_territorial` | Split/Merge/Redistribute with `parent_id == child_id` |
| **error** | `transient_unit` / `resurrected_unit` | A unit created and consumed in one event year, or re-created after it ceased |
| warning | `namechange_distinct_ids` | NameChange where `parent_id != child_id` |
| info | `multi_parent_split` | Child appears as Split-child of multiple parents |
| warning | `split_looks_like_rename` | Single-child Split with same parent/child name |
| info | `leaf_count` | Reports # of units appearing only as territorial children (modern-leaf sanity check) |
| warning | `namechange_unit_name_mismatch` | NameChange whose `parent_name` is not the unit's name at that point |

### Group 2: 3 stats × lineage validators (fire in `aggregate_stats`)

| Severity | Check | Catches |
|---|---|---|
| **error** | `stats_unit_id_malformed` | Sentinel values, empty strings, NaN unit_ids |
| warning | `stats_unit_id_unknown` | unit_ids not in lineage AND not in modern shapefile |
| warning | `stats_post_cease_reporting` | unit_ids that ceased per a territorial event but keep reporting |

### Group 3: 1 contiguity validator (fires in `build_boundaries`)

`validate_stable_group_contiguity` — flags stable groups whose modern members aren't spatially connected (`touches` self-join). Usually means a homonym name-matching bug.

### `LineageIssue` dataclass

```python
severity: "error" | "warning" | "info"
category: str
message: str
detail: str
rows: list[int]      # event-row indices
ids: list[str]       # implicated unit_ids
```

### `format_issues(issues)` + `validation_report()`

Errors block construction (raise `LineageDataError` with the formatted report). Warnings/infos surface via `ln.validation_report()` / `sb.validation_report()` / `mb.validation_report()`.

---

## 15. `fnid.py` + `unit_defs.py` (FEWS interop)

### FNID format

```
<ISO><YYYY>A<LEVEL><CODE>
IN2003A20107   ← India, 2003, admin 2, state 07 + district 01
VN1991A101     ← Vietnam, 1991, admin 1, state 01
```

CODE = 2 chars for admin1 (SS), 4 chars for admin2 (SS + DD).

### Code alphabet

359 entries: `01..99`, then `A0..A9`, `B0..B9`, ..., `Z9`. `FNIDOverflowError` if a country needs more.

### `build_admin1_code_map`

Deterministic SS per admin1, sorted by `(first_appearance_year, admin1_id)`. **Retired-code rule**: ceased admin1s keep their slots; new admin1s get fresh slots after the highest used. Produces visible gaps in the sequence wherever a unit was retired.

### `build_admin2_code_map`

Per admin2: `(SS, DD)` where SS = origin admin1 (walking back through events to earliest ancestor), DD = per-origin index. **Path-dependent** — a district that was historically in AP but is now in Telangana keeps the AP `SS`.

**DD is minted per revision and is NOT stable across lineage corrections.** SS gets the retired-code rule, so a state's code survives its neighbours changing; DD does not. DD is `sorted(unit_ids)` then `enumerate`, re-packed on every build, so inserting or removing a single unit shifts the code of every district that sorts after it *within that state*. Correcting one spurious Rajasthan district re-pointed 41 district codes across India's 2024 and 2025 vintages — the same FNID naming a different district, which joins silently rather than failing.

This is forced, not chosen. DD holds 359 codes per state, while ADM2 ids are globally sequential (India reaches `01127`), so 24 of India's 32 origin states hold ids whose ordinal overflows the DD space and the retired-code rule cannot be applied. Making DD stable would need a persisted `unit_id → DD` map checked in alongside the lineage.

Consequence for downstream users: **join on `unit_id`, not FNID, across vintages.** FNIDs are for handing to FEWS, and a bundle's FNIDs are only meaningful against the lineage revision that produced them.

### `assign_fnids(stats, code_map, *, iso, level, year_col, unit_id_col)`

Joins code map onto stats, computes per-row `build_fnid(iso, year, level, ss, dd)`. Rows whose unit_id isn't in the code map get empty FNID.

### `unit_defs.py` — per-year unit-definition tables

```python
write_unit_defs_files(
    graph, baseline,
    admin1_code_map=, admin2_code_map=,
    iso=, admin0=, years=, levels=(1, 2),
    out_dir="unit_defs/",
)
# Writes one CSV per (level, year):
#   unit_defs/admin1/IN_Admin1_1991.csv
#   unit_defs/admin1/IN_Admin1_1992.csv
#   ... etc
```

Per file, one row per active admin unit at that year. Admin1 schema: `FNID, EFF_YEAR, COUNTRY, admin0, admin1`. Admin2 adds `admin2` column.

`build_admin1_attribution_table` walks events forward from baseline to compute `(year, unit_id) → (admin1_id, admin1_name)` — the per-year admin1 attribution that's not trivially in the lineage.

---

## 16. End-to-end recipe: FEWS RT → unit_defs

```python
from stablebound import (
    Lineage, build_admin1_code_map, assign_fnids, write_unit_defs_files,
)

# 1. Load FEWS RT (the hand-authored fixture under tests/fixtures/rt_convert/
#    is a small real example of the format)
ln = Lineage.from_legacy_rt(
    "relationshiptable_XX.csv",
    country="XX", admin_level=1,
    name_change_log_path=None,
)

# 2. Inspect
print(ln.validation_report())

# 3. Attach modern shapefile (propose → review → attach if needed)
proposal = ln.propose_shapefile_mapping(
    "xx_provinces.geojson", name_column="NAME", coarse_column="REGION",
)
proposal.to_csv("review.csv")
# ... review in Excel ...
ln.attach_shapefile("xx_provinces.geojson", mapping="review.csv",
                    name_column="NAME")

# 4. Build FNID code maps
admin1_codes = build_admin1_code_map(ln.lineage, ln.baseline, iso="XX")
# admin2_codes = build_admin2_code_map(ln.lineage, ln.baseline, iso="XX")  # if coarse cols exist

# 5. Attach FNIDs to stats (assumes stats already have unit_id)
stats_with_fnid = assign_fnids(
    stats_with_ids, admin1_codes, iso="XX", level=1,
    year_col="year", unit_id_col="unit_id",
)
stats_with_fnid.to_csv("xx_stats_with_fnid.csv", index=False)

# 6. Write per-year unit_defs files
write_unit_defs_files(
    ln.lineage, ln.baseline,
    admin1_code_map=admin1_codes,
    iso="XX", admin0="Exampleland",
    years=ln.years, levels=(1,),
    out_dir="xx_unit_defs/",
)
```

Optionally then run the products:

```python
from stablebound import StableBoundary, ModernBoundary
sb = StableBoundary(ln, target_year=ln.min_year, output_dir="xx_out/")
sb.build_boundaries()
sb.aggregate_stats(stats=stats_with_ids)
mb = ModernBoundary(ln, target_year=ln.min_year, output_dir="xx_out/")
mb.aggregate_stats(stats=stats_with_ids)
```

---

## 17. `breakpoint.py` (BEAST changepoint detection)

`breakpoint.py`. Optional dependency: `pip install stablebound[analysis]` (Rbeast).

```python
from stablebound import analyze_breakpoints

result = analyze_breakpoints(
    stats,                                  # long-form DataFrame
    value_col="value",
    group_cols=("unit_id", "crop"),
    year_col="year",
    season_col="season",
    season_filter="Total Year",             # None to skip filtering
    min_years=8,                            # skip shorter series
)
```

Returns `BreakpointResult` with:
```python
result.per_series                            # (unit_id, crop, year, posterior)
result.multicrop(reduce_cols=("crop",))      # collapse to (unit_id, year, multicrop_posterior, n_series)
result.flagged(threshold=0.5)                # filter to high-posterior rows
```

### Under the hood

For each per-`(group_cols)` series:
1. Reindex to a regular annual grid (BEAST requires no NaNs)
2. Linear-interpolate gaps
3. Run BEAST with `season="none"` (no seasonality decomposition)
4. Return per-year changepoint-occurrence posterior

Per-series failures are silently dropped (one bad series shouldn't abort a sweep).

### Multicrop reducer

Only `method="product"` implemented:
```
P_multi(year) = ∏ over series-present P_i(year)
```

**Empirically too harsh** — median 18 crops/district means even one low-posterior crop collapses the product. **Use per-crop posteriors as the real signal.** A softer reducer (geomean / any-of) would be a small extension if needed.

### Typical use patterns

- **Run on pre-stable raw stats** — find candidate boundary artifacts in upstream data
- **Run on post-stable aggregated stats** — see which artifacts the stable product fixed
- **Diff the two** — on India, the StableBound paper reports the share of name-matching breakpoints that disappear under stable aggregation, with the "fixed" years aligning with real reorganization events

---

## 18. Performance & progress feedback

| Operation | India scale | Notes |
|---|---|---|
| `Lineage("IN")` construction | < 10ms | No disk reads |
| `.lineage` first access | ~50ms | Parses + validates the RT |
| `ln.snapshot(year=Y)` | ~5ms | Re-walks every call (no caching) |
| `sb.build_boundaries()` | ~15s | 26 years (1997–2022) × dissolve per year |
| `sb.aggregate_stats(...)` | ~10s | ~750,000 input rows |
| `mb.aggregate_stats(...)` | seconds | Modern algorithm (vectorized pre-pivot) |

Stderr progress lines (no stdout pollution):
```
[stablebound] building stable boundaries for 29 years (1997-2025)...    # >= 15 years
[stablebound] boundaries built in 14.5s (29 years).
[stablebound] aggregating 748,426 stats rows (27 years × 106 variables)... # >= 10K rows
[stablebound] aggregation done in 10.9s (572,643 output rows).
```

Exampleland and synthetic test data stay quiet.

---

## 19. Things to know for follow-up work

### Diagnostic systems that are deliberately outside the correctness path

Both **reconciliation** and **breakpoint detection** are diagnostics that won't ever be perfectly solved. The package's choice:
- Reconciliation is implemented but not wired in (`reconcile_mode` accepts only `"off"`); call `stablebound.reconcile.reconcile` directly to experiment.
- Breakpoint detection is opt-in entirely (separate module, not auto-run).
- Neither is in the critical correctness path.

### Late-reporting taxonomy (3 subtypes from India analysis)

When `late_reporting=True` rows appear in stable, or `late_reporting.csv` is non-empty in modern, three distinct causes:
1. **Post-dissolution late reports** — unit dissolved, source kept filing under old code
2. **Missing-from-modern-shapefile** — unit exists in baseline/lineage but not in attached shapefile (shapefile gap)
3. **Pre-split parent with persistent stale code** — pre-split parent ID kept being assigned by upstream preprocessing instead of the post-split same-name successor

On India most late reports were of subtypes 2 and 3, i.e. fixable upstream, rather than genuine post-dissolution filings.

### Deferred features worth knowing

- **Aggregate-to-parent rerouting mode** for late_reporting — pending real-data evidence
- **Configurable preference for absorber-over-area** in modern fractions — opinionated; current default is data-trumps-geometry
- **Warning aggregation** for chatty partial-window logs on India
- **Soft multicrop reducer** for breakpoints (geomean / any-of)
- **`cause` column** in `late_reporting.csv` distinguishing the 3 subtypes
- **`Lineage(..., normalizer=)` slot** for custom (non-bundled) countries

### Where the canonical schemas live

- `STATS_REQUIRED_COLUMNS` = `(unit_id, year, season, variable, value)` — `season` is **required** (no nulls — raises SchemaError)
- `RT_REQUIRED_COLUMNS` = `(event_year, event_type, parent_id, parent_name, child_id, child_name)`
- `BASELINE_REQUIRED_COLUMNS` = `(unit_id, name)`; `year`, `coarse_id`, `coarse_name` optional
- `VALID_EVENT_TYPES` = `{Split, Merge, Redistribute, NameChange, Coarse}`

### Where the canonical conventions live

- **Event-year convention**: event at year T happens DURING T. Snapshot at start-of-T is pre-event; effect visible from start-of-(T+1).
- **NameChange invariant**: `parent_id == child_id`. The lineage loader enforces this; the validator (`namechange_distinct_ids`) catches violations.
- **stable_id = min(group_members)**: lex-smallest member, deterministic across runs.
- **Singleton fallback**: every `additional_units` unit appears in the remap at minimum as its own group.
