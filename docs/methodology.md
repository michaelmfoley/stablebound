# StableBound — Algorithm Reference

A plain-English walkthrough of every algorithm in the package, paired to
the section and algorithm numbers of the StableBound paper (Foley et al.,
*Earth System Science Data*, in review) and to the modules that implement
each piece. For the user-facing API, see [`USAGE.md`](USAGE.md).

## The two-product framework

StableBound takes a canonical relationship table, a modern shapefile, an
optional baseline snapshot, and an optional long-form statistics table,
and produces a **stable boundary product**: a set of dissolved shapefiles
(one per snapshot year) on a geometry that is consistent across the
analysis window, plus an aggregated long-form statistics table on the
same geometry.

The *modern boundary product* (disaggregation onto today's geometry) is
[`ModernBoundary`](methodology_modern.md). This document covers the stable
product only; see [`methodology_modern.md`](methodology_modern.md) for the
modern one.

## Inputs (canonical schemas)

The package commits to a single canonical schema for each input.
Researchers reshape their data once into these schemas — typical FEWS NET
relationship tables, GADM/Geolocet shapefiles, and country statistics
files all need a one-time conversion. Once converted, the algorithm is
country-agnostic.

The five inputs and their schemas are documented in [`USAGE.md`](USAGE.md).

## Lineage validation (`stablebound.validate`)

`LineageGraph.from_dataframe` runs `validate_lineage` automatically at
load time. **Errors block** (raise `LineageDataError`); warnings and infos
are logged. The check categories:

Eight checks run, in this order:

| Severity | Category | Detects |
|---|---|---|
| error | `duplicate_row` | Same `(event_year, event_type, parent_id, child_id)` row appearing twice. |
| error | `self_referential_territorial` | `Split`/`Merge`/`Redistribute` with `parent_id == child_id` (probably a rename mistakenly tagged territorial). |
| error | `transient_unit` / `resurrected_unit` | A unit created and consumed in the same event year (it would never appear in any snapshot yet stay alive beside its successors), or a unit re-created after it ceased. |
| warning | `namechange_distinct_ids` | `NameChange` row with `parent_id != child_id`. Algorithmically harmless but semantically suspect. |
| info | `multi_parent_split` | A `Split` child has multiple distinct parents — structurally a Redistribute. Every parent of such a child lands in the same stable group, which costs geographic resolution; retagging as `Redistribute` makes that intent explicit. |
| warning | `split_looks_like_rename` | A 1-to-1 Split where parent_name == child_name. Either a sibling row is missing, or the row should be a NameChange. |
| info | `leaf_count` | Count of units that only appear as territorial children, never parents (modern leaves). |
| warning | `namechange_unit_name_mismatch` | A `NameChange` whose `parent_name` is not the name the unit carried at that point in the lineage. |

The reports are formatted with row numbers and explicit "Fix:" prose so
researchers can locate and correct issues directly.

## Algorithm 1 — `BuildSnapshot` (`stablebound.snapshot.build_snapshot`)

Returns the set of unit IDs active in a given year. Walks territorial
events in chronological order: each Split / Merge / Redistribute removes
its parents and adds its children to the active set; NameChange and
Coarse leave the set unchanged.

The `additional_units` parameter extends the **initial** active set with
units that don't appear anywhere in the relationship table — typical of
FEWS NET / HarvestStat data, where many districts have existed unchanged
since the start of the timeline. Such "always-alive" units are treated as
present from the dawn of time and never removed by any event.

**Filter.** Units passed in `additional_units` that DO appear in the RT
are excluded from the always-alive treatment — their existence is
governed by the event timeline (so a unit created mid-window doesn't
appear before its creation year).

## Algorithm 2 — `InferYear` (`stablebound.snapshot.infer_year`)

For each candidate year, computes the symmetric difference between the
shapefile's units and the year's snapshot, returns the year with smallest
mismatch. Useful when a shapefile's effective vintage is uncertain (GADM
and GAUL releases often lag official boundary changes by years).

Tie-break: earliest candidate year wins.

## Algorithm 3 — `BuildStableGroups` (`stablebound.groups.build_stable_groups`)

The core construction. For each base-year unit, traces forward through
post-base-year territorial events to find every reachable descendant,
then unions descendants into the base unit's group via Union-Find.

The stable_id of a group is the lexicographically smallest unit ID in it
— deterministic across runs.

### Multi-parent children force ancestor groups to merge

When a child unit has more than one parent in the post-base territorial
graph, Union-Find unions the child with each of its parents. Because
unions are associative, those parents — and everything in their
respective ancestor groups — collapse into a single stable group.

This is a deliberate choice. A multi-parent child means territory was
exchanged across what would otherwise be separate stable groups, and we
have no way to recover what fraction of the territory came from where
without a map. Keeping the parents as separate stable groups would
force a downstream allocation assumption to attribute the multi-parent
child's reported data to one parent vs. another. That defeats the
stable boundary's "no allocation assumptions, only sums of reported
values" contract.

Cost: lost geographic resolution. India's Assam region (BTAD
reorganizations) collapses ~25 modern units into one stable group,
because the territorial-flow DAG of those events can't be cleanly
decomposed.

Benefit: data sums cleanly. At every year of the analysis window, the
stable group's value for any extensive variable is just the sum of its
constituent units' reported values — no fractions, no estimation.

Worked example. Suppose A, B, D each contribute territory to a new unit
C in the same year; D also contributes to E; A and B also contribute to
F. The bipartite event graph connects {A, B, D} through C, and the
2013 stable polygon for that region must be the union of A, B, and D
rather than three independent polygons.

The same logic covers clean Merge events: `A + B → C` produces one
stable group `{A, B, C}` rather than three separate ones. India's 2022
Andhra Pradesh reorganization (Tirupati has parents Chittoor and SPS
Nellore) likewise consolidates Chittoor + SPS Nellore + Tirupati into
one group.

### The singleton fallback

After tracing, the Union-Find universe is augmented with all
`additional_units` (typically modern shapefile + baseline IDs). A unit
that doesn't trace to any base via the safe rule lands in its own
**singleton** group rather than disappearing from the remap entirely.

This ensures geographic completeness in the dissolved output: every
modern shapefile feature contributes a polygon at every year. The
tradeoff: a post-base-year modern unit (e.g., Korea Sejong, created 2012)
appears as its own polygon at pre-creation years. This is a known
limitation — the alternative (dropping it) leaves geographic gaps in the
year-y dissolved shapefile, which is worse for most downstream uses.
Researchers needing strict per-year geographic accuracy should slice the
year-y dissolved shapefile against the year-y snapshot manually.

## Geometry dissolution (`stablebound.dissolve.dissolve`)

For each snapshot year, the per-year remap is applied to the modern
shapefile. Each stable group's polygon is the `unary_union` of its modern
constituent polygons; invalid geometries are repaired with `make_valid`
before unioning. Output: a GeoDataFrame with one row per stable group,
columns `stable_id`, `n_modern`, `source_ids`, `geometry`.

## Algorithm 4 — `ReconcileConflicts` (`stablebound.reconcile.reconcile`)

After geometric grouping, the package optionally inspects the stats table
for symptoms of stale reporting around split and merge events. Two
diagnostics are computed per (event, variable) over a rolling pre/post
window of length `window` (default 3 years).

### Drop test (paper eq. 3)

For a split event affecting parent *i* in year *s* with successors *j*:

```
r_t  = Σ_j Y_{j, post-mean}  /  Y_{i, pre-mean}        expected fractional drop
d_t  = 1 − Y_{i, post-mean}  /  Y_{i, pre-mean}        observed fractional drop
flag if  d_t < r_t − tau_drop
```

The parent's observed drop falls short of the expected territorial loss by
more than the tolerance.

### Sum-jump test (paper eq. 4)

```
flag if  (Y_{i, post-mean} + Σ_j Y_{j, post-mean}) / Y_{i, pre-mean}  >  1 + tau_sum
```

Post-event sum of parent + successors exceeds the pre-event baseline by
more than the tolerance.

For merge events, the diagnostics apply with parent/successor roles
reversed: persisting-parent rows play the parent role; the new merged-unit
row plays the successor role.

### Status: implemented, not wired in

`stablebound.reconcile` implements the two tests and three repair modes
(`flag`, `merge`, `subtract`), but `aggregate_stats(reconcile_mode=...)`
accepts only `"off"`, which is the default; every other value raises
`NotImplementedError`, and no `reconciliation_flags.csv` is written. On
India, `merge` mode deleted legitimate post-event child rows in the large
majority of flagged cases — series looked smoother because roughly half
their post-event data was gone — so the diagnostic was disabled pending a
rework rather than shipped with a known false-positive rate.

The underlying difficulty is that a stale-reporting artifact and a real
change look alike over a short window: drought years, bumper years, late
reporting and source-data quirks all trip the tests. The
breakpoint-detection workflow (`stablebound.analyze_breakpoints`) is a
complementary and equally imperfect signal, and is opt-in for the same
reason. Redistribute events are in any case unreconcilable from totals
alone — the territory transferred depends on land use that reported
totals do not reveal (paper §3.3, last paragraph).

## Algorithm 5 — `StableStats` (`stablebound.stats.aggregate`)

Each (unit_id, year, season, variable) row in the stats table is routed
to its stable polygon via the remap and summed. The package's late-/early-
reporting detection is **snapshot-aware**: it checks whether each
reporting unit is alive in the snapshot at the row's year (with a one-
event-year grace on each side so a child reporting in its creation year
or a parent reporting in its dissolution year isn't flagged as a false
positive).

Three routing cases:

1. **Standard.** `unit_id` is alive at the row's year (per
   `snapshot(year)` or `snapshot(year+1)`). Routes to `remap[unit_id]`.
2. **Early reporting.** A unit reports before its official creation year
   and isn't covered by the one-year grace. The row is flagged
   `late_reporting=True` and surfaced under `stable_id=unit_id` rather
   than rerouted.
3. **Late reporting.** A unit continues to report after its official
   dissolution year (paper §3.4). Same handling as early reporting:
   flagged + surfaced.

A `UserWarning` is emitted at the end of aggregation if any flagged rows
are present, with a count and a sample. The diagnostic is visible in
notebooks even when the user doesn't read the output CSV.

Output columns:
- `year, season, variable, stable_id, value`
- `n_constituents`: distinct unit_ids that contributed
- `n_in_group`: count of remap members alive at the row's year (NA for
  late-reporting rows — the concept doesn't apply)
- `complete`: True iff every alive-at-year member reported
- `missing_unit_ids`: comma-joined list of alive-at-year members who
  didn't report
- `constituent_ids`: audit trail of the reporters
- `late_reporting`

**Strict NaN-season validation.** `validate_stats` rejects rows with
missing `season` values — use an explicit sentinel like `"Annual"` for
non-seasonal data. The previous lenient behavior grouped NaN-season
rows into their own bucket, which left consumers unsure whether the
rows were annual data, missing-tag rows, or a schema bug.

**Future improvement** (not yet implemented): an opt-in rerouting mode
that maps late-reporting rows back to a parent's stable group when the
relationship is unambiguous. The current "flag and surface" behavior is
the honest default; rerouting is opinionated and may not match the
user's data convention.

## Intensive variables (`stablebound.stats.aggregate(intensive=...)`)

Intensive quantities (yield, density, etc.) cannot be summed or averaged
across constituent units — they must be recomputed from aggregated
extensives. Researchers declare pairs in `aggregate_stats(intensive=...)`:

```python
intensive={"yield_mt_ha": ("production_mt", "area_ha")}
```

Since v0.1.4 the ratio is built from **matched constituents**. For each
declared pair and each cell (year, season, stable polygon), let *I* be the
set of constituent units that reported both the numerator and the
denominator in that (unit, year, season). Then

    value = Σ_{u∈I} numerator_u / Σ_{u∈I} denominator_u

The extensive rows still sum every reporter, so the area implied by a yield
row can be smaller than the cell's area row when some unit reported area
without production. The yield row's completeness columns say exactly which
units were used: `constituent_ids` = *I*, `n_constituents` = |*I*|,
`missing_unit_ids` = alive members not in *I*, `complete` = no alive member
missing, `completeness` = (n_in_group − |missing|) / n_in_group. If *I* is
empty no row is emitted for that cell; a zero denominator gives NaN, never
infinity. Input rows whose `variable` already carries an intensive name are
dropped before anything is summed.

Implementation: `aggregate` runs its routing-and-grouping helper once on all
rows (the extensive pass) and once per pair on the paired subset (the
intensive pass), then calls `derive_intensive` on the paired aggregate, so
the two passes share one routing, one snapshot and one completeness rule.
`derive_intensive` alone divides whatever sums it is handed; the modern
product still uses it that way because its frame has no per-unit rows.

## Name history (`stablebound.lineage.name_history`)

Audit table for paper §3.2.3. For each year in the analysis window, lists
every (stable_id, year, unit_id, name) triple. Lets researchers check
exactly which historical units (under which historical names) each
stable group represents over the full window.

## Working invariants

A few non-negotiable rules baked into the algorithm:

- **`min_year == base_year`.** Stable boundaries are only valid forward
  from the base year. Stats rows with `year < base_year` are rejected at
  validation.
- **The remap is canonical.** `aggregate_stats` always loads the remap
  produced by `build_boundaries` rather than recomputing it. Union-Find
  root nondeterminism would otherwise produce silently inconsistent
  stable-group IDs across runs.
- **Intensives are never summed.** Ratios come only from
  `aggregate(intensive=...)` (matched constituents) or, for frames with no
  per-unit rows, `derive_intensive`.
- **Multi-parent children consolidate ancestor groups.** Whenever a
  child has more than one parent, all of those parents' base groups
  merge into a single stable group. Geographic resolution is sacrificed
  for assumption-free comparability across years. See Algorithm 3.
- **Singleton fallback.** Every modern shapefile feature lands in the
  remap, possibly as its own singleton stable group. Geographic
  completeness over strict per-year temporal accuracy.
- **Validation is mandatory by default.** `LineageGraph.from_dataframe`
  raises on data errors; pass `validate=False` to opt out.
