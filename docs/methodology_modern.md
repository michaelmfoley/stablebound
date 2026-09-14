# StableBound — Modern Boundary Algorithm [EXPERIMENTAL]

The narrative reference for the modern boundary product. Pairs with
[`methodology.md`](methodology.md) (the stable boundary's algorithm
reference) and [`USAGE.md`](USAGE.md) (user-facing API).

## Conceptual summary

The stable boundary product fixes geometry at the **base year** and
projects history forward — every observation across the analysis window
is aggregated upward onto base-year polygons. Stable boundaries are the
right answer when you want a temporally consistent geography that loses
as little spatial resolution as possible to historical events.

The modern boundary product fixes geometry at the **modern shapefile**
and rescales history backward — every historical observation is
disaggregated downward onto today's polygons via post-event fractions.
Modern boundaries are the right answer when you want to use today's
familiar geography and are willing to trade some attribution
uncertainty for it.

Both products read the same canonical inputs: the relationship table,
modern shapefile, optional baseline, and stats. The two products
produce different (but mutually consistent) outputs from the same data
and lineage.

## Algorithm steps

For each modern unit `M`, we want a single long-form time series of
stats covering `[base_year, max_year]` whose values represent "what we
estimate this stat would have been on `M`'s present-day extent at the
historical year."

The procedure walks the relationship table forward in time and carries
a per-unit ledger. Each ledger cell tracks the running value, the set
of historical unit_ids that have contributed (`sources`), a
`has_nan_fraction` flag, a `lineage_depth` counter, a `fraction_method`
string indicating the worst tier of the cascade ever applied to the
cell, and a `late_report_redistributed` flag.

1. **Initialize.** Every reported (unit_id, year, season, variable,
   value) row in the input stats seeds a ledger cell with depth 0 and
   no fraction applied.

2. **Walk territorial events chronologically.** Group rows of the
   relationship table by event year. NameChange and Coarse rows are
   skipped — they don't move territory, so the ledger flows through
   unchanged.

3. **Find connected components.** Within a single event year there can
   be multiple independent events (e.g., 2014 in India had three
   unrelated splits). For each year, decompose the rows into
   independent (parents, children) bipartite-graph components.

4. **Per-event fractions with the three-tier cascade.** For each child
   of a component, pick a fraction by trying:

   1. **Seasonal (common years)** — child's mean reported value for
      `(variable, season)` computed over the INTERSECTION of years
      all children reported in the post-event window. Normalize across
      siblings. Apples-to-apples comparison; not biased by one child's
      extra unique-year reports. If at least one sibling has zero
      reports for this `(var, season)` in window, the intersection
      collapses → tier fails → cascade.

   2. **Total-year fallback (common cells)** — if (1) is NaN. Per
      season, take the intersection of years across siblings; sum
      per-season means over those common years for each child;
      normalize. Captures the case where a child reports Kharif rice
      but not Rabi: their Rabi pre-event data gets attributed by
      their share of the whole-year total over comparable cells.

   3. **Modern-area fallback** — if (2) is NaN. Each child's share of
      the modern shapefile area / sum of siblings' areas. Doesn't
      depend on data; final tier to ensure no parent data is lost.

   If all three tiers fail (no common reports AND no modern shapefile
   areas), the fraction is `undefined` and that cell's data is lost.
   This is intentional: an honest "we can't compute this" beats a
   silent "the only reporting sibling absorbs everything." Direct
   callers of `build_modern_ledger` who don't supply `modern_areas`
   accept this risk; `StableBoundary` / `ModernBoundary` always pass
   areas from the attached shapefile so the area tier catches it.

   The chosen fraction's tier is recorded as `fraction_method`. The
   audit table records the fraction, method, `n_common_observations`
   (common years/cells used — same across siblings within an event),
   and `n_individual_observations` (this child's in-window report
   count — reveals when one child's incomplete reporting reduced the
   common-set). Inspecting where `n_individual > n_common` shows
   exactly which children would benefit from gap-filling.

   The window length is configurable per variable via the
   `modern_window` kwarg on `ModernBoundary.aggregate_stats`, falling
   back to `modern_window_default`. Default is 5 years.

5. **Pool parents.** For each parent, sum its pre-event ledger cells
   into a single pool keyed by (year, season, variable). Pool inclusion
   is `cell_year <= event_year` (the parent is alive throughout the
   event year under the canonical convention). Late-reporting cells
   (cell_year > event_year) stay on the parent for Step D below.

   Multi-parent events (Redistribute, multi-parent Splits) pool all
   parents into one. The per-parent share of each cell is intentionally
   lost; the package does not attempt spatial-connectedness reasoning.

6. **Distribute pool to children.** For each child, multiply each pool
   cell by the child's chosen fraction (from Step 4) and add to the
   child's ledger. The `lineage_depth` increments by 1; the worst-tier
   `fraction_method` so far is recorded.

7. **Drop pre-event cells from parents.** Once the pool is distributed,
   the parent's pre-event ledger is redundant. We delete it so the
   parent's only remaining cells are post-event reports.

8. **Step D — Late reports (redistribution disabled).** After the
   chronological event walk, a parent unit may still hold ledger cells
   with `year > terminal_event_year` — reports filed under its old code
   after the unit was reorganized. These rows are emitted to
   `late_reporting.csv` and are **not** redistributed onto the modern
   children, so they do not contribute to `stats_modern.csv`. A
   redistribution pass exists (`_redistribute_late_reports`, replaying
   the terminal event's fractions) but is not called: on India it
   double-counted real observations because the upstream id mapping files
   one report under several canonical ids. `late_report_redistributed`
   is therefore always False in the current release.

9. **Sequential composition emerges.** When a modern unit `M` descends
   through multiple events, each event multiplies the inherited ledger
   by that event's child-fraction. The `fraction_method` records the
   worst tier across all composed events.

10. **Step E — Total Year derived rows.** For every
    `(modern_id, year, variable)`, append a row with
    `season='Total Year'` whose value is the NaN-aware sum across
    explicit seasons. Yields are recomputed via `derive_intensive` on
    the Total Year subframe — `yield_mt_ha` for `season='Total Year'`
    is `production_mt_TY / area_ha_TY`, never an average of seasonal
    yields.

    **This is the season where conservation holds strictly**: for any
    `(variable, year)`, sum across all modern units for
    `season='Total Year'` equals sum across all stable units
    (excluding rows with no modern destination). Per-season
    conservation may NOT hold — the cascade reallocates parent data
    across seasons when the seasonal tier fails.

11. **Output.** Every modern unit's ledger is pivoted to long-form
    `stats_modern.csv`. Audit fractions go to `event_fractions.csv`.
    Late-reporting originals go to `late_reporting.csv`. Run metadata
    + counts go to `summary.json`.

## Season-year convention

The package stores values under their **sowing year**, matching
India's official agricultural-ministry standard:

- Kharif T → sown June–Sep T, harvested Oct–Dec T → year = T
- Rabi T → sown Oct–Dec T, harvested Mar–May T+1 → year = T (Rabi
  2010 spans Oct 2010 – Jan 2011, stored with `year=2010`)
- Summer T → sown Mar–May T, harvested Jun–Aug T → year = T
- Whole Year T → annual aggregate → year = T

The package does no year shifting. If your raw data uses harvest-year
attribution, shift it during preparation before passing to
`read_stats`.

## Outputs

| File | Description |
|---|---|
| `stats_modern.csv` | One row per (modern_id, year, season, variable). Columns: `value`, `sources` (comma-joined contributing unit_ids), `lineage_depth` (events composed), `has_nan_fraction` (NaN-attribution flag), `fraction_method` (worst cascade tier applied), `late_report_redistributed` (always False; see Step D). |
| `event_fractions.csv` | Audit table: per (event, child, variable, season) row recording the fraction applied, the window length used, `n_common_observations` and `n_individual_observations`, and a computed_at timestamp. |
| `late_reporting.csv` | Reports filed under a unit's id after its terminal event. Not redistributed; absent from `stats_modern.csv`. |
| `summary.json` | Run metadata: country, year range, event count, modern unit count, NaN-cell count, late-reporting count, default window. |

## Edge cases

- **Cascading events.** Sequential composition handles arbitrary depth.
  India's deepest chains (Telangana, Chhattisgarh, parts of the
  Northeast) reach depth 3.

- **Insufficient post-event data.** A child with only N < window years
  of data uses what's available. Researchers should check
  `n_common_observations` and `n_individual_observations` in
  `event_fractions.csv` to identify high-uncertainty fractions.

- **One sibling has no coverage.** Common in real data (a newly promoted
  city that does not report agricultural statistics, for instance). The
  non-reporting child's fraction is NaN; the reporting child's fraction
  normalizes to 1.0. All pre-event data lands on the reporting child.

- **Late-reporting on the parent's old code.** Common during
  reorganization transitions (India's Andhra Pradesh/Telangana split).
  Surfaced in `late_reporting.csv` rather than silently re-routed.

- **Units that never change.** Their ledger is just their own reports;
  no events ever touch them. They flow straight through with
  `lineage_depth = 0`.

- **Modern units not in the lineage.** Singletons that exist in the
  modern shapefile but have no relationship-table ancestors. They
  appear in `stats_modern.csv` with only their own reports — no
  inherited data.

## Comparison to the stable product

| Property | Stable | Modern |
|---|---|---|
| Geometry | Base-year polygons (one per stable group) | Today's polygons |
| Aggregation direction | Many → few (modern → stable) | Few → many (historical → modern) |
| Attribution accuracy | Exact for additive variables | Estimated via post-event fractions |
| Geographic resolution | Coarsest at base year | Finest (modern shapefile) |
| Loss surface | Aggregation: deliberate spatial smoothing | Disaggregation: estimated fractions |
| Reconciliation pass | Implemented but not wired in (paper Algorithm 4; see `methodology.md`) | No — different statistical operation |
| Output schema | Long-form on `stable_id` | Long-form on `modern_id` |

Both can be run from the same `Lineage`. Researchers often run both
products on the same dataset and pick the one that matches their
research question.

## Implementation pointers

- Pure-function core: `src/stablebound/modern_algorithm.py`. Three
  helpers (`_connected_components`, `_compute_fractions`,
  `_pool_parents` + `_distribute_pool_to_children`) plus the
  driver `build_modern_ledger`.
- Orchestrator: `src/stablebound/modern.py`. `ModernBoundary` class
  wraps the algorithm with config loading, caching, and CSV writing.
  Mirrors `StableBoundary` rather than extending it (the two products
  have different schemas and different caching semantics).
- The same `derive_intensive` helper from the stable product
  (`src/stablebound/stats.py`) is reused for yield recomputation —
  intensives are computed identically across both products.
