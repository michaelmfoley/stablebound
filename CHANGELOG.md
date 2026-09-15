# Changelog

The public repository begins at the 0.1.5 root commit; entries below it record
the development history and refer to tags that exist only in the authors'
private archive.

## Unreleased

### Added

- `analyze_breakpoints(..., missing="native")`: hand BEAST the regular annual
  grid with its gaps as NaN instead of interpolating them (the default,
  `"interpolate"`, is unchanged and is what every published number used).
  Rbeast accepts missing values; the grid is a choice, not a constraint, and
  the option exists to measure what the choice costs. Also passes Rbeast's
  `print_param` instead of a `print_options` keyword it never had.
- `assign_unit_ids` (module `stablebound.assign_ids`): mint canonical unit
  ids for a relationship table authored in names, given a baseline whose
  ids are already right. The id format is inferred from the baseline; events
  are replayed in order (territorial children get fresh ids, one per child
  name per year; `NameChange` / `Coarse` keep the parent's id); the
  name-change log takes part in the replay (an optional `coarse_name`
  column in the log tells apart two living units with one old name); coarse
  ids are resolved from the baseline and, optionally, an already id'd
  coarse lineage. `mode="fill"`
  (default) keeps existing ids and mints only blanks; `mode="rebuild"`
  re-mints every non-baseline id chronologically. Pinned by the synthetic
  registry (blank, re-mint, same snapshots and groups) and by the bundled
  India files (blank every id, recover the same history).

## 0.1.5

### Removed — the repository is now India-only, not just the wheel

0.1.4 narrowed the *wheel* to India; 0.1.5 narrows the *repository*. The draft
relationship tables and registry entries for Korea, Vietnam, Thailand and
Bangladesh, the example directories for those countries and the Philippines,
the Korea test fixtures, the per-year FEWS unit-definition example, the
archived developer regression harnesses and the pre-July India input archive
are all removed from the public tree. They are preserved in the authors'
private archive. `stablebound.data.EXPERIMENTAL_COUNTRIES_DIR` and
`BundledCountry.validated` are gone with them; `Lineage("KR")` now raises the
same `ValueError` as any other unbundled code, naming the bundled countries.

### Changed — tests

The legacy-dialect importer and the FEWS exporter, whose only real-data
coverage was the removed countries, are now pinned by a hand-authored fixture
pair: `tests/fixtures/synthetic/legacy_admin1/` (canonical) and
`tests/fixtures/rt_convert/relationshiptable_XX.csv` (the FEWS dialect). The
two are each other's oracle and neither is generated from the other; the README
beside the CSV explains the row blocks. The coverage-window tests register a
synthetic country through a new `bundle_synthetic` conftest fixture, since
India cannot express a coverage end later than its last event. The sdist now
ships `tests/fixtures/` and `tests/conftest.py` (`MANIFEST.in`), so its test
suite can actually run.

### Documentation

- Reconciliation is described as implemented but not wired in everywhere it
  was previously described as defaulting to `flag`.
- Late-report redistribution in the modern product is described as disabled
  everywhere it was previously described as running.
- The lineage validator runs eight checks, not six.
- Internal working documents (plans, review logs, reading lists, a
  reproducibility assessment) are no longer part of the repository.
- Author and licence metadata name the author rather than "StableBound
  contributors"; `CITATION.cff` carries the paper as the preferred citation.

## 0.1.4

### Changed — yield and other intensives are computed from matched constituents

`StableBoundary.aggregate_stats(intensive=...)` now derives each ratio from a
separate aggregation of only the units that reported both the numerator and
the denominator in a given (unit, year, season). Previously the ratio divided
the full sums, so a district reporting production without area pushed the
group's yield up, and one reporting area without production pushed it down;
the completeness columns recorded the mismatch but the value did not. Now
`constituent_ids` on a yield row names exactly the units in the ratio,
`missing_unit_ids` the alive members not in it, and the value is a ratio of
matched sums. `stats.aggregate` gained an `intensive` keyword and a shared
routing helper so the extensive and intensive passes cannot drift;
`derive_intensive` is unchanged for callers holding an already-aggregated
frame (the modern product). Input rows whose `variable` is an intensive name
are dropped before summing. On Exampleland every unit reports both inputs, so
nothing moves there; on India the yield now differs from production/area
wherever a district reported one without the other.

### Changed — release scope is India only

The StableBound paper describes India. The draft relationship tables for
Korea, Vietnam, Thailand and Bangladesh are no longer packaged: their files
moved from `src/stablebound/data/` to `experimental/countries/`, the wheel's
package data is narrowed to `data/IN/*`, and their registry entries carry
`validated=False`. From a source checkout `Lineage("KR")` still loads, with a
`UserWarning`; from a wheel it raises `FileNotFoundError` naming the
experimental directory. The developer regression harnesses moved to
`experimental/regression/` and the India data generators to `tools/india/`.
The pre-move tree is tagged `archive/multi-country-2026-09-04`.

### Known limitations

- Validating and re-shipping the non-India tables is deferred to v0.2.

## 0.1.3

### Corrections — wrong ids on the India map

The previous release shipped two polygons carrying the wrong district's
identifier. That is not a labelling slip: a polygon is dissolved into the
stable group its id names, so the geometry itself was wrong.

- Chhattisgarh's Bilaspur carried Himachal Pradesh's id and dissolved into a
  stable group 1,100 km away. It was the whole of the remaining 0.111% of
  misplaced area.
- A Raigarh polygon carried an id the 2022 Sarangarh-Bilaigarh split had
  already retired, so a live district held a dead identifier.

Both are fixed by matching on (name, state) rather than name alone. The
published India shapefile now carries a `state` column, built by spatial join
against the official 2021 state boundaries, and `attach_shapefile` /
`propose_shapefile_mapping` / `attach_shapefile_ids` accept `coarse_column`
to use it. The India pipeline's twelve-entry `HOMONYM_OVERRIDES`, keyed by
shapefile *feature index*, is deleted rather than corrected: index keys
silently repoint after any shapefile rewrite, and the state column derives
what they asserted. Every geometry check now reports zero — 0 violating ids,
0 km2 in the wrong stable group, 0 lineage units without a polygon.

### Corrections — state names as of the wrong year

- A snapshot built for year Y reported its districts' states using whatever
  name was current when each district last changed, or the baseline's name
  where it never changed. India's Odisha districts read "Orissa" in 2024,
  thirteen years after the rename, and Uttarakhand's read "Uttaranchal"
  seventeen years on. Upper-admin renames are now applied on all three paths
  that produce a state name — a district's own past events, the forward
  fallback for a district that never changed, and the matcher's baseline
  fallback. Fixing one of the three, as an earlier attempt did, leaves the
  other two frozen.
- `_build_snapshot_lookup` preferred the baseline's state name over the
  year-aware one, so a lookup built for 2024 answered in 1991 vocabulary.
  Reversed.
- `infer_year` was called without seeding the baseline, so every baseline unit
  the relationship table never mentions counted as a mismatch in every
  candidate year: 230 reported discrepancies against 3 real ones on India.
- `india.matcher._build_parents_map` raised nothing when the DESAGRI frame
  arrived with lowercase headers — `Series.get` returned an empty mapping and
  every district silently failed to match. It now raises and names the rename
  to apply.

### Added

- `validate_shapefile_lineage_consistency`, run automatically by
  `attach_shapefile` (opt out with `validate=False`). Attaching ids checked
  only that every feature *got* one; these four checks ask whether the id is
  real, unique, and in force at the map's vintage, which is what a wrong id
  passes. Findings arrive as one `UserWarning` and on
  `Lineage.shapefile_issues` / `shapefile_report()`; nothing is raised,
  because a deliberately historical map fails them for a good reason.

### Documentation

- `fnid.py`'s "a district's short code never moves" contract is scoped to the
  codes it actually covers. It holds for admin1 SS and for the origin SS
  inside an admin2 code; it does **not** hold for an admin2's DD, which is
  assigned by sort position and re-packed on every build. Correcting one
  spurious Rajasthan district re-pointed 41 district codes across India's 2024
  and 2025 vintages. This is forced by the code space — 359 codes per state
  against globally sequential ADM2 ids reaching 01127 — so downstream users
  should join on `unit_id`, not FNID, across vintages.

## 0.1.2

### Reproducibility

- `analyze_breakpoints` now seeds BEAST's sampler (`DEFAULT_MCMC_SEED = 1`,
  overridable per call). BEAST estimates changepoint posteriors by MCMC and
  previously seeded from the clock, so two runs over the same series disagreed
  by up to 0.16 against a 0.9 threshold and any thresholded count drifted
  between runs. Pass `mcmc_seed=0` for BEAST's own random seeding, or vary it
  deliberately to measure how much of a reported count is sampling noise.

### Corrections

- `validate_lineage`'s multi-parent-split finding no longer tells users that
  "the redistribute-safe rule already handles it correctly". That rule was
  removed on 2026-04-30. The message now says what actually happens: every
  parent of a multi-parent child lands in the same stable group, which costs
  geographic resolution. Still info severity — these are real administrative
  reorganizations, not data defects.
- Bundled India ADM2 lineage: `Giridh` corrected to `Giridih` (five rows).
  `IN.ADM2.00475` had carried the misspelling from 1992 to 2025.

### Regression

- Three new India audit phases (perturbation, ablation, FEWS provenance)
  generate the relationship-table validation numbers the paper reports.

## 0.1.1

Version bump only, recorded retroactively — the release commit did not update
this file. An earlier 0.1.0 wheel had been distributed 29 commits behind the
tree, including a correctness fix in the stats matcher; two different 0.1.0
wheels in circulation would have been a trap, so the corrected build shipped
as 0.1.1.

## 0.1.0

First test release, installable from GitHub.

### Public API

- `Lineage` / `StableBoundary` / `ModernBoundary` triad, with
  human-in-the-loop name matching (`propose_shapefile_mapping`,
  `propose_stats_mapping`, `MatchProposal` review artifact).
- `Lineage.from_legacy_rt(...)` builds a lineage directly from a legacy
  FEWS NET relationship table, deriving the baseline from the earliest
  hierarchical snapshot.
- Year-aware stats matching: `propose_stats_mapping(year_aware=True)`
  matches each name against its own year's snapshot and walks the lineage
  to a year-compatible ancestor when the match was not yet alive. Every
  such remap is flagged in `method`, in `notes`, and in
  `MatchProposal.remapped`.

### FEWS deliverables

- New `stablebound.fews_export`: country-agnostic builders for the three
  FEWS upload files — per-year `{ISO}_Admin_Definitions_{year}.xlsx`,
  `{ISO}_GeographicUnitRelationship.csv`, and `{ISO}_AgStats_*.xlsx`.
  Reproduces India's frozen deliverable bundle byte for byte.
- Admin1-only country support for the deliverable path
  (`build_admin1_only_code_map`, `build_admin1_defs_table`), so
  single-level lineages such as Vietnam, Thailand, Bangladesh and Korea
  can export without a coarse layer.
- FNID SS/DD codes are ID-stable: a unit keeps its code even when its
  parent admin1 changes, and a retired unit's slot is never reassigned.

### Bundled data

- Lineage + baseline bundled for India (`"IN"`), Korea (`"KR"`), Vietnam
  (`"VN"`), Thailand (`"TH"`) and Bangladesh (`"BD"`). India additionally
  ships a name-change log and a country-specific name normalizer.
- India source-data fixes applied upstream rather than patched in code:
  Bihar admin1 assignment, a phantom Gujarat unit, the Aurangabad rename,
  and the Assam 2022–2023 transient district duplicates.

### Country submodules

- `stablebound.india` carries the DESAGRI stats matcher, so the India
  pipeline has no required code outside the package. The legacy
  `fnid_assigner` is retired in favour of the package encoder.

### Examples, regression, docs

- Synthetic Exampleland fixture plus a getting-started notebook.
- India and Korea end-to-end regression suites under `regression/`,
  including `run_deliverables.py`, which asserts the FEWS bundle against
  a frozen reference.
- `docs/`: usage guide, code walkthrough, user manual, algorithm
  references for both products, and a running paper-discrepancy log.

### Known limitations

- Reconciliation is implemented but not wired in: every
  `reconcile_mode` other than `"off"` raises. See Known Issues in the
  README.
- Late-report redistribution is disabled in the modern product.
- FEWS export covers admin levels 1 and 2 only.
