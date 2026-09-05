"""Modern boundary product (paper Algorithm 6 — to be added).

Where the stable boundary product fixes geometry at the *base year* and
projects history forward, the **modern boundary product** fixes geometry
at *today's shapefile* and rescales every historical observation backward
onto modern units. For each modern unit ``M`` the result is one long-form
time series covering ``[base_year, max_year]`` whose values represent
"what we estimate this stat would have been on ``M``'s present-day
extent at the historical year."

The algorithm in five steps:

    Step A — Build the lineage chain forward.
        Walk territorial events (Split, Merge, Redistribute) in
        chronological order. NameChange and Coarse rows are skipped
        (they don't move territory). Carry a ``ledger`` per unit: cells
        indexed by ``(year, season, variable)`` accumulate values +
        provenance.

    Step B — At each event, compute fractions with a three-tier cascade.
        For every child of the event, pick a fraction in this order:

          1. **Seasonal** — child's mean reported value for
             (variable, season) in the post-event window divided by the
             sum of all siblings' means for the same (variable, season).
             This is the most accurate when data is available.

          2. **Total-year fallback** — if (1) is NaN (no children
             reported that exact (var, season) combo). Sum each child's
             means across ALL seasons for the variable, divide by the
             sum across siblings. Captures the case where a child
             reports Kharif rice but not Rabi: their Rabi pre-event
             data gets attributed by their share of the variable's
             whole-year total.

          3. **Modern-area fallback** — if (2) is NaN too. Each child's
             share of modern shapefile area / sum of siblings' areas.
             Doesn't depend on data; final fallback to ensure no parent
             data is lost.

        Each fraction records its ``fraction_method``
        ∈ {'seasonal', 'total_year', 'area', 'undefined'}. The fourth
        only happens if a child has no reports anywhere AND no modern
        shapefile entry — extremely rare.

    Step C — Pool parents' ledgers, distribute to children by fraction.
        Sum the parents' pre-event ledgers cell-by-cell (the "pool"),
        then for each child multiply each pool cell by the matching
        fraction and add to the child's ledger. Each cell tracks the
        worst fraction method applied across its lineage.

    Step D — Late reports (post-walk pass; redistribution DISABLED).
        After all events, parent units may still have ledger cells with
        ``year > terminal_event_year`` — reports filed under their old
        code after the unit was dissolved. These rows are written to
        ``late_reporting.csv`` for inspection and are NOT redistributed
        onto modern descendants, so they do not contribute to
        ``stats_modern.csv``; ``late_report_redistributed`` is therefore
        always False in the current release. The redistribution pass
        (``_redistribute_late_reports``) is retained but not called — on
        India it double-counted real observations because the upstream id
        mapping files the same report under several canonical ids.

    Step E — Total Year derived rows.
        After event processing and intensives, append per
        ``(modern_id, year, variable)`` rows with ``season='Total Year'``
        whose value is the sum across explicit seasons. This is the
        season where conservation must hold strictly (sum across modern
        units == sum across stable units, modulo unattributable
        orphans). Yields are recomputed via ``derive_intensive``.

Edge cases handled:

- **Multi-parent children** — pooled in Step C; the per-parent split
  of the pool is intentionally lost (no spatial-connectedness reasoning).
- **Insufficient post-event data** — partial windows still compute
  fractions and warn.
- **All-zero or all-NaN siblings** — kicks the cascade. Total-year then
  area fallback typically saves the day.
- **Late-reporting** — Step D redistributes onto modern children
  retroactively.
- **Units that never change** — flow straight through.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Iterable

import pandas as pd

from .lineage import LineageGraph

_LOGGER = logging.getLogger(__name__)


# Fraction-method ranking for "worst-of" composition through cascading
# events. Higher rank = "worse" (more uncertainty). When a cell's lineage
# touches multiple events, its recorded method is the max-rank one.
_METHOD_RANK = {
    "": 0,            # direct report, no fraction applied
    "seasonal": 1,    # tier 1 — best
    "total_year": 2,  # tier 2 — fallback A
    "area": 3,        # tier 3 — fallback B
    "undefined": 4,   # all tiers failed
}


def _worst_method(*methods: str) -> str:
    """Return the worst (highest-rank) method among the given ones.

    Used when composing fractions across multiple events: a cell that
    inherited via 'seasonal' at one event and 'total_year' at the next
    is reported as 'total_year' overall.
    """
    valid = [m for m in methods if m]  # skip empty strings
    if not valid:
        return ""
    return max(valid, key=lambda m: _METHOD_RANK.get(m, 0))


# --- Cell construction helpers ---------------------------------------------
#
# Cell schema:
#     {
#         "value": float,                   # may be math.nan
#         "sources": set[str],              # contributing unit_ids
#         "has_nan_fraction": bool,         # any composed fraction was NaN
#         "depth": int,                     # max events touched
#         "fraction_method": str,           # worst tier used in lineage
#         "late_report_redistributed": bool, # came via Step D pass
#         "min_n_common": float,            # fewest common observations any
#                                           # fraction on this cell's path
#                                           # rested on; inf = direct report
#     }
#
# ``min_n_common`` is the binding constraint on how well-estimated a cell is.
# A cell composed through three events is only as good as the *weakest* of the
# three fractions, so the minimum is what a user should filter on. The full
# per-event detail stays in ``event_fractions.csv``; this is the one number
# that travels with the value.


def _new_cell(value, source: str) -> dict:
    """Make a fresh ledger cell from a single reported observation."""
    return {
        "value": float(value) if value is not None else math.nan,
        "sources": {source},
        "has_nan_fraction": False,
        "depth": 0,
        "fraction_method": "",  # direct report — no fraction applied
        "late_report_redistributed": False,
        "min_n_common": math.inf,
    }


def _empty_pool_cell() -> dict:
    """Make a fresh pool-aggregation cell, used while summing parents."""
    return {
        "value": 0.0,
        "sources": set(),
        "has_nan_fraction": False,
        "depth": 0,
        "fraction_method": "",
        "late_report_redistributed": False,
        "min_n_common": math.inf,
        "_observed": False,
    }


def _is_nan(x) -> bool:
    """NaN check tolerant to None / pd.NA / float NaN / strings."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


# --- Connected-component decomposition -------------------------------------


def _connected_components(
    year_events: pd.DataFrame,
) -> list[tuple[set[str], set[str]]]:
    """Return components ``[(parents, children), ...]`` for one year's events.

    Within a single year there can be multiple unrelated events; we
    process each as its own (parents, children) pool independently.
    """
    parents_of: dict[str, set[str]] = defaultdict(set)
    children_of: dict[str, set[str]] = defaultdict(set)
    for row in year_events.itertuples(index=False):
        parents_of[row.child_id].add(row.parent_id)
        children_of[row.parent_id].add(row.child_id)

    visited_parents: set[str] = set()
    visited_children: set[str] = set()
    components: list[tuple[set[str], set[str]]] = []

    for seed in sorted(children_of.keys()):
        if seed in visited_parents:
            continue
        comp_parents: set[str] = set()
        comp_children: set[str] = set()
        queue: list[tuple[str, str]] = [("parent", seed)]
        while queue:
            kind, node = queue.pop()
            # Walk back and forth between old districts and new ones until
            # the group closes. Reorganisations are rarely tidy: two
            # districts can both give territory to a third, which makes all
            # three part of one event even though no single record says so.
            # Sharing out their figures only works if they are handled
            # together, so this finds everything that has to move at once.
            if kind == "parent":
                if node in visited_parents:
                    continue
                visited_parents.add(node)
                comp_parents.add(node)
                # From an old district, reach everything it fed into...
                for c in children_of[node]:
                    if c not in visited_children:
                        queue.append(("child", c))
            else:
                # ...and from a new one, back to everything that fed it.
                if node in visited_children:
                    continue
                visited_children.add(node)
                comp_children.add(node)
                for p in parents_of[node]:
                    if p not in visited_parents:
                        queue.append(("parent", p))
        components.append((comp_parents, comp_children))
    return components


# --- Fraction computation (three-tier cascade) -----------------------------


def _window_means(
    children: set[str],
    event_year: int,
    var: str,
    season: object,
    window: int,
    stats_by_unit_var: dict[tuple[str, str], list[tuple[int, object, float]]],
) -> tuple[dict[str, float | None], int, dict[str, int]]:
    """For each child, compute mean reported value of (var, season) over the
    INTERSECTION of years all children reported in the post-event window
    ``(event_year, event_year + window]``.

    Common-years semantics (not per-child available years). Rationale: if
    child B reports 5 years and child C reports 4 years, computing each
    mean over its own available set compares apples to oranges — the
    fraction encodes B's unique-year noise. The intersection produces a
    fraction over identical years for both, eliminating that bias.

    Returns ``(means, n_common, n_individual)`` where:

    - ``means[child]`` is the child's mean over the common years
      (``None`` if no common years exist for this group).
    - ``n_common`` is the number of common years used (same for all
      children by construction).
    - ``n_individual[child]`` is the number of in-window reports the
      child made under (var, season) — exposes when a child's data was
      excluded from the common set.

    If at least one child has zero in-window reports for (var, season),
    no common years exist → all means are None → caller's cascade falls
    through to the total_year tier.

    Iterates only the rows that match ``(child, var)`` thanks to the
    pre-pivoted ``stats_by_unit_var`` index — orders of magnitude faster
    than scanning every cell each child has reported.
    """
    upper = event_year + window
    per_child_year_value: dict[str, dict[int, float]] = {}
    for child in children:
        yr_to_val: dict[int, float] = {}
        for (y, s, v) in stats_by_unit_var.get((child, var), ()):
            if s == season and event_year < y <= upper and not _is_nan(v):
                yr_to_val[y] = float(v)
        per_child_year_value[child] = yr_to_val

    n_individual = {c: len(d) for c, d in per_child_year_value.items()}

    # Intersection across all children. If any child has zero in-window
    # reports, the intersection is empty and the tier fails honestly.
    year_sets = [set(d.keys()) for d in per_child_year_value.values()]
    if year_sets and all(year_sets):
        common_years = set.intersection(*year_sets)
    else:
        common_years = set()

    means: dict[str, float | None] = {}
    for child in children:
        if common_years:
            vals = [per_child_year_value[child][y] for y in common_years]
            means[child] = sum(vals) / len(vals)
        else:
            means[child] = None

    return means, len(common_years), n_individual


def _total_year_means(
    children: set[str],
    event_year: int,
    var: str,
    window: int,
    stats_by_unit_var: dict[tuple[str, str], list[tuple[int, object, float]]],
) -> tuple[dict[str, float | None], int, dict[str, int]]:
    """For each child, compute the SUM ACROSS SEASONS of per-season means,
    where each per-season mean uses ONLY the years all children reported
    in that season.

    Common-cells semantics (same rationale as ``_window_means``): a
    fraction comparing children with different reporting completeness
    encodes the difference rather than the underlying territorial share.
    Per season we take the intersection of years across all children;
    each child's total-year value sums per-season means over that
    common-years set.

    Returns ``(means, n_common, n_individual)`` where:

    - ``means[child]`` is the child's summed per-season means (``None``
      if no season had any common years across children).
    - ``n_common`` is the total number of common (year, season) cells
      used across all seasons (same for all children).
    - ``n_individual[child]`` is the child's count of in-window
      (year, season) cells under ``var`` — surfaces excluded data.
    """
    upper = event_year + window
    # Per child: dict[(year, season) -> value], for in-window non-NaN
    per_child_cells: dict[str, dict[tuple[int, object], float]] = {}
    for child in children:
        cells: dict[tuple[int, object], float] = {}
        for (y, s, v) in stats_by_unit_var.get((child, var), ()):
            if event_year < y <= upper and not _is_nan(v):
                cells[(y, s)] = float(v)
        per_child_cells[child] = cells

    n_individual = {c: len(d) for c, d in per_child_cells.items()}

    # Per season, find years all children reported.
    seasons: set[object] = set()
    for cells in per_child_cells.values():
        for (_y, s) in cells:
            seasons.add(s)

    common_years_per_season: dict[object, set[int]] = {}
    for season in seasons:
        year_sets = [
            {y for (y, s) in per_child_cells[c] if s == season}
            for c in children
        ]
        if year_sets and all(year_sets):
            common_years_per_season[season] = set.intersection(*year_sets)
        else:
            common_years_per_season[season] = set()

    total_common_cells = sum(len(ys) for ys in common_years_per_season.values())

    means: dict[str, float | None] = {}
    for child in children:
        per_season_means: list[float] = []
        for season, common_yrs in common_years_per_season.items():
            if not common_yrs:
                continue
            vals = [per_child_cells[child][(y, season)] for y in common_yrs]
            per_season_means.append(sum(vals) / len(vals))
        if per_season_means:
            means[child] = sum(per_season_means)
        else:
            means[child] = None

    return means, total_common_cells, n_individual


def _normalize(
    means: dict[str, float | None],
) -> dict[str, float]:
    """Convert per-child means into fractions that sum to 1 across the
    non-NaN siblings.

    If ``total > 0``: child fraction = mean / total (None children get NaN).
    If ``total == 0``: all fractions are NaN (mathematically undefined).
    """
    non_nan = {c: m for c, m in means.items() if m is not None}
    total = sum(non_nan.values()) if non_nan else 0.0
    out: dict[str, float] = {}
    for child, mean in means.items():
        if total > 0 and mean is not None:
            out[child] = mean / total
        else:
            out[child] = math.nan
    return out


def _compute_fractions(
    children: set[str],
    parents: set[str],
    event_year: int,
    stats_lookup: dict[str, dict[tuple[int, object, str], float]],
    stats_by_unit_var: dict[tuple[str, str], list[tuple[int, object, float]]],
    stats_vars_by_unit: dict[str, set[str]],
    parent_pool_keys: set[tuple[int, object, str]],
    window_per_var: dict[str, int],
    default_window: int,
    modern_areas: dict[str, float] | None = None,
) -> tuple[dict[str, dict[tuple[str, object], tuple[float, int, str]]], list[dict]]:
    """Compute three-tier cascading fractions for each child.

    Returns:
        ``(fractions, audit_rows)`` where:

        - ``fractions[child][(var, season)] = (fraction, n_common, method)``
          is the chosen fraction for that (child, var, season) combo,
          tagged with which tier produced it. ``n_common`` is the count
          of common observations used (same across siblings within an
          event, since the fraction is computed over their intersection).

        - ``audit_rows`` — one dict per (child, var, season) combo with
          full provenance. Includes the chosen fraction + method,
          ``n_common_observations`` (per-event), and
          ``n_individual_observations`` (per-child; reveals when a
          child's reports were excluded from the intersection).

    The cascade tries:
      1. Seasonal (per (var, season) within the window).
      2. Total-year per variable (sum across all seasons, same window).
      3. Modern-area share (no data needed).
      4. NaN with method='undefined' (only if all three fail).

    Discovery of (var, season) pairs to compute fractions for combines
    THREE sources, so that we always have fractions for any cell that
    could appear in the pool:

      a. Children's in-window reports (drives the seasonal tier numerator).
      b. Parents' pre-event reports (parent could pool any of these).
      c. The pool's actual cells (catches inherited values from prior
         events that may not be in parents' direct stats_lookup but ARE
         in the parent's ledger).

    Without (b) and (c), recent events with no post-event reporting yet
    would leave parent pool data unattributable.
    """
    # --- Pass 1: discover scope ----------------------------------------
    # Use the (unit_id, var) index to iterate only the variables a unit
    # actually reports, instead of scanning every cell. The full key
    # `(year, season, var)` would otherwise be enumerated per child per
    # event — extremely slow on India (~750K cells × 700+ events).
    varseasons: set[tuple[str, object]] = set()
    variables: set[str] = set()
    # (a) From children's in-window reports
    for child in children:
        for var in stats_vars_by_unit.get(child, ()):
            window = window_per_var.get(var, default_window)
            upper = event_year + window
            for (y, s, _v) in stats_by_unit_var.get((child, var), ()):
                if event_year < y <= upper:
                    varseasons.add((var, s))
                    variables.add(var)
    # (b) From parents' pre-event reports
    for parent in parents:
        for var in stats_vars_by_unit.get(parent, ()):
            for (y, s, _v) in stats_by_unit_var.get((parent, var), ()):
                if y <= event_year:
                    varseasons.add((var, s))
                    variables.add(var)
    # (c) From the pool's actual cells (covers inherited values that
    # parents themselves never reported but received at prior events)
    for (_cy, cs, cv) in parent_pool_keys:
        varseasons.add((cv, cs))
        variables.add(cv)

    # --- Pass 2: seasonal fractions per (var, season) ---
    # Each entry stores (fractions, n_common, n_individual_per_child).
    # n_common is per-event (same across siblings by construction);
    # n_individual lets the audit reveal when a child's data was
    # excluded from the common-years intersection.
    seasonal_fracs: dict[tuple[str, object], dict[str, float]] = {}
    seasonal_n_common: dict[tuple[str, object], int] = {}
    seasonal_n_individual: dict[tuple[str, object], dict[str, int]] = {}
    for var, season in varseasons:
        window = window_per_var.get(var, default_window)
        means, n_common, n_indiv = _window_means(
            children, event_year, var, season, window, stats_by_unit_var
        )
        seasonal_fracs[(var, season)] = _normalize(means)
        seasonal_n_common[(var, season)] = n_common
        seasonal_n_individual[(var, season)] = n_indiv

    # --- Pass 3: total-year fractions per variable ---
    total_year_fracs: dict[str, dict[str, float]] = {}
    total_year_n_common: dict[str, int] = {}
    total_year_n_individual: dict[str, dict[str, int]] = {}
    for var in variables:
        window = window_per_var.get(var, default_window)
        means, n_common, n_indiv = _total_year_means(
            children, event_year, var, window, stats_by_unit_var
        )
        total_year_fracs[var] = _normalize(means)
        total_year_n_common[var] = n_common
        total_year_n_individual[var] = n_indiv

    # --- Pass 4: modern-area fractions ---
    area_frac_per_child: dict[str, float] = {}
    if modern_areas:
        in_modern = {c: float(modern_areas[c]) for c in children if c in modern_areas}
        total_area = sum(in_modern.values())
        for child in children:
            if total_area > 0 and child in in_modern:
                area_frac_per_child[child] = in_modern[child] / total_area
            else:
                area_frac_per_child[child] = math.nan
    else:
        for child in children:
            area_frac_per_child[child] = math.nan

    # --- Pass 5: pick ONE tier per (var, season) for the WHOLE event ---
    # This is critical for conservation: all siblings must share the same
    # tier so their fractions sum to 1. If we let each child pick its own
    # tier, a reporting sibling at seasonal=1.0 plus a non-reporting
    # sibling at area=0.5 would over-allocate the pool to 1.5x.
    #
    # The rule per (var, season):
    #   - If ANY sibling has a non-NaN seasonal fraction → use seasonal
    #     for all. Non-reporting siblings get fraction = 0 (reporter
    #     absorbed their share — same semantics as the original code
    #     where the reporter normalized to 1.0 alone).
    #   - Else if ANY sibling has a non-NaN total-year fraction → use
    #     total_year for all. Non-contributors get 0.
    #   - Else if ANY sibling has a non-NaN area fraction → use area for
    #     all. (Modern-shapefile orphans get 0.)
    #   - Else → undefined; all children get NaN.
    fractions: dict[str, dict[tuple[str, object], tuple[float, int, str]]] = {
        c: {} for c in children
    }
    audit_rows: list[dict] = []

    for var, season in sorted(varseasons, key=lambda k: (str(k[0]), str(k[1]))):
        window = window_per_var.get(var, default_window)
        seas_per = seasonal_fracs[(var, season)]
        seas_n_common = seasonal_n_common[(var, season)]
        seas_n_indiv = seasonal_n_individual[(var, season)]
        ty_per = total_year_fracs.get(var, {})
        ty_n_common = total_year_n_common.get(var, 0)
        ty_n_indiv = total_year_n_individual.get(var, {})

        # Determine the tier to use across the WHOLE event.
        any_seasonal = any(not _is_nan(seas_per[c]) for c in children)
        any_total_year = any(not _is_nan(ty_per.get(c, math.nan)) for c in children)
        any_area = any(not _is_nan(area_frac_per_child[c]) for c in children)

        if any_seasonal:
            method = "seasonal"
        elif any_total_year:
            method = "total_year"
        elif any_area:
            method = "area"
        else:
            method = "undefined"

        for child in children:
            # n_common is per-event (same across siblings); n_individual
            # is per-child (varies). Both go into the audit so the user
            # can see whether one child's incomplete reporting reduced
            # the common-years set.
            if method == "seasonal":
                f = seas_per[child]
                if _is_nan(f):
                    f, n_common = 0.0, 0  # absorbed by reporting sibling
                else:
                    n_common = seas_n_common
                n_indiv = seas_n_indiv.get(child, 0)
            elif method == "total_year":
                f = ty_per.get(child, math.nan)
                if _is_nan(f):
                    f, n_common = 0.0, 0
                else:
                    n_common = ty_n_common
                n_indiv = ty_n_indiv.get(child, 0)
            elif method == "area":
                f = area_frac_per_child[child]
                if _is_nan(f):
                    f, n_common = 0.0, 0
                else:
                    n_common = 0
                n_indiv = 0
            else:
                f, n_common, n_indiv = math.nan, 0, 0

            fractions[child][(var, season)] = (f, n_common, method)

            # Shares worked out from only a year or two of overlap are worth
            # saying out loud. They are still the best available answer, but
            # a single unusual harvest can skew them badly, and someone
            # reading a surprising number later deserves to know that.
            if method == "seasonal" and 0 < n_common < window:
                _LOGGER.warning(
                    "modern fraction for child=%s var=%s season=%s at "
                    "event_year=%d uses only %d/%d common post-event years",
                    child, var, season, event_year, n_common, window,
                )

            audit_rows.append({
                "event_year": event_year,
                "child_id": child,
                "variable": var,
                "season": season,
                "window_used": window,
                "n_common_observations": n_common,
                "n_individual_observations": n_indiv,
                "fraction": f,
                "fraction_method": method,
            })

    return fractions, audit_rows


def _lookup_fraction(
    fractions: dict[str, dict[tuple[str, object], tuple[float, int, str]]],
    child: str,
    var: str,
    season: object,
    total_year_lookup: dict[str, dict[str, float]] | None = None,
    area_lookup: dict[str, float] | None = None,
) -> tuple[float, str]:
    """Look up the fraction for ``(child, var, season)``, applying cascade
    fallback live if the (var, season) wasn't precomputed.

    Used in two places:
      1. Distribution at event time — fractions are precomputed for
         in-scope (var, season) pairs; this function returns those.
      2. Late-report redistribution — a (var, season) on a late cell
         may not have been in-scope at the terminal event. We fall back
         to total_year and area on the fly.

    Returns ``(fraction, method)``.
    """
    child_fracs = fractions.get(child)
    if child_fracs is not None and (var, season) in child_fracs:
        frac, _, method = child_fracs[(var, season)]
        if not _is_nan(frac):
            return frac, method

    # Live fallback: total_year for this var
    if total_year_lookup is not None:
        ty = total_year_lookup.get(var, {}).get(child)
        if ty is not None and not _is_nan(ty):
            return ty, "total_year"

    # Live fallback: area
    if area_lookup is not None:
        ar = area_lookup.get(child)
        if ar is not None and not _is_nan(ar):
            return ar, "area"

    return math.nan, "undefined"


# --- Pool + distribute -----------------------------------------------------


def _pool_parents(
    parents: set[str],
    event_year: int,
    ledger: dict[str, dict[tuple[int, object, str], dict]],
) -> dict[tuple[int, object, str], dict]:
    """Sum parents' pre-event ledgers cell-by-cell into a single pool.

    Pool inclusion is ``year <= event_year`` (parent alive through year T
    under the canonical convention). Late-reporting cells (``year >
    event_year``) stay on the parent for Step D redistribution.
    """
    # When districts are reorganised, everything they reported beforehand is
    # tipped into a common pot, to be shared out among whatever replaced
    # them. Add up each parent's figures for the same crop, season and year.
    pool: dict[tuple[int, object, str], dict] = defaultdict(_empty_pool_cell)
    for parent in parents:
        parent_cells = ledger.get(parent)
        if not parent_cells:
            continue
        for key, cell in parent_cells.items():
            year = key[0]
            # Only what was reported up to and including the year of the
            # change. Anything a dissolved district filed afterwards is left
            # where it is and dealt with separately, since it is a different
            # problem — a district still reporting after it ceased to exist.
            if year > event_year:
                continue
            pcell = pool[key]
            v = cell["value"]
            # A missing figure is not a zero. Record that something is
            # unknown and carry that fact forward, rather than adding
            # nothing and letting the total look complete.
            if _is_nan(v):
                pcell["has_nan_fraction"] = True
            else:
                pcell["value"] += float(v)
                pcell["_observed"] = True
            # The rest travels with the figure so the finished result can say
            # where it came from and how well founded it is: which districts
            # contributed, whether anything was missing, how many
            # reorganisations it has been through, and -- taking the weakest
            # of the parents -- how the shares were arrived at.
            pcell["sources"] |= cell["sources"]
            pcell["has_nan_fraction"] |= cell["has_nan_fraction"]
            pcell["depth"] = max(pcell["depth"], cell["depth"])
            pcell["fraction_method"] = _worst_method(
                pcell["fraction_method"], cell["fraction_method"]
            )
            pcell["min_n_common"] = min(
                pcell["min_n_common"], cell.get("min_n_common", math.inf)
            )
            pcell["late_report_redistributed"] |= cell.get(
                "late_report_redistributed", False
            )
    for key, pcell in pool.items():
        if not pcell["_observed"]:
            pcell["value"] = math.nan
            pcell["has_nan_fraction"] = True
    return pool


def _add_to_ledger_cell(
    ledger: dict[str, dict[tuple[int, object, str], dict]],
    child: str,
    key: tuple,
    value: float,
    sources: set,
    has_nan_fraction: bool,
    depth: int,
    fraction_method: str,
    late_report_redistributed: bool,
    min_n_common: float = math.inf,
) -> None:
    """Add ``value`` to ``ledger[child][key]``, NaN-aware. Composes
    fraction_method, sources, and flags.
    """
    # A district can receive from more than one direction — territory from
    # two different reorganisations, or several parents in the same one — so
    # arriving figures are added to whatever is already there rather than
    # replacing it. First arrival just moves in.
    existing = ledger[child].get(key)
    if existing is None:
        ledger[child][key] = {
            "value": value,
            "sources": set(sources),
            "has_nan_fraction": has_nan_fraction,
            "depth": depth,
            "fraction_method": fraction_method,
            "late_report_redistributed": late_report_redistributed,
            "min_n_common": min_n_common,
        }
        return
    # Otherwise add to it. If either side is unknown the total is unknown —
    # adding a known figure to an unknown one does not produce a known
    # answer, and treating the gap as zero would understate the district.
    ev = existing["value"]
    if _is_nan(ev) or _is_nan(value):
        existing["value"] = math.nan
    else:
        existing["value"] = float(ev) + float(value)
    # The accompanying record merges pessimistically throughout: every
    # contributing district is remembered, any missing input taints the
    # result, the depth is the longest chain involved, and the method is the
    # weakest one used. A figure is only as trustworthy as its worst part.
    existing["sources"] |= sources
    existing["has_nan_fraction"] = existing["has_nan_fraction"] or has_nan_fraction
    existing["depth"] = max(existing["depth"], depth)
    existing["fraction_method"] = _worst_method(
        existing["fraction_method"], fraction_method
    )
    existing["late_report_redistributed"] = (
        existing["late_report_redistributed"] or late_report_redistributed
    )
    existing["min_n_common"] = min(
        existing.get("min_n_common", math.inf), min_n_common
    )


def _distribute_pool_to_children(
    children: set[str],
    pool: dict[tuple[int, object, str], dict],
    fractions: dict[str, dict[tuple[str, object], tuple[float, int, str]]],
    ledger: dict[str, dict[tuple[int, object, str], dict]],
) -> set[tuple[object, str]]:
    """For each child and each pool cell, apply the cascading fraction and
    add to the child's ledger. Records fraction_method per cell.

    Returns the set of ``(season, variable)`` pairs that were
    *undistributable* (method="undefined" — no child has a non-NaN
    seasonal / total_year / area fraction). Caller MUST exclude these
    keys from `_drop_pre_event_cells` so the parent's data is preserved
    (routes to ``stats_modern`` if parent still in the modern shapefile,
    else ``late_reporting``). Added 2026-06-05 to fix the C7 silent-drop
    bug: 2023 boundary changes whose new child IDs aren't yet in the
    modern shapefile were silently losing 184M MT/ha of parent data.
    """
    undistributable: set[tuple[object, str]] = set()
    # Detect undistributable (var, season) pairs upfront. Because the
    # method tag is chosen per-event (same for all children at a given
    # (var, season), see _compute_fractions Pass 5), checking any single
    # child suffices.
    any_child = next(iter(children), None)
    if any_child is not None:
        for (var, season), (_frac, _n, method) in fractions.get(any_child, {}).items():
            if method == "undefined":
                undistributable.add((season, var))

    for child in children:
        child_fracs = fractions.get(child, {})
        for key, pcell in pool.items():
            year, season, var = key
            if (season, var) in undistributable:
                # Skip — leaving the parent cell intact preserves the
                # data via the parent's own modern/late routing.
                continue
            # How much of the pot this district gets, for this crop and
            # season. The share was worked out earlier from what the
            # districts reported once they existed separately.
            frac_info = child_fracs.get((var, season))
            if frac_info is None:
                # No share was worked out for this crop at all, which should
                # not happen — every crop in the pot was covered earlier.
                # Produce an explicit unknown rather than guessing.
                inherited_value = math.nan
                has_nan = True
                method = "undefined"
                n_common = 0
            else:
                frac, n_common, method = frac_info
                # An unknown share, or an unknown total to apply it to, gives
                # an unknown answer either way.
                if _is_nan(frac):
                    inherited_value = math.nan
                    has_nan = True
                elif _is_nan(pcell["value"]):
                    inherited_value = math.nan
                    has_nan = True
                else:
                    # The ordinary case: this district's slice of the pot.
                    inherited_value = pcell["value"] * frac
                    has_nan = pcell["has_nan_fraction"]

            # Compose this event's fraction method with anything already
            # in the pool cell's lineage (e.g., earlier-event fallbacks).
            composed_method = _worst_method(pcell["fraction_method"], method)
            # Weakest link along the path, not this event's figure alone.
            composed_min_n = min(pcell["min_n_common"], n_common)

            _add_to_ledger_cell(
                ledger,
                child=child,
                key=key,
                value=inherited_value,
                sources=pcell["sources"],
                has_nan_fraction=has_nan,
                depth=pcell["depth"] + 1,
                fraction_method=composed_method,
                late_report_redistributed=pcell["late_report_redistributed"],
                min_n_common=composed_min_n,
            )
    return undistributable


def _drop_pre_event_cells(
    parents: set[str],
    event_year: int,
    ledger: dict[str, dict[tuple[int, object, str], dict]],
    skip_keys: set[tuple[object, str]] | None = None,
) -> None:
    """Remove pre-event cells from each parent's ledger after distribution.

    Late-reporting cells (year > T) stay on the parent for Step D.

    ``skip_keys`` (``(season, variable)`` tuples): if provided, cells
    whose (season, variable) is in this set are NOT dropped — they were
    not distributed to children, so the parent must retain them to
    preserve the data (silent-drop fix, 2026-06-05).
    """
    skip = skip_keys or set()
    for parent in parents:
        if parent not in ledger:
            continue
        keys_to_drop = [
            k for k in ledger[parent]
            if k[0] <= event_year and (k[1], k[2]) not in skip
        ]
        for k in keys_to_drop:
            del ledger[parent][k]


# --- Step D: late-report redistribution ------------------------------------


def _redistribute_late_reports(
    ledger: dict[str, dict[tuple[int, object, str], dict]],
    parent_terminal_event: dict[str, int],
    parent_event_fractions: dict[
        tuple[str, int],
        dict[str, dict[tuple[str, object], tuple[float, int, str]]],
    ],
    parent_event_total_year: dict[tuple[str, int], dict[str, dict[str, float]]],
    parent_event_area: dict[tuple[str, int], dict[str, float]],
    parent_event_children: dict[tuple[str, int], set[str]],
) -> int:
    """Redistribute parent ledger cells filed AFTER the parent's terminal
    event onto the modern children, using the same fractions computed at
    that event (with cascade fallback).

    .. warning::

        **2026-06-04: this routine is not currently invoked from
        ``build_modern_ledger``.** Empirical audit on India 1997-2022
        showed that 91.3% of the redistributed cells double-count real
        successor data that's independently reported through the
        correct canonical IDs (driven by an upstream LGD-to-canonical-ID
        mapping bug). The routine is left in the module so the math can
        be inspected and so a future fix can wire it back up after the
        underlying assumption ("post-terminal parent reports = orphan
        data that must be redistributed") is corrected. The audit that
        showed this is in the authors' diagnostic notebook (not part of this repository).

    Mutates ``ledger`` in place. Returns the count of redistributed cells.

    Implementation notes:
      - The original late-reporting cell stays under the parent's ID for
        ``late_reporting.csv``; we don't delete it here.
      - For (var, season) combos not seen at the terminal event, the
        cascade fallback (total_year → area → undefined) is applied via
        ``_lookup_fraction``.
    """
    redistributed_count = 0

    for parent_id, terminal_year in parent_terminal_event.items():
        parent_cells = ledger.get(parent_id)
        if not parent_cells:
            continue

        children = parent_event_children.get((parent_id, terminal_year), set())
        if not children:
            continue

        fractions = parent_event_fractions.get((parent_id, terminal_year), {})
        ty_lookup = parent_event_total_year.get((parent_id, terminal_year), {})
        ar_lookup = parent_event_area.get((parent_id, terminal_year), {})

        late_keys = [k for k in parent_cells if k[0] > terminal_year]
        for key in late_keys:
            year, season, var = key
            cell = parent_cells[key]
            v = cell["value"]
            if _is_nan(v):
                # No data to redistribute; leave the late-reporting flag
                # as-is.
                continue

            for child in children:
                frac, method = _lookup_fraction(
                    fractions, child, var, season,
                    total_year_lookup=ty_lookup,
                    area_lookup=ar_lookup,
                )
                if _is_nan(frac):
                    # Even cascade failed — child has no data and no
                    # modern shapefile entry. Skip this child for this
                    # cell (data is not lost in aggregate; just unallocated
                    # for this particular child).
                    continue

                inherited_value = float(v) * frac
                _add_to_ledger_cell(
                    ledger,
                    child=child,
                    key=key,
                    value=inherited_value,
                    sources=cell["sources"] | {parent_id},
                    has_nan_fraction=cell["has_nan_fraction"],
                    depth=cell["depth"] + 1,
                    fraction_method=_worst_method(cell["fraction_method"], method),
                    late_report_redistributed=True,
                )
                redistributed_count += 1

    return redistributed_count


# --- Step E: Total Year derived rows ---------------------------------------


def _append_total_year_rows(stats_modern: pd.DataFrame) -> pd.DataFrame:
    """Append per (modern_id, year, variable) summed-across-seasons rows
    with ``season='Total Year'`` to ``stats_modern``.

    Vectorized via pandas groupby agg — was the bottleneck at ~280s on
    India before this rewrite (Python-level per-group loop over 1.6M
    groups). The agg pass handles the easy aggregations natively;
    sources/fraction_method use 'first' (cells in one (modern_id, year,
    variable) usually share lineage so 'first' is representative).

    NaN semantics: pandas .sum(skipna=True) treats NaN as 0, so a
    group where every seasonal value is NaN comes out as 0. We
    detect that case and promote to NaN explicitly.
    """
    if stats_modern.empty:
        return stats_modern

    keys = ["modern_id", "year", "variable"]
    g = stats_modern.groupby(keys, dropna=False, sort=False)
    # Vectorized aggs (these are the vast majority of the work).
    agg = g.agg(
        value=("value", "sum"),
        has_nan_fraction=("has_nan_fraction", "any"),
        late_report_redistributed=("late_report_redistributed", "any"),
        lineage_depth=("lineage_depth", "max"),
        fraction_method=("fraction_method", "first"),
        sources=("sources", "first"),
        # "first" to stay consistent with the `sources` string above, which
        # is also first-wins. (Strictly the Total Year row's sources are the
        # union across seasons; that pre-existing simplification is left
        # alone here rather than changed as a side effect.)
        n_sources=("n_sources", "first"),
        # A Total Year value sums the seasons, so it is only as
        # well-estimated as the weakest season that fed it.
        min_n_common_observations=("min_n_common_observations", "min"),
        # Track count of non-NaN values so we can detect all-NaN groups.
        _n_non_nan=("value", "count"),
    ).reset_index()

    # Promote sum-of-all-NaN (which pandas reports as 0) to NaN.
    all_nan_mask = agg["_n_non_nan"] == 0
    agg.loc[all_nan_mask, "value"] = math.nan
    agg = agg.drop(columns=["_n_non_nan"])

    agg["season"] = "Total Year"

    # Reorder and concat.
    ty_df = agg[stats_modern.columns]
    return (
        pd.concat([stats_modern, ty_df], ignore_index=True)
        .sort_values(
            ["variable", "year", "season", "modern_id"], na_position="last"
        )
        .reset_index(drop=True)
    )


# --- Top-level driver ------------------------------------------------------


def build_modern_ledger(
    graph: LineageGraph,
    stats_df: pd.DataFrame,
    modern_unit_ids: Iterable[str],
    base_year: int,
    max_year: int,
    *,
    window_per_var: dict[str, int] | None = None,
    default_window: int = 5,
    modern_areas: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the modern boundary product.

    Args:
        graph: Parsed ``LineageGraph``.
        stats_df: Long-form stats with the canonical schema.
        modern_unit_ids: IDs of units in today's shapefile.
        base_year, max_year: Inclusive analysis window.
        window_per_var: Per-variable post-event window override.
        default_window: Default post-event window length.
        modern_areas: Optional ``{unit_id: area}`` dict for the
            modern-area fallback (third tier of the cascade). If omitted,
            the cascade reduces to seasonal → total_year → undefined.

    Returns:
        ``(stats_modern, fractions, late_reporting)`` — three DataFrames.
        See ``docs/USAGE.md`` for the column reference.
    """
    window_per_var = dict(window_per_var or {})
    modern_set = set(modern_unit_ids)
    modern_areas = dict(modern_areas) if modern_areas else None

    # --- Step 0: reorganise the statistics for fast lookup -------------
    # The rest of this function asks "what did district X report for this
    # crop, season and year?" many thousands of times as it walks through
    # history. Answering that by searching the table each time is what makes
    # a naive version of this unusably slow, so it is arranged up front into
    # a structure that answers the question directly.
    stats_lookup: dict[str, dict[tuple[int, object, str], float]] = defaultdict(dict)
    for row in stats_df.itertuples(index=False):
        # Ignore anything outside the period being built.
        year = int(row.year)
        if year < base_year or year > max_year:
            continue
        key = (year, row.season, row.variable)
        existing = stats_lookup[str(row.unit_id)].get(key)
        v = row.value
        if existing is None:
            stats_lookup[str(row.unit_id)][key] = (
                math.nan if _is_nan(v) else float(v)
            )
        elif _is_nan(existing):
            stats_lookup[str(row.unit_id)][key] = (
                math.nan if _is_nan(v) else float(v)
            )
        elif _is_nan(v):
            pass
        else:
            stats_lookup[str(row.unit_id)][key] = float(existing) + float(v)

    # --- Step 0b: build the (unit_id, var) index ----------------------
    # The cascade fraction logic queries "what's unit X's report for
    # var V in years (T+1, T+w]?" thousands of times per India run.
    # Without this index, each query scans ALL of unit X's cells (any
    # var) to filter to var V — quadratic in the worst case. Pre-pivot
    # by (unit_id, var) so the query is O(rows for that var only),
    # cutting India's modern build from ~165s to a few seconds.
    stats_by_unit_var: dict[tuple[str, str], list[tuple[int, object, float]]] = defaultdict(list)
    stats_vars_by_unit: dict[str, set[str]] = defaultdict(set)
    for unit_id, cells in stats_lookup.items():
        for (y, s, var), v in cells.items():
            stats_by_unit_var[(unit_id, var)].append((y, s, v))
            stats_vars_by_unit[unit_id].add(var)

    # --- Step 1: initialize ledger -------------------------------------
    ledger: dict[str, dict[tuple[int, object, str], dict]] = defaultdict(dict)
    for unit_id, cells in stats_lookup.items():
        for key, value in cells.items():
            ledger[unit_id][key] = _new_cell(value, unit_id)

    # --- Step 2: walk events; track per-event fractions for Step D ----
    territorial = graph.territorial
    in_window = territorial[
        (territorial["event_year"] >= base_year)
        & (territorial["event_year"] <= max_year)
    ]
    fractions_audit_rows: list[dict] = []

    # For Step D: per (parent_id, event_year), capture the fraction tables
    # so we can replay them on late reports.
    parent_terminal_event: dict[str, int] = {}
    parent_event_fractions: dict[
        tuple[str, int],
        dict[str, dict[tuple[str, object], tuple[float, int, str]]],
    ] = {}
    parent_event_total_year: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    parent_event_area: dict[tuple[str, int], dict[str, float]] = {}
    parent_event_children: dict[tuple[str, int], set[str]] = {}

    for year, year_events in in_window.groupby("event_year"):
        year_int = int(year)
        for parents, children in _connected_components(year_events):
            # Per-parent processing (2026-06-05 fix for E1 inflation):
            # for each parent in the connected component, distribute ITS
            # pool only to ITS OWN children per the lineage edges. The
            # previous logic pooled across parents in the same component
            # and split to ALL children — which over-allocated to
            # children that have no lineage edge from one of the parents
            # (Redistribute events). Children shared between parents
            # (e.g. Anupgarh from {Bikaner, Sri Ganganagar} in 2023)
            # correctly accumulate one share per parent edge.
            for parent in sorted(parents):
                parent_children: set[str] = set(
                    year_events.loc[year_events["parent_id"] == parent, "child_id"]
                )
                if not parent_children:
                    continue

                pool = _pool_parents({parent}, year_int, ledger)
                pool_keys = set(pool.keys())

                child_fractions, audit_rows = _compute_fractions(
                    children=parent_children,
                    parents={parent},
                    event_year=year_int,
                    stats_lookup=stats_lookup,
                    stats_by_unit_var=stats_by_unit_var,
                    stats_vars_by_unit=stats_vars_by_unit,
                    parent_pool_keys=pool_keys,
                    window_per_var=window_per_var,
                    default_window=default_window,
                    modern_areas=modern_areas,
                )

                for r in audit_rows:
                    r["parent_ids"] = parent
                    r["event_type"] = (
                        "Split" if len(parent_children) > 1 else "Coarse"
                    )
                fractions_audit_rows.extend(audit_rows)

                # Build live fallback lookups for Step D (per-parent basis).
                # Step D is currently disabled (see comment below), but we
                # keep the bookkeeping in case it's re-enabled.
                ty_lookup: dict[str, dict[str, float]] = defaultdict(dict)
                ar_lookup: dict[str, float] = {}
                variables_in_event: set[str] = set()
                for child, fmap in child_fractions.items():
                    for (var, _season) in fmap.keys():
                        variables_in_event.add(var)
                for var in variables_in_event:
                    window = window_per_var.get(var, default_window)
                    ty_means, _, _ = _total_year_means(
                        parent_children, year_int, var, window, stats_by_unit_var
                    )
                    ty_normalized = _normalize(ty_means)
                    for c, f in ty_normalized.items():
                        ty_lookup[var][c] = f
                # Last resort, if nothing was ever reported separately: split
                # by how much ground each new district covers. Crude — it
                # assumes farmland is spread evenly, which it is not — but it
                # is better than abandoning the figures entirely, and the
                # result records that this is how it was arrived at.
                if modern_areas:
                    in_modern = {
                        c: modern_areas[c]
                        for c in parent_children if c in modern_areas
                    }
                    total_area = sum(in_modern.values())
                    for c in parent_children:
                        if total_area > 0 and c in in_modern:
                            ar_lookup[c] = in_modern[c] / total_area
                        else:
                            # No area known for this one, so no share can be
                            # given. Left explicitly unknown.
                            ar_lookup[c] = math.nan

                parent_terminal_event[parent] = max(
                    parent_terminal_event.get(parent, year_int), year_int
                )
                parent_event_fractions[(parent, year_int)] = child_fractions
                parent_event_total_year[(parent, year_int)] = dict(ty_lookup)
                parent_event_area[(parent, year_int)] = dict(ar_lookup)
                parent_event_children[(parent, year_int)] = set(parent_children)

                undistributable = _distribute_pool_to_children(
                    children=parent_children,
                    pool=pool,
                    fractions=child_fractions,
                    ledger=ledger,
                )
                _drop_pre_event_cells(
                    {parent}, year_int, ledger, skip_keys=undistributable
                )

    # --- Step D: late-report redistribution (DISABLED 2026-06-04) ------
    # Empirical audit on India 1997-2022 found that 91.3% of redistributed
    # cells stack phantom data on top of real successor data that's
    # independently reported through the correct canonical IDs (the
    # upstream LGD-to-canonical-ID lookup files the successor's data
    # twice — under both the post-split successor ID and the pre-split
    # parent's stale ID). Redistribution then doubles the real value at
    # the modern-district level, sometimes by orders of magnitude
    # (e.g. Kamrup jute 2005: parent_late=3,322 ha vs child reports 69 ha;
    # Kancheepuram sugarcane 2022: parent_late=128,639 MT vs children 59,354 MT).
    # The BEAST evidence on affected modern_id × crop series confirms the
    # signal: redistribution makes 91% of changed series choppier, with
    # mean breakpoint count rising from 0.27 (no redistribution) to 1.01.
    # Disabled in step with the reconcile teardown. The orphan rows are
    # still surfaced in `late_reporting.csv` for transparency; they no
    # longer appear in `stats_modern.csv` (no `late_report_redistributed`
    # rows). The audit is in the authors' diagnostic notebook (not part of this repository).
    # (Step D used to bind `n_redistributed` here and thread it into the
    # summary. With the step disabled the binding was dead — nothing reads
    # it — so it has been removed rather than pinned at a misleading 0.
    # Re-enabling Step D means restoring both the call and the binding.)

    # --- Step 3: assemble outputs --------------------------------------
    stats_rows: list[dict] = []
    late_rows: list[dict] = []
    for unit_id, cells in ledger.items():
        is_modern = unit_id in modern_set
        # Flatten everything into rows for the output file. Each figure
        # carries its own provenance alongside it — which districts it came
        # from, how thin the evidence for its share was, how many
        # reorganisations it has passed through — so a reader can judge any
        # single number without having to rerun anything.
        for (year, season, var), cell in cells.items():
            row = {
                "year": year,
                "season": season,
                "variable": var,
                "modern_id": unit_id,
                "value": cell["value"],
                "sources": ",".join(sorted(cell["sources"])),
                "n_sources": len(cell["sources"]),
                # inf means no fraction was ever applied (a direct report);
                # NA reads better than "inf" in a CSV a human will open.
                "min_n_common_observations": (
                    pd.NA if math.isinf(cell.get("min_n_common", math.inf))
                    else int(cell["min_n_common"])
                ),
                "lineage_depth": cell["depth"],
                "has_nan_fraction": cell["has_nan_fraction"],
                "fraction_method": cell["fraction_method"],
                "late_report_redistributed": cell.get(
                    "late_report_redistributed", False
                ),
            }
            if is_modern:
                stats_rows.append(row)
            else:
                row.pop("modern_id")
                row["unit_id"] = unit_id
                late_rows.append(row)

    stats_columns = [
        "year", "season", "variable", "modern_id", "value",
        "sources", "n_sources", "min_n_common_observations",
        "lineage_depth", "has_nan_fraction",
        "fraction_method", "late_report_redistributed",
    ]
    stats_modern = pd.DataFrame(stats_rows, columns=stats_columns)
    if not stats_modern.empty:
        stats_modern = stats_modern.sort_values(
            ["variable", "year", "season", "modern_id"], na_position="last"
        ).reset_index(drop=True)

    # --- Step E: append Total Year rows --------------------------------
    stats_modern = _append_total_year_rows(stats_modern)

    late_columns = [
        "year", "season", "variable", "unit_id", "value",
        "sources", "n_sources", "min_n_common_observations",
        "lineage_depth", "has_nan_fraction",
        "fraction_method", "late_report_redistributed",
    ]
    late_reporting = pd.DataFrame(late_rows, columns=late_columns)
    if not late_reporting.empty:
        late_reporting = late_reporting.sort_values(
            ["unit_id", "variable", "year", "season"], na_position="last"
        ).reset_index(drop=True)

    fractions = pd.DataFrame(
        fractions_audit_rows,
        columns=[
            "event_year", "event_type", "parent_ids", "child_id",
            "variable", "season", "window_used",
            "n_common_observations", "n_individual_observations",
            "fraction", "fraction_method",
        ],
    )
    if not fractions.empty:
        fractions = fractions.sort_values(
            ["event_year", "child_id", "variable", "season"],
            na_position="last",
        ).reset_index(drop=True)

    return stats_modern, fractions, late_reporting
