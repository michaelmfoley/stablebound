"""Dataset-level completeness summaries for both products.

The per-row completeness columns (``n_constituents``, ``n_in_group``,
``completeness``, ``complete``, ``missing_unit_ids``) are the ground truth,
but nobody assesses a dataset by scanning 700,000 rows. This module collapses
them to one row per ``(variable, year)`` — the shape you would put in a paper
appendix or scan before deciding a country is ready to publish.

Both products call :func:`summarize`, which adapts to whichever columns the
frame carries:

- **Stable** rows have the full set, so the summary reports the reporting
  ratio, how many cells are complete, and which groups are chronically worst.
- **Modern** rows have ``n_sources`` and ``min_n_common_observations``
  instead — there is no "expected members" denominator for a modern unit, so
  the summary reports source counts and the weakest fraction evidence rather
  than inventing a ratio.

Late-reporting rows are excluded from the stable summary: their completeness
columns are NA by design (a unit reporting outside its lifespan has no group
to be complete against), so including them would drag every average down for
a reason unrelated to coverage. They are counted separately instead.
"""
from __future__ import annotations

import pandas as pd

# One row per (variable, year) unless the caller asks otherwise.
DEFAULT_KEYS = ["variable", "year"]


def summarize(
    df: pd.DataFrame,
    *,
    by: str | list[str] | None = None,
    worst_n: int = 3,
) -> pd.DataFrame:
    """Collapse per-row completeness columns into a scannable summary.

    Args:
        df: A stable ``stats_aggregated`` frame or a modern ``stats_modern``
            frame. The available columns decide which summary is produced.
        by: Grouping keys. Defaults to ``["variable", "year"]``. Pass
            ``"stable_id"`` to find chronically incomplete units instead of
            chronically incomplete years.
        worst_n: How many worst-offending ids to name per group.

    Returns:
        One row per group. Empty frame (with no columns) if ``df`` is empty.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    keys = DEFAULT_KEYS if by is None else ([by] if isinstance(by, str) else list(by))
    keys = [k for k in keys if k in df.columns]
    if not keys:
        raise ValueError(
            f"none of the requested grouping keys are present; frame has "
            f"{list(df.columns)}"
        )

    if "completeness" in df.columns:
        return _summarize_stable(df, keys, worst_n)
    if "n_sources" in df.columns:
        return _summarize_modern(df, keys)
    raise ValueError(
        "frame carries neither 'completeness' (stable) nor 'n_sources' "
        f"(modern); got {list(df.columns)}"
    )


def _summarize_stable(df: pd.DataFrame, keys: list[str], worst_n: int) -> pd.DataFrame:
    late_col = df["late_reporting"].astype(bool) if "late_reporting" in df else None
    live = (df[~late_col] if late_col is not None else df).copy()
    # `completeness` and `complete` arrive as object dtype because they mix
    # floats/bools with pd.NA. Coerce once here: pandas' numeric methods
    # (nsmallest, mean) reject object columns outright.
    live["_completeness"] = pd.to_numeric(live["completeness"], errors="coerce")
    live["_complete"] = live["complete"].map(
        lambda v: bool(v) if v is True or v is False else False
    )
    # Group over ALL live rows, not just the scored ones. A group whose
    # completeness is entirely unknown (e.g. an intensive whose inputs were
    # both absent) must still appear, reporting NA — dropping it would make a
    # whole variable silently vanish from the report.
    rows = []
    for gkey, grp in live.groupby(keys, dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        vals = grp["_completeness"].dropna() if len(grp) else pd.Series(dtype=float)
        complete_mask = grp["_complete"] if len(grp) else []
        row = dict(zip(keys, gkey))
        row.update({
            "n_cells": len(grp),
            "n_complete": int(sum(complete_mask)) if len(grp) else 0,
            "pct_complete": (float(sum(complete_mask)) / len(grp)) if len(grp) else pd.NA,
            "mean_completeness": float(vals.mean()) if len(vals) else pd.NA,
            "min_completeness": float(vals.min()) if len(vals) else pd.NA,
            # A group with no reporters at all is a different problem from a
            # partially-reported one, so it gets its own count.
            "n_cells_zero_reporters": int((grp["n_constituents"] == 0).sum())
            if "n_constituents" in grp else 0,
            "worst_ids": _worst_ids(grp, worst_n),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    # Districts reporting outside their lifetime are counted separately
    # rather than folded into the completeness figure. They are a different
    # kind of problem — not a district that failed to report, but data
    # arriving under a name that should no longer exist — and averaging the
    # two together would hide both.
    if late_col is not None:
        n_late = (
            df[late_col].groupby([df.loc[late_col, k] for k in keys]).size()
            if late_col.any() else None
        )
        out["n_late_reporting_rows"] = (
            _late_counts(df, late_col, keys, out) if n_late is not None else 0
        )
    return out.sort_values(keys).reset_index(drop=True)


def _worst_ids(grp: pd.DataFrame, worst_n: int) -> str:
    """The ids with the lowest completeness in this group, worst first."""
    id_col = "stable_id" if "stable_id" in grp.columns else "modern_id"
    if id_col not in grp.columns or "_completeness" not in grp.columns:
        return ""
    sub = grp[grp["_completeness"].notna()]
    if sub.empty:
        return ""
    worst = sub.nsmallest(worst_n, "_completeness")
    return ",".join(str(v) for v in worst[id_col])


def _late_counts(df, late_col, keys, out) -> list[int]:
    """Late-row counts aligned to ``out``'s group order."""
    late = df[late_col]
    counts = late.groupby(keys, dropna=False).size() if len(late) else {}
    result = []
    for _, r in out.iterrows():
        k = tuple(r[k] for k in keys)
        k = k[0] if len(k) == 1 else k
        try:
            result.append(int(counts.get(k, 0)))
        except AttributeError:
            result.append(0)
    return result


def _summarize_modern(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Modern summary: source counts and fraction evidence, no ratio.

    A modern unit has no "expected members" denominator — its value is a
    redistribution, not an aggregation — so reporting a completeness ratio
    would be inventing one. What a user can act on is how many historical
    units fed each cell and how thin the weakest fraction's evidence was.
    """
    rows = []
    for gkey, grp in df.groupby(keys, dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        n_src = pd.to_numeric(grp["n_sources"], errors="coerce")
        minc = pd.to_numeric(
            grp.get("min_n_common_observations", pd.Series(dtype=float)),
            errors="coerce",
        )
        row = dict(zip(keys, gkey))
        row.update({
            "n_cells": len(grp),
            "mean_n_sources": float(n_src.mean()) if n_src.notna().any() else pd.NA,
            "min_n_sources": float(n_src.min()) if n_src.notna().any() else pd.NA,
            "min_n_common_observations": float(minc.min()) if minc.notna().any() else pd.NA,
            "n_cells_nan_fraction": int(
                grp["has_nan_fraction"].fillna(False).astype(bool).sum()
            ) if "has_nan_fraction" in grp else 0,
            "worst_fraction_method": _worst_present_method(grp),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _worst_present_method(grp: pd.DataFrame) -> str:
    """The least-trustworthy fraction tier present in this group."""
    from .modern_algorithm import _METHOD_RANK

    if "fraction_method" not in grp.columns:
        return ""
    methods = [m for m in grp["fraction_method"].dropna().unique() if m]
    if not methods:
        return ""
    return max(methods, key=lambda m: _METHOD_RANK.get(m, 0))


def format_report(summary: pd.DataFrame, title: str = "Completeness") -> str:
    """One-screen text rendering, in the style of MatchProposal.summary()."""
    if summary.empty:
        return f"{title}: no rows."
    lines = [f"== {title} — {len(summary)} group(s) =="]
    if "pct_complete" in summary.columns:
        overall = summary["pct_complete"].astype(float).mean()
        lines.append(f"mean pct_complete across groups: {overall:.1%}")
        worst = summary.nsmallest(5, "pct_complete")
        lines.append("")
        lines.append("Least complete groups:")
        for _, r in worst.iterrows():
            label = " ".join(str(r[k]) for k in summary.columns[:2])
            lines.append(
                f"  {label:<40s} {float(r['pct_complete']):.1%} complete "
                f"of {int(r['n_cells'])} cells   worst: {r.get('worst_ids', '')}"
            )
    else:
        lines.append(
            f"mean sources per cell: "
            f"{summary['mean_n_sources'].astype(float).mean():.2f}"
        )
    return "\n".join(lines)
