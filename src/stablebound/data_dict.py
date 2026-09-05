"""Stats data dictionary.

A :class:`DataDictionary` captures everything the package needs to
know about a long-form stats file's *contents* (as opposed to its
file path). It is passed to ``aggregate_stats(...)`` on either product
to declare:

    - which columns map to canonical (``stats_columns``),
    - which variables are extensive and should be summed
      (``extensive``),
    - which variables are intensive and derived after aggregation
      (``intensive``),
    - which variables should be dropped (``ignore``).

Defined once, reused across stable + modern + multiple stats files
that share the same conventions. Construction validates shape so
mistakes surface immediately rather than during the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataDictionary:
    """Schema declaration for a long-form stats file.

    Fields:

        stats_columns: ``{user_column: canonical_column}`` rename map.
            The canonical column names are
            ``unit_id, year, season, variable, value``. Omit any
            user-side names that already match.

        extensive: variable values to keep and sum during aggregation.
            ``None`` (default) means "sum all variables", except those
            listed in ``ignore``. If non-None, variables not in this
            list are dropped (along with anything in ``ignore``).
            Variables that appear as numerator/denominator of an
            ``intensive`` entry are automatically retained even if
            they aren't listed here.

        intensive: ``{intensive_name: (numerator_var, denominator_var)}``.
            Computed post-aggregation by ``derive_intensive``. The
            numerator and denominator must be summable extensives in
            the same frame.

        ignore: variable values to drop entirely. Useful for things
            like price columns that aren't analytically interesting.
    """

    stats_columns: dict[str, str] = field(default_factory=dict)
    extensive: list[str] | None = None
    intensive: dict[str, tuple[str, str]] = field(default_factory=dict)
    ignore: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Each intensive entry must be a 2-tuple. Same shape check as
        # the legacy Config used.
        for name, pair in self.intensive.items():
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"intensive[{name!r}] must be a (numerator_var, "
                    f"denominator_var) tuple; got {pair!r}."
                )

        # ``extensive`` and ``ignore`` should not overlap — that's a
        # contradiction (sum the variable AND drop it).
        if self.extensive is not None:
            overlap = set(self.extensive) & set(self.ignore)
            if overlap:
                raise ValueError(
                    f"extensive and ignore overlap: {sorted(overlap)}. "
                    f"A variable cannot be both summed and dropped."
                )

    # --- Helpers -----------------------------------------------------

    def intensive_input_vars(self) -> set[str]:
        """All variables referenced as a numerator or denominator."""
        s: set[str] = set()
        for num, den in self.intensive.values():
            s.add(num)
            s.add(den)
        return s

    def variables_to_keep(self) -> set[str] | None:
        """Variables to retain during aggregation.

        Returns ``None`` (sentinel for "keep all") when ``extensive``
        is unset. Otherwise the union of ``extensive`` plus any
        intensive inputs.
        """
        if self.extensive is None:
            return None
        return set(self.extensive) | self.intensive_input_vars()


__all__ = ["DataDictionary"]
