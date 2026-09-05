"""FEWS NET FNID assignment.

This module turns the package's lineage + baseline into FEWS-spec FNIDs
for any country, at admin level 1 or 2. The FNID format is::

    <ISO><YYYY>A<LEVEL><CODE>

where ``CODE`` is a 2-char ``SS`` for admin1 and a 4-char ``SS+DD`` pair
for admin2. Codes are drawn from a 359-entry alphabet
(``"01".."99"`` then ``"A0".."Z9"``) so every slot fits in 2 characters.

Two code maps live here:

- ``build_admin1_code_map`` — every admin1_id that has ever existed in
  the country's lineage gets a unique 2-char SS via deterministic sort
  + the retired-code rule (a ceased state keeps its slot vacant; new
  states get fresh slots after the highest used). This produces the
  gappy ``01..60`` sequence visible in FEWS's distributed Vietnam admin-1 definitions.

- ``build_admin2_code_map`` — port of the legacy India ``fnid_assigner``.
  Each admin2 unit gets ``(SS, DD)`` where SS = origin admin1 (the
  territorial-history admin1, walking back through Split/Merge/
  Redistribute events to the earliest ancestor) and DD = a per-origin
  index. The retired-code rule applies within each origin state.

Both maps are stable per ``(iso, lineage)`` — re-running on the same
inputs yields identical codes — and the FNID year is taken per-row from
the stats data, not from the code map. A district observed in 2003 and
2015 gets the same SS+DD with different YYYY prefixes.

Public API:

- ``CODE_ALPHABET``, ``encode_code``, ``FNIDOverflowError``
- ``build_admin1_code_map(graph, baseline, *, iso) -> pd.DataFrame``
- ``build_admin2_code_map(graph, baseline, *, iso) -> pd.DataFrame``
- ``build_fnid(iso, year, level, ss, dd="", unit_type="A") -> str``
- ``parse_fnid(fnid) -> FnidParts`` / ``validate_fnid(fnid, *, level, iso)``
- ``assign_fnids(stats, code_map, *, iso, level, year_col, unit_id_col) -> pd.DataFrame``
"""

from __future__ import annotations

from typing import NamedTuple

from collections import defaultdict
from typing import Iterable

import pandas as pd

from .lineage import LineageGraph


# ---------------------------------------------------------------------------
# Code alphabet (359 entries: 01..99 then A0..Z9)
# ---------------------------------------------------------------------------

def _build_code_alphabet() -> list[str]:
    codes = [f"{i:02d}" for i in range(1, 100)]
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for digit in "0123456789":
            codes.append(f"{letter}{digit}")
    return codes


CODE_ALPHABET: list[str] = _build_code_alphabet()
CODE_CAPACITY: int = len(CODE_ALPHABET)  # 359


class FNIDOverflowError(Exception):
    """Raised when a slot exceeds the 2-char code capacity (359)."""


def encode_code(index: int) -> str:
    """Map a zero-based position (0..358) to its 2-char code."""
    if not 0 <= index < CODE_CAPACITY:
        raise FNIDOverflowError(
            f"code index {index} out of range [0, {CODE_CAPACITY})"
        )
    return CODE_ALPHABET[index]


def _fews_id_slots(ids: Iterable[str]) -> dict[str, int] | None:
    """Map each id to its canonical FEWS code slot (numeric suffix minus one).

    e.g. ``IN.ADM1.00011`` -> slot ``10`` (``SS '11'``). FEWS admin codes *are*
    the id ordinal, so keying the slot off the id — rather than a re-packed sort
    position — is the retired-code rule: a merged/removed id leaves its slot
    vacant and a new id claims its own slot, neither perturbing any other id's
    code.

    Returns ``None`` when the ids are not clean FEWS ordinals (non-numeric
    suffix, out of range, or colliding slots), signalling the caller to fall
    back to positional slot assignment — which keeps arbitrary "bring your own"
    admin ids working.
    """
    # Public contract, and note its SCOPE: it holds for the codes this
    # function assigns -- admin1 SS, and the origin-admin1 SS carried by an
    # admin2 code. Once such a code has been published it identifies that unit
    # for good, so a dissolved unit leaves its code empty rather than freeing
    # it up. Shuffling codes to close a gap would silently re-point every
    # published figure at a different place.
    #
    # It does NOT hold for an admin2's DD. `build_admin2_code_map` assigns DD
    # by sort position within the origin admin1 and never calls this function
    # for it, so DD re-packs whenever a unit is added or removed. That is
    # forced by the code space: DD holds 359 codes per state while ADM2 ids
    # are globally sequential (India reaches 01127), so the ordinals overflow
    # in most states. Do not read the guarantee below as covering districts.
    #
    # Where identifiers already end in a number, use that number as the code
    # so the two stay in step for good.
    slots: dict[str, int] = {}
    for a in ids:
        tail = str(a).rsplit(".", 1)[-1]
        try:
            slots[a] = int(tail) - 1
        except ValueError:
            # Not numbered — someone's own identifiers. Say so, and the
            # caller falls back to numbering them by position instead.
            return None
    vals = list(slots.values())
    # Also refuse if the numbers run past what a code can hold, or if two
    # districts want the same one. Both would produce codes that cannot be
    # told apart, which is worse than not using this shortcut at all.
    if any(not 0 <= s < CODE_CAPACITY for s in vals) or len(set(vals)) != len(vals):
        return None
    return slots


# ---------------------------------------------------------------------------
# FNID string construction
# ---------------------------------------------------------------------------

def build_fnid(
    iso: str, year: int, level: int, ss: str, dd: str = "", unit_type: str = "A"
) -> str:
    """Format a single FNID: ``IN2003A20107``, ``VN1991A101``, etc.

    For admin1 pass ``dd=""`` (or omit it). The level-1 FNID has 10 chars
    total; the level-2 FNID has 12.

    ``unit_type`` is the character after the year. ``"A"`` (administrative)
    is what this package emits. FEWS also publishes ``"R"`` for crop regions
    — Sri Lanka's statistics are keyed on ``LK1978R20101``-style ids — which
    the package cannot currently *generate* code maps for, but the parameter
    exists so parsing and round-tripping such ids does not require string
    surgery at every call site.
    """
    return f"{iso}{int(year):04d}{unit_type}{int(level)}{ss}{dd}"


class FnidParts(NamedTuple):
    """The decomposed pieces of an FNID. See :func:`parse_fnid`."""

    iso: str
    year: int
    unit_type: str
    level: int
    code: str

    @property
    def ss(self) -> str:
        """The admin1 slice of the code."""
        return self.code[:2]

    @property
    def dd(self) -> str:
        """The admin2 slice of the code, empty at level 1."""
        return self.code[2:4]


# Code width per admin level. Level 3+ is deliberately absent: the only
# real ADM3 FNIDs found (DR Congo's CD1997A30910) carry a 4-char code, the
# same width as ADM2, which contradicts a straightforward SS+DD+EE
# extension. Until FEWS confirms the layout, guessing would bake a wrong
# assumption into every id we emit.
_CODE_WIDTH = {1: 2, 2: 4}


def parse_fnid(fnid: str) -> FnidParts:
    """Decompose an FNID into its parts, or raise ``ValueError``.

    ``IN2015A20102`` -> ``FnidParts("IN", 2015, "A", 2, "0102")``

    Structural checks only: it does not verify the code corresponds to a
    real unit. Use it instead of positional slicing so a malformed id is
    reported rather than silently yielding nonsense — ``fnid[8:12]`` on a
    short string returns a short string, not an error.
    """
    s = str(fnid).strip()
    if len(s) < 9:
        raise ValueError(f"FNID too short to parse: {fnid!r}")
    iso, year_s, unit_type, level_s, code = s[:2], s[2:6], s[6], s[7], s[8:]
    if not iso.isalpha():
        raise ValueError(f"FNID {fnid!r}: expected a 2-letter ISO code, got {iso!r}")
    if not year_s.isdigit():
        raise ValueError(f"FNID {fnid!r}: expected a 4-digit year, got {year_s!r}")
    if not unit_type.isalpha():
        raise ValueError(
            f"FNID {fnid!r}: expected a unit-type letter at position 7, "
            f"got {unit_type!r}"
        )
    if not level_s.isdigit():
        raise ValueError(f"FNID {fnid!r}: expected a digit level, got {level_s!r}")
    return FnidParts(iso.upper(), int(year_s), unit_type.upper(), int(level_s), code)


def validate_fnid(
    fnid: str, *, level: int | None = None, iso: str | None = None
) -> FnidParts:
    """Parse and check an FNID's shape; return its parts or raise.

    Verifies the code width matches the level (2 chars at admin1, 4 at
    admin2) and, when given, that ``level`` and ``iso`` are what the caller
    expected. Levels beyond 2 raise :class:`NotImplementedError` rather than
    a guess — see ``_CODE_WIDTH``.
    """
    parts = parse_fnid(fnid)
    if parts.level not in _CODE_WIDTH:
        raise NotImplementedError(
            f"FNID {fnid!r} is admin level {parts.level}; this package supports "
            f"levels {sorted(_CODE_WIDTH)} only. The code width for deeper "
            "levels is unconfirmed (the only known ADM3 ids use a 4-char code, "
            "not the 6 an SS+DD+EE extension would imply)."
        )
    want = _CODE_WIDTH[parts.level]
    if len(parts.code) != want:
        raise ValueError(
            f"FNID {fnid!r}: admin level {parts.level} needs a {want}-char code, "
            f"got {len(parts.code)} ({parts.code!r})"
        )
    if level is not None and parts.level != level:
        raise ValueError(
            f"FNID {fnid!r} is admin level {parts.level}, expected {level}"
        )
    if iso is not None and parts.iso != iso.upper():
        raise ValueError(f"FNID {fnid!r} is for {parts.iso}, expected {iso.upper()}")
    return parts


# ---------------------------------------------------------------------------
# Admin1 code map
# ---------------------------------------------------------------------------

def _admin1_universe(graph: LineageGraph, baseline: pd.DataFrame) -> dict[str, str]:
    """Every admin1_id that has ever existed → its most-recent name.

    Sources:
      - Baseline: ``coarse_id``/``coarse_name`` columns (initial-year
        admin1s).
      - Lineage events: ``parent_coarse_id``/``parent_coarse_name`` and
        ``child_coarse_id``/``child_coarse_name`` (covers admin1s that
        were created or retired mid-window — Telangana, Uttarakhand,
        Chhattisgarh, Jharkhand for India).

    Name resolution: walk the events in chronological order; the latest
    name we see for an admin1_id wins. The baseline acts as the floor —
    if an event never updates the name, baseline's name stands.
    """
    names: dict[str, str] = {}

    if "coarse_id" in baseline.columns:
        for _, row in baseline.iterrows():
            cid = row.get("coarse_id")
            cname = row.get("coarse_name")
            if pd.notna(cid):
                names[str(cid)] = str(cname) if pd.notna(cname) else str(cid)

    ev = graph.events
    if "parent_coarse_id" in ev.columns:
        for _, row in ev.sort_values("event_year").iterrows():
            for id_col, name_col in (
                ("parent_coarse_id", "parent_coarse_name"),
                ("child_coarse_id", "child_coarse_name"),
            ):
                cid = row.get(id_col)
                cname = row.get(name_col)
                if pd.notna(cid):
                    names[str(cid)] = str(cname) if pd.notna(cname) else str(cid)

    return names


def _admin1_first_appearance(
    graph: LineageGraph,
    baseline: pd.DataFrame,
) -> dict[str, int]:
    """admin1_id → year it first appears (in baseline or as an event's child_coarse_id).

    The retired-code rule needs an ordering: admin1s that existed at
    baseline get their slots first; new admin1s that appear later get
    fresh slots. Within each cohort, IDs are sorted lexicographically
    for deterministic output.
    """
    first: dict[str, int] = {}

    if "coarse_id" in baseline.columns:
        # Baseline establishes the year-0 cohort. Use the smallest year
        # in the baseline (if multi-year) or treat all as the same year
        # if there's no year column — either way they're the founding set.
        if "year" in baseline.columns and not baseline["year"].isna().all():
            base_year = int(baseline["year"].min())
        else:
            base_year = graph.min_event_year - 1
        for cid in baseline["coarse_id"].dropna().unique():
            first[str(cid)] = base_year

    ev = graph.events
    if "child_coarse_id" in ev.columns:
        # An admin1 that never appears in the baseline first shows up
        # as a `child_coarse_id` of the event that creates it. The
        # earliest such event year is the admin1's birth year.
        for _, row in ev.sort_values("event_year").iterrows():
            cid = row.get("child_coarse_id")
            if pd.notna(cid):
                cid = str(cid)
                year = int(row["event_year"])
                if cid not in first or year < first[cid]:
                    first[cid] = year

    return first


def build_admin1_code_map(
    graph: LineageGraph,
    baseline: pd.DataFrame,
    *,
    iso: str,
) -> pd.DataFrame:
    """Assign each admin1_id its canonical 2-char SS (the retired-code rule).

    ``SS`` is the admin1_id's own ordinal slot (``_id_slot`` — the id's numeric
    suffix, 1-based, mapped through ``CODE_ALPHABET``). This is the FEWS
    convention: ``IN.ADM1.00011`` -> ``SS '11'``. Keying off the id rather than a
    re-packed sort position means a merged/removed admin1 leaves its slot vacant,
    and a new admin1 (Telangana 2014, Bihar's post-split successor, etc.) claims
    its own slot — neither perturbs any other admin1's code. For the bundled
    lineages (ids assigned densely in creation order) this reproduces the
    previous position-based codes exactly.

    For non-FEWS (arbitrary-string) admin1 ids the code falls back to the
    original positional scheme: sort by ``(first_appearance_year, admin1_id)``
    and assign ``CODE_ALPHABET[i]``.

    Output columns: ``ISO, ADMIN1_ID, ADMIN1_NAME, SS``.
    """
    universe = _admin1_universe(graph, baseline)

    slots = _fews_id_slots(universe.keys())
    if slots is None:
        first = _admin1_first_appearance(graph, baseline)
        if len(universe) > CODE_CAPACITY:
            raise FNIDOverflowError(
                f"Country {iso} has {len(universe)} admin1 ids over its lineage; "
                f"max {CODE_CAPACITY}. Extend the code alphabet."
            )
        ordered = sorted(universe, key=lambda a: (first.get(a, 0), a))
        slots = {a: i for i, a in enumerate(ordered)}

    rows = []
    for admin1_id in sorted(universe):
        rows.append({
            "ISO": iso,
            "ADMIN1_ID": admin1_id,
            "ADMIN1_NAME": universe[admin1_id],
            "SS": encode_code(slots[admin1_id]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Admin1 code map — single-level (admin1-only) lineages
# ---------------------------------------------------------------------------

def _unit_first_appearance(
    graph: LineageGraph,
    baseline: pd.DataFrame,
) -> dict[str, int]:
    """unit_id → first year it appears (baseline year floor, else first event).

    Only used to order ids for the positional-slot fallback, so an
    approximate first-appearance (event year for units seen only as an
    event parent/child) is fine — it just needs to be deterministic.
    """
    # Work out roughly when each district first appears, so they can be put
    # in a sensible order. Districts present from the start all share the
    # earliest year; the rest are dated from when they were created.
    out: dict[str, int] = {}
    if "year" in baseline.columns and not baseline["year"].isna().all():
        base_year = int(baseline["year"].min())
    elif not graph.events.empty:
        # No starting year recorded, so treat the founding districts as
        # predating the first change.
        base_year = graph.min_event_year - 1
    else:
        base_year = 0
    for _, row in baseline.iterrows():
        out[str(row["unit_id"])] = base_year
    # Then take the earliest change that mentions each district, working
    # through in order so the first mention wins.
    if not graph.events.empty:
        for _, row in graph.events.sort_values("event_year").iterrows():
            y = int(row["event_year"])
            for id_col in ("parent_id", "child_id"):
                uid = row.get(id_col)
                if pd.notna(uid):
                    uid = str(uid)
                    if uid not in out or y < out[uid]:
                        out[uid] = y
    return out


def build_admin1_only_code_map(
    graph: LineageGraph,
    baseline: pd.DataFrame,
    *,
    iso: str,
) -> pd.DataFrame:
    """Admin1 code map for a *single-level* lineage — each unit IS an admin1.

    The two-level :func:`build_admin1_code_map` derives its admin1 universe
    from the ``coarse_id``/``coarse_name`` columns (the admin2 → admin1
    attribution). Admin1-only countries have no coarse columns —
    the lineage's units are themselves the admin1s. This variant takes the
    universe directly from the unit ids (baseline plus every event
    parent/child), resolves each unit's most-recent name, then applies the
    *same* id-stable ``SS`` assignment (``_fews_id_slots``, with the
    positional fallback for arbitrary ids).

    Output columns match :func:`build_admin1_code_map`:
    ``ISO, ADMIN1_ID, ADMIN1_NAME, SS``.
    """
    # Some countries are only described down to state level, with no
    # districts beneath. There is then no separate state list to consult —
    # the units in the records ARE the states — so gather them directly.
    #
    # Each is labelled with the most recent name it went by: start from the
    # opening list, then let later changes overwrite as history is replayed
    # in order.
    names: dict[str, str] = {}
    for _, row in baseline.iterrows():
        uid = str(row["unit_id"])
        nm = row.get("name")
        names[uid] = str(nm) if pd.notna(nm) else uid
    if not graph.events.empty:
        for _, row in graph.events.sort_values("event_year").iterrows():
            for id_col, name_col in (
                ("parent_id", "parent_name"),
                ("child_id", "child_name"),
            ):
                uid = row.get(id_col)
                if pd.isna(uid):
                    continue
                nm = row.get(name_col)
                names[str(uid)] = str(nm) if pd.notna(nm) else str(uid)

    slots = _fews_id_slots(names.keys())
    if slots is None:
        if len(names) > CODE_CAPACITY:
            raise FNIDOverflowError(
                f"Country {iso} has {len(names)} admin1 ids over its lineage; "
                f"max {CODE_CAPACITY}. Extend the code alphabet."
            )
        first = _unit_first_appearance(graph, baseline)
        ordered = sorted(names, key=lambda a: (first.get(a, 0), a))
        slots = {a: i for i, a in enumerate(ordered)}

    return pd.DataFrame([
        {
            "ISO": iso,
            "ADMIN1_ID": uid,
            "ADMIN1_NAME": names[uid],
            "SS": encode_code(slots[uid]),
        }
        for uid in sorted(names)
    ])


# ---------------------------------------------------------------------------
# Admin2 code map (origin-state SS, retired DD per origin)
# ---------------------------------------------------------------------------

TERRITORIAL_EVENT_TYPES = {"Split", "Merge", "Redistribute"}


def _build_reverse_parent_map(graph: LineageGraph) -> dict[str, set[str]]:
    """child_id → set(parent_ids) restricted to territorial events.

    Coarse and NameChange rows are ignored — they don't move territory,
    so they don't affect a unit's territorial origin.
    """
    reverse: dict[str, set[str]] = defaultdict(set)
    for _, row in graph.territorial.iterrows():
        reverse[str(row["child_id"])].add(str(row["parent_id"]))
    return dict(reverse)


def _earliest_appearance_with_admin1(
    graph: LineageGraph,
    baseline: pd.DataFrame,
) -> dict[str, tuple[int, str, str]]:
    """unit_id → (earliest_year, admin1_id, admin1_name).

    The "earliest year" is the first time this admin2 appears either:
      - in the baseline (its initial-year admin1 is ``coarse_id``), or
      - as a parent or child in a lineage event (its admin1 at that
        event is ``parent_coarse_id`` or ``child_coarse_id``
        respectively).

    For a unit that appears in both, the earlier wins. For a unit that
    appears only in events, the first event row gives its admin1 — for
    splits this is the parent's admin1 if the unit appears as a parent,
    or the post-split admin1 if it appears as a child.
    """
    # Every district's code is built from the state it started life in, so
    # this works out that state and never revisits it. A district moved
    # between states later keeps its original code — the code is an
    # identifier, not a description of where the place is now.
    out: dict[str, tuple[int, str, str]] = {}

    # First choice: the state each district was recorded in at the start.
    if "coarse_id" in baseline.columns:
        if "year" in baseline.columns and not baseline["year"].isna().all():
            base_year = int(baseline["year"].min())
        else:
            base_year = graph.min_event_year - 1
        for _, row in baseline.iterrows():
            uid = str(row["unit_id"])
            cid = row.get("coarse_id")
            cname = row.get("coarse_name")
            if pd.isna(cid):
                continue
            cand = (base_year, str(cid), str(cname) if pd.notna(cname) else str(cid))
            if uid not in out or cand[0] < out[uid][0]:
                out[uid] = cand

    # Then districts created later, which the starting list cannot cover.
    # Working through in date order and keeping only the earliest mention
    # means a district that later moves between states does not overwrite
    # where it began.
    ev = graph.events
    if "parent_coarse_id" in ev.columns:
        for _, row in ev.sort_values("event_year").iterrows():
            year = int(row["event_year"])
            # Both sides of a change name a state, and either may be the
            # first time we hear of that district.
            for uid_col, cid_col, cname_col in (
                ("parent_id", "parent_coarse_id", "parent_coarse_name"),
                ("child_id", "child_coarse_id", "child_coarse_name"),
            ):
                uid = row.get(uid_col)
                cid = row.get(cid_col)
                cname = row.get(cname_col)
                if pd.isna(uid) or pd.isna(cid):
                    continue
                uid = str(uid)
                cand = (year, str(cid), str(cname) if pd.notna(cname) else str(cid))
                if uid not in out or cand[0] < out[uid][0]:
                    out[uid] = cand

    return out


def _build_origin_admin1_map(
    graph: LineageGraph,
    baseline: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """unit_id → (origin_admin1_id, origin_admin1_name).

    The origin admin1 is the admin1 of the unit's earliest territorial
    ancestor — walk back through Split/Merge/Redistribute parents,
    pick the ancestor with the earliest year of appearance, return its
    admin1 at that earliest-year snapshot.

    Consequence (India): Uttarakhand districts carry UP's origin SS,
    Telangana districts carry AP's, Chhattisgarh's carry MP's,
    Jharkhand's carry BR's.

    Coarse events do NOT change territorial origin (they reassign admin1
    without moving territory), so a Coarse event from AP→Telangana
    leaves the district's origin SS as AP.
    """
    reverse = _build_reverse_parent_map(graph)
    earliest = _earliest_appearance_with_admin1(graph, baseline)

    origin: dict[str, tuple[str, str]] = {}
    for unit_id in earliest:
        # Walk all ancestors via territorial parents; pick the ancestor
        # with the earliest snapshot year (lex-tie-break on unit_id).
        visited: set[str] = set()
        stack = [unit_id]
        best: tuple[int, str, str, str] | None = None
        while stack:
            uid = stack.pop()
            if uid in visited:
                continue
            visited.add(uid)
            if uid in earliest:
                year, admin1_id, admin1_name = earliest[uid]
                cand = (year, admin1_id, admin1_name, uid)
                if best is None or cand[0] < best[0] or (
                    cand[0] == best[0] and cand[3] < best[3]
                ):
                    best = cand
            for parent in reverse.get(uid, ()):
                if parent not in visited:
                    stack.append(parent)
        if best is None:
            continue
        _, admin1_id, admin1_name, _ = best
        origin[unit_id] = (admin1_id, admin1_name)

    return origin


def _latest_unit_name(graph: LineageGraph, baseline: pd.DataFrame) -> dict[str, str]:
    """unit_id → most-recent name we have for it.

    Used only for audit readability in the code map. Source order:
    baseline (low precedence), then events ordered by event_year (later
    wins). For a unit that's both parent and child of the same event,
    child_name wins (it's the post-event name).
    """
    latest: dict[str, str] = {}
    if {"unit_id", "name"}.issubset(baseline.columns):
        for _, row in baseline.iterrows():
            if pd.notna(row.get("name")):
                latest[str(row["unit_id"])] = str(row["name"])

    ev = graph.events.sort_values("event_year")
    for _, row in ev.iterrows():
        if pd.notna(row.get("parent_id")) and pd.notna(row.get("parent_name")):
            latest[str(row["parent_id"])] = str(row["parent_name"])
    for _, row in ev.iterrows():
        if pd.notna(row.get("child_id")) and pd.notna(row.get("child_name")):
            latest[str(row["child_id"])] = str(row["child_name"])
    return latest


def build_admin2_code_map(
    graph: LineageGraph,
    baseline: pd.DataFrame,
    *,
    iso: str,
) -> pd.DataFrame:
    """Assign deterministic ``(SS, DD)`` codes per admin2 ``unit_id``.

    Algorithm:
      1. Resolve each unit_id's origin admin1 (territorial history).
      2. SS: the origin admin1's canonical slot (``_id_slot``) — the same
         retired-code rule as ``build_admin1_code_map``, so admin2 SS always
         matches the origin's admin1 SS and a removed origin leaves its slot
         vacant instead of re-packing later origins.
      3. DD: within each origin admin1, sort unit_ids ascending; assign
         codes 0, 1, 2, ... from ``CODE_ALPHABET``.

    Returns a DataFrame with columns:
    ``ISO, ADMIN2_ID, ADMIN2_NAME, ORIGIN_ADMIN1_ID, ORIGIN_ADMIN1_NAME, SS, DD``.
    """
    origin = _build_origin_admin1_map(graph, baseline)
    latest_name = _latest_unit_name(graph, baseline)

    origin_ids = sorted({admin1_id for admin1_id, _ in origin.values()})
    slots = _fews_id_slots(origin_ids)
    if slots is None:
        if len(origin_ids) > CODE_CAPACITY:
            raise FNIDOverflowError(
                f"Too many origin admin1 codes for {iso}: {len(origin_ids)} "
                f"(max {CODE_CAPACITY}). Extend the code alphabet."
            )
        slots = {a: i for i, a in enumerate(origin_ids)}
    ss_by_origin: dict[str, str] = {a: encode_code(slots[a]) for a in origin_ids}

    by_origin: dict[str, list[str]] = defaultdict(list)
    for unit_id, (admin1_id, _) in origin.items():
        by_origin[admin1_id].append(unit_id)

    dd_by_unit: dict[str, str] = {}
    for admin1_id, unit_ids in by_origin.items():
        unit_ids.sort()
        if len(unit_ids) > CODE_CAPACITY:
            origin_name = next(
                (n for a, n in origin.values() if a == admin1_id),
                admin1_id,
            )
            raise FNIDOverflowError(
                f"Origin admin1 {admin1_id} ({origin_name}) has {len(unit_ids)} "
                f"admin2 ids; max {CODE_CAPACITY}. Extend the code alphabet."
            )
        for i, uid in enumerate(unit_ids):
            dd_by_unit[uid] = encode_code(i)

    rows = []
    for unit_id, (admin1_id, admin1_name) in origin.items():
        rows.append({
            "ISO": iso,
            "ADMIN2_ID": unit_id,
            "ADMIN2_NAME": latest_name.get(unit_id, ""),
            "ORIGIN_ADMIN1_ID": admin1_id,
            "ORIGIN_ADMIN1_NAME": admin1_name,
            "SS": ss_by_origin[admin1_id],
            "DD": dd_by_unit[unit_id],
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["ORIGIN_ADMIN1_ID", "ADMIN2_ID"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# FNID assignment to a stats DataFrame
# ---------------------------------------------------------------------------

def assign_fnids(
    stats: pd.DataFrame,
    code_map: pd.DataFrame,
    *,
    iso: str,
    level: int,
    year_col: str = "year",
    unit_id_col: str = "unit_id",
    fnid_col: str = "FNID",
) -> pd.DataFrame:
    """Add an ``FNID`` column to ``stats`` by joining on ``unit_id_col``.

    ``code_map`` is the output of ``build_admin1_code_map`` (level=1) or
    ``build_admin2_code_map`` (level=2). The join key is auto-detected
    from the code map: ``ADMIN1_ID`` for level=1, ``ADMIN2_ID`` for level=2.

    Rows whose unit id is missing from the code map (e.g. a stats row
    referencing a unit that was filtered out of the lineage) get an
    empty FNID. The original frame is not mutated.
    """
    if level == 1:
        key_col = "ADMIN1_ID"
        cm = code_map[[key_col, "SS"]].copy()
        cm["DD"] = ""
    elif level == 2:
        key_col = "ADMIN2_ID"
        cm = code_map[[key_col, "SS", "DD"]].copy()
    else:
        raise ValueError(f"unsupported level={level}; expected 1 or 2")

    cm = cm.drop_duplicates(subset=[key_col])
    out = stats.merge(
        cm, how="left",
        left_on=unit_id_col, right_on=key_col,
        suffixes=("", "__from_codemap"),
    )
    if f"{key_col}__from_codemap" in out.columns:
        out = out.drop(columns=[f"{key_col}__from_codemap"])
    if key_col != unit_id_col and key_col in out.columns:
        out = out.drop(columns=[key_col])

    def _row_fnid(r: pd.Series) -> str:
        ss = r["SS"]
        dd = r["DD"]
        year = r[year_col]
        if pd.isna(ss) or pd.isna(year):
            return ""
        return build_fnid(iso, int(year), level, str(ss), str(dd) if pd.notna(dd) else "")

    out[fnid_col] = out.apply(_row_fnid, axis=1)
    return out.drop(columns=["SS", "DD"])
