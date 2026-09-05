"""India DESAGRI crop-statistics → lineage ADM2_ID matcher.

Country submodule (``stablebound.india``). Resolves each raw stats row's
``(Admin 1, Admin 2, Year)`` triple to a canonical lineage ``ADM2_ID`` with a
year-aware, multi-pass name matcher: exact / normalized / fuzzy matching
against the per-year admin snapshot, a manual-override table for known aliases,
and a U2 ancestor walk that maps a name reported under a post-split id back to
the year-compatible ancestor.

Public entry point::

    match_stats_to_lineage(stats_df, snapshots, name_log, lineage=lineage)
        -> (match_map, unmatched, match_log, ancestor_diag)

Moved into the package 2026-07 from the legacy standalone script
``scripts/india/.../2_stats_pipeline_IN_v2.py`` so the India stats pipeline has
no required code outside ``stablebound``. Only the matcher moved; that script's
stable-boundary aggregation is superseded by ``stablebound.stats`` /
``stablebound.boundary`` and was left behind.
"""

import re

import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher


# Columns in stats file
STATS_STATE_COL = "Admin 1"
STATS_DIST_COL = "Admin 2"
STATS_YEAR_COL = "Year"
STATS_SEASON_COL = "Season"
STATS_CROP_COL = "Source crop"
STATS_AREA_PLANTED_COL = "Area Planted: ha"
STATS_AREA_HARVESTED_COL = "Area Harvested: ha"
STATS_YIELD_REPORTED_COL = "Yield: MT/ha (Reported)"
STATS_YIELD_CALC_COL = "Yield: MT/ha (Calculated)"
STATS_PRODUCTION_COL = "Quantity Produced: MT"

TERRITORY_EVENTS = {"Split", "Merge", "Redistribute"}


# ============================================================
# STATE NAME NORMALIZATION
# ============================================================

# Maps stats state names to snapshot state names
STATE_NAME_MAP = {
    "Dadra and Nagar Haveli": "Dadra And Nagar Haveli",
    "Daman and Diu": "Daman And Diu",
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Delhi": "NCT Of Delhi",
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand",
    "Pondicherry": "Pondicherry U.T.",
    "Puducherry": "Puducherry",  # State name changed from Pondicherry U.T. to Puducherry in 2007
}

def normalize_state(state_name):
    """Normalize a stats state name to match snapshot conventions."""
    s = state_name.strip()
    return STATE_NAME_MAP.get(s, s)


# ============================================================
# DISTRICT NAME NORMALIZATION & MATCHING
# ============================================================

def parse_stats_district(admin2_str):
    """Parse 'District Name (XX)' → (district_name, state_abbrev)."""
    m = re.match(r'^(.+?)\s*\(([A-Z]{2,3})\)$', str(admin2_str))
    if m:
        return m.group(1).strip(), m.group(2)
    return str(admin2_str).strip(), None


def normalize_district(name):
    """Normalize district name for matching."""
    s = str(name).lower().strip()
    s = s.replace(' district', '').replace(' dist.', '').replace(' dist', '')
    s = s.replace('-', ' ').replace('  ', ' ')
    for pat in [r'\bdr\.\s*', r'\bdr\s+', r'\bsri\s+', r'\bshri\s+']:
        s = re.sub(pat, '', s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return s.strip()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Hand-written translations from the name a district goes by in the published
# statistics to the name the boundary records use for the same place. Each one
# was added because a real row failed to match and a person checked what it
# should have been; none is guessable, which is why they are listed rather
# than derived.
#
# Three kinds of entry, and the difference matters when maintaining this:
#   - different spellings of one name ("Ahmedabad" / "Ahmadabad"), which are
#     safe and permanent;
#   - a district renamed since ("Bengaluru" for "Bangalore"), which the rename
#     records ought to cover and which can be removed once they do;
#   - names the statistics use for something that is not a district at all,
#     handled at the end of the table.
#
# The state is always checked alongside these, so an entry can never move a
# figure into a different state by accident.
STATS_MANUAL_OVERRIDES = {
    # Spelling variants
    "ahmedabad": "Ahmadabad",
    "ahmednagar": "Ahmad Nagar",
    "angul": "Anugul",
    "anuppur": "Anupur",
    "bagpat": "Baghpat",
    "balasore": "Baleshwar",
    "balrampur ramanujganj": "Balrampur",
    "banaskantha": "Banas Kantha",
    "bara banki": "Barabanki",
    "beed": "Bid",
    "bengaluru rural": "Bangalore Rural",
    "bengaluru urban": "Bangalore",
    "budgam": "Badgam",
    "buldhana": "Buldana",
    "chikballapur": "Chikkaballapura",
    "chittorgarh": "Chittaurgarh",
    "dahod": "Dohad",
    "dangs": "The Dangs",
    "darjeeling": "Darjiling",
    "dindigul": "Dindigul Anna",
    "b r ambedkar konaseema": "Konaseema",
    "br ambedkar konaseema": "Konaseema",
    "gaurella pendra marwahi": "Gaurela-Pendra-Marwahi",
    "haridwar": "Hardwar",
    "hazaribagh": "Hazaribag",
    "hooghly": "Hugli",
    "howrah": "Haora",
    "jagatsinghpur": "Jagatsinghapur",
    "kanchipuram": "Kancheepuram",
    "kanyakumari": "Kanniyakumari",
    "kasargod": "Kasaragod",
    "khairagarh": "Khairagarh-Chhuikhadan-Gandai",
    "lahaul and spiti": "Lahul And Spiti",
    "lakhimpur kheri": "Kheri",
    "malda": "Maldah",
    "manendragarh": "Manendragarh-Chirmiri-Bharatpur",
    "manendragarh chirmiri bharatpur mcb": "Manendragarh-Chirmiri-Bharatpur",
    "medchalmalkajgiri": "Medchal-Malkajgiri",
    "medchal malkajgiri": "Medchal-Malkajgiri",
    "mohla manpur": "Mohla-Manpur-Ambagarh Chowki",
    "mohla manpur ambagarh chowki": "Mohla-Manpur-Ambagarh Chowki",
    "mumbai city": "Mumbai",
    "muzzafarpur": "Muzaffarpur",
    "nabarangpur": "Nabarangapur",
    "narsinghpur": "Narsimhapur",
    "nilgiris": "The Nilgiris",
    "north twentyfour paraganas": "North Twenty Four Parganas",
    "panchmahal": "Panch Mahals",
    "paschim champaran": "Pashchim Champaran",
    "paschim singhbhum": "Pashchimi Singhbhum",
    "poonch": "Punch",
    "puducherry": "Pondicherry U.T.",
    "purba singhbhum": "Purbi Singhbhum",
    "purulia": "Puruliya",
    "raebareli": "Rae Bareli",
    "raigad": "Raigarh",
    "rajouri": "Rajauri",
    "ranga reddy": "Rangareddi",
    "ri bhoi": "Ri Bhoi",
    "rudra prayag": "Rudraprayag",
    "seraikela kharsawan": "Saraikela-Kharsawan",
    "shahid bhagat singh nagar": "Shaheed Bhagat Singh Nagar",
    "shakti": "Shakti Nagar",
    "shravasti": "Shrawasti",
    "siddharth nagar": "Siddharthnagar",
    "sivasagar": "Sibsagar",
    "south andamans": "South Andaman",
    "south twentyfour paraganas": "South Twenty Four Parganas",
    "thiruvarur": "Tiruvarur",
    "tiruchirappalli": "Tiruchchirappalli",
    "tiruvallur": "Tiruvallur",
    "tiruvannamalai": "Tiruvannamalai",
    "uttar kannad": "Uttara Kannada",
    "villupuram": "Viluppuram",
    "yamuna nagar": "Yamunanagar",
    "ysr kadapa": "Y.S.R.",
    "north and middle andaman": "North And Middle Andaman",
    # Aggregate to filter
    "delhi_total": "__FILTER__",
    "delhitotal": "__FILTER__",
}


# ============================================================
# STAGE 1: NAME MATCHING
# ============================================================

# ------------------------------------------------------------------
# Year-validation helpers (added 2026-06-05 for U2 fix).
# After each match pass picks an ADM2_ID, verify the ID was alive in
# the row's year. If not, walk the lineage's ancestor chain for a
# year-compatible alternative.
# ------------------------------------------------------------------

def _build_year_alive_index(snapshots) -> dict[str, set[int]]:
    """{ADM2_ID: set(years in which the ID appears in any snapshot)}."""
    out: dict[str, set[int]] = defaultdict(set)
    for _, row in snapshots.iterrows():
        out[row['ADM2_ID']].add(int(row['YEAR']))
    return dict(out)


def _build_parents_map(lineage) -> dict[str, set[str]]:
    """{child_id: set(parent_ids)} from lineage events (excluding self-loops).

    Raises on the wrong column case rather than returning an empty map.

    This used ``e.get('PARENT_ID')``, and ``Series.get`` returns None for a
    missing key instead of raising. Handed a frame using the package's
    canonical lowercase columns -- which is what ``LineageGraph.events`` and
    ``read_relationship_table`` both produce -- every row failed the notna
    check and the function returned ``{}`` with no error at all.

    The ancestor walk then found no candidates for any row, so
    ``match_stats_to_lineage`` tagged them ``no_ancestor`` and dropped them.
    A silently empty index does not fail; it quietly discards data, which is
    the worst way for this to go wrong.
    """
    required = {'PARENT_ID', 'CHILD_ID'}
    missing = required - set(lineage.columns)
    if missing:
        lower = {c.lower() for c in lineage.columns}
        hint = (
            "  The frame appears to use the canonical lowercase schema. This "
            "matcher expects the UPPERCASE source schema; rename with "
            "`df.rename(columns={'parent_id': 'PARENT_ID', 'child_id': 'CHILD_ID'})` "
            "before calling."
            if {c.lower() for c in required} <= lower else
            f"  Columns present: {sorted(lineage.columns)}"
        )
        raise KeyError(
            f"_build_parents_map requires {sorted(required)}; missing "
            f"{sorted(missing)}.\n{hint}"
        )

    out: dict[str, set[str]] = defaultdict(set)
    for _, e in lineage.iterrows():
        pid, cid = e['PARENT_ID'], e['CHILD_ID']
        if pd.notna(pid) and pd.notna(cid) and pid != cid:
            out[cid].add(pid)
    return dict(out)


# The ancestor walk itself is shared with the generic matcher rather than
# copied. Two copies of the code that decides *which unit a row's data lands
# on* is precisely the drift that produces silent misattribution: the two
# would diverge on a fix applied to one and not the other, and nothing would
# fail. Before consolidating, the two sets were checked equivalent over 9,000
# randomised inputs (0 mismatches); importing makes divergence impossible
# rather than merely unlikely.
#
# Only the index *builders* stay here, because they differ in input shape:
# these read India's uppercase snapshot/lineage DataFrames, while the generic
# ones read the matcher's internal lookup structures.
from ..match import (  # noqa: E402
    _earliest_created_ancestor,
    _walk_year_compat_ancestors,
    _year_compatible,
)


def build_snapshot_lookup(snapshots):
    """
    Build a year-aware lookup: (normalized_name, normalized_state) → ADM2_ID
    for each year. Also build an all-years lookup for fallback.

    Returns:
      year_lookup: {year: {(norm_name, norm_state): (ADM2_ID, ADM2_NAME)}}
      all_lookup: {(norm_name, norm_state): (ADM2_ID, ADM2_NAME)}
      name_only_lookup: {norm_name: [(ADM2_ID, ADM2_NAME, ADM1_NAME)]}
    """
    year_lookup = defaultdict(dict)
    all_lookup = {}
    name_only_lookup = defaultdict(list)

    for _, row in snapshots.iterrows():
        norm_name = normalize_district(row['ADM2_NAME'])
        norm_state = row['ADM1_NAME'].strip().lower()
        year = int(row['YEAR'])
        entry = (row['ADM2_ID'], row['ADM2_NAME'])

        year_lookup[year][(norm_name, norm_state)] = entry
        all_lookup[(norm_name, norm_state)] = entry
        name_only_lookup[norm_name].append((row['ADM2_ID'], row['ADM2_NAME'], row['ADM1_NAME']))

    # Deduplicate name_only_lookup
    for k in name_only_lookup:
        name_only_lookup[k] = list(set(name_only_lookup[k]))

    return dict(year_lookup), all_lookup, dict(name_only_lookup)


def build_name_change_reverse(name_log, snapshots):
    """Build old_name → (ADM2_ID, current_name) lookup from name change log."""
    lookup = {}
    for _, row in name_log.iterrows():
        if row['LEVEL'] == 2:
            unit_id = str(row['UNIT_ID']).strip()
            old_norm = normalize_district(str(row['OLD_OFFICIAL_NAME']))
            new_name = str(row['NEW_OFFICIAL_NAME'])
            lookup[old_norm] = (unit_id, new_name)
    return lookup


def match_stats_to_lineage(stats_districts, snapshots, name_log, lineage=None):
    """
    Match each unique (Admin 1, Admin 2, Year) combo from stats to a lineage ADM2_ID.

    YEAR-AWARE: For stats year Y, first try matching against snapshot year Y,
    then fall back to all-years lookup. This ensures that e.g. "Anantapur" in
    year 2000 maps to the pre-Telangana ID, not the post-reorganization one.

    U2 fix (2026-06-05): when a year-unaware pass (exact_all, name_only,
    fuzzy) returns an ID that did NOT exist in stats_year, walk the
    lineage ancestor chain for a year-compatible alternative.
        - 1 compatible ancestor → use it; method tagged "+ancestor_walk"
        - 2+ compatible ancestors → pick earliest-created, tag "+ambiguous_pick_earliest"
        - 0 compatible ancestors → mark unmatched (no_ancestor)
    Requires `lineage` for the ancestor walk; if not provided, year-
    validation is skipped (legacy behavior).

    Returns:
      match_map: {(Admin1, Admin2, Year): (ADM2_ID, match_method)}
      unmatched: [(Admin1, Admin2, Year, best_candidate, score)]
      match_log: full audit log (includes year-validation tags)
      ancestor_diag: [{...}]  per-row diagnostic for ambiguous/no_ancestor cases
    """
    year_lookup, all_lookup, name_only_lookup = build_snapshot_lookup(snapshots)
    ncl_lookup = build_name_change_reverse(name_log, snapshots)

    # U2 fix: build year-validation structures if lineage was provided
    if lineage is not None:
        year_alive_index = _build_year_alive_index(snapshots)
        parents_map = _build_parents_map(lineage)
        do_year_validate = True
    else:
        year_alive_index = {}
        parents_map = {}
        do_year_validate = False

    ancestor_diag: list[dict] = []

    def _validate(adm2_id, method, stats_state, stats_dist_raw, stats_year, dist_name):
        """Returns (adm2_id_to_use, method_tag, status, candidates_list).
        status in {'ok', 'remapped', 'ambiguous', 'no_ancestor'}.
        Logs to ancestor_diag for ambiguous/no_ancestor cases."""
        if not do_year_validate:
            return adm2_id, method, 'ok', []
        # The district we landed on existed when the figure was reported, so
        # the match stands as-is.
        if _year_compatible(adm2_id, stats_year, year_alive_index):
            return adm2_id, method, 'ok', []
        # It didn't exist yet — usually an old figure carrying a name that
        # now belongs to a district created later. Walk back up the family
        # tree for the district that held that ground at the time.
        candidates = _walk_year_compat_ancestors(
            adm2_id, stats_year, parents_map, year_alive_index)
        # Exactly one fits: move the figure there and mark that we did, so
        # the change is visible rather than silent.
        if len(candidates) == 1:
            return candidates[0], f"{method}+ancestor_walk", 'remapped', candidates
        # Several fit. Take the oldest and write the alternatives to the
        # diagnostic file, since this was a judgement call, not a fact.
        if len(candidates) > 1:
            chosen = _earliest_created_ancestor(candidates, year_alive_index)
            ancestor_diag.append({
                'stats_state': stats_state, 'stats_district': stats_dist_raw,
                'year': stats_year, 'parsed_name': dist_name,
                'original_match_id': adm2_id, 'original_method': method,
                'status': 'ambiguous_picked_earliest',
                'chosen_id': chosen,
                'all_candidates': ';'.join(sorted(candidates)),
                'n_candidates': len(candidates),
            })
            return chosen, f"{method}+ambiguous_pick_earliest", 'ambiguous', candidates
        # Nothing fits at all, so we cannot say where this figure belongs.
        # Give it up rather than attach it to a district that did not exist,
        # and record why so the row can be chased down later.
        ancestor_diag.append({
            'stats_state': stats_state, 'stats_district': stats_dist_raw,
            'year': stats_year, 'parsed_name': dist_name,
            'original_match_id': adm2_id, 'original_method': method,
            'status': 'no_ancestor_dropped',
            'chosen_id': '',
            'all_candidates': '',
            'n_candidates': 0,
        })
        return None, f"{method}+no_ancestor", 'no_ancestor', []

    # Get unique (state, district, year) combos
    unique_triples = stats_districts[
        [STATS_STATE_COL, STATS_DIST_COL, STATS_YEAR_COL]
    ].drop_duplicates()

    match_map = {}
    unmatched = []
    match_log = []

    for _, row in unique_triples.iterrows():
        stats_state = row[STATS_STATE_COL]
        stats_dist_raw = row[STATS_DIST_COL]
        stats_year = int(row[STATS_YEAR_COL])

        dist_name, state_abbr = parse_stats_district(stats_dist_raw)
        dist_norm = normalize_district(dist_name)
        snap_state = normalize_state(stats_state).lower()

        key = (stats_state, stats_dist_raw, stats_year)

        # Check for filter
        if dist_norm in STATS_MANUAL_OVERRIDES and STATS_MANUAL_OVERRIDES[dist_norm] == "__FILTER__":
            match_map[key] = ("__FILTER__", "filter")
            match_log.append({
                'stats_state': stats_state, 'stats_district': stats_dist_raw,
                'year': stats_year, 'parsed_name': dist_name,
                'adm2_id': '__FILTER__', 'snap_name': '', 'method': 'filter'
            })
            continue

        matched = False

        # Compare against the district list as it stood in the year the
        # figure is from. If we have no list for that exact year, use the
        # most recent earlier one — the districts as they were last known,
        # rather than as they became later.
        available_years = sorted(year_lookup.keys())
        use_year = stats_year
        if use_year not in year_lookup:
            candidates = [y for y in available_years if y <= stats_year]
            use_year = candidates[-1] if candidates else available_years[0]

        yr_lookup = year_lookup.get(use_year, {})

        # What follows is a ladder of eight attempts, from the most certain
        # to the least. Each one either claims the row and stops, or hands it
        # down to the next.
        #
        # Every rung has the same shape, so it is worth reading once here
        # rather than eight times below: look the name up one particular way;
        # if it hits, check the district actually existed in the year of the
        # figure, walking back to an earlier district if not; then record
        # either the match or the reason it failed. A row that finds a
        # district but can't be traced back to one that existed at the time
        # is recorded as unmatched rather than attached to the wrong place.

        # 1. The name and state match a district on the list for that year.
        #    Nothing to interpret.
        if (dist_norm, snap_state) in yr_lookup:
            adm2_id, snap_name = yr_lookup[(dist_norm, snap_state)]
            new_id, new_method, status, _ = _validate(
                adm2_id, "exact_year", stats_state, stats_dist_raw, stats_year, dist_name)
            if status == 'no_ancestor':
                unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                })
            else:
                match_map[key] = (new_id, new_method)
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                })
            matched = True
            continue

        # 2. Someone has written down by hand what this name really refers
        #    to. Try that against the year's list, then against any year.
        if dist_norm in STATS_MANUAL_OVERRIDES:
            override_name = STATS_MANUAL_OVERRIDES[dist_norm]
            override_norm = normalize_district(override_name)
            if (override_norm, snap_state) in yr_lookup:
                adm2_id, snap_name = yr_lookup[(override_norm, snap_state)]
                new_id, new_method, status, _ = _validate(
                    adm2_id, "manual_year", stats_state, stats_dist_raw, stats_year, dist_name)
                if status == 'no_ancestor':
                    unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                    })
                else:
                    match_map[key] = (new_id, new_method)
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                    })
                matched = True
                continue
            # Try all-years with state
            if (override_norm, snap_state) in all_lookup:
                adm2_id, snap_name = all_lookup[(override_norm, snap_state)]
                new_id, new_method, status, _ = _validate(
                    adm2_id, "manual_all", stats_state, stats_dist_raw, stats_year, dist_name)
                if status == 'no_ancestor':
                    unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                    })
                else:
                    match_map[key] = (new_id, new_method)
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                    })
                matched = True
                continue
            # Try without state
            if override_norm in name_only_lookup:
                candidates = name_only_lookup[override_norm]
                if len(candidates) == 1:
                    adm2_id, snap_name, _ = candidates[0]
                    new_id, new_method, status, _x = _validate(
                        adm2_id, "manual_nostate", stats_state, stats_dist_raw, stats_year, dist_name)
                    if status == 'no_ancestor':
                        unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                        match_log.append({
                            'stats_state': stats_state, 'stats_district': stats_dist_raw,
                            'year': stats_year, 'parsed_name': dist_name,
                            'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                        })
                    else:
                        match_map[key] = (new_id, new_method)
                        match_log.append({
                            'stats_state': stats_state, 'stats_district': stats_dist_raw,
                            'year': stats_year, 'parsed_name': dist_name,
                            'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                        })
                    matched = True
                    continue

        # 3. Same as the first attempt, but against every year at once. This
        #    catches a district that existed at some point but not in the
        #    particular year we compared against.
        if (dist_norm, snap_state) in all_lookup:
            adm2_id, snap_name = all_lookup[(dist_norm, snap_state)]
            new_id, new_method, status, _ = _validate(
                adm2_id, "exact_all", stats_state, stats_dist_raw, stats_year, dist_name)
            if status == 'no_ancestor':
                unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                })
            else:
                match_map[key] = (new_id, new_method)
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                })
            matched = True
            continue

        # 4. The name is one a district used to go by, according to the list
        #    of official renames.
        if dist_norm in ncl_lookup:
            adm2_id, new_name = ncl_lookup[dist_norm]
            new_id, new_method, status, _ = _validate(
                adm2_id, "name_change", stats_state, stats_dist_raw, stats_year, dist_name)
            if status == 'no_ancestor':
                unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, new_name, 'no_ancestor'))
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': '', 'snap_name': new_name, 'method': 'UNMATCHED:no_ancestor'
                })
            else:
                match_map[key] = (new_id, new_method)
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': new_id, 'snap_name': new_name, 'method': new_method
                })
            matched = True
            continue

        # 5. Drop the state and match on the district name alone — but only
        #    when exactly one district in the country goes by it, so there is
        #    no risk of picking the wrong one. This is what rescues rows
        #    where the state is missing or recorded wrongly.
        if dist_norm in name_only_lookup:
            candidates = name_only_lookup[dist_norm]
            if len(candidates) == 1:
                adm2_id, snap_name, _ = candidates[0]
                new_id, new_method, status, _x = _validate(
                    adm2_id, "name_only", stats_state, stats_dist_raw, stats_year, dist_name)
                if status == 'no_ancestor':
                    unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                    })
                else:
                    match_map[key] = (new_id, new_method)
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                    })
                matched = True
                continue

        # 6. Nothing matched outright, so start guessing — but only within
        #    the right state, which keeps a misspelling from being answered
        #    by a similarly-named district on the other side of the country.
        #    Try the year's own list first.
        if not matched:
            best_score = 0
            best_match = None

            for (sn_norm, sn_state), (sid, sname) in yr_lookup.items():
                if sn_state == snap_state:
                    score = similarity(dist_norm, sn_norm)
                    if score > best_score:
                        best_score = score
                        best_match = (sid, sname)

            if best_score >= 0.75 and best_match:
                adm2_id, snap_name = best_match
                base_method = f"fuzzy_year({best_score:.2f})"
                new_id, new_method, status, _ = _validate(
                    adm2_id, base_method, stats_state, stats_dist_raw, stats_year, dist_name)
                if status == 'no_ancestor':
                    unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                    })
                else:
                    match_map[key] = (new_id, new_method)
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': new_id, 'snap_name': snap_name, 'method': new_method
                    })
                matched = True
                continue

            # 7. Same guess, widened to districts from any year. Note the
            #    running best score carries over, so a better match found
            #    here has to beat what the year's own list already offered.
            for (sn_norm, sn_state), (sid, sname) in all_lookup.items():
                if sn_state == snap_state:
                    score = similarity(dist_norm, sn_norm)
                    if score > best_score:
                        best_score = score
                        best_match = (sid, sname)

            if best_score >= 0.75 and best_match:
                adm2_id, snap_name = best_match
                base_method = f"fuzzy_all({best_score:.2f})"
                new_id, new_method, status, _ = _validate(
                    adm2_id, base_method, stats_state, stats_dist_raw, stats_year, dist_name)
                if status == 'no_ancestor':
                    unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name, snap_name, 'no_ancestor'))
                    match_log.append({
                        'stats_state': stats_state, 'stats_district': stats_dist_raw,
                        'year': stats_year, 'parsed_name': dist_name,
                        'adm2_id': '', 'snap_name': snap_name, 'method': 'UNMATCHED:no_ancestor'
                    })
                    matched = True
                    continue
                adm2_id = new_id  # use the validated id for the rest of the assignment
                match_map[key] = (new_id, new_method)
                match_log.append({
                    'stats_state': stats_state, 'stats_district': stats_dist_raw,
                    'year': stats_year, 'parsed_name': dist_name,
                    'adm2_id': new_id, 'snap_name': snap_name,
                    'method': new_method
                })
                matched = True
                continue

        # Unmatched
        if not matched:
            unmatched.append((stats_state, stats_dist_raw, stats_year, dist_name,
                            best_match[1] if best_match else "???", best_score))
            match_log.append({
                'stats_state': stats_state, 'stats_district': stats_dist_raw,
                'year': stats_year, 'parsed_name': dist_name, 'adm2_id': '',
                'snap_name': f'BEST: {best_match[1] if best_match else "???"} ({best_score:.2f})',
                'method': 'UNMATCHED'
            })

    return match_map, unmatched, match_log, ancestor_diag

