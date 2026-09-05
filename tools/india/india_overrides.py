"""India-specific name-matching overrides.

Lifted directly from the legacy production script
``scripts/india/india_hybrid_methodology/stable/production/1_build_stable_boundaries_IN_v3.py``.

This is the country-specific knowledge that has to come from the researcher.
The package itself stays out of the matching business; this file encodes
India's accumulated experience with district name variants -- spellings the
statistics use that the lineage does not ("Kutch"/"Kachchh"), and renames the
lineage records under the older name ("Mysuru"/"Mysore").

Homonyms -- one name, two states -- used to need a second dict here. They no
longer do: the shapefile carries a `state` column and the matcher resolves them
from it. See the retirement note at the foot of this file for why that dict was
deleted rather than kept up to date.
"""

from __future__ import annotations

# Shapefile name (normalized) → snapshot name.
# Common India spelling variants and renames.
MANUAL_OVERRIDES = {
    "beed": "Bid",
    "bengaluru urban": "Bangalore",
    "cooch behar": "Koch Bihar",
    "dahod": "Dohad",
    "dang": "The Dangs",
    "dindigul": "Dindigul Anna",
    "dr b r ambedkar konaseema": "Konaseema",
    "b r ambedkar konaseema": "Konaseema",
    "east champaran": "Purba Champaran",
    "east delhi": "East",
    "east singhbhum": "Purbi Singhbhum",
    "hooghly": "Hugli",
    "howrah": "Haora",
    "kutch": "Kachchh",
    "lakhimpur kheri": "Kheri",
    "medchalmalkajgiri": "Medchal-Malkajgiri",
    "mumbai city": "Mumbai",
    "mysuru": "Mysore",
    "ntr": "NT Rama Rao",
    "north 24 parganas": "North Twenty Four Parganas",
    "north delhi": "North",
    "north east delhi": "North East",
    "north west delhi": "North West",
    "poonch": "Punch",
    "puducherry": "Pondicherry U.T.",
    "south 24 parganas": "South Twenty Four Parganas",
    "south delhi": "South",
    "south east delhi": "South East",
    "south west delhi": "South West",
    "uttar bastar kanker": "Kanker",
    "west champaran": "Pashchim Champaran",
    "west delhi": "West",
    "west singhbhum": "Pashchimi Singhbhum",
    "raigad": "Raigarh",
    # RETIRED 2026-09-02 — the 2023 Rajasthan districts. These mapped nine real
    # districts onto their parents, which was a workaround for resolving ids at
    # max_year (2025) where the 2023 districts no longer exist. Attaching at the
    # shapefile's own 2024 vintage makes them resolve to their own ids, so seven
    # of the nine had already gone inert: Anupgarh reaches 01066, Kekri 01069,
    # Sanchore 01097, Shahpura 01093, Dudu 01087, Neem Ka Thana 01067 and
    # Gangapur City 01106 by exact match, before any override is consulted.
    #
    # The two that still fired were Jaipur(Rural) and Jodhpur(Rural), where the
    # normalizer strips the parenthesis without inserting a space -- "jaipurrural"
    # against the lineage's "jaipur rural". That scores 0.957, far above the 0.80
    # fuzzy threshold, so the matcher resolves them correctly once the override
    # stops sending them to the parent.
    "central delhi": "Central",         # sibling of "east delhi": "East" etc.
    "chhatrapati sambhajinagar": "Aurangabad",
}


# HOMONYM_OVERRIDES — RETIRED 2026-09-02.
#
# Twelve entries mapping a shapefile FEATURE INDEX to the right id for districts
# whose name is shared across states (two Hamirpurs, two Pratapgarhs, two
# Balrampurs, two Raigarhs, two Aurangabads, Delhi's North and South).
#
# They are gone because the problem is: the published shapefile now carries a
# `state` column (built by prepare_geolocet.py from the official 2021 state
# boundaries), and `propose_shapefile_mapping(coarse_column="state")` derives
# what these entries used to assert. Every one of the twelve now resolves by
# exact name+state match, and the `homonym` match method no longer appears in
# the log at all.
#
# Deleting them WITH the column, rather than after, was the point. Keying on
# feature index means any edit that reorders or filters the shapefile silently
# repoints an override at a different district -- wrong id, no error, no
# warning. Regenerating the shapefile is exactly such an edit. One entry was
# already stale before this change: index 310 asserted Chhattisgarh's Raigarh
# but the file had moved, and Chhattisgarh's Bilaspur -- the district actually
# responsible for the 0.111% of misplaced area -- was never listed here at all.
#
# If a future shapefile arrives without a usable state column, add the column;
# do not restore index-keyed overrides.
