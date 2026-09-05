"""India-specific StableBound helpers.

Country submodule. As more countries are onboarded, mirror this layout
(``stablebound/<country>/...``) rather than adding country-specific logic to the
core modules. The first occupant is the DESAGRI crop-statistics → lineage
``ADM2_ID`` name matcher.
"""

from .matcher import match_stats_to_lineage

__all__ = ["match_stats_to_lineage"]
