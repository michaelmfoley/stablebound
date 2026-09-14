"""StableBound — reproducible framework for harmonizing data across
administrative boundary changes.

See ``docs/USAGE.md`` for the practical guide,
``docs/methodology.md`` and ``docs/methodology_modern.md`` for the
narrative description of the algorithms, and ``docs/USER_MANUAL.md``
for the source-tour reference.
"""

from __future__ import annotations

from .assess import ReadinessReport, assess_country, detect_dialect
from .assign_ids import IdAssignment, IdAssignmentError, assign_unit_ids
from .boundary import StableBoundary
from .data import BUNDLED_COUNTRIES, BundledCountry
from .data_dict import DataDictionary
from .fnid import (
    FNIDOverflowError,
    FnidParts,
    assign_fnids,
    build_admin1_code_map,
    build_admin1_only_code_map,
    build_admin2_code_map,
    build_fnid,
    parse_fnid,
    validate_fnid,
)
from .breakpoint import BreakpointResult, analyze_breakpoints
from .fews_export import (
    DeliverableValidationError,
    RelationshipLevelInput,
    attach_stats_fnids,
    build_deliverables,
    build_legacy_relationship_table,
    build_relationship_table,
    validate_no_within_admin1_duplicate,
    write_admin_definitions_workbooks,
    write_agstats_workbook,
    write_legacy_relationship_table,
    write_relationship_table,
)
from .lineage import LineageGraph, name_of_unit_in_year
from .lineage_class import Lineage
from .match import (
    MatchProposal,
    attach_shapefile_ids,
    attach_stats_ids,
    normalize_name,
    propose_shapefile_mapping,
    propose_stats_mapping,
    read_mapping,
)
from .modern import ModernBoundary
from .rt_convert import (
    convert_relationship_table_to_lineage,
    read_legacy_relationship_table,
)
from .provenance import provenance_record, write_provenance
from .schemas import SchemaError
from .unit_defs import (
    build_admin1_attribution_table,
    build_admin1_defs_table,
    build_unit_defs_table,
    write_unit_defs_files,
)
from .validate import (
    LineageDataError,
    LineageIssue,
    format_issues,
    validate_coarse_references,
    validate_lineage,
    validate_stable_group_contiguity,
    validate_stats_lineage_consistency,
)

__version__ = "0.1.5"

__all__ = [
    "provenance_record",
    "write_provenance",
    "assess_country",
    "ReadinessReport",
    "detect_dialect",
    "assign_unit_ids",
    "IdAssignment",
    "IdAssignmentError",
    "StableBoundary",
    "ModernBoundary",
    "Lineage",
    "LineageGraph",
    "DataDictionary",
    "SchemaError",
    "LineageDataError",
    "LineageIssue",
    "validate_lineage",
    "validate_coarse_references",
    "validate_stats_lineage_consistency",
    "validate_stable_group_contiguity",
    "format_issues",
    "FNIDOverflowError",
    "assign_fnids",
    "build_admin1_code_map",
    "build_admin1_only_code_map",
    "build_admin2_code_map",
    "build_fnid",
    "parse_fnid",
    "validate_fnid",
    "FnidParts",
    "build_admin1_attribution_table",
    "build_admin1_defs_table",
    "build_unit_defs_table",
    "write_unit_defs_files",
    "name_of_unit_in_year",
    "build_deliverables",
    "build_relationship_table",
    "write_relationship_table",
    "build_legacy_relationship_table",
    "write_legacy_relationship_table",
    "write_admin_definitions_workbooks",
    "attach_stats_fnids",
    "write_agstats_workbook",
    "validate_no_within_admin1_duplicate",
    "DeliverableValidationError",
    "RelationshipLevelInput",
    "convert_relationship_table_to_lineage",
    "read_legacy_relationship_table",
    "analyze_breakpoints",
    "BreakpointResult",
    "BUNDLED_COUNTRIES",
    "BundledCountry",
    "MatchProposal",
    "propose_shapefile_mapping",
    "propose_stats_mapping",
    "attach_shapefile_ids",
    "attach_stats_ids",
    "read_mapping",
    "normalize_name",
    "__version__",
]
