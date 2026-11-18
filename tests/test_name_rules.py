"""Name-tidying rules written as data reproduce India's normalizer exactly.

India's rules are leaving the package's code for the ``stablebound.json``
that ships with India's tables. These tests pin that the data form gives the
same answer as the function it replaces on every name in the India tables
and on a set chosen to hit each rule, and that each rule actually matters
(dropping any one changes some output — otherwise the comparison could not
tell the rules apart from no rules at all).
"""

from __future__ import annotations

import pytest

from stablebound import CountryFolderError, normalize_name, normalizer_from_rules
from stablebound.data import _india_normalizer
from tests.india_reference import AWKWARD_NAMES, INDIA_NAME_RULES, india_names


@pytest.fixture(scope="module")
def names() -> list[str]:
    return india_names()


def test_india_rules_equal_the_india_normalizer_on_every_name(names):
    rules = normalizer_from_rules(INDIA_NAME_RULES)
    # 892 distinct names on 18 Nov 2026 (every unit, state and renamed name in
    # the four tables, plus the awkward set). A much smaller number means the
    # tables were not read, and the comparison below would prove little.
    assert len(names) > 850, f"fixture drift: only {len(names)} names read"
    diff = [(n, rules(n), _india_normalizer(n)) for n in names if rules(n) != _india_normalizer(n)]
    assert diff == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Villupuram (TN)", "villupuram"),
        ("Hamirpur District", "hamirpur"),
        ("Bijapur Dist.", "bijapur"),
        ("Dr. B.R. Ambedkar Konaseema", "br ambedkar konaseema"),
        ("Sri Potti Sriramulu Nellore", "potti sriramulu nellore"),
        ("Shri Ganganagar", "ganganagar"),
    ],
)
def test_india_rules_known_outputs(raw, expected):
    # Pinned values, recorded from the bundled India normalizer on 18 Nov
    # 2026 (the unbundle plan's B0 anchors), so this keeps meaning something
    # after that function is removed from the package.
    assert normalizer_from_rules(INDIA_NAME_RULES)(raw) == expected


@pytest.mark.parametrize("kind", sorted(INDIA_NAME_RULES))
def test_each_rule_kind_matters(kind):
    """Positive control: removing any one kind of rule changes some output."""
    full = normalizer_from_rules(INDIA_NAME_RULES)
    reduced = normalizer_from_rules({k: v for k, v in INDIA_NAME_RULES.items() if k != kind})
    assert any(full(n) != reduced(n) for n in AWKWARD_NAMES)


def test_state_suffix_must_be_stripped_before_lowercasing():
    """The '(TN)' rule only works while the capitals are still there."""
    moved = {
        "remove_patterns": INDIA_NAME_RULES["strip_before_lowercase"]
        + INDIA_NAME_RULES["remove_patterns"],
        "remove_substrings": INDIA_NAME_RULES["remove_substrings"],
    }
    assert normalizer_from_rules(moved)("Villupuram (TN)") == "villupuram tn"
    assert normalizer_from_rules(INDIA_NAME_RULES)("Villupuram (TN)") == "villupuram"


def test_no_rules_is_the_package_default():
    assert normalizer_from_rules(None) is normalize_name
    assert normalizer_from_rules({}) is normalize_name


def test_unknown_rule_kind_is_refused():
    with pytest.raises(CountryFolderError, match="unknown kind"):
        normalizer_from_rules({"remove_suffixes": [" district"]})


def test_bad_pattern_is_refused():
    with pytest.raises(CountryFolderError, match="not a valid pattern"):
        normalizer_from_rules({"remove_patterns": ["(unclosed"]})


def test_comment_keys_are_ignored():
    with_comment = {"_why": "DESAGRI appends state codes", **INDIA_NAME_RULES}
    assert normalizer_from_rules(with_comment)("Beed (MH)") == "beed"
