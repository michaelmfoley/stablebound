"""Name-tidying rules written as data reproduce India's normalizer exactly.

India's rules left the package's code in 0.2.0 for the ``stablebound.json``
that ships with India's tables. These tests pin that the data form gives the
same answer as the function it replaced on every name in India's tables and
on a set chosen to hit each rule — against outputs recorded from that
function before it was removed (tests/fixtures/india_name_rules_expected.json),
not against the code under test — and that each rule actually matters
(dropping any one changes some output; otherwise the comparison could not tell
the rules apart from no rules at all).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stablebound import CountryFolderError, normalize_name, normalizer_from_rules
from tests.india_reference import AWKWARD_NAMES, INDIA_NAME_RULES, india_names

EXPECTED = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "india_name_rules_expected.json")
    .read_text(encoding="utf-8")
)["expected"]


def test_the_recorded_outputs_cover_every_india_name():
    """Guard: the frozen expectations name every name the fixture holds."""
    names = india_names()
    # 892 distinct names on 18-19 Nov 2026. A much smaller number means the
    # tables were not read, and the comparison below would prove little.
    assert len(names) > 850, f"fixture drift: only {len(names)} names read"
    assert set(names) == set(EXPECTED)


def test_india_rules_reproduce_the_recorded_outputs_on_every_name():
    rules = normalizer_from_rules(INDIA_NAME_RULES)
    diff = [(n, rules(n), want) for n, want in EXPECTED.items() if rules(n) != want]
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
    # Recorded from the then-bundled India normalizer on 18 Nov 2026 (the
    # unbundle plan's B0 anchors).
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
