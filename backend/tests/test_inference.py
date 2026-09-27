"""Unit tests for pure NLP helper logic (no model download required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ""))

from app.inference import (  # noqa: E402
    entity_statistics,
    normalize_label,
    to_display_type,
)


def test_normalize_flat_labels():
    assert normalize_label("Person") == ("Person", "FLAT")
    assert normalize_label("Other") == (None, None)
    assert normalize_label("O") == (None, None)
    assert normalize_label("") == (None, None)


def test_normalize_iob2_labels():
    assert normalize_label("B-Person") == ("Person", "B")
    assert normalize_label("I-Location") == ("Location", "I")


def test_normalize_maha_iob_codes():
    assert normalize_label("BNEP") == ("Person", "B")
    assert normalize_label("INETI") == ("Time", "I")
    assert normalize_label("BED") == ("Designation", "B")
    assert normalize_label("BNED") == ("Date", "B")


def test_display_type_uppercases():
    assert to_display_type("Person") == "PERSON"
    assert to_display_type("Organization") == "ORGANIZATION"


def test_entity_statistics_counts_per_type():
    ents = [
        {"label": "PERSON"},
        {"label": "PERSON"},
        {"label": "LOCATION"},
    ]
    stats = entity_statistics(ents)
    assert stats == {"PERSON": 2, "LOCATION": 1}


def test_entity_statistics_empty():
    assert entity_statistics([]) == {}
