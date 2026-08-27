"""
tests/test_temporal_graph.py
─────────────────────────────
Unit tests for services/temporal_graph.py

Tests:
- TemporalEvent canonical ordering
- TemporalGraph event add + sort
- Temporal split correctness
- build_temporal_events factory function
- Year fallback for papers without year
"""

from __future__ import annotations

import numpy as np
import pytest

from services.temporal_graph import (
    TemporalEvent,
    TemporalGraph,
    build_temporal_events,
)


# ─────────────────────────────────────────────────────────────
# TemporalEvent tests
# ─────────────────────────────────────────────────────────────

def test_temporal_event_canonical_order():
    """source and target are always ordered (smaller, larger) lexicographically."""
    emb = np.zeros(384)
    event = TemporalEvent("Zebra", "Apple", "paper_001", 2022, emb)
    assert event.source_concept == "Apple"
    assert event.target_concept == "Zebra"


def test_temporal_event_pair_property():
    emb = np.zeros(384)
    event = TemporalEvent("A", "B", "paper_001", 2021, emb)
    assert event.pair == ("A", "B")


def test_temporal_event_already_ordered():
    emb = np.zeros(384)
    event = TemporalEvent("Alpha", "Beta", "paper_001", 2023, emb)
    assert event.source_concept == "Alpha"
    assert event.target_concept == "Beta"


# ─────────────────────────────────────────────────────────────
# TemporalGraph tests
# ─────────────────────────────────────────────────────────────

def _make_event(src: str, dst: str, year: int) -> TemporalEvent:
    return TemporalEvent(src, dst, f"paper_{year}", year, np.zeros(384))


def _populate_graph() -> TemporalGraph:
    tg = TemporalGraph()
    tg.add_event(_make_event("AI", "Healthcare", 2021))
    tg.add_event(_make_event("ML", "Healthcare", 2022))
    tg.add_event(_make_event("AI", "Robotics", 2023))
    tg.add_event(_make_event("GNN", "NLP", 2024))
    tg.sort()
    return tg


def test_temporal_graph_len():
    tg = _populate_graph()
    assert len(tg) == 4


def test_temporal_graph_sorted():
    tg = _populate_graph()
    timestamps = [e.timestamp for e in tg.events]
    assert timestamps == sorted(timestamps)


def test_temporal_graph_unique_concepts():
    tg = _populate_graph()
    concepts = tg.unique_concepts()
    assert "AI" in concepts
    assert "Healthcare" in concepts
    assert isinstance(concepts, list)
    assert concepts == sorted(concepts)


def test_temporal_graph_year_range():
    tg = _populate_graph()
    y_min, y_max = tg.year_range()
    assert y_min == 2021
    assert y_max == 2024


def test_temporal_graph_events_before():
    tg = _populate_graph()
    before_2023 = tg.events_before(2023)
    assert all(e.timestamp < 2023 for e in before_2023)
    assert len(before_2023) == 2  # 2021, 2022


def test_temporal_graph_positive_pairs():
    tg = _populate_graph()
    pairs = tg.positive_pairs()
    assert ("AI", "Healthcare") in pairs
    assert ("GNN", "NLP") in pairs


def test_temporal_graph_is_empty():
    tg = TemporalGraph()
    assert tg.is_empty()
    tg.add_event(_make_event("A", "B", 2020))
    assert not tg.is_empty()


# ─────────────────────────────────────────────────────────────
# Temporal split tests
# ─────────────────────────────────────────────────────────────

def test_temporal_split_basic():
    """With 4+ years, split should produce non-empty train set."""
    tg = _populate_graph()
    train, val, test = tg.temporal_split(val_ratio=0.25, test_ratio=0.25)
    assert len(train) > 0


def test_temporal_split_no_future_in_train():
    """Training events must have earlier timestamps than test events."""
    tg = _populate_graph()
    train, val, test = tg.temporal_split()
    if test:
        max_train_t = max(e.timestamp for e in train)
        min_test_t = min(e.timestamp for e in test)
        assert max_train_t <= min_test_t


def test_temporal_split_insufficient_years():
    """With < 4 distinct years, all events go to train; val and test are empty."""
    tg = TemporalGraph()
    tg.add_event(_make_event("A", "B", 2022))
    tg.add_event(_make_event("C", "D", 2022))
    tg.add_event(_make_event("E", "F", 2023))
    tg.sort()
    train, val, test = tg.temporal_split()
    assert len(val) == 0
    assert len(test) == 0
    assert len(train) == 3


def test_temporal_split_empty_graph():
    tg = TemporalGraph()
    train, val, test = tg.temporal_split()
    assert train == [] and val == [] and test == []


# ─────────────────────────────────────────────────────────────
# build_temporal_events tests
# ─────────────────────────────────────────────────────────────

def _make_papers() -> list[dict]:
    return [
        {
            "paper_id": "paper_001",
            "title": "GNN for Healthcare",
            "year": 2021,
            "year_estimated": False,
            "concepts": ["Graph Neural Network", "Healthcare", "Drug Discovery"],
        },
        {
            "paper_id": "paper_002",
            "title": "NLP in Medicine",
            "year": 2022,
            "year_estimated": False,
            "concepts": ["Natural Language Processing", "Healthcare", "Clinical Notes"],
        },
        {
            "paper_id": "paper_003",
            "title": "Deep Learning Survey",
            "year": 2023,
            "year_estimated": False,
            "concepts": ["Deep Learning", "Graph Neural Network", "Natural Language Processing"],
        },
    ]


def test_build_temporal_events_basic():
    papers = _make_papers()
    emb = {c: np.random.rand(384).astype(np.float32) for p in papers for c in p["concepts"]}
    tg, warnings = build_temporal_events(papers, emb)
    assert len(tg) > 0
    assert len(tg.unique_concepts()) >= 4


def test_build_temporal_events_sorted():
    papers = _make_papers()
    emb = {c: np.random.rand(384).astype(np.float32) for p in papers for c in p["concepts"]}
    tg, _ = build_temporal_events(papers, emb)
    timestamps = [e.timestamp for e in tg.events]
    assert timestamps == sorted(timestamps)


def test_build_temporal_events_no_year_fallback():
    """Paper without a year should receive a fallback year and generate a warning."""
    papers = [
        {"paper_id": "p001", "title": "Test", "year": 2022, "year_estimated": False,
         "concepts": ["Alpha", "Beta"]},
        {"paper_id": "p002", "title": "No Year", "year": None, "year_estimated": True,
         "concepts": ["Gamma", "Delta"]},
    ]
    emb = {}
    tg, warnings = build_temporal_events(papers, emb)
    # Should still produce events, not crash
    assert len(tg) > 0
    # Should produce at least one warning about missing year
    assert any("year" in w.lower() or "estimated" in w.lower() for w in warnings)


def test_build_temporal_events_empty_concepts():
    """Papers with no concepts produce no events (no crash)."""
    papers = [{"paper_id": "p001", "year": 2022, "year_estimated": False, "concepts": []}]
    tg, _ = build_temporal_events(papers, {})
    assert len(tg) == 0
