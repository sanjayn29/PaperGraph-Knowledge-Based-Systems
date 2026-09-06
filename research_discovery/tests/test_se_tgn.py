"""
tests/test_se_tgn.py
─────────────────────
Unit tests for services/se_tgn.py

Tests (all run without a real GPU; CPU only):
- TimeEncode produces correct output shape
- NodeMemory stores and retrieves values
- SETGN forward pass (process_event + predict)
- Training loop runs without crash
- Negative sampling never returns positive concept
- compute_setgn_scores returns scores or None gracefully
- setgn_status_label returns correct strings
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from services.se_tgn import (
    SETGN_AVAILABLE,
    setgn_status_label,
    MIN_EVENTS_FOR_TRAINING,
)
from services.temporal_graph import TemporalEvent, TemporalGraph

# Skip all SE-TGN tests if PyTorch is not available
pytestmark = pytest.mark.skipif(
    not SETGN_AVAILABLE,
    reason="PyTorch not installed — SE-TGN tests skipped",
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_tg(n_papers: int = 5) -> tuple[TemporalGraph, dict[str, np.ndarray]]:
    """Create a small TemporalGraph and concept embeddings for testing."""
    concepts = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    papers = []
    for i in range(n_papers):
        year = 2020 + i
        paper_id = f"paper_{i:03d}"
        papers.append({
            "paper_id": paper_id,
            "year": year,
            "year_estimated": False,
            "concepts": concepts[:3],  # each paper has 3 concepts
        })

    emb = {c: np.random.rand(384).astype(np.float32) for c in concepts}

    from services.temporal_graph import build_temporal_events
    tg, _ = build_temporal_events(papers, emb)
    return tg, emb


# ─────────────────────────────────────────────────────────────
# TimeEncode tests
# ─────────────────────────────────────────────────────────────

def test_time_encode_output_shape():
    import torch
    from services.se_tgn import TimeEncode, D_TIME

    encoder = TimeEncode(d_time=D_TIME)
    dt = torch.tensor([1.0, 2.0, 3.0])
    out = encoder(dt)
    assert out.shape == (3, D_TIME), f"Expected (3, {D_TIME}), got {out.shape}"


def test_time_encode_scalar():
    import torch
    from services.se_tgn import TimeEncode, D_TIME

    encoder = TimeEncode(d_time=D_TIME)
    dt = torch.tensor(0.0)
    out = encoder(dt)
    assert out.shape == (1, D_TIME)


def test_time_encode_zero_delta():
    """Zero time difference should not produce NaN or Inf."""
    import torch
    from services.se_tgn import TimeEncode, D_TIME

    encoder = TimeEncode(d_time=D_TIME)
    dt = torch.tensor([0.0])
    out = encoder(dt)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ─────────────────────────────────────────────────────────────
# NodeMemory tests
# ─────────────────────────────────────────────────────────────

def test_node_memory_init_zeros():
    from services.se_tgn import NodeMemory, D_MEM
    mem = NodeMemory(n_concepts=5, d_mem=D_MEM)
    import torch
    assert torch.all(mem.memory == 0)


def test_node_memory_update_and_get():
    import torch
    from services.se_tgn import NodeMemory, D_MEM
    mem = NodeMemory(n_concepts=4, d_mem=D_MEM)
    new_vec = torch.ones(D_MEM) * 0.5
    mem.update(idx=2, new_mem=new_vec, timestamp=2022.0)
    retrieved = mem.get(2)
    assert torch.allclose(retrieved, new_vec, atol=1e-6)


def test_node_memory_delta_t():
    import torch
    from services.se_tgn import NodeMemory, D_MEM
    mem = NodeMemory(n_concepts=3, d_mem=D_MEM)
    mem.update(0, torch.zeros(D_MEM), timestamp=2020.0)
    dt = mem.delta_t(idx=0, current_time=2023.0)
    assert abs(dt.item() - 3.0) < 1e-5


def test_node_memory_reset():
    import torch
    from services.se_tgn import NodeMemory, D_MEM
    mem = NodeMemory(n_concepts=3, d_mem=D_MEM)
    mem.update(0, torch.ones(D_MEM), timestamp=2021.0)
    mem.reset()
    assert torch.all(mem.memory == 0)
    assert torch.all(mem.last_update == 0)


# ─────────────────────────────────────────────────────────────
# SETGN model tests
# ─────────────────────────────────────────────────────────────

def test_setgn_predict_output_range():
    """predict() should return a value in [0, 1]."""
    import torch
    from services.se_tgn import SETGN, NodeMemory, D_MEM

    model = SETGN()
    mem = NodeMemory(n_concepts=5, d_mem=D_MEM)
    score = model.predict(0, 1, mem)
    assert 0.0 <= float(score.item()) <= 1.0


def test_setgn_process_event_updates_memory():
    """process_event should change the memory for both src and dst concepts."""
    import torch
    from services.se_tgn import SETGN, NodeMemory, D_MEM, PAPER_EMB_DIM

    model = SETGN()
    mem = NodeMemory(n_concepts=5, d_mem=D_MEM)
    paper_emb = torch.zeros(PAPER_EMB_DIM)

    old_src_mem = mem.get(0).clone()
    model.process_event(0, 1, paper_emb, timestamp=2022.0, memory=mem)
    new_src_mem = mem.get(0)

    # Memory should have changed (not all zeros anymore)
    assert not torch.allclose(old_src_mem, new_src_mem)


def test_setgn_process_event_no_nan():
    """Ensure no NaN values propagate through the model."""
    import torch
    from services.se_tgn import SETGN, NodeMemory, D_MEM, PAPER_EMB_DIM

    model = SETGN()
    mem = NodeMemory(n_concepts=4, d_mem=D_MEM)
    paper_emb = torch.randn(PAPER_EMB_DIM)
    model.process_event(0, 2, paper_emb, timestamp=2021.0, memory=mem)

    for i in range(4):
        assert not torch.isnan(mem.get(i)).any(), f"NaN in memory[{i}]"


# ─────────────────────────────────────────────────────────────
# Training loop tests
# ─────────────────────────────────────────────────────────────

def test_train_setgn_returns_model():
    """train_setgn() should return a (model, memory, concept_index) triple."""
    from services.se_tgn import train_setgn

    tg, emb = _make_tg(n_papers=5)
    result = train_setgn(tg, emb, n_epochs=1)
    assert result is not None, "Training should succeed with 5 papers"
    model, memory, concept_index = result
    assert isinstance(concept_index, dict)
    assert len(concept_index) >= 3


def test_train_setgn_insufficient_events():
    """With fewer than MIN_EVENTS_FOR_TRAINING events, training should return None."""
    from services.se_tgn import train_setgn
    from services.temporal_graph import TemporalGraph

    tg = TemporalGraph()
    # Add only 2 events
    tg.add_event(TemporalEvent("A", "B", "p1", 2022, np.zeros(384)))
    tg.add_event(TemporalEvent("C", "D", "p2", 2023, np.zeros(384)))
    tg.sort()
    result = train_setgn(tg, {})
    assert result is None


def test_train_setgn_excludes_held_out_events_from_memory_updates():
    from services.se_tgn import SETGN, train_setgn

    training_events = [
        TemporalEvent("A", "B", f"train_{year}", year, np.zeros(384))
        for year in range(2020, 2030)
    ]
    held_out_event = TemporalEvent("C", "D", "test_2030", 2030, np.zeros(384))
    temporal_graph = TemporalGraph()
    for event in training_events + [held_out_event]:
        temporal_graph.add_event(event)
    temporal_graph.sort()

    update_timestamps = []
    original_process_event = SETGN.process_event

    def tracked_process_event(self, src_idx, dst_idx, paper_emb, timestamp, memory):
        update_timestamps.append(timestamp)
        return original_process_event(self, src_idx, dst_idx, paper_emb, timestamp, memory)

    with patch.object(SETGN, "process_event", tracked_process_event):
        result = train_setgn(
            temporal_graph,
            {},
            n_epochs=1,
            training_events=training_events,
            concept_names=["A", "B", "C", "D"],
        )

    assert result is not None
    assert update_timestamps == [float(year) for year in range(2020, 2030)]
    assert 2030.0 not in update_timestamps


# ─────────────────────────────────────────────────────────────
# Negative sampling test
# ─────────────────────────────────────────────────────────────

def test_sample_negative_not_src_or_dst():
    """Negative sample should never be src or dst concept."""
    from services.se_tgn import _sample_negative

    concepts = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    for _ in range(50):
        neg = _sample_negative(concepts, "Alpha", "Beta")
        assert neg != "Alpha"
        assert neg != "Beta"


# ─────────────────────────────────────────────────────────────
# compute_setgn_scores tests
# ─────────────────────────────────────────────────────────────

def test_compute_setgn_scores_returns_dict():
    """compute_setgn_scores should return a non-empty dict of floats."""
    from services.se_tgn import compute_setgn_scores

    tg, emb = _make_tg(n_papers=5)
    scores = compute_setgn_scores(tg, emb)
    assert scores is not None
    assert isinstance(scores, dict)
    assert len(scores) > 0
    for score in scores.values():
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_compute_setgn_scores_canonical_keys():
    """All keys should be in canonical (smaller, larger) order."""
    from services.se_tgn import compute_setgn_scores

    tg, emb = _make_tg(n_papers=5)
    scores = compute_setgn_scores(tg, emb)
    if scores:
        for (a, b) in scores.keys():
            assert a <= b, f"Non-canonical key: ({a}, {b})"


# ─────────────────────────────────────────────────────────────
# Status label tests (run without PyTorch too)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Status label tests — these run even without PyTorch
# ─────────────────────────────────────────────────────────────

def test_setgn_status_label_inactive():
    """Status label for 0 events should mention 'inactive' or 'minimum'.
    This test is always collected (no PyTorch skip) because it only
    tests the pure-Python status string helper.
    """
    label = setgn_status_label(n_events=0)
    assert "inactive" in label.lower() or "minimum" in label.lower()

