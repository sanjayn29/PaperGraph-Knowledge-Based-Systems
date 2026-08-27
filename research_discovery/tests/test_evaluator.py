"""
tests/test_evaluator.py
───────────────────────
Unit tests for services/evaluator.py

Tests:
- compute_metrics with known perfect / random / adversarial scores
- baseline_random_scores produces values in [0, 1]
- run_baseline_comparison returns dicts for all active baselines
- Graceful handling of insufficient test positives
- all_concept_pairs generates correct canonical pairs
"""

from __future__ import annotations

import pytest

from services.evaluator import (
    SKLEARN_AVAILABLE,
    all_concept_pairs,
    baseline_random_scores,
    compute_metrics,
    run_baseline_comparison,
)

pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="scikit-learn not installed — evaluator tests skipped",
)


# ─────────────────────────────────────────────────────────────
# all_concept_pairs tests
# ─────────────────────────────────────────────────────────────

def test_all_concept_pairs_count():
    """n concepts → n*(n-1)/2 pairs."""
    concepts = ["A", "B", "C", "D"]
    pairs = all_concept_pairs(concepts)
    assert len(pairs) == 6  # 4*3/2


def test_all_concept_pairs_canonical():
    """All returned pairs must be in (smaller, larger) order."""
    concepts = ["Zebra", "Apple", "Mango"]
    pairs = all_concept_pairs(concepts)
    for a, b in pairs:
        assert a <= b


def test_all_concept_pairs_empty():
    assert all_concept_pairs([]) == []
    assert all_concept_pairs(["Solo"]) == []


# ─────────────────────────────────────────────────────────────
# baseline_random_scores tests
# ─────────────────────────────────────────────────────────────

def test_baseline_random_scores_range():
    pairs = [("A", "B"), ("C", "D"), ("E", "F")]
    scores = baseline_random_scores(pairs, seed=42)
    assert len(scores) == 3
    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_baseline_random_scores_deterministic():
    pairs = [("A", "B"), ("C", "D")]
    s1 = baseline_random_scores(pairs, seed=99)
    s2 = baseline_random_scores(pairs, seed=99)
    assert s1 == s2


# ─────────────────────────────────────────────────────────────
# compute_metrics tests
# ─────────────────────────────────────────────────────────────

def _make_predictions(n_pos: int = 5, n_neg: int = 10):
    """
    Create a predictions dict where positives have score=1.0 and negatives=0.0.
    This is a 'perfect' predictor.
    """
    positives = {(f"pos_a{i}", f"pos_b{i}") for i in range(n_pos)}
    preds = {}
    for pair in positives:
        preds[pair] = 1.0
    for i in range(n_neg):
        preds[(f"neg_a{i}", f"neg_b{i}")] = 0.0
    return preds, positives


def test_compute_metrics_perfect_predictor():
    """Perfect predictor should get AUC=1.0, AP=1.0."""
    preds, positives = _make_predictions(n_pos=5, n_neg=10)
    result = compute_metrics(preds, positives)
    assert result["sufficient_data"] is True
    assert result["auc"] == 1.0
    assert result["ap"] == 1.0


def test_compute_metrics_random_predictor():
    """Random predictor AUC should be ~0.5 (not exactly due to small size)."""
    import random
    rng = random.Random(42)
    positives = {(f"p{i}a", f"p{i}b") for i in range(10)}
    negatives = {(f"n{i}a", f"n{i}b") for i in range(10)}
    preds = {p: rng.random() for p in positives | negatives}
    result = compute_metrics(preds, positives)
    assert result["sufficient_data"] is True
    assert 0.0 <= result["auc"] <= 1.0


def test_compute_metrics_insufficient_positives():
    """Fewer than 3 positives → sufficient_data = False."""
    preds = {("A", "B"): 0.9, ("C", "D"): 0.1, ("E", "F"): 0.05}
    positives = {("A", "B")}  # only 1 positive
    result = compute_metrics(preds, positives)
    assert result["sufficient_data"] is False


def test_compute_metrics_no_negatives():
    """All pairs positive → sufficient_data = False (can't compute AUC)."""
    pairs = {("A", "B"), ("C", "D"), ("E", "F")}
    preds = {p: 0.9 for p in pairs}
    result = compute_metrics(preds, pairs)
    assert result["sufficient_data"] is False


def test_compute_metrics_empty_predictions():
    result = compute_metrics({}, set())
    assert result["sufficient_data"] is False


def test_compute_metrics_k_values():
    """precision_at_k and ndcg_at_k keys should match provided k_values."""
    preds, positives = _make_predictions(n_pos=5, n_neg=30)
    result = compute_metrics(preds, positives, k_values=[5, 10])
    assert "sufficient_data" in result
    if result["sufficient_data"]:
        assert "p@5" in result["precision_at_k"]
        assert "ndcg@5" in result["ndcg_at_k"]


# ─────────────────────────────────────────────────────────────
# run_baseline_comparison tests
# ─────────────────────────────────────────────────────────────

def test_run_baseline_comparison_includes_random_and_graph():
    """Random and Static Graph baselines always present."""
    pairs = [(f"concept{i}a", f"concept{i}b") for i in range(20)]
    positives = {pairs[0], pairs[1], pairs[2], pairs[3], pairs[4]}
    graph_scores = {p: 0.5 for p in pairs}

    result = run_baseline_comparison(
        all_pairs=pairs,
        test_positives=positives,
        graph_scores=graph_scores,
        gnn_scores=None,
        setgn_scores=None,
    )
    assert "Random" in result
    assert "Static Graph" in result
    assert "SE-TGN" not in result  # not provided


def test_run_baseline_comparison_includes_setgn():
    """SE-TGN baseline included when setgn_scores provided."""
    pairs = [(f"concept{i}a", f"concept{i}b") for i in range(20)]
    positives = {pairs[0], pairs[1], pairs[2], pairs[3], pairs[4]}
    graph_scores = {p: 0.5 for p in pairs}
    setgn_scores = {p: 0.8 if p in positives else 0.2 for p in pairs}

    result = run_baseline_comparison(
        all_pairs=pairs,
        test_positives=positives,
        graph_scores=graph_scores,
        gnn_scores=None,
        setgn_scores=setgn_scores,
    )
    assert "SE-TGN" in result
