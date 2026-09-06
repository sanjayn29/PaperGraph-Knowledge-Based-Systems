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

import numpy as np
import pytest

from services.evaluator import (
    SKLEARN_AVAILABLE,
    all_concept_pairs,
    baseline_random_scores,
    compute_metrics,
    run_evaluation_variant,
    run_baseline_comparison,
    semantic_similarity_scores,
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


def test_compute_metrics_recall_and_hits_at_k():
    predictions = {
        ("N0", "N0"): 1.00,
        ("P0", "P0"): 0.90,
        ("N1", "N1"): 0.80,
        ("P1", "P1"): 0.70,
        ("N2", "N2"): 0.60,
        ("N3", "N3"): 0.50,
        ("N4", "N4"): 0.40,
        ("N5", "N5"): 0.30,
        ("N6", "N6"): 0.20,
        ("N7", "N7"): 0.10,
        ("P2", "P2"): 0.09,
        ("P3", "P3"): 0.08,
        ("P4", "P4"): 0.07,
    }
    positives = {(f"P{i}", f"P{i}") for i in range(5)}

    result = compute_metrics(predictions, positives)

    assert result["recall_at_k"]["recall@10"] == 0.4
    assert result["recall_at_k"]["recall@50"] == 1.0
    assert result["recall_at_k"]["recall@100"] == 1.0
    assert result["hits_at_k"]["hits@10"] == 1.0
    assert result["hits_at_k"]["hits@50"] == 1.0
    assert result["hits_at_k"]["hits@100"] == 1.0
    assert "mrr" not in result


def test_compute_metrics_recall_and_hits_edge_cases():
    predictions = {
        ("P0", "P0"): 0.9,
        ("P1", "P1"): 0.8,
        ("P2", "P2"): 0.7,
        ("P3", "P3"): 0.6,
        ("P4", "P4"): 0.5,
        ("N0", "N0"): 0.1,
        ("N1", "N1"): 0.0,
    }
    positives = {(f"P{i}", f"P{i}") for i in range(5)}

    result = compute_metrics(predictions, positives, k_values=[10])

    assert result["recall_at_k"]["recall@10"] == 1.0
    assert result["hits_at_k"]["hits@10"] == 1.0

    no_positive_result = compute_metrics(predictions, set())
    assert no_positive_result["sufficient_data"] is False


def test_evaluation_variants_share_pairs_and_labels():
    pairs = [(f"c{i}", f"d{i}") for i in range(6)]
    positives = {pairs[0], pairs[1], pairs[2]}
    graph_scores = {pair: 0.8 - index * 0.05 for index, pair in enumerate(pairs)}
    semantic_scores = {pair: 0.7 - index * 0.04 for index, pair in enumerate(pairs)}
    setgn_scores = {pair: 0.6 - index * 0.03 for index, pair in enumerate(pairs)}

    results = {
        variant: run_evaluation_variant(
            variant=variant,
            all_pairs=pairs,
            test_positives=positives,
            graph_scores=graph_scores,
            semantic_scores=semantic_scores,
            setgn_scores=setgn_scores,
        )
        for variant in ("full", "graph_only", "semantic_only")
    }

    assert all(result["sufficient_data"] for result in results.values())
    assert {result["n_positive"] for result in results.values()} == {3}
    assert {result["n_negative"] for result in results.values()} == {3}
    assert {result["n_total"] for result in results.values()} == {6}


def test_unsupported_se_tgn_ablation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported evaluation variant"):
        run_evaluation_variant(
            variant="setgn_without_temporal",
            all_pairs=[],
            test_positives=set(),
            graph_scores={},
        )


def test_semantic_scores_use_only_concept_embeddings():
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    embeddings = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([1.0, 0.0]),
        "C": np.array([0.0, 1.0]),
    }

    scores = semantic_similarity_scores(pairs, embeddings)

    assert scores[("A", "B")] == 1.0
    assert scores[("A", "C")] == 0.0
    assert scores[("B", "C")] == 0.0


def test_semantic_variant_ignores_graph_and_setgn_scores():
    pairs = [(f"c{i}", f"d{i}") for i in range(6)]
    positives = {pairs[0], pairs[1], pairs[2]}
    semantic_scores = {pair: 1.0 - index * 0.1 for index, pair in enumerate(pairs)}

    result_a = run_evaluation_variant(
        variant="semantic_only",
        all_pairs=pairs,
        test_positives=positives,
        graph_scores={pair: 0.0 for pair in pairs},
        setgn_scores={pair: 0.0 for pair in pairs},
        semantic_scores=semantic_scores,
    )
    result_b = run_evaluation_variant(
        variant="semantic_only",
        all_pairs=pairs,
        test_positives=positives,
        graph_scores={pair: 100.0 for pair in pairs},
        setgn_scores={pair: 100.0 for pair in pairs},
        semantic_scores=semantic_scores,
    )

    assert result_a == result_b


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
