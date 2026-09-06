"""
evaluator.py
────────────
Evaluation metrics for PaperGraph's link-prediction task.

Implements the evaluation protocol from the base paper:
  - AUC (Area Under the ROC Curve)
  - Average Precision (AP)
  - Precision@K (P@K) for K ∈ {10, 50, 100}
    - Recall@K and Hits@K for K ∈ {10, 50, 100}
  - NDCG@K for K ∈ {10, 50, 100}

The evaluator currently represents one global ranked list of concept pairs,
not multiple source-node queries. Query-level MRR is therefore not reported.

Temporal split evaluation
─────────────────────────
Evaluation uses historical concept pairs for training and held-out
FUTURE pairs (from later papers) as test positives. Random concept
pairs not seen in any paper serve as negatives.

This matches the base paper's future-link-prediction setup.

Baselines
─────────
1. Random           — random uniform scores
2. Static graph     — centrality-based graph_score from graph_analyzer
3. SE-TGN           — trained temporal GNN scores

Graceful degradation
────────────────────
- If scikit-learn is not installed → returns a message dict only
- If < 10 events in test set → returns "Insufficient data" message
- Never fabricates metrics
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum test-set positives required to report meaningful metrics
_MIN_TEST_POSITIVES = 3
_DEFAULT_K_VALUES = [10, 50, 100]
SUPPORTED_ABLATION_VARIANTS = {"full", "graph_only", "semantic_only"}

# ─────────────────────────────────────────────────────────────
# Optional scikit-learn import
# ─────────────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        ndcg_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info(
        "scikit-learn not installed — evaluation metrics unavailable. "
        "Install with: pip install scikit-learn"
    )


# ─────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: dict[tuple[str, str], float],
    positive_pairs: set[tuple[str, str]],
    k_values: list[int] = _DEFAULT_K_VALUES,
) -> dict:
    """
    Compute AUC, AP, P@K, Recall@K, Hits@K, and NDCG@K for link predictions.

    Parameters
    ----------
    predictions   : {(concept_a, concept_b) → score ∈ [0,1]}
    positive_pairs: set of ground-truth positive concept pairs
    k_values      : list of K values for P@K and NDCG@K

    Returns
    -------
    dict with keys: auc, ap, precision_at_k, recall_at_k, hits_at_k,
    ndcg_at_k, sufficient_data
    """
    if not SKLEARN_AVAILABLE:
        return {
            "sufficient_data": False,
            "message": "scikit-learn not installed. Run: pip install scikit-learn",
        }

    if not predictions:
        return {"sufficient_data": False, "message": "No predictions available."}

    # Build aligned y_true and y_score arrays
    pairs = list(predictions.keys())
    y_score = np.array([predictions[p] for p in pairs], dtype=float)
    y_true = np.array([1 if p in positive_pairs else 0 for p in pairs], dtype=int)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos < _MIN_TEST_POSITIVES:
        return {
            "sufficient_data": False,
            "message": (
                f"Only {n_pos} positive test pairs found "
                f"(minimum required: {_MIN_TEST_POSITIVES}). "
                "Upload papers spanning more years for temporal evaluation."
            ),
        }

    if n_neg == 0:
        return {
            "sufficient_data": False,
            "message": "No negative pairs found — cannot compute AUC.",
        }

    try:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    except Exception as exc:
        logger.warning("AUC/AP computation failed: %s", exc)
        return {"sufficient_data": False, "message": f"Metric computation error: {exc}"}

    # P@K and NDCG@K
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_true = y_true[sorted_indices]

    precision_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    hits_at_k: dict[str, float] = {}
    ndcg_at_k: dict[str, float] = {}

    for k in k_values:
        actual_k = min(k, len(sorted_true))
        top_k = sorted_true[:actual_k]
        precision_at_k[f"p@{k}"] = float(top_k.sum() / actual_k) if actual_k > 0 else 0.0
        recall_at_k[f"recall@{k}"] = float(top_k.sum() / n_pos) if n_pos > 0 else 0.0
        hits_at_k[f"hits@{k}"] = 1.0 if top_k.sum() > 0 else 0.0

        try:
            # ndcg_score expects 2D arrays
            ndcg_at_k[f"ndcg@{k}"] = float(
                ndcg_score(
                    y_true.reshape(1, -1),
                    y_score.reshape(1, -1),
                    k=min(k, len(y_score)),
                )
            )
        except Exception:
            ndcg_at_k[f"ndcg@{k}"] = 0.0

    return {
        "sufficient_data": True,
        "auc": round(auc, 4),
        "ap": round(ap, 4),
        "precision_at_k": {k: round(v, 4) for k, v in precision_at_k.items()},
        "recall_at_k": {k: round(v, 4) for k, v in recall_at_k.items()},
        "hits_at_k": {k: round(v, 4) for k, v in hits_at_k.items()},
        "ndcg_at_k": {k: round(v, 4) for k, v in ndcg_at_k.items()},
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_total": len(predictions),
    }


# ─────────────────────────────────────────────────────────────
# Baseline comparisons
# ─────────────────────────────────────────────────────────────

def semantic_similarity_scores(
    all_pairs: list[tuple[str, str]],
    concept_embeddings: dict[str, np.ndarray],
) -> Optional[dict[tuple[str, str], float]]:
    """Build semantic-only scores for an existing pair universe."""
    if not concept_embeddings:
        return None

    scores: dict[tuple[str, str], float] = {}
    for concept_a, concept_b in all_pairs:
        embedding_a = concept_embeddings.get(concept_a)
        embedding_b = concept_embeddings.get(concept_b)
        if embedding_a is None or embedding_b is None:
            scores[(concept_a, concept_b)] = 0.0
            continue

        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        if norm_a <= 1e-8 or norm_b <= 1e-8:
            scores[(concept_a, concept_b)] = 0.0
        else:
            scores[(concept_a, concept_b)] = float(
                np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
            )
    return scores


def run_evaluation_variant(
    variant: str,
    all_pairs: list[tuple[str, str]],
    test_positives: set[tuple[str, str]],
    graph_scores: dict[tuple[str, str], float],
    setgn_scores: Optional[dict[tuple[str, str], float]] = None,
    semantic_scores: Optional[dict[tuple[str, str], float]] = None,
    k_values: list[int] = _DEFAULT_K_VALUES,
) -> dict:
    """Evaluate one component variant over a shared pair universe and labels."""
    if variant not in SUPPORTED_ABLATION_VARIANTS:
        raise ValueError(
            f"Unsupported evaluation variant '{variant}'. "
            f"Supported variants: {sorted(SUPPORTED_ABLATION_VARIANTS)}"
        )

    if variant == "full":
        if setgn_scores is None:
            raise ValueError("The full variant requires leakage-free SE-TGN scores.")
        scores = setgn_scores
    elif variant == "graph_only":
        scores = graph_scores
    else:
        if semantic_scores is None:
            raise ValueError("The semantic_only variant requires concept embeddings.")
        scores = semantic_scores

    predictions = {pair: scores.get(pair, 0.0) for pair in all_pairs}
    return compute_metrics(predictions, test_positives, k_values)

def baseline_random_scores(
    all_pairs: list[tuple[str, str]],
    seed: int = 42,
) -> dict[tuple[str, str], float]:
    """Assign random uniform scores to all pairs (Random baseline)."""
    rng = random.Random(seed)
    return {pair: rng.random() for pair in all_pairs}


def run_baseline_comparison(
    all_pairs: list[tuple[str, str]],
    test_positives: set[tuple[str, str]],
    graph_scores: dict[tuple[str, str], float],
    gnn_scores: Optional[dict[tuple[str, str], float]],
    setgn_scores: Optional[dict[tuple[str, str], float]],
    k_values: list[int] = _DEFAULT_K_VALUES,
) -> dict:
    """
    Run evaluation for all available baselines and return a comparison dict.

    Parameters
    ----------
    all_pairs      : all (concept_a, concept_b) pairs being evaluated
    test_positives : ground-truth positive pairs from held-out test set
    graph_scores   : centrality-based scores from graph_analyzer (always available)
    gnn_scores     : static GCN-based scores (None if GNN inactive)
    setgn_scores   : SE-TGN scores (None if SE-TGN inactive)
    k_values       : K values for P@K and NDCG@K

    Returns
    -------
    dict mapping baseline_name → metrics dict
    """
    results: dict[str, dict] = {}

    # Random baseline
    random_scores = baseline_random_scores(all_pairs)
    results["Random"] = compute_metrics(random_scores, test_positives, k_values)

    # Static graph baseline
    # Restrict to all_pairs only (some pairs may not have graph scores)
    graph_pred = {p: graph_scores.get(p, 0.0) for p in all_pairs}
    results["Static Graph"] = compute_metrics(graph_pred, test_positives, k_values)

    # GCN baseline (optional)
    if gnn_scores is not None:
        gnn_pred = {p: gnn_scores.get(p, 0.0) for p in all_pairs}
        results["GCN (untrained)"] = compute_metrics(gnn_pred, test_positives, k_values)

    # SE-TGN (the main model)
    if setgn_scores is not None:
        setgn_pred = {p: setgn_scores.get(p, 0.0) for p in all_pairs}
        results["SE-TGN"] = compute_metrics(setgn_pred, test_positives, k_values)

    return results


# ─────────────────────────────────────────────────────────────
# Utility: build all candidate pairs from a concept list
# ─────────────────────────────────────────────────────────────

def all_concept_pairs(concepts: list[str]) -> list[tuple[str, str]]:
    """Generate all canonical (smaller, larger) pairs from a concept list."""
    from itertools import combinations
    return [(min(a, b), max(a, b)) for a, b in combinations(concepts, 2)]


# ─────────────────────────────────────────────────────────────
# Status helper
# ─────────────────────────────────────────────────────────────

def evaluator_status() -> str:
    """Return a human-readable status string for the evaluator."""
    if not SKLEARN_AVAILABLE:
        return "Evaluation inactive (scikit-learn not installed — pip install scikit-learn)"
    return "Evaluation active (scikit-learn available)"
