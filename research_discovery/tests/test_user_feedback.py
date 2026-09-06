"""
test_user_feedback.py
──────────────────────
Unit tests for human-in-the-loop feedback and personalized reranking.
"""

from unittest.mock import patch

import pytest
from services.user_feedback import (
    record_feedback,
    compute_personalized_rankings,
    clear_user_profile,
    get_user_profile_summary,
    WEIGHT_ORIGINAL,
    WEIGHT_PREFERENCE,
    WEIGHT_SIMILARITY,
)


def test_user_feedback_and_personalization():
    clear_user_profile()

    candidates = [
        {"candidate_id": "c1", "concept_a": "GNN", "concept_b": "NLP", "candidate_score": 0.85},
        {"candidate_id": "c2", "concept_a": "GNN", "concept_b": "Drug Discovery", "candidate_score": 0.80},
        {"candidate_id": "c3", "concept_a": "CNN", "concept_b": "Vision", "candidate_score": 0.75},
    ]

    concept_domains = {
        "GNN": {"domain": "Artificial Intelligence"},
        "NLP": {"domain": "Computer Science"},
        "Drug Discovery": {"domain": "Chemistry & Materials"},
        "CNN": {"domain": "Artificial Intelligence"},
        "Vision": {"domain": "Artificial Intelligence"},
    }

    # Baseline before feedback: original order preserved
    ranked_initial = compute_personalized_rankings(candidates, concept_domains)
    assert ranked_initial[0]["candidate_id"] == "c1"

    # User gives strong positive feedback to Drug Discovery
    custom_profile = {
        "preferred_concepts": {"Drug Discovery": 2.0},
        "preferred_domains": {"Chemistry & Materials": 1.0},
        "disliked_concepts": {"NLP": 1.5},
        "feedback_history": [],
    }

    ranked_personalized = compute_personalized_rankings(
        candidates, concept_domains, custom_profile=custom_profile
    )

    # After feedback, c2 (Drug Discovery) should jump to rank #1 and c1 (NLP) should drop
    assert ranked_personalized[0]["candidate_id"] == "c2"
    assert ranked_personalized[0]["personalized_rank"] == 1
    assert ranked_personalized[0]["rank_delta"] > 0  # Moved up


def test_feedback_isolated_between_contexts_and_persisted(tmp_path):
    feedback_file = tmp_path / "user_feedback.json"
    candidates = [
        {"candidate_id": "c1", "concept_a": "Vision", "concept_b": "NLP", "candidate_score": 0.85},
        {"candidate_id": "c2", "concept_a": "GNN", "concept_b": "Drug Discovery", "candidate_score": 0.80},
    ]
    domains = {
        "Vision": {"domain": "Artificial Intelligence"},
        "GNN": {"domain": "Artificial Intelligence"},
        "NLP": {"domain": "Computer Science"},
        "Drug Discovery": {"domain": "Chemistry & Materials"},
    }

    with patch("services.user_feedback._FEEDBACK_FILE", feedback_file):
        clear_user_profile("context_a")
        clear_user_profile("context_b")
        record_feedback(
            "c2", "GNN", "Drug Discovery", "highly_relevant",
            "Artificial Intelligence", "Chemistry & Materials",
            context_id="context_a",
        )

        ranked_a = compute_personalized_rankings(
            candidates, domains, context_id="context_a"
        )
        ranked_b = compute_personalized_rankings(
            candidates, domains, context_id="context_b"
        )
        summary_a = get_user_profile_summary("context_a")
        summary_b = get_user_profile_summary("context_b")

    assert ranked_a[0]["candidate_id"] == "c2"
    assert [candidate["candidate_id"] for candidate in ranked_b] == ["c1", "c2"]
    assert summary_a["total_feedbacks"] == 1
    assert summary_b["total_feedbacks"] == 0


def test_feedback_weights_remain_unchanged():
    assert WEIGHT_ORIGINAL == 0.70
    assert WEIGHT_PREFERENCE == 0.15
    assert WEIGHT_SIMILARITY == 0.15
