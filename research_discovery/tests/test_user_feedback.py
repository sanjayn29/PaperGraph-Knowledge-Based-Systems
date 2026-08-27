"""
test_user_feedback.py
──────────────────────
Unit tests for human-in-the-loop feedback and personalized reranking.
"""

import pytest
from services.user_feedback import (
    record_feedback,
    compute_personalized_rankings,
    clear_user_profile,
    get_user_profile_summary,
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
