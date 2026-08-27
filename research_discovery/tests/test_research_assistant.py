"""
test_research_assistant.py
──────────────────────────
Unit tests for interactive research assistant context formatting and fallback logic.
"""

import pytest
from services.research_assistant import (
    build_assistant_context,
    query_research_assistant,
    PRESET_QUESTIONS,
)


def test_build_assistant_context():
    mock_result = {
        "papers": [
            {"paper_id": "p1", "title": "Attention Is All You Need", "year": 2017, "concepts": ["Transformer", "Attention"]}
        ],
        "candidates": [
            {"concept_a": "Transformer", "concept_b": "GNN", "candidate_score": 0.89, "setgn_score": 0.92}
        ],
        "research_gaps": [
            {"concept_a": "Transformer", "concept_b": "GNN", "gap_score": 0.84, "explanation": "Novel synergy."}
        ],
        "final_result": {
            "connection": "Transformer + GNN",
            "novelty": 5,
            "impact": 5,
            "research_direction": "Hybrid relational transformers",
        },
    }

    context = build_assistant_context(mock_result)
    assert "Attention Is All You Need" in context
    assert "Transformer + GNN" in context
    assert "Hybrid relational transformers" in context


def test_fallback_research_assistant():
    mock_result = {
        "papers": [{"title": "GNN in Healthcare", "year": 2024}],
        "candidates": [
            {"concept_a": "GNN", "concept_b": "Healthcare", "candidate_score": 0.91, "semantic_similarity": 0.88, "setgn_score": 0.90}
        ],
        "final_result": {
            "connection": "GNN + Healthcare",
            "research_questions": ["How does GNN scale for EHR graphs?"],
        },
    }

    # Deterministic answer for "why recommended?"
    ans_why = query_research_assistant(PRESET_QUESTIONS["why_recommended"], mock_result)
    assert "GNN + Healthcare" in ans_why
    assert "Candidate Score" in ans_why

    # Deterministic answer for research questions
    ans_rq = query_research_assistant(PRESET_QUESTIONS["research_questions"], mock_result)
    assert "EHR graphs" in ans_rq
