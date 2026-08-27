"""
test_research_gap_detector.py
─────────────────────────────
Unit tests for Research Gap Detection scoring and heuristics.
"""

import networkx as nx
import pytest
from services.research_gap_detector import (
    compute_structural_opportunity,
    compute_research_gap_score,
    detect_research_gaps,
)


def test_compute_structural_opportunity():
    G = nx.Graph()
    G.add_edge("A", "B", weight=10)  # Saturated edge
    G.add_edge("A", "C", weight=1)
    G.add_edge("B", "D", weight=1)
    # C and D are unconnected, but central through A and B

    # Directly connected with heavy weight has lower gap opportunity than missing edge
    opp_ab = compute_structural_opportunity("A", "B", G)
    opp_cd = compute_structural_opportunity("C", "D", G)

    assert 0.0 <= opp_ab <= 1.0
    assert 0.0 <= opp_cd <= 1.0


def test_compute_research_gap_score():
    score = compute_research_gap_score(
        semantic_compatibility=0.90,
        temporal_emergence=0.80,
        structural_opportunity=0.85,
        cross_domain_potential=0.70,
        novelty_potential=0.90,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.75


def test_detect_research_gaps():
    G = nx.Graph()
    G.add_nodes_from(["GNN", "Bioactivity", "Transformer"])
    G.add_edge("GNN", "Transformer", weight=5)

    candidates = [
        {
            "concept_a": "GNN",
            "concept_b": "Bioactivity",
            "semantic_similarity": 0.88,
        },
        {
            "concept_a": "GNN",
            "concept_b": "Transformer",
            "semantic_similarity": 0.95,
        },
    ]

    concept_domains = {
        "GNN": {"domain": "Artificial Intelligence"},
        "Bioactivity": {"domain": "Biology & Bioinformatics"},
        "Transformer": {"domain": "Artificial Intelligence"},
    }

    papers = [
        {"paper_id": "p1", "title": "GNN Paper", "year": 2024, "concepts": ["GNN"]},
        {"paper_id": "p2", "title": "Bio Paper", "year": 2025, "concepts": ["Bioactivity"]},
    ]

    gaps = detect_research_gaps(
        candidates=candidates,
        G=G,
        concept_domains=concept_domains,
        papers=papers,
        min_gap_score=0.30,
    )

    assert len(gaps) >= 1
    top_gap = gaps[0]
    assert "gap_score" in top_gap
    assert "explanation" in top_gap
    assert top_gap["concept_a"] == "GNN"
