"""
test_provenance.py
──────────────────
Unit tests for evidence and provenance compilation.
"""

import networkx as nx
import numpy as np
import pytest
from services.provenance import build_candidate_provenance, compile_all_provenance
from services.temporal_graph import TemporalEvent, TemporalGraph


def test_build_candidate_provenance():
    G = nx.Graph()
    G.add_edge("GNN", "Bioactivity", weight=1)
    G.add_edge("GNN", "Drug Design", weight=3)
    G.add_edge("Bioactivity", "Drug Design", weight=2)

    papers = [
        {
            "paper_id": "p1",
            "title": "GNN for Molecules",
            "year": 2024,
            "authors": ["Author A"],
            "concepts": ["GNN", "Bioactivity", "Drug Design"],
        },
        {
            "paper_id": "p2",
            "title": "Bioactivity Analysis",
            "year": 2025,
            "authors": ["Author B"],
            "concepts": ["Bioactivity"],
        },
    ]

    candidate = {
        "concept_a": "GNN",
        "concept_b": "Bioactivity",
        "candidate_score": 0.85,
        "semantic_similarity": 0.90,
        "setgn_score": 0.88,
        "graph_score": 0.80,
    }

    prov = build_candidate_provenance(candidate, G, papers)

    assert prov["evidence_strength"] in ("HIGH", "MEDIUM", "EXPLORATORY")
    assert len(prov["supporting_papers"]) >= 1
    assert prov["supporting_papers"][0]["evidence_type"] == "direct_cooccurrence"
    assert "graph_evidence" in prov
    assert "narrative_points" in prov
    assert len(prov["narrative_points"]) > 0


def test_compile_all_provenance():
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C"])
    papers = [{"paper_id": "p1", "title": "Paper 1", "concepts": ["A", "B"]}]
    candidates = [{"concept_a": "A", "concept_b": "B", "candidate_score": 0.7}]

    all_prov = compile_all_provenance(candidates, G, papers)
    assert "A + B" in all_prov


def test_temporal_provenance_uses_temporal_event_api():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    candidate = {"concept_a": "A", "concept_b": "B"}

    empty_provenance = build_candidate_provenance(
        candidate, G, [], temporal_graph=TemporalGraph()
    )
    assert empty_provenance["temporal_evidence"] == {
        "concept_a_events": 0,
        "concept_b_events": 0,
        "first_seen_year_a": None,
        "first_seen_year_b": None,
        "shared_event_years": [],
    }

    temporal_graph = TemporalGraph()
    temporal_graph.add_event(
        TemporalEvent("A", "C", "paper_001", 2020, np.zeros(384))
    )
    temporal_graph.add_event(
        TemporalEvent("B", "C", "paper_002", 2021, np.zeros(384))
    )
    temporal_graph.add_event(
        TemporalEvent("A", "B", "paper_003", 2022, np.zeros(384))
    )
    temporal_graph.sort()

    temporal_evidence = build_candidate_provenance(
        candidate, G, [], temporal_graph=temporal_graph
    )["temporal_evidence"]
    assert temporal_evidence["concept_a_events"] == 2
    assert temporal_evidence["concept_b_events"] == 2
    assert temporal_evidence["first_seen_year_a"] == 2020
    assert temporal_evidence["first_seen_year_b"] == 2021
    assert temporal_evidence["shared_event_years"] == [2022]
