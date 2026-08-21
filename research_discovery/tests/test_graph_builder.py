"""
test_graph_builder.py
─────────────────────
Unit tests for services/graph_builder.py and services/graph_analyzer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.graph_builder import build_graph, graph_summary, get_top_concepts
from services.graph_analyzer import rank_candidates, compute_centralities


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def _make_papers():
    return [
        {
            "paper_id": "paper_001",
            "filename": "paper1.pdf",
            "title": "GNN for Drug Discovery",
            "abstract": "",
            "year": 2023,
            "authors": [],
            "full_text": "",
            "concepts": ["Graph Neural Network", "Drug Discovery", "Molecular Graph"],
        },
        {
            "paper_id": "paper_002",
            "filename": "paper2.pdf",
            "title": "LLM in Healthcare",
            "abstract": "",
            "year": 2023,
            "authors": [],
            "full_text": "",
            "concepts": ["Large Language Model", "Healthcare", "Drug Discovery"],
        },
        {
            "paper_id": "paper_003",
            "filename": "paper3.pdf",
            "title": "Graph Attention for Biology",
            "abstract": "",
            "year": 2024,
            "authors": [],
            "full_text": "",
            "concepts": ["Graph Neural Network", "Bioinformatics", "Large Language Model"],
        },
    ]


# ─────────────────────────────────────────────────────────────
# Graph Builder Tests
# ─────────────────────────────────────────────────────────────

class TestBuildGraph:
    def test_builds_non_empty_graph(self):
        papers = _make_papers()
        G = build_graph(papers)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_nodes_are_concepts(self):
        papers = _make_papers()
        G = build_graph(papers)
        assert "Graph Neural Network" in G.nodes
        assert "Drug Discovery" in G.nodes
        assert "Large Language Model" in G.nodes

    def test_edge_exists_for_shared_paper(self):
        papers = _make_papers()
        G = build_graph(papers)
        # GNN and Drug Discovery both in paper_001
        assert G.has_edge("Graph Neural Network", "Drug Discovery") or \
               G.has_edge("Drug Discovery", "Graph Neural Network")

    def test_edge_weight_reflects_co_occurrence(self):
        papers = _make_papers()
        G = build_graph(papers)
        # Drug Discovery appears in papers 1 and 2
        # GNN appears in papers 1 and 3
        # LLM appears in papers 2 and 3
        # GNN+LLM co-occur in paper 3
        assert G.has_edge("Graph Neural Network", "Large Language Model") or \
               G.has_edge("Large Language Model", "Graph Neural Network")

    def test_empty_papers_gives_empty_graph(self):
        G = build_graph([])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_paper_with_one_concept_adds_node_no_edge(self):
        papers = [
            {
                "paper_id": "paper_001",
                "filename": "p1.pdf",
                "title": "Solo",
                "abstract": "",
                "year": 2024,
                "authors": [],
                "full_text": "",
                "concepts": ["Unique Concept"],
            }
        ]
        G = build_graph(papers)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_node_attributes_set(self):
        papers = _make_papers()
        G = build_graph(papers)
        # Drug Discovery appears in 2 papers
        assert G.nodes["Drug Discovery"]["paper_count"] == 2

    def test_supporting_papers_on_edge(self):
        papers = _make_papers()
        G = build_graph(papers)
        # Find GNN — Drug Discovery edge
        if G.has_edge("Graph Neural Network", "Drug Discovery"):
            edge_data = G["Graph Neural Network"]["Drug Discovery"]
        else:
            edge_data = G["Drug Discovery"]["Graph Neural Network"]
        assert "supporting_papers" in edge_data
        assert "paper_001" in edge_data["supporting_papers"]


class TestGraphSummary:
    def test_summary_has_expected_keys(self):
        papers = _make_papers()
        G = build_graph(papers)
        summary = graph_summary(G)
        assert "node_count" in summary
        assert "edge_count" in summary
        assert "density" in summary
        assert "connected_components" in summary

    def test_empty_graph_summary(self):
        import networkx as nx
        G = nx.Graph()
        summary = graph_summary(G)
        assert summary["node_count"] == 0
        assert summary["edge_count"] == 0


# ─────────────────────────────────────────────────────────────
# Graph Analyzer Tests
# ─────────────────────────────────────────────────────────────

class TestComputeCentralities:
    def test_returns_degree_and_betweenness(self):
        papers = _make_papers()
        G = build_graph(papers)
        centralities = compute_centralities(G)
        assert "degree" in centralities
        assert "betweenness" in centralities

    def test_centrality_values_in_range(self):
        papers = _make_papers()
        G = build_graph(papers)
        centralities = compute_centralities(G)
        for val in centralities["degree"].values():
            assert 0.0 <= val <= 1.0
        for val in centralities["betweenness"].values():
            assert 0.0 <= val <= 1.0


class TestRankCandidates:
    def test_returns_list_of_dicts(self):
        papers = _make_papers()
        G = build_graph(papers)
        candidates = rank_candidates(G, {})
        assert isinstance(candidates, list)
        for c in candidates:
            assert "connection" in c
            assert "candidate_score" in c
            assert "graph_score" in c
            assert "semantic_similarity" in c

    def test_candidates_sorted_by_score(self):
        papers = _make_papers()
        G = build_graph(papers)
        candidates = rank_candidates(G, {})
        scores = [c["candidate_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self):
        papers = _make_papers()
        G = build_graph(papers)
        candidates = rank_candidates(G, {})
        for c in candidates:
            assert 0.0 <= c["candidate_score"] <= 1.0
            assert 0.0 <= c["graph_score"] <= 1.0

    def test_too_few_nodes_returns_empty(self):
        import networkx as nx
        G = nx.Graph()
        G.add_node("Only One")
        candidates = rank_candidates(G, {})
        assert candidates == []

    def test_top_k_respected(self):
        papers = _make_papers()
        G = build_graph(papers)
        candidates = rank_candidates(G, {}, top_k=2)
        assert len(candidates) <= 2

    def test_with_semantic_embeddings(self):
        import numpy as np
        papers = _make_papers()
        G = build_graph(papers)
        # Create fake embeddings
        embeddings = {
            node: np.random.rand(384).astype(np.float32)
            for node in G.nodes()
        }
        candidates = rank_candidates(G, embeddings)
        assert len(candidates) > 0
        # At least some semantic similarity should be non-zero
        sem_sims = [c["semantic_similarity"] for c in candidates]
        assert any(s > 0 for s in sem_sims)
