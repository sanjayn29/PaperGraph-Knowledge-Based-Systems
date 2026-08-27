"""
graph_analyzer.py
─────────────────
Generates and scores candidate concept-pair connections from the knowledge graph.

Role in the pipeline
--------------------
Provides graph-structural features (centrality, betweenness, novelty) that act as
supporting signals alongside SE-TGN's temporal link-prediction score.

Candidate score formula (3-tier priority):
─────────────────────────────────────────
1. SE-TGN ACTIVE (temporal model available):
       candidate_score = 0.50 × setgn_score
                       + 0.30 × graph_score
                       + 0.20 × semantic_similarity

2. GCN ACTIVE, SE-TGN inactive (GCN baseline):
       candidate_score = 0.40 × graph_score
                       + 0.30 × semantic_similarity
                       + 0.30 × gnn_score

3. GRAPH ONLY (pure fallback — no PyTorch):
       candidate_score = 0.57 × graph_score
                       + 0.43 × semantic_similarity
─────────────────────────────────────────────────────────────────────────────

graph_score is derived from:
  - Degree centrality of both nodes (high-degree concepts = structurally important)
  - Betweenness centrality (bridge concepts = potential cross-domain connectors)
  - Inverse edge weight (pairs NOT strongly connected = potentially underexplored)
  - Concept frequency
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Scoring weight constants
# ── SE-TGN active (primary temporal signal) ──────────────────
_W_SETGN_SETGN = 0.50
_W_SETGN_GRAPH = 0.30
_W_SETGN_SEM   = 0.20
# ── GCN active (optional baseline, no temporal) ───────────────
_W_GCN_GRAPH = 0.40
_W_GCN_SEM   = 0.30
_W_GCN_GNN   = 0.30
# ── Graph only (pure fallback) ────────────────────────────────
_W_GRAPH_ONLY_GRAPH = 0.57
_W_GRAPH_ONLY_SEM   = 0.43


def compute_centralities(G: nx.Graph) -> dict[str, dict[str, float]]:
    """
    Compute normalized degree and betweenness centrality for all nodes.

    Returns
    -------
    dict with keys 'degree' and 'betweenness', each mapping node → float in [0, 1]
    """
    if G.number_of_nodes() == 0:
        return {"degree": {}, "betweenness": {}}

    degree_centrality = nx.degree_centrality(G)

    # Betweenness is expensive for large graphs; use approximate version if needed
    n = G.number_of_nodes()
    if n > 200:
        # Approximate betweenness with k samples
        k = min(100, n)
        betweenness = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
    else:
        betweenness = nx.betweenness_centrality(G, normalized=True)

    return {"degree": degree_centrality, "betweenness": betweenness}


def _graph_score_for_pair(
    node_a: str,
    node_b: str,
    G: nx.Graph,
    centralities: dict[str, dict[str, float]],
    max_edge_weight: float,
) -> float:
    """
    Compute a graph_score in [0, 1] for a concept pair (node_a, node_b).

    High score when:
    - Both concepts are central in the graph (appear in many papers)
    - The direct edge between them is WEAK or absent (= potentially underexplored)
    - At least one concept bridges multiple clusters (high betweenness)

    Formula:
        centrality_score = mean(degree_a, degree_b)
        bridge_score     = max(betweenness_a, betweenness_b)
        novelty_bonus    = 1 - (edge_weight / max_edge_weight)  [0 if no edge]
        graph_score      = 0.45 * centrality + 0.30 * bridge + 0.25 * novelty_bonus
    """
    deg = centralities["degree"]
    bet = centralities["betweenness"]

    centrality_score = (deg.get(node_a, 0.0) + deg.get(node_b, 0.0)) / 2.0
    bridge_score = max(bet.get(node_a, 0.0), bet.get(node_b, 0.0))

    if G.has_edge(node_a, node_b):
        edge_weight = G[node_a][node_b].get("weight", 1)
        novelty_bonus = 1.0 - (edge_weight / max_edge_weight) if max_edge_weight > 0 else 0.0
    else:
        # No direct edge → maximum novelty bonus
        novelty_bonus = 1.0

    score = 0.45 * centrality_score + 0.30 * bridge_score + 0.25 * novelty_bonus
    return float(np.clip(score, 0.0, 1.0))


def rank_candidates(
    G: nx.Graph,
    concept_embeddings: dict[str, np.ndarray],
    gnn_scores: Optional[dict[tuple[str, str], float]] = None,
    setgn_scores: Optional[dict[tuple[str, str], float]] = None,
    prediction_time: Optional[int] = None,
    top_k: int = 20,
    exclude_strong_edges: bool = True,
    strong_edge_threshold: int = 3,
) -> list[dict]:
    """
    Generate and score all candidate concept pairs, returning the top-k.

    Score priority (descending):
    1. SE-TGN active  → 0.50 × setgn_score + 0.30 × graph_score + 0.20 × semantic
    2. GCN active     → 0.40 × graph_score + 0.30 × semantic + 0.30 × gnn_score
    3. Graph only     → 0.57 × graph_score + 0.43 × semantic

    Parameters
    ----------
    G                    : the concept co-occurrence graph
    concept_embeddings   : dict from embed_concepts() (may be empty if unavailable)
    gnn_scores           : optional dict (node_a, node_b) → float from gnn_model (GCN)
    setgn_scores         : optional dict (node_a, node_b) → float from se_tgn (SE-TGN)
    prediction_time      : the future year being predicted (displayed in UI)
    top_k                : number of top candidates to return
    exclude_strong_edges : if True, skip pairs with very strong co-occurrence
    strong_edge_threshold: minimum edge weight to exclude as "already known"

    Returns
    -------
    List of candidate dicts sorted by candidate_score descending.
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        logger.warning("Graph has fewer than 2 nodes — no candidates can be generated.")
        return []

    setgn_active = setgn_scores is not None and len(setgn_scores) > 0
    gnn_active = (not setgn_active) and gnn_scores is not None and len(gnn_scores) > 0
    centralities = compute_centralities(G)

    # Max edge weight for normalization
    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_edge_weight = max(edge_weights) if edge_weights else 1.0

    candidates: list[dict] = []

    for node_a, node_b in combinations(nodes, 2):
        # Optionally skip pairs with very strong direct co-occurrence
        if exclude_strong_edges and G.has_edge(node_a, node_b):
            if G[node_a][node_b].get("weight", 0) >= strong_edge_threshold:
                continue

        graph_score = _graph_score_for_pair(
            node_a, node_b, G, centralities, max_edge_weight
        )

        # Semantic similarity from embeddings (0 if embeddings unavailable)
        if concept_embeddings:
            emb_a = concept_embeddings.get(node_a)
            emb_b = concept_embeddings.get(node_b)
            if emb_a is not None and emb_b is not None:
                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)
                if norm_a > 0 and norm_b > 0:
                    semantic_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                else:
                    semantic_sim = 0.0
            else:
                semantic_sim = 0.0
        else:
            semantic_sim = 0.0

        canonical_key = (min(node_a, node_b), max(node_a, node_b))

        # GCN baseline score
        gnn_score = (gnn_scores or {}).get(canonical_key, 0.0)

        # SE-TGN score (primary when available)
        setgn_score = (setgn_scores or {}).get(canonical_key, 0.0)

        # Combined candidate score — priority: SE-TGN > GCN > graph-only
        if setgn_active:
            candidate_score = (
                _W_SETGN_SETGN * setgn_score
                + _W_SETGN_GRAPH * graph_score
                + _W_SETGN_SEM   * semantic_sim
            )
        elif gnn_active:
            candidate_score = (
                _W_GCN_GRAPH * graph_score
                + _W_GCN_SEM   * semantic_sim
                + _W_GCN_GNN   * gnn_score
            )
        else:
            candidate_score = (
                _W_GRAPH_ONLY_GRAPH * graph_score
                + _W_GRAPH_ONLY_SEM * semantic_sim
            )

        candidates.append(
            {
                "connection": f"{node_a} + {node_b}",
                "concept_a": node_a,
                "concept_b": node_b,
                "graph_score": round(graph_score, 4),
                "semantic_similarity": round(semantic_sim, 4),
                "gnn_score": round(gnn_score, 4) if gnn_active else None,
                "setgn_score": round(setgn_score, 4) if setgn_active else None,
                "candidate_score": round(float(np.clip(candidate_score, 0.0, 1.0)), 4),
                "gnn_active": gnn_active,
                "setgn_active": setgn_active,
                "prediction_time": prediction_time,
            }
        )

    candidates.sort(key=lambda x: x["candidate_score"], reverse=True)
    return candidates[:top_k]


def get_candidate_context(
    candidate: dict,
    G: nx.Graph,
    papers: list[dict],
) -> str:
    """
    Build a context string for the LLM evaluation stage.
    Summarizes what the graph knows about this candidate pair.
    """
    node_a = candidate["concept_a"]
    node_b = candidate["concept_b"]

    # Papers that mention each concept
    papers_a = [
        p["title"] or p["filename"]
        for p in papers
        if node_a in p.get("concepts", [])
    ]
    papers_b = [
        p["title"] or p["filename"]
        for p in papers
        if node_b in p.get("concepts", [])
    ]

    # Shared papers
    ids_a = set(p["paper_id"] for p in papers if node_a in p.get("concepts", []))
    ids_b = set(p["paper_id"] for p in papers if node_b in p.get("concepts", []))
    shared_ids = ids_a & ids_b
    shared_papers = [
        p["title"] or p["filename"]
        for p in papers
        if p["paper_id"] in shared_ids
    ]

    edge_info = ""
    if G.has_edge(node_a, node_b):
        w = G[node_a][node_b].get("weight", 0)
        edge_info = f"They co-occur directly in {w} paper(s)."
    else:
        edge_info = "They do NOT directly co-occur in any single paper."

    context = (
        f"Concept A: '{node_a}' — mentioned in {len(papers_a)} paper(s): "
        f"{', '.join(papers_a[:3])}{'...' if len(papers_a) > 3 else ''}.\n"
        f"Concept B: '{node_b}' — mentioned in {len(papers_b)} paper(s): "
        f"{', '.join(papers_b[:3])}{'...' if len(papers_b) > 3 else ''}.\n"
        f"Shared papers: {', '.join(shared_papers) if shared_papers else 'None'}.\n"
        f"{edge_info}\n"
        f"Graph score: {candidate['graph_score']:.3f}. "
        f"Semantic similarity: {candidate['semantic_similarity']:.3f}."
    )
    return context
