"""
research_gap_detector.py
────────────────────────
Research Gap Detection Engine for PaperGraph.

Detects underexplored scientific relationships where concepts are semantically
compatible, structurally significant, but weakly connected in the literature.

Formula:
ResearchGapScore =
    0.30 * SemanticCompatibility
  + 0.25 * TemporalEmergence
  + 0.20 * StructuralOpportunity
  + 0.15 * CrossDomainPotential
  + 0.10 * NoveltyPotential
"""

from __future__ import annotations

import logging
from typing import Optional
import networkx as nx
import numpy as np

from services.domain_analyzer import get_domain_distance

logger = logging.getLogger(__name__)

# Configurable weights
WEIGHT_SEMANTIC = 0.30
WEIGHT_TEMPORAL = 0.25
WEIGHT_STRUCTURAL = 0.20
WEIGHT_CROSS_DOMAIN = 0.15
WEIGHT_NOVELTY = 0.10


def compute_temporal_emergence(
    concept_a: str,
    concept_b: str,
    temporal_graph,
) -> float:
    """
    Compute the temporal emergence score based on the recency and growth of concepts
    in the temporal event stream. Returns a normalized float [0.0, 1.0].
    """
    if temporal_graph is None or temporal_graph.is_empty:
        return 0.5

    years = temporal_graph.year_range
    if years[0] == 0:
        return 0.5

    min_yr, max_yr = years
    year_span = max(1, max_yr - min_yr)

    # Find events for concept_a and concept_b
    events_a = [e.timestamp for e in temporal_graph.events if e.source == concept_a or e.target == concept_a]
    events_b = [e.timestamp for e in temporal_graph.events if e.source == concept_b or e.target == concept_b]

    if not events_a or not events_b:
        return 0.4

    # Calculate average normalized timestamp (higher if in recent years)
    avg_a = (np.mean(events_a) - min_yr) / year_span
    avg_b = (np.mean(events_b) - min_yr) / year_span

    emergence = 0.5 * (avg_a + avg_b)
    return round(float(np.clip(emergence, 0.0, 1.0)), 4)


def compute_structural_opportunity(
    u: str,
    v: str,
    G: nx.Graph,
) -> float:
    """
    Evaluate structural opportunity: high if both nodes are individually well-connected
    or central, but share few or zero direct co-occurrences.
    """
    if not G.has_node(u) or not G.has_node(v):
        return 0.5

    deg_u = G.degree(u)
    deg_v = G.degree(v)
    max_deg = max((d for _, d in G.degree()), default=1) or 1

    # Individual importance (normalized degree)
    importance = (deg_u + deg_v) / (2.0 * max_deg)

    # Edge directness penalty
    if G.has_edge(u, v):
        weight = G[u][v].get("weight", 1)
        # Low weight has higher gap opportunity than saturated edge
        edge_penalty = min(1.0, weight / 5.0)
    else:
        # No direct edge = high structural gap opportunity
        try:
            path_len = nx.shortest_path_length(G, u, v)
            edge_penalty = 0.1 if path_len == 2 else 0.0
        except nx.NetworkXNoPath:
            edge_penalty = 0.0

    opportunity = 0.6 * importance + 0.4 * (1.0 - edge_penalty)
    return round(float(np.clip(opportunity, 0.0, 1.0)), 4)


def compute_research_gap_score(
    semantic_compatibility: float,
    temporal_emergence: float,
    structural_opportunity: float,
    cross_domain_potential: float,
    novelty_potential: float,
) -> float:
    """
    Calculate the unified Research Gap Score normalized to [0.0, 1.0].
    """
    score = (
        WEIGHT_SEMANTIC * np.clip(semantic_compatibility, 0.0, 1.0)
        + WEIGHT_TEMPORAL * np.clip(temporal_emergence, 0.0, 1.0)
        + WEIGHT_STRUCTURAL * np.clip(structural_opportunity, 0.0, 1.0)
        + WEIGHT_CROSS_DOMAIN * np.clip(cross_domain_potential, 0.0, 1.0)
        + WEIGHT_NOVELTY * np.clip(novelty_potential, 0.0, 1.0)
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def _generate_explanation(
    concept_a: str,
    concept_b: str,
    dom_a: str,
    dom_b: str,
    existing_conn: int,
    semantic_score: float,
    gap_score: float,
) -> str:
    """Generate a concise, human-interpretable rationale for the identified gap."""
    if dom_a != dom_b and dom_a != "Unknown" and dom_b != "Unknown":
        domain_str = f"an interdisciplinary bridge between {dom_a} and {dom_b}"
    else:
        domain_str = f"within {dom_a}"

    if existing_conn == 0:
        conn_str = "no direct co-occurrence in the current corpus"
    elif existing_conn == 1:
        conn_str = "only 1 shared paper in the corpus"
    else:
        conn_str = f"limited direct co-occurrence ({existing_conn} shared papers)"

    return (
        f"'{concept_a}' and '{concept_b}' exhibit high semantic compatibility ({semantic_score:.2f}) "
        f"and represent {domain_str}, yet have {conn_str}. "
        f"This indicates an underexplored research frontier with a gap score of {gap_score:.2f}."
    )


def detect_research_gaps(
    candidates: list[dict],
    G: nx.Graph,
    temporal_graph=None,
    concept_domains: Optional[dict[str, dict]] = None,
    papers: Optional[list[dict]] = None,
    min_gap_score: float = 0.50,
    top_k: int = 15,
) -> list[dict]:
    """
    Detect, score, and rank research gaps from candidate concept pairs.

    Returns
    -------
    List of research gap dictionaries sorted by gap_score descending.
    """
    gaps: list[dict] = []
    concept_domains = concept_domains or {}

    for idx, cand in enumerate(candidates):
        ca = cand.get("concept_a", "")
        cb = cand.get("concept_b", "")
        if not ca or not cb:
            continue

        # 1. Semantic compatibility
        semantic_sim = float(cand.get("semantic_similarity", 0.5))

        # 2. Temporal emergence
        temporal_score = compute_temporal_emergence(ca, cb, temporal_graph)

        # 3. Structural opportunity
        structural_score = compute_structural_opportunity(ca, cb, G)

        # 4. Cross-domain potential
        dom_a = concept_domains.get(ca, {}).get("domain", "Unknown")
        dom_b = concept_domains.get(cb, {}).get("domain", "Unknown")
        cross_domain_pot = get_domain_distance(dom_a, dom_b)

        # 5. Novelty potential (inversely related to direct co-occurrence weight)
        existing_conn = 0
        if G.has_edge(ca, cb):
            existing_conn = G[ca][cb].get("weight", 1)
        novelty_score = max(0.1, 1.0 - (existing_conn / 5.0))

        gap_score = compute_research_gap_score(
            semantic_compatibility=semantic_sim,
            temporal_emergence=temporal_score,
            structural_opportunity=structural_score,
            cross_domain_potential=cross_domain_pot,
            novelty_potential=novelty_score,
        )

        if gap_score >= min_gap_score:
            # Find supporting papers mentioning either or both
            supporting_papers = []
            if papers:
                for p in papers:
                    p_concepts = set(p.get("concepts", []))
                    if ca in p_concepts or cb in p_concepts:
                        supporting_papers.append({
                            "paper_id": p.get("paper_id", ""),
                            "title": p.get("title", "Untitled"),
                            "year": p.get("year"),
                            "contains": [c for c in (ca, cb) if c in p_concepts],
                        })

            explanation = _generate_explanation(
                ca, cb, dom_a, dom_b, existing_conn, semantic_sim, gap_score
            )

            status = "highly_promising" if gap_score >= 0.75 else "underexplored"

            gaps.append({
                "gap_id": f"gap_{idx + 1:03d}",
                "concept_a": ca,
                "concept_b": cb,
                "domain_a": dom_a,
                "domain_b": dom_b,
                "gap_score": gap_score,
                "semantic_score": round(semantic_sim, 4),
                "temporal_score": round(temporal_score, 4),
                "structural_score": round(structural_score, 4),
                "cross_domain_score": round(cross_domain_pot, 4),
                "novelty_score": round(novelty_score, 4),
                "existing_connections": existing_conn,
                "supporting_papers": supporting_papers[:5],
                "explanation": explanation,
                "status": status,
            })

    gaps.sort(key=lambda x: x["gap_score"], reverse=True)
    return gaps[:top_k]
