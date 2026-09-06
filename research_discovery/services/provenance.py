"""
provenance.py
─────────────
Evidence & Provenance Tracking Engine for PaperGraph.

Ensures every recommendation, research gap, and generated insight is fully
traceable to concrete papers, extracted concepts, graph topology, and model metrics.
"""

from __future__ import annotations

import logging
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)


def build_candidate_provenance(
    candidate: dict,
    G: nx.Graph,
    papers: list[dict],
    temporal_graph=None,
    research_gaps: Optional[list[dict]] = None,
) -> dict:
    """
    Construct a structured provenance record for a single candidate connection.

    Parameters
    ----------
    candidate     : candidate dict with 'concept_a', 'concept_b', and model scores
    G             : NetworkX knowledge graph
    papers        : list of paper dicts
    temporal_graph: optional TemporalGraph object
    research_gaps : optional list of detected research gaps

    Returns
    -------
    Provenance dictionary with full traceability evidence.
    """
    ca = candidate.get("concept_a", "")
    cb = candidate.get("concept_b", "")

    # 1. Paper-level evidence
    supporting_papers: list[dict] = []
    direct_papers: list[str] = []
    ca_papers: list[str] = []
    cb_papers: list[str] = []

    for p in papers:
        pid = p.get("paper_id", "")
        title = p.get("title", "Untitled")
        year = p.get("year")
        p_concepts = set(p.get("concepts", []))

        has_a = ca in p_concepts
        has_b = cb in p_concepts

        if has_a and has_b:
            direct_papers.append(title)
            supporting_papers.append({
                "paper_id": pid,
                "title": title,
                "year": year,
                "authors": p.get("authors", []),
                "evidence_type": "direct_cooccurrence",
                "matched_concepts": [ca, cb],
            })
        elif has_a:
            ca_papers.append(title)
            supporting_papers.append({
                "paper_id": pid,
                "title": title,
                "year": year,
                "authors": p.get("authors", []),
                "evidence_type": "concept_a_context",
                "matched_concepts": [ca],
            })
        elif has_b:
            cb_papers.append(title)
            supporting_papers.append({
                "paper_id": pid,
                "title": title,
                "year": year,
                "authors": p.get("authors", []),
                "evidence_type": "concept_b_context",
                "matched_concepts": [cb],
            })

    # 2. Graph topology evidence
    graph_evidence: dict = {
        "concept_a_degree": G.degree(ca) if G.has_node(ca) else 0,
        "concept_b_degree": G.degree(cb) if G.has_node(cb) else 0,
        "direct_edge": G.has_edge(ca, cb),
        "cooccurrence_weight": G[ca][cb].get("weight", 0) if G.has_edge(ca, cb) else 0,
        "common_neighbors": [],
        "shortest_path_length": None,
    }

    if G.has_node(ca) and G.has_node(cb):
        try:
            graph_evidence["common_neighbors"] = list(nx.common_neighbors(G, ca, cb))[:8]
        except Exception:
            pass

        try:
            graph_evidence["shortest_path_length"] = nx.shortest_path_length(G, ca, cb)
        except nx.NetworkXNoPath:
            graph_evidence["shortest_path_length"] = "No path"

    # 3. Temporal event evidence
    temporal_evidence: dict = {
        "concept_a_events": 0,
        "concept_b_events": 0,
        "first_seen_year_a": None,
        "first_seen_year_b": None,
        "shared_event_years": [],
    }

    if temporal_graph and not temporal_graph.is_empty():
        years_a = [
            e.timestamp
            for e in temporal_graph.events
            if e.source_concept == ca or e.target_concept == ca
        ]
        years_b = [
            e.timestamp
            for e in temporal_graph.events
            if e.source_concept == cb or e.target_concept == cb
        ]

        if years_a:
            temporal_evidence["concept_a_events"] = len(years_a)
            temporal_evidence["first_seen_year_a"] = min(years_a)
        if years_b:
            temporal_evidence["concept_b_events"] = len(years_b)
            temporal_evidence["first_seen_year_b"] = min(years_b)

        shared_years = [
            e.timestamp for e in temporal_graph.events
            if (e.source_concept == ca and e.target_concept == cb)
            or (e.source_concept == cb and e.target_concept == ca)
        ]
        temporal_evidence["shared_event_years"] = sorted(list(set(shared_years)))

    # 4. Model scores summary
    model_scores: dict = {
        "candidate_score": candidate.get("candidate_score", 0.0),
        "semantic_similarity": candidate.get("semantic_similarity", 0.0),
        "graph_score": candidate.get("graph_score", 0.0),
        "setgn_score": candidate.get("setgn_score"),
        "gnn_score": candidate.get("gnn_score"),
    }

    # Match research gap score if available
    gap_info = None
    if research_gaps:
        for g in research_gaps:
            if (g["concept_a"] == ca and g["concept_b"] == cb) or (g["concept_a"] == cb and g["concept_b"] == ca):
                gap_info = g
                break
    if gap_info:
        model_scores["research_gap_score"] = gap_info.get("gap_score")

    # 5. Assess evidence strength
    if direct_papers and candidate.get("candidate_score", 0) >= 0.70:
        strength = "HIGH"
    elif len(supporting_papers) >= 2 or candidate.get("semantic_similarity", 0) >= 0.65:
        strength = "MEDIUM"
    else:
        strength = "EXPLORATORY"

    # 6. Traceability narrative summary
    narrative_points = []
    if direct_papers:
        narrative_points.append(
            f"Supported directly in {len(direct_papers)} paper(s): {', '.join(direct_papers[:2])}"
        )
    else:
        narrative_points.append(
            f"Indirect connection across literature (Shortest path: {graph_evidence['shortest_path_length']})"
        )

    if graph_evidence["common_neighbors"]:
        narrative_points.append(
            f"Bridged via common concepts: {', '.join(graph_evidence['common_neighbors'][:3])}"
        )

    if candidate.get("setgn_score") is not None:
        narrative_points.append(
            f"SE-TGN temporal link probability: {candidate['setgn_score']:.3f}"
        )

    narrative_points.append(
        f"Semantic embedding compatibility: {candidate.get('semantic_similarity', 0.0):.3f}"
    )

    return {
        "candidate": f"{ca} + {cb}",
        "concept_a": ca,
        "concept_b": cb,
        "evidence_strength": strength,
        "supporting_papers": supporting_papers,
        "graph_evidence": graph_evidence,
        "temporal_evidence": temporal_evidence,
        "semantic_evidence": {
            "similarity": candidate.get("semantic_similarity", 0.0),
            "embedding_dimension": 384,
            "metric": "Cosine Similarity",
        },
        "model_scores": model_scores,
        "narrative_points": narrative_points,
    }


def compile_all_provenance(
    candidates: list[dict],
    G: nx.Graph,
    papers: list[dict],
    temporal_graph=None,
    research_gaps: Optional[list[dict]] = None,
) -> dict[str, dict]:
    """
    Compile provenance records for all candidates.

    Returns
    -------
    { "concept_a + concept_b": provenance_dict }
    """
    provenance_map: dict[str, dict] = {}
    for cand in candidates:
        key = f"{cand.get('concept_a', '')} + {cand.get('concept_b', '')}"
        prov = build_candidate_provenance(cand, G, papers, temporal_graph, research_gaps)
        provenance_map[key] = prov

    return provenance_map
