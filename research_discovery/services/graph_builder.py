"""
graph_builder.py
────────────────
Builds the concept co-occurrence knowledge graph using NetworkX.

Role in the pipeline
--------------------
Plays the structural role of the co-occurrence event stream preparation in the
base paper. Instead of a multi-year temporal event stream, we build a single
snapshot graph from 5–10 uploaded PDFs.

Graph schema
------------
Nodes : unique canonical concept strings
  - attributes: frequency (total co-occurrence count), paper_count (# papers)

Edges : connect any two concepts that appear together in the same paper
  - attributes:
      weight          : number of papers in which BOTH concepts appear
      co_occurrence   : same as weight (explicit alias)
      supporting_papers: list of paper_ids where both concepts co-occur
"""

from __future__ import annotations

import logging
from itertools import combinations


import networkx as nx

logger = logging.getLogger(__name__)


def build_graph(
    papers: list[dict],
) -> nx.Graph:
    """
    Build a concept co-occurrence graph from the processed papers.

    Parameters
    ----------
    papers               : list of paper dicts (each must have 'concepts' list)

    Returns
    -------
    nx.Graph — undirected weighted co-occurrence graph
    """
    G = nx.Graph()

    # ── Collect all concepts and their paper membership ───────
    concept_freq: dict[str, int] = {}          # concept → total mentions
    concept_papers: dict[str, set[str]] = {}   # concept → set of paper_ids

    for paper in papers:
        pid = paper["paper_id"]
        for concept in paper.get("concepts", []):
            concept_freq[concept] = concept_freq.get(concept, 0) + 1
            if concept not in concept_papers:
                concept_papers[concept] = set()
            concept_papers[concept].add(pid)

    # ── Add nodes ─────────────────────────────────────────────
    for concept, freq in concept_freq.items():
        G.add_node(
            concept,
            frequency=freq,
            paper_count=len(concept_papers.get(concept, set())),
        )

    # ── Add edges (co-occurrence within same paper) ───────────
    # edge_data: (concept_a, concept_b) → {"weight": int, "supporting_papers": list}
    edge_data: dict[tuple[str, str], dict] = {}

    for paper in papers:
        pid = paper["paper_id"]
        concepts = paper.get("concepts", [])
        # Deduplicate within this paper
        unique_concepts = list(dict.fromkeys(concepts))

        for c_a, c_b in combinations(unique_concepts, 2):
            # Canonical ordering to avoid (A,B) vs (B,A) duplicates
            key = (min(c_a, c_b), max(c_a, c_b))
            if key not in edge_data:
                edge_data[key] = {"weight": 0, "co_occurrence": 0, "supporting_papers": []}
            edge_data[key]["weight"] += 1
            edge_data[key]["co_occurrence"] += 1
            edge_data[key]["supporting_papers"].append(pid)

    for (c_a, c_b), attrs in edge_data.items():
        # Deduplicate supporting_papers
        attrs["supporting_papers"] = sorted(set(attrs["supporting_papers"]))
        G.add_edge(c_a, c_b, **attrs)

    logger.info(
        "Knowledge graph built: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def graph_summary(G: nx.Graph) -> dict:
    """
    Return a summary dict of basic graph statistics.
    """
    if G.number_of_nodes() == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "density": 0.0,
            "connected_components": 0,
            "avg_degree": 0.0,
        }

    degrees = [d for _, d in G.degree()]
    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "connected_components": nx.number_connected_components(G),
        "avg_degree": round(sum(degrees) / len(degrees), 2),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
    }


def get_top_concepts(G: nx.Graph, n: int = 10) -> list[tuple[str, int]]:
    """Return the top-n concepts by degree centrality."""
    degree_dict = dict(G.degree())
    return sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:n]


# ─────────────────────────────────────────────────────────────
# Temporal graph factory — convenience re-export
# ─────────────────────────────────────────────────────────────
# Imported here so callers can do:
#     from services.graph_builder import build_graph, build_temporal_events
# without needing to know about the separate temporal_graph module.

def build_temporal_events(papers: list[dict], concept_embeddings: dict) -> tuple:
    """
    Build a TemporalGraph from processed papers.

    Delegates to services.temporal_graph.build_temporal_events().

    Returns
    -------
    (temporal_graph, warnings)  — see temporal_graph.build_temporal_events docstring.
    """
    from services.temporal_graph import build_temporal_events as _build
    return _build(papers, concept_embeddings)

