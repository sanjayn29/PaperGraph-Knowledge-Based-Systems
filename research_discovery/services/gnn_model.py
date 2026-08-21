"""
gnn_model.py
────────────
Optional lightweight GNN encoder component for PaperGraph.

Role in the pipeline
--------------------
Provides an ADDITIONAL scoring signal analogous (in spirit only) to the
graph encoder inside SE-TGN from the base paper. This is NOT a trained
temporal GNN — it is a small 2-layer GCN/GAT that runs a single forward
pass over the concept co-occurrence graph to produce node representations.
These are used only as an extra signal; the system works without them.

Important caveats (displayed in the UI's Technical Details section):
- No training corpus, no training loop — this is a lightweight structural
  encoder, not a validated link-prediction model.
- The GNN component is inactive when:
    (a) PyTorch or PyTorch Geometric is not installed
    (b) The graph has fewer than MIN_NODES_FOR_GNN nodes
    (c) Any runtime error occurs during the forward pass
- When inactive, candidate_score falls back to the 2-weight formula:
      0.57 × graph_score + 0.43 × semantic_similarity

Installation (optional):
    pip install torch torch-geometric
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Minimum graph size for the GNN to be meaningful
MIN_NODES_FOR_GNN = 4

# ─────────────────────────────────────────────────────────────
# Optional PyTorch + PyG imports
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F  # noqa: N812

    try:
        from torch_geometric.data import Data as PyGData
        from torch_geometric.nn import GCNConv, GATConv

        GNN_AVAILABLE = True
        logger.info("PyTorch Geometric available — GNN component ACTIVE.")
    except ImportError:
        GNN_AVAILABLE = False
        logger.info(
            "torch_geometric not installed — GNN component INACTIVE. "
            "Install with: pip install torch-geometric"
        )
except ImportError:
    GNN_AVAILABLE = False
    logger.info(
        "PyTorch not installed — GNN component INACTIVE. "
        "Install with: pip install torch"
    )


# ─────────────────────────────────────────────────────────────
# GCN model definition (only defined when PyG is available)
# ─────────────────────────────────────────────────────────────
if GNN_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    from torch_geometric.nn import GCNConv

    class _LightweightGCN(nn.Module):  # type: ignore[misc]
        """
        A 2-layer Graph Convolutional Network used as a structural encoder.

        Input features: node degree + betweenness centrality + frequency
        (all normalized to [0, 1]).
        Output: node embeddings of dimension `hidden_dim`.
        """

        def __init__(self, in_channels: int, hidden_dim: int = 32):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        def forward(self, x, edge_index):  # type: ignore[override]
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            return x


def compute_gnn_scores(
    G: nx.Graph,
    concept_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> Optional[dict[tuple[str, str], float]]:
    """
    Run the lightweight GCN encoder on the concept graph and return
    pairwise similarity scores for all node pairs.

    Parameters
    ----------
    G                  : the concept co-occurrence graph (NetworkX)
    concept_embeddings : optional pre-computed sentence-transformer embeddings
                         used to augment node features (may be None)

    Returns
    -------
    dict mapping (node_a, node_b) in canonical order → similarity float in [0, 1]
    Returns None if GNN is inactive or any error occurs.
    """
    if not GNN_AVAILABLE:
        return None

    nodes = list(G.nodes())
    n = len(nodes)

    if n < MIN_NODES_FOR_GNN:
        logger.info(
            "Graph has %d nodes (< %d) — GNN component INACTIVE for this analysis.",
            n,
            MIN_NODES_FOR_GNN,
        )
        return None

    try:
        return _run_gcn(G, nodes, concept_embeddings)
    except Exception as exc:
        logger.warning("GNN forward pass failed (%s) — falling back to graph-only scoring.", exc)
        return None


def _run_gcn(
    G: nx.Graph,
    nodes: list[str],
    concept_embeddings: Optional[dict[str, np.ndarray]],
) -> dict[tuple[str, str], float]:
    """Internal: build PyG data, run GCN, compute pairwise cosine similarities."""
    import torch  # local re-import to satisfy type checkers

    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    # ── Node features ──────────────────────────────────────────
    # Feature vector per node: [degree_norm, betweenness_norm, freq_norm]
    # + optionally the first 8 dims of the sentence-transformer embedding
    degree_centrality = nx.degree_centrality(G)
    try:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        betweenness = {node: 0.0 for node in nodes}

    max_freq = max((G.nodes[node].get("frequency", 1) for node in nodes), default=1)

    features: list[list[float]] = []
    for node in nodes:
        deg_c = degree_centrality.get(node, 0.0)
        bet_c = betweenness.get(node, 0.0)
        freq_c = G.nodes[node].get("frequency", 1) / max_freq

        base_feats = [deg_c, bet_c, freq_c]

        # Optionally append embedding dimensions
        if concept_embeddings and node in concept_embeddings:
            emb = concept_embeddings[node][:8].tolist()  # first 8 dims
            base_feats.extend(emb)
        else:
            base_feats.extend([0.0] * 8)

        features.append(base_feats)

    x = torch.tensor(features, dtype=torch.float)

    # ── Edge index ────────────────────────────────────────────
    edge_list: list[list[int]] = []
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        edge_list.append([i, j])
        edge_list.append([j, i])  # undirected → add both directions

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Forward pass ─────────────────────────────────────────
    in_dim = x.shape[1]
    model = _LightweightGCN(in_channels=in_dim, hidden_dim=32)
    model.eval()
    with torch.no_grad():
        node_embs = model(x, edge_index)  # (n, 32)

    # ── Pairwise cosine similarity ────────────────────────────
    node_embs_np = node_embs.numpy()
    # L2-normalize
    norms = np.linalg.norm(node_embs_np, axis=1, keepdims=True) + 1e-8
    node_embs_norm = node_embs_np / norms

    scores: dict[tuple[str, str], float] = {}
    for na, nb in combinations(nodes, 2):
        i, j = node_index[na], node_index[nb]
        sim = float(np.dot(node_embs_norm[i], node_embs_norm[j]))
        # Map from [-1, 1] → [0, 1]
        sim_norm = (sim + 1.0) / 2.0
        key = (min(na, nb), max(na, nb))
        scores[key] = round(sim_norm, 4)

    logger.info("GNN encoder produced scores for %d concept pairs.", len(scores))
    return scores


def gnn_status_label(G: Optional[nx.Graph] = None) -> str:
    """
    Return a human-readable label for the current GNN component status.
    Used by the UI to label the analysis mode.
    """
    if not GNN_AVAILABLE:
        return "Lightweight Graph Analysis (GNN inactive — PyTorch/PyG not installed)"
    if G is not None and G.number_of_nodes() < MIN_NODES_FOR_GNN:
        return f"Lightweight Graph Analysis (GNN inactive — graph too small: {G.number_of_nodes()} nodes < {MIN_NODES_FOR_GNN})"
    return "Lightweight GNN-based Graph Analysis"
