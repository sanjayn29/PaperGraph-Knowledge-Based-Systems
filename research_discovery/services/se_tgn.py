"""
se_tgn.py
─────────
Practical SE-TGN (Semantic-Enhanced Temporal Graph Network) implementation
for PaperGraph, inspired by the base paper:

    "Uncovering novel scientific insights with a synergistic GNN-LLM framework"
    Knowledge-Based Systems, 2025.

IMPORTANT — Academic honesty note
──────────────────────────────────
This is a practical B.Tech student implementation of the SE-TGN concept.
It captures the core ideas (temporal encoding, node memory, message passing,
GRU update, link prediction with BCE loss) but differs significantly from
the base paper's full implementation:
  - The base paper trains on tens of thousands of papers over multiple years.
  - This version trains on 5–10 uploaded PDFs (a few dozen to a few hundred events).
  - No temporal attention mechanism; uses mean aggregation for simplicity.
    - 3 training epochs per analysis session (fast for small datasets).
  - CPU-only; no distributed training.

Architecture
────────────
TimeEncode(Δt) → 32-dim sinusoidal positional encoding of time differences

NodeMemory:
  memory[concept_idx] ∈ R^64   (GRU hidden state per concept)
  last_update_time[concept_idx] (timestamp of last update)

MessageFunction:
  msg = concat(memory[src], memory[dst], paper_emb, TimeEncode(Δt))
  msg_dim = 64 + 64 + 384 + 32 = 544

MemoryUpdate (GRU):
  GRU(input=msg, hidden=memory[node]) → new_memory[node]

LinkPredictor:
  concat(memory[A], memory[B]) → Linear(128→64) → ReLU → Linear(64→1) → sigmoid

Training:
  Positive pairs: real co-occurrence events
  Negative pairs: random concept not in the positive pair (1:1 ratio)
  Loss: Binary Cross Entropy
  Chronological event processing (no future leakage)

Graceful degradation
────────────────────
- PyTorch not installed → SETGN_AVAILABLE = False; module is a no-op
- < MIN_EVENTS_FOR_TRAINING events → training skipped; returns None scores
- Any runtime error → logged + returns None; caller falls back to graph scoring
"""

from __future__ import annotations

import logging
import math
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────

D_MEM = 64           # Node memory dimension
D_TIME = 32          # Temporal encoding dimension
D_HIDDEN = 64        # Link predictor hidden layer dimension
PAPER_EMB_DIM = 384  # Sentence-transformer embedding dimension

D_MSG = D_MEM + D_MEM + PAPER_EMB_DIM + D_TIME  # = 544

MIN_EVENTS_FOR_TRAINING = 10   # Minimum events to attempt training
TRAINING_EPOCHS = 3            # Epochs per analysis session (fast on small data)
LEARNING_RATE = 5e-3
NEG_SAMPLE_RATIO = 1           # Negatives per positive event

# ─────────────────────────────────────────────────────────────
# Optional PyTorch import
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    SETGN_AVAILABLE = True
    logger.info("PyTorch available — SE-TGN component ACTIVE.")
except ImportError:
    SETGN_AVAILABLE = False
    logger.info(
        "PyTorch not installed — SE-TGN component INACTIVE. "
        "Install with: pip install torch"
    )


# ─────────────────────────────────────────────────────────────
# Model components (only defined when PyTorch is available)
# ─────────────────────────────────────────────────────────────

if SETGN_AVAILABLE:
    import torch
    import torch.nn as nn

    class TimeEncode(nn.Module):
        """
        Sinusoidal temporal encoding from Temporal Graph Networks (Rossi et al., 2020).

        Encodes a scalar time difference Δt into a d_time-dimensional vector using
        learned frequency parameters. Captures temporal periodicity.

        Input shape:  (batch_size,) or scalar float
        Output shape: (batch_size, d_time) or (1, d_time)
        """

        def __init__(self, d_time: int = D_TIME) -> None:
            super().__init__()
            self.d_time = d_time
            # Learnable frequency parameter (initialized to [1/10^(i/d_time)])
            freqs = torch.zeros(d_time)
            for i in range(d_time):
                freqs[i] = 1.0 / (10000.0 ** (2.0 * i / d_time))
            self.freq = nn.Parameter(freqs)  # shape: (d_time,)

        def forward(self, delta_t: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            delta_t : shape () or (B,)  — time differences in years

            Returns
            -------
            Tensor of shape (1, d_time) or (B, d_time)
            """
            if delta_t.dim() == 0:
                delta_t = delta_t.unsqueeze(0)  # (1,)
            # Outer product: (B,) × (d_time,) → (B, d_time)
            angles = delta_t.unsqueeze(-1) * self.freq.unsqueeze(0)  # (B, d_time)
            # Alternate sin/cos
            sin_enc = torch.sin(angles[:, 0::2])
            cos_enc = torch.cos(angles[:, 1::2])
            # Interleave
            enc = torch.zeros(delta_t.shape[0], self.d_time, device=delta_t.device)
            enc[:, 0::2] = sin_enc
            enc[:, 1::2] = cos_enc
            return enc  # (B, d_time)

    class NodeMemory:
        """
        Per-concept stateful memory module.

        Stores a GRU hidden state (d_mem-dim) per concept node.
        The GRU is defined externally (in SETGN) and updated here
        by calling update().

        Parameters
        ----------
        n_concepts : total number of unique concepts in the graph
        d_mem      : memory vector dimension
        """

        def __init__(self, n_concepts: int, d_mem: int = D_MEM) -> None:
            self.n_concepts = n_concepts
            self.d_mem = d_mem
            # Memory matrix: one row per concept
            self.memory = torch.zeros(n_concepts, d_mem)
            # Timestamp of last update per concept
            self.last_update = torch.zeros(n_concepts)

        def get(self, idx: int) -> "torch.Tensor":
            """Return the memory vector for concept `idx`. Shape: (d_mem,)."""
            return self.memory[idx]

        def update(self, idx: int, new_mem: "torch.Tensor", timestamp: float) -> None:
            """Update memory for concept `idx` with new_mem vector."""
            self.memory[idx] = new_mem.detach()
            self.last_update[idx] = timestamp

        def reset(self) -> None:
            """Zero-initialize all memories (called before each training epoch)."""
            self.memory.zero_()
            self.last_update.zero_()

        def delta_t(self, idx: int, current_time: float) -> "torch.Tensor":
            """Return time since last update for concept `idx` as a scalar tensor."""
            return torch.tensor(
                float(current_time - self.last_update[idx].item()),
                dtype=torch.float32,
            )

    class SETGN(nn.Module):
        """
        SE-TGN: Semantic-Enhanced Temporal Graph Network.

        Components
        ──────────
        time_encoder  : TimeEncode(d_time)
        gru           : GRU(input=d_msg, hidden=d_mem)
        link_predictor: MLP(2*d_mem → d_hidden → 1)

        Usage
        ─────
        1. Call process_event(event, memory) for each chronological event
           → updates memory for both source and target concepts.
        2. Call predict(idx_a, idx_b, memory) → link probability in [0, 1].
        """

        def __init__(
            self,
            d_mem: int = D_MEM,
            d_time: int = D_TIME,
            d_msg: int = D_MSG,
            d_hidden: int = D_HIDDEN,
        ) -> None:
            super().__init__()

            self.d_mem = d_mem
            self.d_time = d_time
            self.d_msg = d_msg

            self.time_encoder = TimeEncode(d_time)

            # GRU: input = message vector, hidden = memory vector
            self.gru = nn.GRUCell(input_size=d_msg, hidden_size=d_mem)

            # Link predictor MLP
            self.link_predictor = nn.Sequential(
                nn.Linear(d_mem * 2, d_hidden),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(d_hidden, 1),
                nn.Sigmoid(),
            )

        def _build_message(
            self,
            mem_src: "torch.Tensor",
            mem_dst: "torch.Tensor",
            paper_emb: "torch.Tensor",
            delta_t: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Build the message vector for a single event.

            msg = concat(mem_src, mem_dst, paper_emb, TimeEncode(Δt))
            shape: (1, D_MSG)
            """
            t_enc = self.time_encoder(delta_t)  # (1, d_time)
            msg = torch.cat([
                mem_src.unsqueeze(0),      # (1, d_mem)
                mem_dst.unsqueeze(0),      # (1, d_mem)
                paper_emb.unsqueeze(0),    # (1, 384)
                t_enc,                     # (1, d_time)
            ], dim=1)  # (1, d_msg)
            return msg

        def process_event(
            self,
            src_idx: int,
            dst_idx: int,
            paper_emb: "torch.Tensor",
            timestamp: float,
            memory: "NodeMemory",
        ) -> None:
            """
            Process a single temporal event and update node memories.

            Parameters
            ----------
            src_idx    : source concept index in NodeMemory
            dst_idx    : target concept index in NodeMemory
            paper_emb  : (PAPER_EMB_DIM,) semantic embedding of the paper
            timestamp  : publication year (float)
            memory     : NodeMemory to update in place
            """
            mem_src = memory.get(src_idx)  # (d_mem,)
            mem_dst = memory.get(dst_idx)  # (d_mem,)

            dt_src = memory.delta_t(src_idx, timestamp)
            dt_dst = memory.delta_t(dst_idx, timestamp)

            # Build messages
            msg_src = self._build_message(mem_src, mem_dst, paper_emb, dt_src)
            msg_dst = self._build_message(mem_dst, mem_src, paper_emb, dt_dst)

            # GRU update
            new_mem_src = self.gru(msg_src, mem_src.unsqueeze(0)).squeeze(0)
            new_mem_dst = self.gru(msg_dst, mem_dst.unsqueeze(0)).squeeze(0)

            memory.update(src_idx, new_mem_src, timestamp)
            memory.update(dst_idx, new_mem_dst, timestamp)

        def predict(
            self,
            idx_a: int,
            idx_b: int,
            memory: "NodeMemory",
        ) -> "torch.Tensor":
            """
            Predict link probability for concept pair (idx_a, idx_b).

            Returns
            -------
            Tensor scalar (0-dim) in [0, 1].
            """
            mem_a = memory.get(idx_a)  # (d_mem,)
            mem_b = memory.get(idx_b)  # (d_mem,)
            pair_repr = torch.cat([mem_a, mem_b]).unsqueeze(0)  # (1, 2*d_mem)
            return self.link_predictor(pair_repr).squeeze()  # scalar


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def train_setgn(
    temporal_graph,  # TemporalGraph from temporal_graph.py
    concept_embeddings: dict[str, np.ndarray],
    n_epochs: int = TRAINING_EPOCHS,
    lr: float = LEARNING_RATE,
    seed: int = 42,
    training_events: Optional[list] = None,
    concept_names: Optional[list[str]] = None,
) -> Optional[tuple["SETGN", "NodeMemory", dict[str, int]]]:
    """
    Train the SE-TGN on a TemporalGraph's events.

    Parameters
    ----------
    temporal_graph     : populated TemporalGraph (sorted chronologically)
    concept_embeddings : {concept → np.ndarray(384)} for semantic signals
    n_epochs           : number of training epochs
    lr                 : learning rate
    seed               : random seed for reproducibility
    training_events    : optional chronological event subset for evaluation
    concept_names      : optional full concept universe for evaluation scoring

    Returns
    -------
    (model, memory, concept_index) if training succeeds, or None on failure.

    concept_index : {concept_str → integer index} for memory lookup
    """
    if not SETGN_AVAILABLE:
        logger.info("SE-TGN training skipped: PyTorch not available.")
        return None

    events = temporal_graph.events if training_events is None else training_events
    if len(events) < MIN_EVENTS_FOR_TRAINING:
        logger.info(
            "SE-TGN training skipped: only %d events (minimum: %d).",
            len(events),
            MIN_EVENTS_FOR_TRAINING,
        )
        return None

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    concepts_set = set(
        temporal_graph.unique_concepts() if concept_names is None else concept_names
    )
    for event in events:
        concepts_set.add(event.source_concept)
        concepts_set.add(event.target_concept)
    concept_index = {concept: i for i, concept in enumerate(sorted(concepts_set))}
    concepts = list(concept_index.keys())
    n_concepts = len(concepts)

    if n_concepts < 2:
        logger.info("SE-TGN training skipped: fewer than 2 concepts.")
        return None

    try:
        model = SETGN()
        memory = NodeMemory(n_concepts)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCELoss()

        model.train()

        for epoch in range(n_epochs):
            memory.reset()
            epoch_loss = 0.0
            n_batches = 0

            for event in events:
                src_idx = concept_index[event.source_concept]
                dst_idx = concept_index[event.target_concept]

                paper_emb_np = event.paper_embedding
                if paper_emb_np is None or len(paper_emb_np) != PAPER_EMB_DIM:
                    paper_emb_np = np.zeros(PAPER_EMB_DIM, dtype=np.float32)
                paper_emb = torch.tensor(paper_emb_np, dtype=torch.float32)

                timestamp = float(event.timestamp)

                # ── Positive prediction ──────────────────────────────
                pos_score = model.predict(src_idx, dst_idx, memory)
                pos_label = torch.tensor(1.0)

                # ── Negative sampling ────────────────────────────────
                neg_concept = _sample_negative(
                    concepts, event.source_concept, event.target_concept
                )
                neg_idx = concept_index[neg_concept]
                neg_score = model.predict(src_idx, neg_idx, memory)
                neg_label = torch.tensor(0.0)

                # ── Loss + backprop ──────────────────────────────────
                optimizer.zero_grad()
                loss = bce(pos_score, pos_label) + bce(neg_score, neg_label)
                loss.backward()
                # Gradient clipping for stability on small datasets
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # ── Memory update (detached — no gradient through memory) ─
                with torch.no_grad():
                    model.process_event(
                        src_idx, dst_idx, paper_emb, timestamp, memory
                    )

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("SE-TGN epoch %d/%d — avg loss: %.4f", epoch + 1, n_epochs, avg_loss)

        model.eval()
        logger.info(
            "SE-TGN training complete: %d concepts, %d events, %d epochs.",
            n_concepts, len(events), n_epochs,
        )
        return model, memory, concept_index

    except Exception as exc:
        logger.warning("SE-TGN training failed: %s", exc, exc_info=True)
        return None


def _sample_negative(
    concepts: list[str],
    src: str,
    dst: str,
    max_tries: int = 20,
) -> str:
    """
    Sample a concept that is neither src nor dst (negative example).
    Falls back to the first concept if no valid negative is found.
    """
    for _ in range(max_tries):
        neg = random.choice(concepts)
        if neg != src and neg != dst:
            return neg
    # Fallback
    return next((c for c in concepts if c != src and c != dst), concepts[0])


# ─────────────────────────────────────────────────────────────
# Link prediction scoring
# ─────────────────────────────────────────────────────────────

def compute_setgn_scores(
    temporal_graph,  # TemporalGraph
    concept_embeddings: dict[str, np.ndarray],
    query_time: Optional[int] = None,
    training_events: Optional[list] = None,
    concept_names: Optional[list[str]] = None,
) -> Optional[dict[tuple[str, str], float]]:
    """
    Train SE-TGN and compute link-prediction scores for all concept pairs.

    Parameters
    ----------
    temporal_graph     : populated TemporalGraph
    concept_embeddings : {concept → np.ndarray(384)}
    query_time         : the "future" time to predict links for.
                         If None, uses max(year) + 1 from the temporal graph.
    training_events    : optional chronological event subset used for evaluation
    concept_names      : optional full concept universe to score

    Returns
    -------
    {(concept_a, concept_b) → score ∈ [0, 1]} in canonical order, or None.
    """
    if not SETGN_AVAILABLE:
        return None

    trained = train_setgn(
        temporal_graph,
        concept_embeddings,
        training_events=training_events,
        concept_names=concept_names,
    )
    if trained is None:
        return None

    model, memory, concept_index = trained
    concepts = list(concept_index.keys())

    if query_time is None:
        _, y_max = temporal_graph.year_range()
        query_time = (y_max or 2024) + 1

    from itertools import combinations as _combs

    scores: dict[tuple[str, str], float] = {}
    model.eval()
    with torch.no_grad():
        for c_a, c_b in _combs(concepts, 2):
            idx_a = concept_index[c_a]
            idx_b = concept_index[c_b]
            score = float(model.predict(idx_a, idx_b, memory).item())
            key = (min(c_a, c_b), max(c_a, c_b))
            scores[key] = round(score, 4)

    logger.info("SE-TGN scores computed for %d concept pairs.", len(scores))
    return scores


# ─────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────

def setgn_status_label(n_events: int = 0) -> str:
    """Return a human-readable status for the SE-TGN component."""
    if not SETGN_AVAILABLE:
        return "SE-TGN inactive (PyTorch not installed — pip install torch)"
    if n_events < MIN_EVENTS_FOR_TRAINING:
        return (
            f"SE-TGN inactive (only {n_events} events; "
            f"minimum {MIN_EVENTS_FOR_TRAINING} required)"
        )
    return f"SE-TGN active ({n_events} training events)"
