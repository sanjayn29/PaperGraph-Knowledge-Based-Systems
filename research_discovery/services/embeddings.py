"""
embeddings.py
─────────────
Sentence-transformer embeddings for PaperGraph.

Loads the lightweight `all-MiniLM-L6-v2` model (384-dim) and produces:
- Per-paper embeddings from title + abstract
- Per-concept embeddings from the concept string itself

All embeddings are held in memory as numpy arrays — no vector DB, no disk
persistence.

Graceful degradation
--------------------
If `sentence-transformers` is not installed, all functions return None and
a warning is logged. The rest of the pipeline continues using only graph
metrics (semantic_similarity will be 0.0 for all pairs in that case).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Optional sentence-transformers import
# ─────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed — semantic similarity will be disabled. "
        "Install with: pip install sentence-transformers"
    )

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional["SentenceTransformer"] = None  # type: ignore[type-arg]


def get_model() -> Optional["SentenceTransformer"]:  # type: ignore[type-arg]
    """
    Lazily load and cache the sentence-transformer model.
    Returns None if sentence-transformers is unavailable.
    """
    global _model
    if not _ST_AVAILABLE:
        return None
    if _model is None:
        logger.info("Loading embedding model '%s' …", _MODEL_NAME)
        try:
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding model loaded.")
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s", exc)
            return None
    return _model


def embed_papers(papers: list[dict]) -> dict[str, np.ndarray]:
    """
    Compute a 384-dim embedding for each paper using title + abstract.

    Returns
    -------
    dict mapping paper_id → np.ndarray of shape (384,)
    Empty dict if the model is unavailable.
    """
    model = get_model()
    if model is None:
        return {}

    paper_texts = [
        f"{p.get('title', '')} {p.get('abstract', '')}".strip()
        for p in papers
    ]
    paper_ids = [p["paper_id"] for p in papers]

    try:
        embeddings = model.encode(paper_texts, show_progress_bar=False, convert_to_numpy=True)
        return {pid: emb for pid, emb in zip(paper_ids, embeddings)}
    except Exception as exc:
        logger.warning("Paper embedding failed: %s", exc)
        return {}


def embed_concepts(concepts: list[str]) -> dict[str, np.ndarray]:
    """
    Compute a 384-dim embedding for each concept string.

    Returns
    -------
    dict mapping concept_name → np.ndarray of shape (384,)
    Empty dict if the model is unavailable.
    """
    model = get_model()
    if model is None:
        return {}

    if not concepts:
        return {}

    try:
        embeddings = model.encode(concepts, show_progress_bar=False, convert_to_numpy=True)
        return {concept: emb for concept, emb in zip(concepts, embeddings)}
    except Exception as exc:
        logger.warning("Concept embedding failed: %s", exc)
        return {}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D numpy vectors.
    Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_similarity(
    concept_a: str,
    concept_b: str,
    concept_embeddings: dict[str, np.ndarray],
) -> float:
    """
    Return cosine similarity between two concepts using their pre-computed embeddings.
    Returns 0.0 if either concept is missing from the embeddings dict.
    """
    emb_a = concept_embeddings.get(concept_a)
    emb_b = concept_embeddings.get(concept_b)
    if emb_a is None or emb_b is None:
        return 0.0
    return cosine_similarity(emb_a, emb_b)


def is_available() -> bool:
    """Return True if sentence-transformers is installed and the model can be loaded."""
    return _ST_AVAILABLE
