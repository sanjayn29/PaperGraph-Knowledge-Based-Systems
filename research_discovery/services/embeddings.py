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



def is_available() -> bool:
    """Return True if sentence-transformers is installed and the model can be loaded."""
    return _ST_AVAILABLE
