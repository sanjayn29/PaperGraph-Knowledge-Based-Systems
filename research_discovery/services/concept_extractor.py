"""
concept_extractor.py
────────────────────
Extracts, normalizes, and filters domain concepts from research papers.

Role in the pipeline
--------------------
Analogous to the keyword co-occurrence event preparation step in the base paper
(SE-TGN expects tokenized concept nodes). Here we produce concept nodes for the
much smaller NetworkX graph built from 5–10 PDFs.

Strategy
--------
1. Extract noun phrases from each paper's title + abstract + full_text using
   regex-based heuristics (no NLTK/spaCy required).
2. Filter using ACADEMIC_STOPWORDS.
3. Normalize synonyms to canonical forms via SYNONYM_MAP.
4. Apply a cross-paper frequency threshold: keep only concepts that appear in
   at least `min_paper_freq` papers (default 1 for small uploads).
5. Optionally ask the LLM to refine/extend the concept list (degrades gracefully
   when no API key is present).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

from utils.text_utils import (
    ACADEMIC_STOPWORDS,
    extract_noun_phrases,
    is_valid_concept,
    normalize_concept,
)

logger = logging.getLogger(__name__)

# Minimum character length for a concept to be kept
_MIN_CONCEPT_LEN = 3
# Maximum character length
_MAX_CONCEPT_LEN = 60
# A concept must appear in at least this many papers to be kept in the graph
_DEFAULT_MIN_PAPER_FREQ = 1


def extract_concepts_from_paper(paper: dict) -> list[str]:
    """
    Extract and normalize concepts from a single paper dict.

    Sources (in priority order):
    - title
    - abstract
    - full_text

    Returns a deduplicated, normalized list of concept strings.
    """
    text_sources = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("full_text", ""),
    ]
    combined = " ".join(s for s in text_sources if s)

    raw_phrases = extract_noun_phrases(combined)

    normalized: dict[str, str] = {}  # lower_key → canonical form
    for phrase in raw_phrases:
        if not is_valid_concept(phrase, _MIN_CONCEPT_LEN, _MAX_CONCEPT_LEN):
            continue
        canonical = normalize_concept(phrase)
        key = canonical.lower()
        if key not in normalized:
            normalized[key] = canonical

    return list(normalized.values())


def extract_all_concepts(
    papers: list[dict],
    min_paper_freq: int = _DEFAULT_MIN_PAPER_FREQ,
    top_n: Optional[int] = 80,
) -> tuple[list[dict], dict[str, list[str]]]:
    """
    Extract concepts from all papers, apply frequency filtering, and
    update each paper dict in-place with its 'concepts' list.

    Parameters
    ----------
    papers         : list of paper dicts (modified in-place)
    min_paper_freq : concept must appear in at least this many papers
    top_n          : keep only the top-N concepts by cross-paper frequency
                     (None = keep all passing the frequency threshold)

    Returns
    -------
    (papers, concept_to_paper_ids)
    papers               : updated in-place with populated 'concepts' list
    concept_to_paper_ids : mapping from canonical concept → list of paper_ids
                           that mention it
    """
    # Step 1: extract raw concepts per paper
    paper_concepts: dict[str, list[str]] = {}  # paper_id → [canonical concepts]
    for paper in papers:
        concepts = extract_concepts_from_paper(paper)
        paper_concepts[paper["paper_id"]] = concepts

    # Step 2: build cross-paper frequency map (concept → set of paper_ids)
    concept_papers: dict[str, set[str]] = defaultdict(set)
    for paper_id, concepts in paper_concepts.items():
        for concept in concepts:
            concept_papers[concept.lower()].add(paper_id)

    # Canonical-form lookup (lower → canonical)
    canonical_lookup: dict[str, str] = {}
    for paper in papers:
        for concept in paper_concepts[paper["paper_id"]]:
            canonical_lookup[concept.lower()] = concept

    # Step 3: filter by min_paper_freq
    qualifying_concepts: dict[str, int] = {
        lower: len(paper_ids)
        for lower, paper_ids in concept_papers.items()
        if len(paper_ids) >= min_paper_freq
    }

    # Step 4: optionally limit to top_n by paper frequency
    if top_n is not None and len(qualifying_concepts) > top_n:
        qualifying_concepts = dict(
            sorted(qualifying_concepts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

    logger.info(
        "Concept extraction: %d total raw concepts → %d after filtering",
        len(canonical_lookup),
        len(qualifying_concepts),
    )

    # Step 5: update paper dicts and build the output mapping
    concept_to_paper_ids: dict[str, list[str]] = {}
    for lower_key in qualifying_concepts:
        canonical = canonical_lookup.get(lower_key, lower_key.title())
        paper_ids = sorted(concept_papers[lower_key])
        concept_to_paper_ids[canonical] = paper_ids

    for paper in papers:
        paper_id = paper["paper_id"]
        paper["concepts"] = [
            canonical_lookup.get(c.lower(), c)
            for c in paper_concepts[paper_id]
            if c.lower() in qualifying_concepts
        ]

    return papers, concept_to_paper_ids



