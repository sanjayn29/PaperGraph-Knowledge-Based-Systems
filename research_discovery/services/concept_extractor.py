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
    SYNONYM_MAP,
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

# These terms are useful for locating document metadata, but are not useful
# concept nodes when they occur in a paper's extracted text.
_BOILERPLATE_PHRASES = {
    "acknowledgments", "acknowledgements", "appendix", "appendices",
    "author contributions", "bibliography", "conflict of interest",
    "copyright", "declaration", "funding", "references", "see appendix",
    "introduction", "related work", "background", "conclusion",
    "see section", "see table", "long beach", "computer science",
    "neural information processing systems", "strategic missions",
    "end-to-end", "mini-batch", "feed-forward", "element-wise",
    "left-to-right", "state-of-the-art", "rouge-l", "two-layer",
    "state-of-the", "of-the-art", "real-world", "large-scale",
    "pre-trained", "pre-train", "input-output", "up-to-date",
    "sub-optimal", "non-parametric", "sequence-to-sequence", "encoder-decoder",
    "sentence-pair", "token-level", "question-answering", "cross-entropy",
    "max-pooling", "point-wise", "per-layer", "set-up", "run-time",
}
_NON_CONCEPT_ACRONYMS = {
    "acm", "arxiv", "cpu", "gpu", "gpus", "ieee", "nips", "sep", "usa",
}
_MALFORMED_TERM_RE = re.compile(r"^(?:[A-Za-z]+(?:base|large|small))$")
_BOILERPLATE_WORDS = {
    "accepted", "available", "chapter", "conference", "copyright",
    "figure", "fig", "international", "issue", "journal", "license",
    "long", "page", "pages", "papers", "permission", "proceedings",
    "published", "section", "short", "submitted", "table", "volume", "workshop",
}
_BOILERPLATE_PREFIXES = (
    "in proceedings", "in advances", "advances in", "published in",
    "proceedings of", "international conference on",
    "international joint conference",
)
_GENERIC_SINGLE_WORDS = {
    "a", "about", "all", "also", "an", "and", "any", "are", "as", "at",
    "based", "be", "because", "been", "before", "being", "between", "both",
    "but", "by", "can", "could", "each", "either", "else", "english", "especially",
    "even", "every", "finally", "first", "for", "from", "further", "given", "has",
    "have", "he", "her", "here", "how", "if", "in", "instead", "into", "is", "it",
    "its", "just", "long", "may", "method", "more", "most", "much", "must", "my",
    "neural", "no", "nor", "not", "note", "of", "on", "one", "only", "or", "other",
    "our", "out", "over", "same", "second", "she", "short", "since", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "using", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "with", "would",
    "you", "your", "advances", "input", "output", "result", "results",
}
_SCIENTIFIC_SINGLE_WORDS = {
    "attention", "clustering", "embeddings", "graph", "optimization", "reasoning",
    "retrieval", "transformer", "transformers",
}
_SECTION_STOP_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*)?(?:references|bibliography|acknowledg(?:e)?ments?|"
    r"appendix|appendices|author contributions|funding|conflict of interest|"
    r"declaration|copyright)\s*[:.]?\s*$",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_MIXED_CASE_TERM_RE = re.compile(r"\b[A-Za-z]+(?:[A-Z][A-Za-z]+)+\b")
_TITLE_HYPHENATED_RE = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b")
_HYPHENATED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-z]+\s+){0,2}[A-Za-z]+(?:-[A-Za-z]+)+"
    r"(?:\s+[A-Z][a-z]+){0,2}\b"
)


def _main_body_text(text: str) -> str:
    """Remove back matter and repeated short PDF layout lines."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    counts: dict[str, int] = defaultdict(int)
    for line in lines:
        key = re.sub(r"\s+", " ", line).lower()
        if line and len(line) <= 120 and len(line.split()) <= 14:
            counts[key] += 1

    kept: list[str] = []
    for line in lines:
        if _SECTION_STOP_RE.match(line):
            break
        if not line or _PAGE_NUMBER_RE.match(line):
            continue
        normalized_line = re.sub(r"\s+", " ", line).lower()
        is_metadata_line = (
            len(line) <= 120
            and not re.search(r"[.!?]", line)
            and any(marker in normalized_line for marker in _BOILERPLATE_PREFIXES)
        )
        if is_metadata_line:
            continue
        key = re.sub(r"\s+", " ", line).lower()
        if counts[key] > 1 and len(line.split()) <= 14:
            continue
        kept.append(line)
    return "\n".join(kept)


def _raw_candidates(text: str) -> list[str]:
    """Collect phrase forms that the lightweight extractor can recognize."""
    candidates: list[str] = []
    for line in text.splitlines() or [text]:
        candidates.extend(extract_noun_phrases(line))
        candidates.extend(_MIXED_CASE_TERM_RE.findall(line))
        candidates.extend(_TITLE_HYPHENATED_RE.findall(line))
        candidates.extend(_HYPHENATED_PHRASE_RE.findall(line))
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _is_boilerplate_phrase(phrase: str) -> bool:
    lower = re.sub(r"\s+", " ", phrase.strip().lower())
    words = set(re.findall(r"[a-z]+", lower))
    if lower in _BOILERPLATE_PHRASES or lower in _BOILERPLATE_PREFIXES:
        return True
    if any(lower.startswith(prefix) for prefix in _BOILERPLATE_PREFIXES):
        return True
    if any(token in lower for token in _BOILERPLATE_PHRASES):
        return True
    if words and words.intersection({"introduction", "references", "appendix", "figure", "table"}) \
            and words != {"figure"}:
        return True
    return bool(words) and words.issubset(_BOILERPLATE_WORDS)


def _is_candidate(phrase: str, source: str) -> bool:
    """Apply context-sensitive quality checks without requiring an NLP model."""
    if not is_valid_concept(phrase, _MIN_CONCEPT_LEN, _MAX_CONCEPT_LEN):
        return False
    if _is_boilerplate_phrase(phrase):
        return False

    normalized = normalize_concept(phrase)
    lower = normalized.lower()
    word_list = re.findall(r"[a-z]+", lower)
    words = set(word_list)
    if len(words) == 1:
        raw = phrase.strip()
        is_acronym = bool(re.fullmatch(r"[A-Z][A-Z0-9-]{1,8}", raw))
        is_mixed_case = bool(_MIXED_CASE_TERM_RE.fullmatch(raw))
        is_known_term = lower in _SCIENTIFIC_SINGLE_WORDS or lower in SYNONYM_MAP
        if (
            lower in _GENERIC_SINGLE_WORDS
            or lower in _BOILERPLATE_WORDS
            or lower in _NON_CONCEPT_ACRONYMS
            or _MALFORMED_TERM_RE.fullmatch(raw)
        ):
            return False
        return is_acronym or is_mixed_case or is_known_term

    if words.intersection(_NON_CONCEPT_ACRONYMS):
        return True

    # A phrase with a venue/document marker is metadata even when it contains
    # a legitimate title such as "Neural Information Processing Systems".
    if any(marker in lower for marker in _BOILERPLATE_PREFIXES):
        return False
    if word_list and word_list[0] in {"journal", "proceedings", "conference", "workshop", "volume"}:
        return False
    return True


def _prefer_long_phrases(concepts: list[str]) -> list[str]:
    """Drop generic fragments when a longer candidate contains them."""
    lower_concepts = [
        (concept, set(re.findall(r"[a-z]+", concept.lower())))
        for concept in concepts
    ]
    result: list[str] = []
    for concept, words in lower_concepts:
        if any(
            len(other_words) > len(words) and words < other_words
            for other_concept, other_words in lower_concepts
            if other_concept != concept
        ):
            continue
        result.append(concept)
    return result


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
        ("title", paper.get("title", "")),
        ("abstract", paper.get("abstract", "")),
        ("body", _main_body_text(paper.get("full_text", ""))),
    ]

    normalized: dict[str, str] = {}  # lower_key → canonical form
    for source, text in text_sources:
        for phrase in _raw_candidates(text):
            if _is_candidate(phrase, source):
                canonical = normalize_concept(phrase)
                normalized.setdefault(canonical.lower(), canonical)

    return _prefer_long_phrases(list(normalized.values()))


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



