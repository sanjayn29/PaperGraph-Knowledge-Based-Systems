"""Focused tests for full-text concept extraction."""

from services.concept_extractor import extract_concepts_from_paper


def test_extract_concepts_from_complete_full_text():
    paper = {
        "title": "Early Title Concept.",
        "abstract": "Neural Network.",
        "full_text": "x" * 3001 + "\nLate Corpus Concept",
    }

    concepts = extract_concepts_from_paper(paper)

    assert "Early Title Concept" in concepts
    assert "Neural Network" in concepts
    assert "Late Corpus Concept" in concepts