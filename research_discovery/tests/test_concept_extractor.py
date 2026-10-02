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


def test_rejects_document_boilerplate_and_reference_regions():
    paper = {
        "title": "Graph Neural Networks for Retrieval",
        "abstract": "Temporal Graph Networks improve Link Prediction.",
        "full_text": (
            "We study Retrieval-Augmented Generation and Knowledge Graphs.\n"
            "Advances in Neural Information Processing Systems\n"
            "Journal Volume 4 Workshop\n"
            "Appendix\n"
            "International Conference on Machine Learning\n"
            "References\n"
            "Zhang Chen Proceedings Journal\n"
        ),
    }

    concepts = extract_concepts_from_paper(paper)

    assert "Graph Neural Network" in concepts
    assert "Temporal Graph Network" in concepts
    assert "Link Prediction" in concepts
    assert "Retrieval-Augmented Generation" in concepts
    assert "Knowledge Graph" in concepts
    assert not {
        "Advances", "In Advances", "Journal", "Long Papers", "Workshop",
        "Appendix", "International Conference", "Neural Information Processing Systems",
    } & set(concepts)


def test_preserves_acronyms_and_technical_single_words():
    paper = {
        "title": "BERT, SBERT, GraphRAG, and Self-RAG",
        "abstract": "Attention and Reasoning support Embeddings and Optimization.",
        "full_text": "Finally, English, Here, and This are ordinary words.",
    }

    concepts = extract_concepts_from_paper(paper)

    assert {"BERT", "SBERT", "GraphRAG", "Self-RAG", "Attention", "Embeddings"} <= set(concepts)
    assert not {"Finally", "English", "Here", "This"} & set(concepts)


def test_rejects_sentence_starters_and_reference_names():
    paper = {
        "title": "GraphRAG for Scientific Discovery",
        "abstract": "This paper studies Knowledge Graphs and Link Prediction.",
        "full_text": (
            "Our method uses Retrieval-Augmented Generation.\n"
            "Finally, Zhang and Chen are cited below.\n"
            "References\n"
            "Zhang Chen. Journal of Machine Learning Research.\n"
        ),
    }

    concepts = extract_concepts_from_paper(paper)

    assert "Retrieval-Augmented Generation" in concepts
    assert not {"This", "Our", "Finally", "Zhang", "Chen", "Journal"} & set(concepts)


def test_rejects_pdf_methodology_and_layout_fragments():
    paper = {
        "title": "Graph Neural Networks",
        "abstract": "Natural Language Processing with Retrieval-Augmented Generation.",
        "full_text": (
            "Introduction Retrieval End-To-End Mini-Batch Feed-Forward Element-Wise\n"
            "See Appendix Rouge-L Two-Layer Left-To-Right State-Of-The-Art\n"
            "Temporal Graph Networks remain useful scientific concepts.\n"
        ),
    }

    concepts = extract_concepts_from_paper(paper)

    assert "Natural Language Processing" in concepts
    assert "Retrieval-Augmented Generation" in concepts
    assert "Temporal Graph Network" in concepts
    assert not {
        "Introduction Retrieval", "End-To-End", "Mini-Batch", "Feed-Forward",
        "Element-Wise", "See Appendix", "Rouge-L", "Two-Layer",
        "Left-To-Right", "State-Of-The-Art",
    } & set(concepts)


def test_rejects_observed_venue_and_malformed_pdf_terms():
    paper = {
        "title": "Natural Language Processing for Graphs",
        "abstract": "Large Language Models and Long Short-Term Memory.",
        "full_text": (
            "Arxiv Gpu Usa Lstms\n"
            "Long Beach Neural Information Processing Systems\n"
            "See Section Computer Science Experiments We\n"
            "Retrieval-Augmented Generation remains useful.\n"
        ),
    }

    concepts = extract_concepts_from_paper(paper)

    assert "Large Language Model" in concepts
    assert "Long Short-Term Memory" in concepts
    assert "Retrieval-Augmented Generation" in concepts
    assert not {
        "Arxiv", "Gpu", "Usa", "Lstms", "Long Beach",
        "Neural Information Processing Systems", "See Section",
        "Computer Science", "Experiments We",
    } & set(concepts)