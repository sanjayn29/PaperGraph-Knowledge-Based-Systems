"""
test_domain_analyzer.py
───────────────────────
Unit tests for domain classification and cross-domain discovery.
"""

import numpy as np
import pytest
from services.domain_analyzer import (
    classify_concept_domain,
    classify_all_concepts,
    get_domain_distance,
    compute_cross_domain_score,
    extract_cross_domain_discoveries,
)


def test_classify_concept_keyword():
    dom1, conf1 = classify_concept_domain("Graph Neural Network")
    assert dom1 == "Artificial Intelligence"
    assert conf1 > 0.6

    dom2, conf2 = classify_concept_domain("CRISPR Genome Editing")
    assert dom2 == "Biology & Bioinformatics"

    dom3, conf3 = classify_concept_domain("Clinical Trial Patient")
    assert dom3 == "Healthcare & Medicine"


def test_domain_distance():
    # Same domain = 0.0
    assert get_domain_distance("Artificial Intelligence", "Artificial Intelligence") == 0.0

    # Closely related
    dist_near = get_domain_distance("Artificial Intelligence", "Computer Science")
    assert 0.0 < dist_near < 0.5

    # Distant domains
    dist_far = get_domain_distance("Artificial Intelligence", "Chemistry & Materials")
    assert dist_far > 0.7


def test_compute_cross_domain_score():
    score = compute_cross_domain_score(
        domain_a="Artificial Intelligence",
        domain_b="Biology & Bioinformatics",
        semantic_compatibility=0.85,
        gap_score=0.80,
        temporal_emergence=0.75,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.6  # High synergy


def test_extract_cross_domain_discoveries():
    candidates = [
        {
            "concept_a": "Graph Neural Network",
            "concept_b": "Drug Discovery",
            "semantic_similarity": 0.85,
            "setgn_score": 0.90,
            "candidate_score": 0.88,
        },
        {
            "concept_a": "Transformer",
            "concept_b": "Attention Mechanism",
            "semantic_similarity": 0.95,
            "setgn_score": 0.92,
            "candidate_score": 0.93,
        },
    ]

    concept_domains = {
        "Graph Neural Network": {"domain": "Artificial Intelligence", "confidence": 0.95},
        "Drug Discovery": {"domain": "Chemistry & Materials", "confidence": 0.90},
        "Transformer": {"domain": "Artificial Intelligence", "confidence": 0.95},
        "Attention Mechanism": {"domain": "Artificial Intelligence", "confidence": 0.90},
    }

    discoveries = extract_cross_domain_discoveries(
        candidates, concept_domains, min_cross_domain_score=0.3
    )

    # Only GNN + Drug Discovery is cross-domain (AI + Chem), Transformer + Attention is same domain
    assert len(discoveries) == 1
    assert discoveries[0]["concept_a"] == "Graph Neural Network"
    assert discoveries[0]["concept_b"] == "Drug Discovery"


def test_extract_cross_domain_discoveries_handles_missing_setgn_score():
    concept_domains = {
        "Graph Neural Network": {"domain": "Artificial Intelligence"},
        "Drug Discovery": {"domain": "Chemistry & Materials"},
    }
    base_candidate = {
        "concept_a": "Graph Neural Network",
        "concept_b": "Drug Discovery",
        "semantic_similarity": 0.85,
        "candidate_score": 0.88,
    }

    missing_score = extract_cross_domain_discoveries(
        [base_candidate], concept_domains, min_cross_domain_score=0.0
    )
    explicit_none = extract_cross_domain_discoveries(
        [{**base_candidate, "setgn_score": None}],
        concept_domains,
        min_cross_domain_score=0.0,
    )
    numeric_score = extract_cross_domain_discoveries(
        [{**base_candidate, "setgn_score": 0.9}],
        concept_domains,
        min_cross_domain_score=0.0,
    )

    assert missing_score[0]["cross_domain_score"] == explicit_none[0]["cross_domain_score"]
    assert numeric_score[0]["cross_domain_score"] != explicit_none[0]["cross_domain_score"]
