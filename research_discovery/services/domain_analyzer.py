"""
domain_analyzer.py
──────────────────
Scientific domain classification and cross-domain discovery engine for PaperGraph.

Classifies extracted concepts into scientific domains and computes cross-domain
synergy scores to uncover interdisciplinary research opportunities.

Domains supported:
  1. Artificial Intelligence & Machine Learning
  2. Computer Science & Information Systems
  3. Healthcare & Medicine
  4. Biology & Bioinformatics
  5. Chemistry & Pharmacology
  6. Physics & Quantum Science
  7. Mathematics & Statistics
  8. Materials Science & Nanotechnology
  9. Environmental & Earth Science
  10. Engineering & Robotics
  11. Social Sciences & Humanities
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Scientific Domain Taxonomy & Keyword Anchors
# ─────────────────────────────────────────────────────────────

DOMAINS: dict[str, list[str]] = {
    "Artificial Intelligence": [
        "neural network", "deep learning", "machine learning", "transformer", "attention",
        "gnn", "graph neural", "llm", "large language model", "reinforcement learning",
        "embedding", "representation learning", "supervised", "unsupervised", "nlp",
        "computer vision", "generative ai", "diffusion", "classification", "clustering",
        "prompt", "latent", "convolutional", "recurrent", "loss function", "backpropagation",
    ],
    "Computer Science": [
        "algorithm", "database", "distributed system", "cloud computing", "cybersecurity",
        "software engineering", "operating system", "network", "compiler", "data structure",
        "parallel computing", "cryptography", "blockchain", "concurrency", "storage",
        "microservices", "api", "web service", "scalability", "latency", "throughput",
    ],
    "Healthcare & Medicine": [
        "clinical", "patient", "disease", "diagnosis", "therapy", "medical imaging",
        "hospital", "healthcare", "oncology", "pathology", "cardiology", "prognosis",
        "radiology", "biomarker", "ehr", "treatment", "electronic health", "surgery",
        "epidemiology", "symptom", "infection", "vaccine", "pharmacology",
    ],
    "Biology & Bioinformatics": [
        "gene", "protein", "dna", "rna", "genome", "genomics", "proteomics",
        "cellular", "molecular", "amino acid", "enzyme", "organism", "pathway",
        "crispr", "sequencing", "biological", "microbiome", "evolution", "mutation",
        "metabolism", "cell biology", "docking", "sequence alignment",
    ],
    "Chemistry & Materials": [
        "molecule", "molecular", "compound", "reaction", "synthesis", "catalysis",
        "polymer", "nanomaterial", "crystal", "organic", "inorganic", "chemical bond",
        "drug design", "ligand", "solvation", "spectroscopy", "thermodynamics",
        "nanoparticle", "graphene", "semiconductor", "battery", "catalyst",
    ],
    "Physics & Quantum Science": [
        "quantum", "quantum mechanics", "particle", "photon", "thermodynamics", "optics",
        "electromagnetism", "relativity", "gravity", "superconductivity", "matter",
        "energy", "plasma", "spin", "wavefunction", "laser", "astrophysics",
    ],
    "Mathematics & Statistics": [
        "optimization", "probability", "statistical", "linear algebra", "calculus",
        "differential equation", "topology", "stochastic", "bayesian", "matrix",
        "eigenvalue", "manifold", "distribution", "variance", "regression",
        "convex", "non-convex", "gradient", "geometry",
    ],
    "Environmental & Climate Science": [
        "climate", "climate change", "carbon", "emission", "ecosystem", "sustainability",
        "renewable energy", "solar", "wind energy", "oceanography", "pollution",
        "biodiversity", "deforestation", "greenhouse", "meteorology", "hydrology",
    ],
    "Engineering & Robotics": [
        "robotics", "control system", "actuator", "sensor", "autonomous", "mechatronics",
        "signal processing", "mechanical", "electrical", "civil engineering", "aerospace",
        "uav", "automation", "kinematics", "dynamics", "hardware", "fpga",
    ],
    "Social Sciences & Economics": [
        "economy", "financial", "policy", "sociology", "psychology", "behavioral",
        "market", "governance", "ethics", "social network", "demographics", "decision making",
        "incentive", "risk assessment", "human factors", "education",
    ],
}

# Domain affinity matrix (0 = closely related disciplines, 1 = distant disciplines)
# Used to calculate domain distance.
_DOMAIN_AFFINITY_PROXIMITY: dict[tuple[str, str], float] = {
    ("Artificial Intelligence", "Computer Science"): 0.2,
    ("Artificial Intelligence", "Mathematics & Statistics"): 0.3,
    ("Artificial Intelligence", "Healthcare & Medicine"): 0.8,
    ("Artificial Intelligence", "Biology & Bioinformatics"): 0.8,
    ("Artificial Intelligence", "Chemistry & Materials"): 0.85,
    ("Artificial Intelligence", "Physics & Quantum Science"): 0.75,
    ("Artificial Intelligence", "Environmental & Climate Science"): 0.8,
    ("Artificial Intelligence", "Engineering & Robotics"): 0.5,
    ("Artificial Intelligence", "Social Sciences & Economics"): 0.7,
    ("Healthcare & Medicine", "Biology & Bioinformatics"): 0.3,
    ("Healthcare & Medicine", "Chemistry & Materials"): 0.4,
    ("Biology & Bioinformatics", "Chemistry & Materials"): 0.35,
    ("Physics & Quantum Science", "Chemistry & Materials"): 0.4,
    ("Physics & Quantum Science", "Mathematics & Statistics"): 0.35,
    ("Mathematics & Statistics", "Computer Science"): 0.3,
    ("Engineering & Robotics", "Computer Science"): 0.35,
    ("Environmental & Climate Science", "Social Sciences & Economics"): 0.5,
}


def get_domain_distance(domain_a: str, domain_b: str) -> float:
    """
    Return the normalized interdisciplinary distance between two domains [0.0, 1.0].
    Same domain = 0.0. Distant domains = up to 1.0.
    """
    if not domain_a or not domain_b or domain_a == domain_b:
        return 0.0

    key = (domain_a, domain_b)
    rev_key = (domain_b, domain_a)

    if key in _DOMAIN_AFFINITY_PROXIMITY:
        return _DOMAIN_AFFINITY_PROXIMITY[key]
    if rev_key in _DOMAIN_AFFINITY_PROXIMITY:
        return _DOMAIN_AFFINITY_PROXIMITY[rev_key]

    # Default distance for unmapped cross-domain pairs
    return 0.9


def classify_concept_domain(
    concept: str,
    concept_embedding: Optional[np.ndarray] = None,
    domain_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> tuple[str, float]:
    """
    Assign a scientific domain to a concept using keyword matching + embedding similarity.

    Returns
    -------
    (domain_name, confidence_score) e.g. ("Artificial Intelligence", 0.92)
    """
    concept_lower = concept.lower().strip()

    # 1. Direct keyword match scoring
    keyword_scores: dict[str, float] = {}
    for domain, keywords in DOMAINS.items():
        score = 0.0
        for kw in keywords:
            if kw in concept_lower:
                # Higher score for whole word or exact match
                if kw == concept_lower:
                    score += 1.0
                elif f" {kw} " in f" {concept_lower} ":
                    score += 0.8
                else:
                    score += 0.5
        if score > 0:
            keyword_scores[domain] = score

    if keyword_scores:
        best_domain = max(keyword_scores.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, 0.6 + 0.1 * keyword_scores[best_domain])
        return best_domain, round(confidence, 3)

    # 2. Embedding similarity fallback (if embeddings provided)
    if concept_embedding is not None and domain_embeddings:
        best_domain = "Computer Science"
        best_sim = -1.0
        norm_c = np.linalg.norm(concept_embedding)
        if norm_c > 1e-8:
            for domain, dom_emb in domain_embeddings.items():
                norm_d = np.linalg.norm(dom_emb)
                if norm_d > 1e-8:
                    sim = float(np.dot(concept_embedding, dom_emb) / (norm_c * norm_d))
                    if sim > best_sim:
                        best_sim = sim
                        best_domain = domain
            confidence = max(0.4, min(0.85, (best_sim + 1.0) / 2.0))
            return best_domain, round(confidence, 3)

    # 3. Default fallback
    return "Computer Science", 0.50


def build_domain_centroid_embeddings(
    concept_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute dense centroid embeddings for each scientific domain from anchor keywords.
    """
    domain_embs: dict[str, np.ndarray] = {}
    for domain, keywords in DOMAINS.items():
        vecs = [
            concept_embeddings[kw]
            for kw in keywords
            if kw in concept_embeddings
        ]
        if vecs:
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid = centroid / norm
            domain_embs[domain] = centroid

    return domain_embs


def classify_all_concepts(
    concepts: list[str],
    concept_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, dict]:
    """
    Classify a list of concepts into scientific domains.

    Returns
    -------
    {concept_name → {"domain": str, "confidence": float}}
    """
    domain_embeddings: dict[str, np.ndarray] = {}
    if concept_embeddings:
        domain_embeddings = build_domain_centroid_embeddings(concept_embeddings)

    classifications: dict[str, dict] = {}
    for c in concepts:
        c_emb = concept_embeddings.get(c) if concept_embeddings else None
        domain, conf = classify_concept_domain(c, c_emb, domain_embeddings)
        classifications[c] = {"domain": domain, "confidence": conf}

    return classifications


def compute_cross_domain_score(
    domain_a: str,
    domain_b: str,
    semantic_compatibility: float,
    gap_score: float = 0.0,
    temporal_emergence: float = 0.0,
) -> float:
    """
    Compute interdisciplinary Cross-Domain Score:
    Score = 0.40 * DomainDistance + 0.30 * SemanticCompatibility
          + 0.20 * ResearchGapScore + 0.10 * TemporalEmergence
    """
    dom_dist = get_domain_distance(domain_a, domain_b)
    score = (
        0.40 * dom_dist
        + 0.30 * max(0.0, min(1.0, semantic_compatibility))
        + 0.20 * max(0.0, min(1.0, gap_score))
        + 0.10 * max(0.0, min(1.0, temporal_emergence))
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def extract_cross_domain_discoveries(
    candidates: list[dict],
    concept_domains: dict[str, dict],
    min_cross_domain_score: float = 0.45,
    top_k: int = 10,
) -> list[dict]:
    """
    Identify and rank cross-domain discoveries from candidate pairs.

    Returns
    -------
    List of cross-domain discovery dictionaries sorted by cross_domain_score descending.
    """
    discoveries: list[dict] = []

    for cand in candidates:
        ca = cand.get("concept_a", "")
        cb = cand.get("concept_b", "")
        dom_a = concept_domains.get(ca, {}).get("domain", "Unknown")
        dom_b = concept_domains.get(cb, {}).get("domain", "Unknown")

        # Must be distinct domains
        if dom_a == dom_b or dom_a == "Unknown" or dom_b == "Unknown":
            continue

        sem_score = float(cand.get("semantic_similarity", 0.5))
        gap_score = float(cand.get("research_gap", {}).get("score", 0.5)) if isinstance(cand.get("research_gap"), dict) else 0.5
        setgn_score = cand.get("setgn_score")
        temp_score = float(setgn_score if setgn_score is not None else 0.5)

        cd_score = compute_cross_domain_score(
            domain_a=dom_a,
            domain_b=dom_b,
            semantic_compatibility=sem_score,
            gap_score=gap_score,
            temporal_emergence=temp_score,
        )

        if cd_score >= min_cross_domain_score:
            discoveries.append({
                "concept_a": ca,
                "domain_a": dom_a,
                "concept_b": cb,
                "domain_b": dom_b,
                "cross_domain_score": cd_score,
                "semantic_compatibility": round(sem_score, 4),
                "domain_distance": round(get_domain_distance(dom_a, dom_b), 3),
                "base_candidate_score": cand.get("candidate_score", 0.0),
                "supporting_papers": cand.get("supporting_papers", []),
            })

    discoveries.sort(key=lambda x: x["cross_domain_score"], reverse=True)
    return discoveries[:top_k]
