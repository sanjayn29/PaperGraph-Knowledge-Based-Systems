"""
text_utils.py
─────────────
Text normalization helpers used throughout PaperGraph.

Provides:
- ACADEMIC_STOPWORDS: generic academic terms to filter out from concept lists
- SYNONYM_MAP: canonical forms for common abbreviations / variants
- normalize_concept(): lower + strip + synonym resolution
- clean_text(): basic whitespace / punctuation cleaning
- is_valid_concept(): filter check combining length + stopword + digit-only tests
"""

from __future__ import annotations

import re
import unicodedata

# ─────────────────────────────────────────────────────────────
# Generic academic stopwords — concepts that appear in nearly
# every research paper and therefore carry no discriminative
# signal for cross-paper relationship discovery.
# ─────────────────────────────────────────────────────────────
ACADEMIC_STOPWORDS: set[str] = {
    # Paper structure
    "paper", "study", "work", "research", "article", "report",
    "review", "survey", "analysis", "analyses", "experiment",
    "experiments", "evaluation", "evaluations", "implementation",
    "approach", "method", "methods", "methodology", "technique",
    "techniques", "framework", "system", "systems", "process",
    "processes", "procedure", "procedures", "algorithm", "algorithms",
    # Generic descriptors
    "novel", "new", "proposed", "existing", "previous", "current",
    "recent", "state", "art", "based", "using", "used", "use",
    "data", "dataset", "datasets", "model", "models", "result",
    "results", "performance", "accuracy", "efficiency", "problem",
    "task", "tasks", "solution", "solutions", "challenge", "challenges",
    "issue", "issues", "aspect", "aspects", "feature", "features",
    "application", "applications", "example", "examples", "case",
    "cases", "scenario", "scenarios", "sample", "samples",
    # Common verbs turned nouns
    "training", "testing", "learning", "classification", "prediction",
    "detection", "recognition", "generation", "extraction", "selection",
    # Quantitative generics
    "number", "value", "values", "set", "sets", "size", "level",
    "rate", "rates", "measure", "measures", "metric", "metrics",
    "score", "scores", "weight", "weights", "parameter", "parameters",
    # Sections
    "introduction", "conclusion", "conclusions", "abstract",
    "discussion", "related", "background", "contribution",
    "contributions", "summary", "overview", "objective", "objectives",
    # Misc
    "however", "therefore", "thus", "furthermore", "moreover",
    "although", "despite", "since", "while", "fact", "show",
    "shows", "shown", "demonstrate", "demonstrates", "demonstrated",
    "improve", "improves", "improved", "improvement", "improvements",
    "achieve", "achieves", "achieved", "obtain", "obtains", "obtained",
    "compare", "compared", "comparison", "outperform", "outperforms",
    "significant", "significantly", "effective", "effectively",
    "efficient", "efficiently", "high", "low", "large", "small",
    "good", "best", "better", "first", "second", "third",
    "also", "well", "different", "various", "many", "several",
    "both", "two", "three", "among", "including", "without",
    "proposed method", "our method", "this paper", "this work",
}

# ─────────────────────────────────────────────────────────────
# Synonym / abbreviation → canonical form mapping.
# Keys are lowercase; values are the display-ready canonical form.
# ─────────────────────────────────────────────────────────────
SYNONYM_MAP: dict[str, str] = {
    # Graph / Network
    "gnn": "Graph Neural Network",
    "graph neural network": "Graph Neural Network",
    "graph neural networks": "Graph Neural Network",
    "gcn": "Graph Convolutional Network",
    "graph convolutional network": "Graph Convolutional Network",
    "graph convolutional networks": "Graph Convolutional Network",
    "gat": "Graph Attention Network",
    "graph attention network": "Graph Attention Network",
    "graph attention networks": "Graph Attention Network",
    "gae": "Graph Autoencoder",
    "graph autoencoder": "Graph Autoencoder",
    "kg": "Knowledge Graph",
    "knowledge graph": "Knowledge Graph",
    "knowledge graphs": "Knowledge Graph",
    # ML / DL
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "nn": "Neural Network",
    "neural network": "Neural Network",
    "neural networks": "Neural Network",
    "dnn": "Deep Neural Network",
    "deep neural network": "Deep Neural Network",
    "deep neural networks": "Deep Neural Network",
    "cnn": "Convolutional Neural Network",
    "convolutional neural network": "Convolutional Neural Network",
    "convolutional neural networks": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "recurrent neural network": "Recurrent Neural Network",
    "recurrent neural networks": "Recurrent Neural Network",
    "lstm": "Long Short-Term Memory",
    "long short-term memory": "Long Short-Term Memory",
    "transformer": "Transformer",
    "transformers": "Transformer",
    "bert": "BERT",
    "gpt": "GPT",
    "llm": "Large Language Model",
    "large language model": "Large Language Model",
    "large language models": "Large Language Model",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "rl": "Reinforcement Learning",
    "reinforcement learning": "Reinforcement Learning",
    "fl": "Federated Learning",
    "federated learning": "Federated Learning",
    "tl": "Transfer Learning",
    "transfer learning": "Transfer Learning",
    # CV
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "object detection": "Object Detection",
    "image segmentation": "Image Segmentation",
    "image classification": "Image Classification",
    # Bio / Med
    "drug discovery": "Drug Discovery",
    "drug design": "Drug Design",
    "molecular graph": "Molecular Graph",
    "protein structure": "Protein Structure",
    "genomics": "Genomics",
    "proteomics": "Proteomics",
    "bioinformatics": "Bioinformatics",
    "electronic health record": "Electronic Health Record",
    "ehr": "Electronic Health Record",
    "clinical trial": "Clinical Trial",
    # NLP sub-tasks
    "named entity recognition": "Named Entity Recognition",
    "ner": "Named Entity Recognition",
    "relation extraction": "Relation Extraction",
    "sentiment analysis": "Sentiment Analysis",
    "question answering": "Question Answering",
    "qa": "Question Answering",
    "text classification": "Text Classification",
    "machine translation": "Machine Translation",
    # General CS
    "rec sys": "Recommender System",
    "recommendation system": "Recommender System",
    "recommender system": "Recommender System",
    "recommender systems": "Recommender System",
    "link prediction": "Link Prediction",
    "node classification": "Node Classification",
    "graph classification": "Graph Classification",
    "anomaly detection": "Anomaly Detection",
    "time series": "Time Series",
    "federated": "Federated Learning",
    "privacy": "Privacy",
    "explainability": "Explainability",
    "interpretability": "Interpretability",
    "xai": "Explainable AI",
    "explainable ai": "Explainable AI",
    "fairness": "Fairness",
    "robustness": "Robustness",
}


def normalize_concept(concept: str) -> str:
    """
    Return the canonical display form of a concept string.

    Steps:
    1. Lowercase + strip whitespace
    2. Collapse internal whitespace
    3. Look up in SYNONYM_MAP
    4. If not found, title-case the stripped form
    """
    raw = concept.strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    if raw in SYNONYM_MAP:
        return SYNONYM_MAP[raw]
    # Title-case as a reasonable default for display
    return concept.strip().title()


def clean_text(text: str) -> str:
    """
    Normalize unicode, collapse whitespace, and remove control characters.
    Preserves sentence-ending punctuation.
    """
    # Normalize unicode (e.g. ligatures, fancy quotes)
    text = unicodedata.normalize("NFKD", text)
    # Remove control characters except newlines and tabs
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    # Collapse multiple spaces / tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_valid_concept(concept: str, min_length: int = 3, max_length: int = 60) -> bool:
    """
    Return True if the concept string passes basic quality filters:
    - Length between min_length and max_length characters
    - Not exclusively digits / punctuation
    - Not in ACADEMIC_STOPWORDS (normalized)
    - Not a single character after stripping
    """
    stripped = concept.strip()
    if len(stripped) < min_length or len(stripped) > max_length:
        return False
    # Reject digit-only tokens
    if re.fullmatch(r"[\d\s\W]+", stripped):
        return False
    lower = stripped.lower()
    # Check stopwords (also check the normalized form)
    if lower in ACADEMIC_STOPWORDS:
        return False
    normalized_lower = normalize_concept(stripped).lower()
    if normalized_lower in ACADEMIC_STOPWORDS:
        return False
    return True


def extract_noun_phrases(text: str) -> list[str]:
    """
    Lightweight regex-based noun-phrase extraction — no external NLP libs required.

    Captures patterns like:
    - Capitalized multi-word sequences (e.g. "Graph Neural Network")
    - Acronym-like tokens (2–8 uppercase letters)
    - Hyphenated technical terms (e.g. "semi-supervised")

    Returns a deduplicated list of raw phrase strings.
    """
    phrases: list[str] = []

    # Pattern 1: sequences of 1–5 capitalized/title-case words
    capitalized_seq = re.findall(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b",
        text,
    )
    phrases.extend(capitalized_seq)

    # Pattern 2: all-caps acronyms (2–8 letters, not mid-sentence noise)
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", text)
    phrases.extend(acronyms)

    # Pattern 3: hyphenated technical terms (e.g. semi-supervised, multi-layer)
    hyphenated = re.findall(
        r"\b[a-z]+-[a-z]+(?:-[a-z]+)?\b",
        text,
    )
    phrases.extend(hyphenated)

    # Deduplicate preserving first-seen order
    seen: set[str] = set()
    result: list[str] = []
    for p in phrases:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result
