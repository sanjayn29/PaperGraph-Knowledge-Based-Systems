# PaperGraph — AI Research Discovery System

> A synergistic **Temporal Graph Neural Network (SE-TGN) + Large Language Model (LLM)** framework extended with **Research Gap Detection, Evidence Provenance, Human-in-the-Loop Personalization, Cross-Domain Discovery, and an Interactive Research Assistant**.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange.svg)](https://networkx.org)
[![License: Academic](https://img.shields.io/badge/License-Academic%20B.Tech-lightgrey.svg)]()

---

## ⚠️ Academic Disclaimer

> This system identifies **potentially interesting relationships** within the uploaded research papers. It does **not** claim to prove that a relationship is experimentally validated. Results are exploratory and designed to assist human researchers in discovering novel cross-domain hypotheses.

---

## 📋 Overview

PaperGraph implements the core methodology from the research paper:

> *"Uncovering novel scientific insights with a synergistic GNN-LLM framework"*  
> **Knowledge-Based Systems, 2025**

And extends it with **five novel research-oriented contributions** designed to enhance transparency, interdisciplinary discovery, researcher agency, and interactive exploration.

---

## 🏛️ Base Paper vs. Our Proposed Extensions

| Component / Feature | Category | Base Paper (KBS 2025) | PaperGraph (Our Extended System) |
|---|---|---|---|
| **Temporal Dynamic Graph** | Base Methodology | Multi-year continuous event streams | Chronologically sorted `TemporalEvent` stream with `TemporalGraph` |
| **SE-TGN Architecture** | Base Methodology | TGN with semantic message passing | PyTorch `NodeMemory` (GRU) + `TimeEncode` (Sinusoidal) + Link Classifier |
| **CREF Rubric Evaluation** | Base Methodology | LLM scoring (Novelty, Impact, Plausibility, Interdisciplinarity) | Gemini-powered CREF prompt pipeline with normalized 1–5 scoring |
| **GIC Insight Generation** | Base Methodology | Research direction, questions, hypotheses, roadmap | Gemini-powered structured hypotheses and validation roadmaps |
| **Quantitative Benchmarking**| Base Methodology | AUC, AP, P@10, NDCG@10 metrics | `scikit-learn` baseline evaluator comparing Random, Graph, GCN, SE-TGN |
| **1. Research Gap Detection**| 🌟 **Our Contribution** | ❌ Not in base paper | Multi-factor gap scoring (semantic, structural, temporal, cross-domain, novelty) |
| **2. Evidence & Provenance** | 🌟 **Our Contribution** | ❌ Not in base paper | Complete citation traceability, graph topology proofs, and evidence strength |
| **3. Human-in-the-Loop Rerank**| 🌟 **Our Contribution** | ❌ Static ranking only | Interactive feedback (👍 ⭐ 👎 🔖) + real-time personalized reranking without retraining |
| **4. Cross-Domain Discovery**| 🌟 **Our Contribution** | ❌ Not in base paper | 11-discipline domain taxonomy, centroid embeddings, & synergy filtering |
| **5. Research Assistant** | 🌟 **Our Contribution** | ❌ Not in base paper | Context-grounded conversational agent with strict anti-hallucination guardrails |

---

## 🏗️ End-to-End System Architecture

```
                    EXISTING BASE PIPELINE
                             │
                             ▼
                       Candidate Pool
                             │
              ┌──────────────┼───────────────┐
              │              │               │
              ▼              ▼               ▼
      Research Gap      Cross-Domain     Evidence /
       Detection         Discovery       Provenance
              │              │               │
              └──────────────┼───────────────┘
                             ▼
                    Candidate Enrichment
                             │
                             ▼
                 CREF + GIC Existing Layer
                             │
                             ▼
                  Human-in-the-Loop Ranking
                             │
                             ▼
                Personalized Recommendations
                             │
                             ▼
               Interactive Research Assistant
```

---

## 🌟 Detailed Academic Extensions

### 1. Research Gap Detection (`services/research_gap_detector.py`)
Detects underexplored scientific gaps where concepts are semantically compatible and prominent, yet weakly connected in the literature:
$$\text{ResearchGapScore} = 0.30 \cdot \text{Semantic} + 0.25 \cdot \text{Temporal} + 0.20 \cdot \text{Structural} + 0.15 \cdot \text{CrossDomain} + 0.10 \cdot \text{Novelty}$$

### 2. Evidence & Provenance System (`services/provenance.py`)
Guarantees full traceability so no insight appears as an ungrounded LLM hallucination:
* **Supporting Papers:** Direct co-occurrences and contextual mentions with exact titles, years, and authors.
* **Graph Topology Evidence:** Degrees, direct edge weights, and common bridge concepts.
* **Evidence Strength:** Classified as `HIGH`, `MEDIUM`, or `EXPLORATORY`.

### 3. Human-in-the-Loop Personalized Ranking (`services/user_feedback.py`)
Allows researchers to rate recommendations (👍 Useful, ⭐ High Value, 👎 Irrelevant, 🔖 Save, "Not My Area"):
$$\text{PersonalizedScore} = 0.70 \cdot \text{OriginalScore} + 0.15 \cdot \text{UserPreferenceScore} + 0.15 \cdot \text{FeedbackSimilarityScore}$$
Maintains a local researcher profile and displays rank movements (e.g. `#7` $\rightarrow$ `#2`, ▲ +5).

### 4. Cross-Domain Discovery (`services/domain_analyzer.py`)
Classifies concepts into 11 scientific disciplines using keyword anchors and dense centroid embeddings:
$$\text{CrossDomainScore} = 0.40 \cdot \text{DomainDistance} + 0.30 \cdot \text{SemanticCompatibility} + 0.20 \cdot \text{GapScore} + 0.10 \cdot \text{TemporalEmergence}$$
Enables filtering by specific discipline pairs (e.g. `AI ↔ Healthcare`, `AI ↔ Biology`).

### 5. Interactive Research Assistant (`services/research_assistant.py`)
A conversational assistant grounded in the current analysis session:
* Answers queries regarding recommendation rationales, supporting papers, gaps, and validation roadmaps.
* **Anti-Hallucination Guardrail:** Explicitly states *"I could not find sufficient evidence in the uploaded research corpus"* when questions exceed corpus evidence.

---

## 🚀 Quick Start

### 1. Navigate to the project directory
```bash
cd research_discovery
```

### 2. Activate virtual environment and install dependencies
```powershell
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scikit-learn
```

### 3. Configure Gemini API Key
Create `.env` in `research_discovery/`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Launch the Streamlit application
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Unit Testing Suite

The repository includes **107 passing unit tests** covering all base modules and the 5 extensions:

```bash
python -m pytest tests/ -v
```

```
tests/test_domain_analyzer.py ........ [PASS]
tests/test_research_gap_detector.py ... [PASS]
tests/test_provenance.py .............. [PASS]
tests/test_user_feedback.py ........... [PASS]
tests/test_research_assistant.py ...... [PASS]
tests/test_history_compatibility.py ... [PASS]
tests/test_temporal_graph.py .......... [PASS]
tests/test_se_tgn.py .................. [PASS]
tests/test_evaluator.py ............... [PASS]
============================= 107 passed in 18.0s =============================
```

---

## 📁 Repository Structure

```
research_discovery/
├── app.py                          # Streamlit UI with 10 interactive tabs
├── requirements.txt                # Core and optional dependencies
├── services/
│   ├── pdf_processor.py            # PDF ingestion & metadata extraction
│   ├── concept_extractor.py        # Regex noun phrase concept extraction
│   ├── embeddings.py               # all-MiniLM-L6-v2 dense embeddings
│   ├── graph_builder.py            # NetworkX co-occurrence graph builder
│   ├── temporal_graph.py           # Temporal event stream & train/val/test splits
│   ├── se_tgn.py                   # PyTorch SE-TGN temporal network
│   ├── gnn_model.py                # Optional GCN structural baseline
│   ├── graph_analyzer.py           # 3-tier candidate scoring formula
│   ├── domain_analyzer.py          # [EXTENSION] 11-domain taxonomy & cross-domain engine
│   ├── research_gap_detector.py    # [EXTENSION] Underexplored research gap detector
│   ├── provenance.py               # [EXTENSION] Citation & graph evidence tracker
│   ├── user_feedback.py            # [EXTENSION] Human-in-the-loop personalized reranker
│   ├── research_assistant.py       # [EXTENSION] Context-grounded conversational agent
│   ├── llm_service.py              # Gemini CREF & GIC service
│   ├── evaluator.py                # Formal quantitative benchmarks
│   ├── analysis_pipeline.py        # End-to-end orchestration pipeline
│   └── history_service.py          # Backward-compatible JSON history persistence
└── tests/                          # 107 automated unit tests
```
