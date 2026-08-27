# 🧠 PaperGraph — Complete Technical Explainer & Viva Defense Guide

> A comprehensive reference for **PaperGraph** — detailing both the **Base Paper Methodology** (KBS 2025) and our **Five Novel Academic Extensions**, complete with mathematical formulations, structured data examples, architecture diagrams, and a viva demonstration script.

---

## 🏛️ 1. Base Paper vs. Our Proposed Extensions

| Category | Component | Base Paper (KBS 2025) | PaperGraph (Our Extension) |
|---|---|---|---|
| **Base Methodology** | **Temporal Graph Modeling** | Continuous timestamped co-occurrences | `TemporalEvent` stream with `TemporalGraph` engine |
| **Base Methodology** | **SE-TGN Architecture** | Temporal GNN with semantic message passing | PyTorch `NodeMemory` (GRU) + `TimeEncode` (Sinusoidal) + Link Classifier |
| **Base Methodology** | **CREF & GIC Layer** | Multi-dimension rubric + hypothesis generation | Gemini-powered CREF scores & GIC research roadmap |
| **Base Methodology** | **Formal Benchmarking** | AUC, AP, P@10, NDCG@10 metrics | `scikit-learn` baseline evaluator comparing Random, Graph, GCN, SE-TGN |
| **🌟 Our Extension 1** | **Research Gap Detection** | ❌ Not in base paper | Multi-factor gap scoring (semantic, structural, temporal, cross-domain, novelty) |
| **🌟 Our Extension 2** | **Evidence & Provenance** | ❌ Not in base paper | Complete citation traceability, graph topology proofs, and evidence strength |
| **🌟 Our Extension 3** | **Human-in-the-Loop Rerank** | ❌ Static ranking only | Interactive feedback (👍 ⭐ 👎 🔖) + real-time personalized reranking without retraining |
| **🌟 Our Extension 4** | **Cross-Domain Discovery** | ❌ Not in base paper | 11-discipline domain taxonomy, centroid embeddings, & synergy filtering |
| **🌟 Our Extension 5** | **Interactive Research Assistant** | ❌ Not in base paper | Context-grounded conversational agent with strict anti-hallucination guardrails |

---

## 🗺️ 2. High-Level Extended Architecture

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

## 🔬 3. Deep-Dive Mathematical Formulations

### Feature 1: Research Gap Detection (`services/research_gap_detector.py`)
Identifies concept pairs $(u, v)$ that have strong latent compatibility but minimal direct co-occurrence in the literature:

$$\text{ResearchGapScore}(u, v) = 0.30 \cdot S_{\text{sem}}(u, v) + 0.25 \cdot S_{\text{temp}}(u, v) + 0.20 \cdot S_{\text{struct}}(u, v) + 0.15 \cdot S_{\text{domain}}(u, v) + 0.10 \cdot S_{\text{nov}}(u, v)$$

* **$S_{\text{sem}}(u, v)$:** Cosine similarity of dense embeddings $\frac{\mathbf{e}_u \cdot \mathbf{e}_v}{\|\mathbf{e}_u\| \|\mathbf{e}_v\|}$.
* **$S_{\text{temp}}(u, v)$:** Temporal emergence score based on recent publication year distribution.
* **$S_{\text{struct}}(u, v)$:** High individual node centrality with low direct edge weight:
  $$S_{\text{struct}}(u, v) = 0.6 \cdot \frac{C_d(u) + C_d(v)}{2} + 0.4 \cdot (1 - \text{EdgePenalty}(u, v))$$
* **$S_{\text{domain}}(u, v)$:** Interdisciplinary distance between assigned domains.
* **$S_{\text{nov}}(u, v)$:** Novelty potential ($1 - \frac{\text{Weight}(u, v)}{5}$).

---

### Feature 2: Evidence & Provenance Tracking (`services/provenance.py`)
Provides verifiable evidence cards linking candidate predictions back to raw corpus data:
* **Supporting Papers:** Identifies papers containing $u$, $v$, or both, with exact titles, years, and authors.
* **Graph Proof:** Reports shortest path length, direct co-occurrence frequency, and common intermediate bridge concepts.
* **Evidence Strength Classification:**
  - `HIGH`: Direct co-occurrence in multiple papers + candidate score $\ge 0.70$.
  - `MEDIUM`: Contextual paper support $\ge 2$ or semantic similarity $\ge 0.65$.
  - `EXPLORATORY`: Distant conceptual leap inferred primarily via graph topology.

---

### Feature 3: Human-in-the-Loop Personalized Ranking (`services/user_feedback.py`)
Enables researchers to iteratively guide recommendations without retraining the SE-TGN network:

$$\text{PersonalizedScore}(u, v) = 0.70 \cdot \text{Score}_{\text{original}}(u, v) + 0.15 \cdot \text{Score}_{\text{pref}}(u, v) + 0.15 \cdot \text{Score}_{\text{sim}}(u, v)$$

* **$\text{Score}_{\text{pref}}$:** Explicit boosts from user ratings (+1.0 for ⭐ High Value, +0.6 for 👍 Useful, -0.8 for 👎 Irrelevant, -0.5 for "Not My Area").
* **$\text{Score}_{\text{sim}}$:** Semantic affinity to previously favored concept embeddings.
* **UI Features:** Live rank delta badges (e.g. `▲ +5 (Orig #7)`) and plain-English explanation of movements.

---

### Feature 4: Cross-Domain Discovery (`services/domain_analyzer.py`)
Classifies concepts into 11 standard scientific disciplines and identifies cross-field synergies:

$$\text{CrossDomainScore}(u, v) = 0.40 \cdot \text{Dist}(\text{Dom}_u, \text{Dom}_v) + 0.30 \cdot S_{\text{sem}}(u, v) + 0.20 \cdot \text{GapScore}(u, v) + 0.10 \cdot S_{\text{temp}}(u, v)$$

Disciplines supported:
1. Artificial Intelligence & Machine Learning
2. Computer Science & Systems
3. Healthcare & Medicine
4. Biology & Bioinformatics
5. Chemistry & Materials
6. Physics & Quantum Science
7. Mathematics & Statistics
8. Environmental & Climate Science
9. Engineering & Robotics
10. Social Sciences & Economics

---

### Feature 5: Interactive Research Assistant (`services/research_assistant.py`)
A context-grounded conversational engine grounded in the uploaded analysis:
* **Grounded Reasoning:** Accesses paper metadata, graph topology, SE-TGN predictions, research gaps, and CREF scores.
* **Strict Anti-Hallucination Guardrail:** Explicitly states:
  > *"I could not find sufficient evidence in the uploaded research corpus."*
  whenever a query seeks information outside the uploaded papers.
* **Preset Accelerators:** Instant buttons for *"Why recommended?"*, *"Supporting papers?"*, *"Why is this a gap?"*, and *"Research questions?"*.

---

## 🎤 4. Viva Defense & Demonstration Guide

When presenting this project to an examiner or review committee, use this step-by-step demonstration flow:

### Step 1: Explain the Base Paper
> *"Our project is based on the 2025 Knowledge-Based Systems paper, which introduced a synergistic GNN-LLM framework for scientific discovery. We implemented their core pipeline: temporal dynamic graph modeling, continuous SE-TGN link prediction with GRU memory, and LLM rubric evaluation using CREF and GIC."*

### Step 2: Highlight Our 5 Academic Contributions
> *"While the base paper focused purely on algorithmic link prediction, we identified five critical limitations in practical research discovery and extended the system with:*
> 1. *Research Gap Detection — to find missing links between prominent concepts.*
> 2. *Evidence Provenance — to make every AI recommendation auditable back to source papers.*
> 3. *Human-in-the-Loop Personalization — allowing researchers to steer rankings in real-time.*
> 4. *Cross-Domain Discovery — identifying interdisciplinary breakthroughs across 11 disciplines.*
> 5. *Interactive Research Assistant — a grounded conversational interface with anti-hallucination guardrails."*

### Step 3: Live UI Demonstration
1. **Upload Papers:** Show multi-year papers in the **Paper Metadata & Year Correction** expander.
2. **Tab 1 (Top Insight):** Show the CREF scores and GIC hypothesis.
3. **Tab 2 (Predictions & Feedback):** Click "⭐ High Value" on a candidate connection and toggle to "Personalized Ranking" to demonstrate real-time ranking adjustments.
4. **Tab 3 (Research Gaps):** Show the gap score breakdown and explain the structural/semantic formula.
5. **Tab 4 (Cross-Domain):** Filter by `AI ↔ Biology` or `AI ↔ Healthcare` to demonstrate interdisciplinary synergy scoring.
6. **Tab 5 (Evidence & Provenance):** Select a candidate and show the exact supporting paper citations and graph proof.
7. **Tab 6 (Research Assistant):** Click *"Why is this a gap?"* or ask an out-of-scope question to demonstrate grounded reasoning and the anti-hallucination guardrail.
8. **Tab 9 (Model Evaluation):** Present the quantitative benchmark table ($AUC$, $AP$, $P@10$, $NDCG@10$) comparing Random, Graph, GCN, and SE-TGN.

---

## 🧪 5. Automated Testing Verification

PaperGraph includes **107 automated unit tests** verifying all mathematical models and UI services:

```bash
python -m pytest tests/ -v
```
```
============================= 107 passed in 18.0s =============================
```
