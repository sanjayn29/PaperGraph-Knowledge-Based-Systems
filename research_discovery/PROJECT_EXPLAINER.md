# 🧠 PaperGraph — Complete Project Explainer

> A step-by-step breakdown of how PaperGraph works, with real examples, tables, and formulas for every stage of the pipeline.

---

## 📌 What is PaperGraph?

PaperGraph is an **AI-powered research discovery tool** built as a B.Tech project. You upload 5–10 research papers (PDFs), and the system:

1. Reads and extracts text from each PDF
2. Pulls out **key research concepts** (like "Graph Neural Network", "Healthcare", "NLP")
3. Builds a **knowledge graph** where concepts are nodes and shared papers are edges
4. **Scores concept pairs** using graph metrics + semantic similarity + optional GNN
5. Asks **Gemini AI** to evaluate and explain the most promising underexplored connection
6. Saves the result as a JSON history file

---

## 🗺️ Pipeline Overview

```
📄 Upload PDFs
     ↓
Step 1 — PDF Extraction       (PyMuPDF)
     ↓
Step 2 — Concept Extraction   (Regex + Stopword filter + Synonym map)
     ↓
Step 3 — Semantic Embeddings  (all-MiniLM-L6-v2, 384 dimensions)
     ↓
Step 4 — Knowledge Graph      (NetworkX, nodes=concepts, edges=co-occurrence)
     ↓
Step 5 — Scoring & Ranking    (Graph metrics + Semantic similarity + Optional GNN)
     ↓
Step 6 — LLM Evaluation       (Gemini CREF → GIC insight generation)
     ↓
📊 Results displayed in Streamlit + saved to data/history/
```

---

## 🔬 Step-by-Step Analysis

---

### ✅ Step 1 — PDF Extraction

**File:** `services/pdf_processor.py`  
**Library:** `PyMuPDF` (also called `fitz`)

The system reads each uploaded PDF and extracts structured data from it.

#### What it extracts:

| Field | Source (in priority order) |
|---|---|
| `title` | PDF metadata → first large text line → filename |
| `abstract` | Regex for "Abstract" heading → first paragraph |
| `year` | PDF metadata date → most frequent 4-digit year in text |
| `authors` | PDF metadata `author` field → text heuristics |
| `full_text` | All pages concatenated (used during analysis only, not saved) |

#### Example Input → Output:

**Input:** A PDF file `transformer_attention.pdf`

**Output (paper dict):**
```json
{
  "paper_id": "paper_001",
  "filename": "transformer_attention.pdf",
  "title": "Attention Is All You Need",
  "abstract": "We propose a new simple network architecture, the Transformer...",
  "year": 2017,
  "authors": "Vaswani et al.",
  "full_text": "Abstract\nWe propose a new simple network..."
}
```

> ⚠️ **Important:** Scanned PDFs (images) will not work — the PDF must have a selectable text layer.

---

### ✅ Step 2 — Concept Extraction

**File:** `services/concept_extractor.py`  
**File:** `utils/text_utils.py`

This step reads each paper's `title + abstract + first 3000 chars of full_text` and pulls out key research concepts using **regex-based noun phrase extraction** (no NLTK or spaCy needed).

#### How it works:

| Sub-step | What happens | Example |
|---|---|---|
| **Extract noun phrases** | Regex finds capitalized multi-word phrases | `"Graph Neural Network"`, `"attention mechanism"` |
| **Filter stopwords** | 150+ generic academic terms removed | `"proposed method"`, `"experimental results"` → ❌ removed |
| **Normalize synonyms** | Maps abbreviations to canonical form | `"GNN"` → `"Graph Neural Network"`, `"ML"` → `"Machine Learning"` |
| **Frequency filter** | Only keep concepts appearing in ≥ 1 paper | rare typos and noise removed |
| **Top-N cap** | Keep only top 80 concepts by cross-paper frequency | ensures graph stays manageable |

#### Synonym Map Examples:

| Raw term found | Canonical form stored |
|---|---|
| `GNN` | `Graph Neural Network` |
| `ML` | `Machine Learning` |
| `DL` | `Deep Learning` |
| `NLP` | `Natural Language Processing` |
| `KG` | `Knowledge Graph` |
| `RL` | `Reinforcement Learning` |
| `CV` | `Computer Vision` |

#### Example:

**Input text (from paper abstract):**
> "In this paper, we propose a GNN-based approach for drug discovery using Knowledge Graphs..."

**Extracted concepts after normalization:**
```
["Graph Neural Network", "Drug Discovery", "Knowledge Graph"]
```

**After cross-paper filtering** (keeping only concepts in ≥ 1 paper):
```
["Graph Neural Network", "Drug Discovery", "Knowledge Graph"]  ✅ all kept
```

> 📌 A real run on 8 papers might extract 231 raw concepts → filtered down to 80 canonical concepts.

---

### ✅ Step 3 — Semantic Embeddings

**File:** `services/embeddings.py`  
**Model:** `all-MiniLM-L6-v2` (from Hugging Face, ~90 MB, Apache 2.0 license)

Every concept is converted into a **384-dimensional vector** that captures its semantic meaning. Concepts with similar meaning will have vectors that point in similar directions.

#### What is a vector embedding?

Think of it like GPS coordinates — but in 384 dimensions. Concepts that mean similar things are "close" in this space.

| Concept A | Concept B | Cosine Similarity | Interpretation |
|---|---|---|---|
| `Graph Neural Network` | `Deep Learning` | 0.82 | Very similar field |
| `Graph Neural Network` | `Drug Discovery` | 0.41 | Different domains |
| `Graph Neural Network` | `Attention Mechanism` | 0.74 | Related technique |
| `Healthcare` | `Drug Discovery` | 0.79 | Same application area |
| `Healthcare` | `Graph Theory` | 0.28 | Unrelated on surface |

> ℹ️ These embeddings are kept **in-memory only** — nothing is written to disk.  
> If `sentence-transformers` isn't installed, semantic similarity defaults to 0 for all pairs.

---

### ✅ Step 4 — Knowledge Graph Construction

**File:** `services/graph_builder.py`  
**Library:** `NetworkX`

A graph is built where:
- **Nodes** = unique concept strings (e.g., `"Graph Neural Network"`)
- **Edges** = two concepts appeared in the **same paper**
- **Edge weight** = how many papers they share

#### Example — 3 papers, 6 concepts:

| Paper | Concepts |
|---|---|
| Paper 1: "GNN for Drug Discovery" | `Graph Neural Network`, `Drug Discovery`, `Healthcare` |
| Paper 2: "Transformer in NLP" | `Natural Language Processing`, `Attention Mechanism`, `Deep Learning` |
| Paper 3: "GNN meets NLP" | `Graph Neural Network`, `Natural Language Processing`, `Deep Learning` |

**Resulting graph edges:**

| Edge (Concept A → Concept B) | Weight (shared papers) |
|---|---|
| `Graph Neural Network` ↔ `Drug Discovery` | 1 |
| `Graph Neural Network` ↔ `Healthcare` | 1 |
| `Drug Discovery` ↔ `Healthcare` | 1 |
| `Natural Language Processing` ↔ `Attention Mechanism` | 1 |
| `Natural Language Processing` ↔ `Deep Learning` | 2 |
| `Graph Neural Network` ↔ `Natural Language Processing` | 1 |
| `Graph Neural Network` ↔ `Deep Learning` | 1 |

**Graph stats from a real 8-paper run:**
```
Nodes: 80 concepts
Edges: 1,613 co-occurrence edges
Density: 0.507 (moderately connected)
```

---

### ✅ Step 5 — Candidate Scoring & Ranking

**File:** `services/graph_analyzer.py`

This is the **core intelligence** of PaperGraph. Every possible pair of concepts is scored to find the most **promising underexplored connection**.

#### 5a. Graph Score

For each concept pair `(A, B)`, three sub-scores are computed:

| Sub-score | Formula | What it measures |
|---|---|---|
| `centrality_score` | `mean(degree_A, degree_B)` | How important both concepts are in the graph |
| `bridge_score` | `max(betweenness_A, betweenness_B)` | Whether either concept bridges different clusters |
| `novelty_bonus` | `1 - (edge_weight / max_edge_weight)` | How *weakly connected* they are (weak = potentially novel) |

```
graph_score = 0.45 × centrality_score
            + 0.30 × bridge_score
            + 0.25 × novelty_bonus
```

> 💡 **Key insight:** A pair scores HIGH when both concepts are important (centrality), at least one connects different research clusters (bridge), and they haven't been deeply explored together yet (novelty).

#### 5b. Combined Candidate Score

| Mode | Formula |
|---|---|
| **Without GNN** (default) | `0.57 × graph_score + 0.43 × semantic_similarity` |
| **With GNN** (if PyTorch installed) | `0.40 × graph_score + 0.30 × semantic_similarity + 0.30 × gnn_score` |

#### Example — Scoring 3 candidate pairs:

| Candidate Pair | Graph Score | Semantic Sim | Candidate Score | Why interesting? |
|---|---|---|---|---|
| `Graph Neural Network` + `Drug Discovery` | 0.71 | 0.41 | **0.58** | High centrality, weak co-occurrence |
| `Deep Learning` + `Healthcare` | 0.65 | 0.55 | **0.61** | Bridge concept, cross-domain |
| `Natural Language Processing` + `Knowledge Graph` | 0.78 | 0.62 | **0.71** ⭐ | Top-ranked |

#### Strong-edge filter:

Pairs that co-occur in **more than 60% of uploaded papers** are automatically excluded — they are already "known" connections, not novel candidates.

```
dynamic_threshold = max(3, int(num_papers × 0.6))
```

| Papers uploaded | Threshold | Meaning |
|---|---|---|
| 5 papers | 3 | Pairs in 3+ papers are excluded |
| 8 papers | 4 | Pairs in 4+ papers are excluded |
| 10 papers | 6 | Pairs in 6+ papers are excluded |

---

### ✅ Step 6 — LLM Evaluation (CREF + GIC)

**File:** `services/llm_service.py`  
**Model:** Google Gemini (default: `gemini-1.5-flash`)  
**Requires:** `GEMINI_API_KEY` in your `.env` file

The top 3 candidates from Step 5 are sent to Gemini for evaluation.

#### 6a — CREF Evaluation (Candidate Relationship Evaluation Framework)

Gemini scores each candidate on **4 dimensions** (1–5 scale):

| Dimension | What it means | Example score |
|---|---|---|
| `novelty` | How underexplored is this connection *within the uploaded papers*? | 4 |
| `impact` | How significant could this be for advancing research? | 5 |
| `plausibility` | How technically feasible and scientifically grounded? | 4 |
| `interdisciplinarity` | How much does it bridge different research domains? | 5 |

> ⚠️ The LLM uses hedged language — it **never** claims the connection is "globally novel" or "scientifically proven". It evaluates within the context of your uploaded papers only.

**CREF output example:**
```json
{
  "candidate": "Natural Language Processing + Knowledge Graph",
  "novelty": 4,
  "impact": 5,
  "plausibility": 4,
  "interdisciplinarity": 5,
  "reasoning": "This connection bridges symbolic reasoning (KG) with statistical NLP methods. Within the uploaded papers, their integration is mentioned but not deeply explored..."
}
```

The candidate with the **highest total CREF score** (novelty + impact + plausibility + interdisciplinarity) is selected for insight generation.

#### 6b — GIC (Generated Insight Content)

The top CREF-ranked candidate gets a full research insight generated:

| Field | Description | Example |
|---|---|---|
| `research_direction` | One-line summary of the proposed direction | "Integrating KG reasoning into NLP transformers for factual grounding" |
| `explanation` | Why this connection is interesting based on uploaded papers | "Paper 3 and Paper 5 both hint at..." |
| `research_questions` | 3 specific questions to explore | "How can KG triples be encoded as attention priors?" |
| `hypothesis` | A testable hypothesis | "KG-augmented transformers will outperform vanilla transformers on fact-intensive QA" |
| `limitations` | What to be careful about | "Based only on uploaded papers; no global novelty claim is made." |
| `supporting_papers` | Which uploaded papers support this | `["paper_003", "paper_005"]` |

---

### ✅ Step 7 — History Save

**File:** `services/history_service.py`

Every analysis is saved automatically to:
```
research_discovery/data/history/analysis_YYYYMMDD_HHMMSS.json
```

> ⚠️ `full_text` is **never saved** to disk — only metadata, concepts, scores, and insights.

---

## 📁 Module Reference Table

| File | Role | Key function |
|---|---|---|
| `app.py` | Streamlit UI — all pages | Main entry point |
| `services/pdf_processor.py` | PDF → paper dict | `process_pdfs()` |
| `services/concept_extractor.py` | Text → concept list | `extract_all_concepts()` |
| `services/embeddings.py` | Concepts → 384-dim vectors | `embed_concepts()` |
| `services/graph_builder.py` | Papers → NetworkX graph | `build_graph()` |
| `services/graph_analyzer.py` | Graph → scored candidates | `rank_candidates()` |
| `services/gnn_model.py` | Optional GCN encoder | `compute_gnn_scores()` |
| `services/llm_service.py` | Gemini CREF + GIC | `evaluate_candidate()`, `generate_insight()` |
| `services/analysis_pipeline.py` | Orchestrates all steps | `run_pipeline()` |
| `services/history_service.py` | Save/load JSON history | `save_analysis()` |
| `utils/text_utils.py` | Stopwords, synonym map, regex | `extract_noun_phrases()` |
| `utils/json_utils.py` | Safe JSON parsing | `extract_json_from_llm_response()` |

---

## 🔄 End-to-End Worked Example

**Scenario:** You upload 6 papers on AI in healthcare.

---

**Step 1 — PDF Extraction:**
```
paper_001: "GNN for Drug Discovery"        (2023)
paper_002: "Transformer Models in NLP"     (2022)
paper_003: "Knowledge Graphs in Medicine"  (2024)
paper_004: "Deep Learning for Diagnosis"   (2023)
paper_005: "NLP for Clinical Notes"        (2024)
paper_006: "Graph Attention Networks"      (2023)
```

---

**Step 2 — Concept Extraction:**
```
231 raw concepts extracted
→ 80 canonical concepts kept after filtering

Sample: [
  "Graph Neural Network", "Drug Discovery", "Knowledge Graph",
  "Natural Language Processing", "Clinical Notes", "Deep Learning",
  "Attention Mechanism", "Healthcare", "Diagnosis", "Drug Target"
]
```

---

**Step 3 — Embeddings:**
```
80 concepts × 384 dimensions = 80 vectors computed
Semantic similarity (Knowledge Graph ↔ Graph Neural Network) = 0.76
Semantic similarity (Clinical Notes ↔ Drug Discovery) = 0.38
```

---

**Step 4 — Knowledge Graph:**
```
Nodes: 80
Edges: 1,422 co-occurrence edges
```

---

**Step 5 — Top 3 Candidates after scoring:**

| Rank | Connection | Graph Score | Semantic Sim | Candidate Score |
|---|---|---|---|---|
| 🥇 1 | `Knowledge Graph + Clinical Notes` | 0.79 | 0.61 | **0.72** |
| 🥈 2 | `Graph Neural Network + Clinical Notes` | 0.74 | 0.58 | **0.67** |
| 🥉 3 | `Drug Discovery + Attention Mechanism` | 0.68 | 0.55 | **0.62** |

---

**Step 6 — CREF Scores from Gemini:**

| Connection | Novelty | Impact | Plausibility | Interdisciplinarity | Total |
|---|---|---|---|---|---|
| `Knowledge Graph + Clinical Notes` | 4 | 5 | 4 | 5 | **18** ⭐ |
| `GNN + Clinical Notes` | 3 | 4 | 4 | 4 | 15 |
| `Drug Discovery + Attention` | 4 | 4 | 3 | 4 | 15 |

---

**Step 6b — GIC Final Output (for top candidate):**

```
Research Direction:
  "Leveraging Knowledge Graphs to structure and query unstructured clinical notes
   for improved diagnostic inference"

Research Questions:
  1. Can KG-structured clinical notes improve rare disease diagnosis accuracy?
  2. How should medical ontologies (SNOMED, ICD) be integrated into the KG?
  3. What graph traversal strategies best surface relevant patient history?

Hypothesis:
  KG-augmented NLP models processing clinical notes will demonstrate
  higher factual consistency than vanilla LLM-based approaches on
  clinical QA benchmarks.

Supporting Papers: paper_003, paper_005

Limitations:
  Based on 6 uploaded papers. No claim of global scientific novelty is made.
  Results require validation by domain experts before any clinical application.
```

---

## ⚖️ What the System Does vs. What It Doesn't Do

| ✅ What it DOES | ❌ What it DOES NOT do |
|---|---|
| Identifies interesting concept pairs within your uploaded papers | Prove that a connection is globally novel |
| Scores candidates using interpretable graph metrics | Replace peer review or expert validation |
| Uses Gemini to generate plausible research directions | Guarantee the generated hypothesis is correct |
| Saves a history of every analysis run | Access the internet or external paper databases |
| Works without the Gemini API key (graph-only mode) | Work on scanned/image PDFs without text layers |

---

## 🔗 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35 | Web UI |
| `PyMuPDF` | ≥ 1.24 | PDF text extraction |
| `sentence-transformers` | ≥ 3.0 | `all-MiniLM-L6-v2` embeddings |
| `networkx` | ≥ 3.3 | Graph construction & centrality |
| `plotly` | ≥ 5.22 | Interactive graph visualization |
| `google-genai` | ≥ 1.0 | Gemini API client |
| `numpy`, `pandas` | ≥ 1.26, ≥ 2.2 | Numerical operations |
| `python-dotenv` | ≥ 1.0 | `.env` file loading |
| `torch` + `torch-geometric` | Optional | GNN encoder (adds ~1.5 GB) |
