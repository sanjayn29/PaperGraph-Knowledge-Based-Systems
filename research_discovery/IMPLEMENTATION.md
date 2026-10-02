# Implementation Corrections

This file lists only statements from the original implementation section that needed correction. All other statements remain supported by the repository.

| Original statement | Correct value |
|---|---|
| "PaperGraph is a local Python 3.11 Streamlit application." | The repository does not enforce a Python version. The current verified environment is Python 3.13.7. Describe Python 3.11 as a target environment only if that is the environment used for deployment or testing. |
| "The models/ directory is empty by design." | The `models/` directory contains `README.md`; it contains no trained GNN checkpoint. |
| "MEDIUM: at least 2 contextual papers." | **MEDIUM:** at least 2 supporting papers, including direct co-occurrence or single-concept contextual evidence, **or** cosine similarity >= 0.65. |

## Corrected Replacement Text

Use the following wording in the paper:

> PaperGraph is a local Python Streamlit application. The repository does not pin or enforce a specific Python version; the current verification environment uses Python 3.13.7. PyTorch and PyTorch Geometric are optional, and the application falls back to graph and semantic scoring when the optional neural components are unavailable. No GPU is required. Embedding weights are downloaded to the sentence-transformers cache. The `models/` directory contains documentation but no trained GNN checkpoint. `GEMINI_API_KEY` and the optional `GEMINI_MODEL` are loaded from `research_discovery/.env`, which is gitignored.

> For provenance cards, MEDIUM evidence means that at least two supporting papers are available, including direct co-occurrence or single-concept contextual evidence, or that cosine similarity is at least 0.65. This indicates corpus-supported evidence but does not establish scientific novelty.
