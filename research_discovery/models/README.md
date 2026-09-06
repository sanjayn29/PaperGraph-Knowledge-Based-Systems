# Models Directory

This directory is reserved for future model weights or configuration files.

## Current Status

**No trained model weights are stored here.**

PaperGraph uses:

1. **`all-MiniLM-L6-v2`** — Downloaded automatically by `sentence-transformers` at runtime.  
   Cache location: `~/.cache/torch/sentence_transformers/` (managed by the library).

2. **Lightweight GCN Encoder** (`services/gnn_model.py`) — A small 2-layer GCN that runs a  
   single forward pass with **randomly initialized weights** over the concept co-occurrence graph.  
   There is no offline training, no saved checkpoint, and no `.pt` file here.

## Why No Trained GNN?

The base paper (SE-TGN) trains a Temporal Graph Network on tens of thousands of timestamped  
co-occurrence events from a large historical corpus, then validates link prediction performance  
on held-out years using AUC/AP/P@K/NDCG@K metrics.

This project operates on **5–10 uploaded PDFs** rather than the base paper's large historical corpus.
It builds a small temporal event stream and supports chronological held-out evaluation when enough
publication years are available. Training a separate GNN in this small setting would still be
statistically limited.
The GCN encoder is used only as a lightweight structural feature extractor, honestly labeled  
as such in the UI.

## If You Add a Pre-trained Model

If you extend this project with a pre-trained GNN checkpoint, place `.pt` files here  
and update `services/gnn_model.py` to load them. Add a `model_card.md` with:
- Model architecture
- Training corpus description  
- Validation metrics and methodology
- Honest scope and limitations
