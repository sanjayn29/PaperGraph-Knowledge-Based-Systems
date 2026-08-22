# 🚀 PaperGraph — Setup Guide for New Contributors

> Everything you need to do **after cloning this repo** to get the app running on your machine.

---

## ✅ Prerequisites

Make sure you have the following installed **before** anything else.

### 1. Python 3.11 or higher

Check your version:

```bash
python --version
```

If you don't have Python 3.11+, download it from:  
👉 https://www.python.org/downloads/

> ⚠️ **Windows users**: During installation, make sure to check **"Add Python to PATH"**.

---

### 2. pip (Python package manager)

pip comes with Python. Verify it works:

```bash
pip --version
```

---

### 3. Git

Check if Git is installed:

```bash
git --version
```

If not, download from: https://git-scm.com/downloads

---

## 📥 Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/PaperGraph.git
cd PaperGraph
```

> Replace `<your-username>` with the actual GitHub username where the repo is hosted.

---

## 📂 Step 2 — Navigate to the App Folder

The Streamlit app lives inside the `research_discovery/` subfolder:

```bash
cd research_discovery
```

All remaining commands should be run from inside this folder.

---

## 🐍 Step 3 — (Recommended) Create a Virtual Environment

This keeps project dependencies isolated from your system Python.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when it's active.

---

## 📦 Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `PyMuPDF` | PDF text extraction |
| `sentence-transformers` | Semantic embeddings (`all-MiniLM-L6-v2`) |
| `networkx` | Knowledge graph construction |
| `plotly` | Interactive graph visualization |
| `google-genai` | Gemini LLM API client |
| `numpy`, `pandas` | Data processing |
| `python-dotenv` | Loading `.env` config |
| `pytest` | Running tests |

> ⏳ `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on **first run** — an internet connection is needed.

---

### Optional: GNN Support

If you want to enable the optional Graph Neural Network component (adds ~1.5 GB for PyTorch):

```bash
pip install torch torch-geometric
```

> The app works perfectly without this — it falls back to graph-only scoring automatically.

---

## 🔑 Step 5 — Set Up Your Gemini API Key

The app uses Google Gemini for AI-powered insight generation. This step is **optional** — the app works without it, but you won't get LLM-generated research insights.

### Get a free API key:
👉 https://aistudio.google.com/app/apikey  
(Sign in with a Google account → click "Create API Key")

### Configure it:

1. Copy the example environment file:

   **Windows:**
   ```bash
   copy .env.example .env
   ```

   **macOS / Linux:**
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in any text editor and fill in your key:

   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. Save the file. **Never share or commit `.env` to GitHub** — it's already in `.gitignore`.

---

## ▶️ Step 6 — Run the App

```bash
python -m streamlit run app.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser and the app is ready!

---

## 📄 Step 7 — Using the App

1. Go to the **"Upload Papers"** tab
2. Upload **5–10 PDF research papers** (must have selectable/copyable text — not scanned images)
3. Click **"Analyse"**
4. View the discovered concept connections, graph visualization, and AI-generated research direction

---

## 🧪 Running Tests (Optional)

To verify everything is working correctly:

```bash
pytest tests/ -v
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` for any package | Run `pip install -r requirements.txt` again inside the `research_discovery/` folder |
| `GEMINI_API_KEY not set` warning | Create your `.env` file as described in Step 5 (app still works without it) |
| App opens but graph is empty | Upload at least 2–3 PDFs with extractable text (not scanned) |
| `torch_geometric not installed` info message | This is just informational — install `torch` + `torch-geometric` if you want GNN mode |
| Port 8501 already in use | Run on a different port: `python -m streamlit run app.py --server.port 8502` |
| Slow first startup | The embedding model (~90 MB) is downloading — only happens once |

---

## 📁 Project Structure (Quick Reference)

```
PaperGraph/
└── research_discovery/        ← All app code lives here
    ├── app.py                 # Main Streamlit UI
    ├── requirements.txt       # Python dependencies
    ├── .env.example           # API key template (copy to .env)
    ├── services/              # Core pipeline modules
    │   ├── pdf_processor.py
    │   ├── concept_extractor.py
    │   ├── embeddings.py
    │   ├── graph_builder.py
    │   ├── graph_analyzer.py
    │   ├── gnn_model.py       # Optional GNN (requires PyTorch)
    │   ├── llm_service.py
    │   ├── history_service.py
    │   └── analysis_pipeline.py
    ├── utils/
    │   ├── text_utils.py
    │   └── json_utils.py
    ├── data/history/          # Saved analysis results (JSON)
    └── tests/                 # pytest test suite
```

---

## 🔗 Useful Links

- [Streamlit Docs](https://docs.streamlit.io)
- [Google AI Studio (get API key)](https://aistudio.google.com/app/apikey)
- [PyMuPDF Docs](https://pymupdf.readthedocs.io)
- [sentence-transformers Docs](https://www.sbert.net)
