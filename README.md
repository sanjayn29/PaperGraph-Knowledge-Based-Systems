# 🚀 PaperGraph — Setup & Contribution Guide

> A synergistic **Temporal Graph Neural Network (SE-TGN) + Large Language Model (LLM)** framework for scientific research discovery.

---

## ✅ Prerequisites

Make sure you have the following installed on your machine:

1. **Python 3.11+**: Check with `python --version`. *(On Windows, make sure "Add Python to PATH" is checked).*
2. **Git**: Check with `git --version`.

---

## 📥 Quick Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/PaperGraph.git
cd PaperGraph
```

### 2. Create and activate a virtual environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
cd research_discovery
pip install -r requirements.txt
```

> **For PyTorch and Graph Neural Network Acceleration (SE-TGN & GCN):**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install torch-geometric scikit-learn
> ```

### 4. Configure your Gemini API key (Optional for LLM features)
Create a `.env` file in `research_discovery/`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 5. Launch the application
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Running Tests

Run the complete test suite (93 unit tests across temporal graphs, SE-TGN, evaluators, and processors):

```bash
python -m pytest tests/ -v
```

---

## 📚 Documentation & Technical Details

* **Technical Reference & Pipeline Math:** Read [`research_discovery/PROJECT_EXPLAINER.md`](research_discovery/PROJECT_EXPLAINER.md)
* **Architecture & Model Details:** Read [`research_discovery/README.md`](research_discovery/README.md)
