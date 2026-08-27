"""
app.py — PaperGraph: AI Research Discovery System
──────────────────────────────────────────────────
Main Streamlit application entry point.

Pages:
  🏠 Home            — Hero, features, quick-start
  🔬 New Analysis    — Upload 5-10 PDFs, run pipeline, view results
  📋 History         — List & view past analyses
  ℹ️  About          — Disclaimer, academic honesty, technical details

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path (needed when running from any CWD)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Page configuration — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperGraph — AI Research Discovery",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Global CSS — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: rgba(22, 27, 34, 0.85);
    --border: rgba(48, 54, 61, 0.8);
    --border-bright: rgba(88, 166, 255, 0.3);
    --accent-blue: #58a6ff;
    --accent-purple: #a371f7;
    --accent-green: #3fb950;
    --accent-orange: #f0883e;
    --accent-red: #f85149;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --gradient-hero: linear-gradient(135deg, #1a1f2e 0%, #0d1117 40%, #141226 100%);
    --gradient-card: linear-gradient(135deg, rgba(88, 166, 255, 0.05), rgba(163, 113, 247, 0.05));
    --shadow: 0 8px 32px rgba(0,0,0,0.4);
    --shadow-hover: 0 12px 40px rgba(88, 166, 255, 0.15);
    --radius: 12px;
    --radius-lg: 16px;
}

/* ── Base ── */
.stApp { background: var(--bg-primary); color: var(--text-primary); font-family: 'Inter', sans-serif; }
.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Sidebar widget overrides (fixes Streamlit JS widgetBackgroundColor/widgetBorderColor/
   skeletonBackgroundColor empty-string warnings — Streamlit internal theming bug) ── */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stMultiSelect > div,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background-color: #1f2937 !important;
    border-color: rgba(48, 54, 61, 0.8) !important;
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] [data-testid="stSkeleton"],
section[data-testid="stSidebar"] .stSkeleton {
    background: linear-gradient(90deg, #161b22 25%, #1f2937 50%, #161b22 75%) !important;
    background-size: 200% 100% !important;
}

/* ── Main content area ── */
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1200px; }

/* ── Typography ── */
h1, h2, h3, h4 { color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; }
p, li, span { color: var(--text-secondary); }
.stMarkdown p { color: var(--text-secondary); line-height: 1.7; }

/* ── Hero section ── */
.hero-container {
    background: var(--gradient-hero);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-lg);
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(88,166,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(135deg, #58a6ff, #a371f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1rem;
}
.hero-subtitle { font-size: 1.15rem; color: var(--text-secondary); max-width: 650px; margin: 0 auto 2rem; }
.hero-badge {
    display: inline-block; padding: 0.35rem 1rem;
    background: rgba(88,166,255,0.1); border: 1px solid rgba(88,166,255,0.3);
    border-radius: 999px; font-size: 0.8rem; color: var(--accent-blue);
    font-weight: 600; letter-spacing: 0.05em; margin: 0.25rem;
}

/* ── Feature cards ── */
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 2rem 0; }
.feature-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.5rem 1.25rem;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(10px);
}
.feature-card:hover {
    border-color: var(--border-bright); transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}
.feature-icon { font-size: 2rem; margin-bottom: 0.75rem; }
.feature-title { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.4rem; }
.feature-desc { font-size: 0.82rem; color: var(--text-muted); line-height: 1.5; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
.metric-card {
    flex: 1; min-width: 130px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.25rem; text-align: center;
    background: var(--gradient-card); backdrop-filter: blur(8px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-1px); box-shadow: var(--shadow-hover); }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--accent-blue); }
.metric-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Score bars ── */
.score-bar-container { margin: 0.5rem 0; }
.score-label { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 0.25rem; }
.score-label-name { color: var(--text-secondary); }
.score-label-value { color: var(--accent-blue); font-weight: 600; }
.score-track { height: 6px; background: var(--bg-secondary); border-radius: 999px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }
.score-fill-blue { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
.score-fill-purple { background: linear-gradient(90deg, #6e40c9, #a371f7); }
.score-fill-green { background: linear-gradient(90deg, #196c2e, #3fb950); }
.score-fill-orange { background: linear-gradient(90deg, #9e4800, #f0883e); }

/* ── Candidate cards ── */
.candidate-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.25rem 1.5rem;
    margin: 0.75rem 0; cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.candidate-card.selected { border-color: var(--accent-blue); box-shadow: var(--shadow-hover); }
.candidate-connection { font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; }
.candidate-score { font-size: 0.82rem; color: var(--text-muted); }

/* ── Result sections ── */
.result-section {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 1.75rem;
    margin: 1rem 0; backdrop-filter: blur(8px);
}
.result-section-title {
    font-size: 1rem; font-weight: 600; color: var(--accent-blue);
    margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
}
.insight-box {
    background: rgba(88,166,255,0.05); border-left: 3px solid var(--accent-blue);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.25rem; margin: 0.75rem 0;
}
.hypothesis-box {
    background: rgba(163,113,247,0.06); border-left: 3px solid var(--accent-purple);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1rem 1.25rem; margin: 0.75rem 0;
}
.warning-box {
    background: rgba(240,136,62,0.08); border-left: 3px solid var(--accent-orange);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.85rem 1.25rem; margin: 0.5rem 0; font-size: 0.85rem;
}
.disclaimer-box {
    background: rgba(248, 81, 73, 0.06); border: 1px solid rgba(248,81,73,0.2);
    border-radius: var(--radius); padding: 1rem 1.25rem; margin: 1rem 0;
    font-size: 0.85rem; color: var(--text-secondary);
}

/* ── Progress step indicator ── */
.step-indicator {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.88rem; color: var(--text-secondary);
    padding: 0.5rem 1rem; background: var(--bg-secondary);
    border-radius: 999px; border: 1px solid var(--border);
    width: fit-content; margin-bottom: 1rem;
}
.step-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-blue); animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── History cards ── */
.history-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.25rem 1.5rem;
    margin: 0.75rem 0; transition: border-color 0.2s, box-shadow 0.2s;
}
.history-card:hover { border-color: var(--border-bright); box-shadow: var(--shadow); }
.history-title { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); }
.history-meta { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.25rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #58a6ff) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── File uploader ── */
.stFileUploader { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

/* ── Tabs ── */
.stTabs [role="tab"] { color: var(--text-secondary) !important; font-family: 'Inter', sans-serif !important; }
.stTabs [role="tab"][aria-selected="true"] { color: var(--accent-blue) !important; font-weight: 600 !important; }

/* ── Expander ── */
.streamlit-expanderHeader { color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; }

/* ── Code blocks ── */
.stCodeBlock { border-radius: var(--radius) !important; }

/* ── Status badges ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.25rem 0.75rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
}
.status-active { background: rgba(63,185,80,0.12); color: var(--accent-green); border: 1px solid rgba(63,185,80,0.3); }
.status-inactive { background: rgba(110,118,129,0.15); color: var(--text-muted); border: 1px solid var(--border); }
.status-warning { background: rgba(240,136,62,0.1); color: var(--accent-orange); border: 1px solid rgba(240,136,62,0.3); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ── Page divider ── */
hr { border-color: var(--border) !important; opacity: 0.5; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='padding: 1rem 0 0.5rem; text-align:center;'>
                <div style='font-size:2.2rem;'>🔬</div>
                <div style='font-weight:800; font-size:1.1rem; color:#e6edf3; margin-top:0.25rem;'>PaperGraph</div>
                <div style='font-size:0.72rem; color:#6e7681; margin-top:0.1rem;'>AI Research Discovery</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        pages = {
            "🏠 Home": "home",
            "🔬 New Analysis": "analysis",
            "📋 History": "history",
            "ℹ️ About": "about",
        }

        if "page" not in st.session_state:
            st.session_state.page = "home"

        for label, page_key in pages.items():
            is_active = st.session_state.page == page_key
            if st.sidebar.button(
                label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = page_key
                st.rerun()

        st.markdown("---")

        # Status badges
        _render_status_sidebar()

        st.markdown(
            """
            <div style='margin-top:auto; padding: 1rem 0 0; font-size:0.72rem; color:#6e7681; text-align:center;'>
            B.Tech Project · Simplified GNN-LLM Framework<br>
            <span style='color:#444c56;'>Results are exploratory, not validated.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_status_sidebar():
    from services.gnn_model import GNN_AVAILABLE, SETGN_AVAILABLE
    from services.embeddings import is_available as emb_available
    import os

    llm_key = bool(os.getenv("GEMINI_API_KEY", ""))

    setgn_label = "SE-TGN Active" if SETGN_AVAILABLE else "SE-TGN Inactive"
    setgn_cls = "status-active" if SETGN_AVAILABLE else "status-inactive"
    gnn_label = "GCN Active" if GNN_AVAILABLE else "GCN Inactive"
    gnn_cls = "status-active" if GNN_AVAILABLE else "status-inactive"
    emb_label = "Embeddings Active" if emb_available() else "Embeddings Inactive"
    emb_cls = "status-active" if emb_available() else "status-inactive"
    llm_label = "LLM Active" if llm_key else "LLM Inactive"
    llm_cls = "status-active" if llm_key else "status-warning"

    st.markdown(
        f"""
        <div style='display:flex; flex-direction:column; gap:0.4rem; padding: 0.25rem 0;'>
            <span class='status-badge {setgn_cls}'>{"🟢" if SETGN_AVAILABLE else "⚪"} {setgn_label}</span>
            <span class='status-badge {gnn_cls}'>{"🟢" if GNN_AVAILABLE else "⚪"} {gnn_label}</span>
            <span class='status-badge {emb_cls}'>{"🟢" if emb_available() else "⚪"} {emb_label}</span>
            <span class='status-badge {llm_cls}'>{"🟢" if llm_key else "🟡"} {llm_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )




# ─────────────────────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────────────────────

def page_home():
    st.markdown(
        """
        <div class='hero-container'>
            <div class='hero-title'>PaperGraph</div>
            <div class='hero-subtitle'>
                Upload 5–10 research papers and discover potentially underexplored connections
                between concepts — powered by SE-TGN temporal graph analysis, semantic embeddings, and LLM reasoning.
            </div>
            <div>
                <span class='hero-badge'>⏱️ SE-TGN Temporal GNN</span>
                <span class='hero-badge'>🕸️ NetworkX Graph Analysis</span>
                <span class='hero-badge'>🧠 Sentence Transformers</span>
                <span class='hero-badge'>📋 CREF Evaluation</span>
                <span class='hero-badge'>💡 GIC Insight Generation</span>
                <span class='hero-badge'>📈 AUC / AP / NDCG@K Metrics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='feature-grid'>
            <div class='feature-card'>
                <div class='feature-icon'>📄</div>
                <div class='feature-title'>PDF Ingestion</div>
                <div class='feature-desc'>Upload 5–10 research PDFs. Automatic extraction of title, abstract, year, authors, and full text. Manual year correction supported.</div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>⏱️</div>
                <div class='feature-title'>SE-TGN Temporal GNN</div>
                <div class='feature-desc'>Concept co-occurrence events are ordered by publication year and fed into a Semantic-Enhanced Temporal Graph Network for future link prediction.</div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🕸️</div>
                <div class='feature-title'>Knowledge Graph</div>
                <div class='feature-desc'>Concepts become nodes; co-occurrence in papers becomes edges. Built with NetworkX. Graph metrics provide supporting structural signals.</div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🧠</div>
                <div class='feature-title'>Semantic Embeddings</div>
                <div class='feature-desc'>all-MiniLM-L6-v2 encodes concepts for semantic similarity scoring. Embeddings also serve as node features in SE-TGN messages.</div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>🤖</div>
                <div class='feature-title'>LLM Evaluation</div>
                <div class='feature-desc'>Gemini evaluates top SE-TGN candidates on novelty, impact, plausibility, and interdisciplinarity (CREF-style).</div>
            </div>
            <div class='feature-card'>
                <div class='feature-icon'>📈</div>
                <div class='feature-title'>Evaluation Metrics</div>
                <div class='feature-desc'>AUC, Average Precision, P@K, and NDCG@K computed across baselines (Random, Graph, GCN, SE-TGN) when sufficient temporal data exists.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔬 Start New Analysis", key="home_new_analysis", use_container_width=True, type="primary"):
            st.session_state.page = "analysis"
            st.rerun()
    with col2:
        if st.button("📋 View History", key="home_history", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div class='disclaimer-box'>
            ⚠️ <strong>Academic Disclaimer:</strong> This system identifies potentially interesting
            relationships within the uploaded research papers. It does not prove that a relationship
            is scientifically novel or that the generated research direction is experimentally validated.
            Results are exploratory and should be treated as candidate directions for human expert review.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Page: New Analysis
# ─────────────────────────────────────────────────────────────

def page_analysis():
    st.markdown("## 🔬 New Analysis")
    st.markdown(
        "<p style='color:#8b949e;'>Upload 5–10 research PDFs to discover potentially underexplored concept connections.</p>",
        unsafe_allow_html=True,
    )

    # ── File upload ───────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload Research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select between 5 and 10 PDF files. Scanned PDFs without extractable text will be skipped.",
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        n = len(uploaded_files)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='color:{'#3fb950' if 5<=n<=10 else '#f85149'};'>{n}</div><div class='metric-label'>PDFs Selected</div></div>",
                unsafe_allow_html=True,
            )
        with col2:
            total_size_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
            st.markdown(
                f"<div class='metric-card'><div class='metric-value'>{total_size_mb:.1f}</div><div class='metric-label'>MB Total</div></div>",
                unsafe_allow_html=True,
            )
        with col3:
            status = "✅ Ready" if 5 <= n <= 10 else ("⬆️ Too many" if n > 10 else "⬇️ Need more")
            st.markdown(
                f"<div class='metric-card'><div class='metric-value' style='font-size:1.3rem;'>{status}</div><div class='metric-label'>Upload Status</div></div>",
                unsafe_allow_html=True,
            )

        # Show file list
        with st.expander("📁 Uploaded Files", expanded=False):
            for f in uploaded_files:
                size_kb = f.size / 1024
                st.markdown(f"- **{f.name}** ({size_kb:.1f} KB)")

        # ── Paper Metadata / Year Override ───────────────────────
        st.markdown("---")
        with st.expander("📅 Paper Metadata & Year Correction", expanded=False):
            st.markdown(
                """
                <div style='font-size:0.85rem; color:#8b949e; margin-bottom:0.75rem;'>
                    SE-TGN uses publication year for temporal ordering. Papers with unreliable
                    year extraction are shown below. Correct any wrong years before running.
                </div>
                """,
                unsafe_allow_html=True,
            )
            from services.pdf_processor import peek_pdf_year

            year_overrides: dict[str, int] = {}
            for idx, f in enumerate(uploaded_files):
                # Read bytes for previewing year without moving stream pointer
                f_bytes = f.getvalue() if hasattr(f, "getvalue") else f.read()
                detected_year, source = peek_pdf_year(f_bytes, f.name)
                default_year = detected_year if detected_year else 2024

                col_name, col_year = st.columns([3, 1])
                with col_name:
                    if source == "metadata":
                        badge = f"<span style='color:#3fb950; font-size:0.75rem;'>[metadata: {detected_year}]</span>"
                    elif source == "text_regex":
                        badge = f"<span style='color:#58a6ff; font-size:0.75rem;'>[text scan: {detected_year}]</span>"
                    else:
                        badge = "<span style='color:#d29922; font-size:0.75rem;'>[estimated: please verify]</span>"

                    st.markdown(
                        f"<div style='padding-top:0.4rem; font-size:0.88rem;'>📄 {f.name} {badge}</div>",
                        unsafe_allow_html=True,
                    )
                with col_year:
                    yr = st.number_input(
                        f"Year for {f.name} ({idx})",
                        min_value=1950,
                        max_value=2030,
                        value=int(default_year),
                        step=1,
                        key=f"year_override_{idx}_{f.name}",
                        label_visibility="collapsed",
                    )
                    year_overrides[f.name] = int(yr)

        if "year_overrides" not in st.session_state:
            st.session_state.year_overrides = {}

        # Validation
        if n < 5:
            st.error(f"⚠️ Please upload at least 5 PDFs (you have {n}). A minimum of 5 papers is required for meaningful co-occurrence analysis.")
            return
        if n > 10:
            st.error(f"⚠️ Please upload at most 10 PDFs (you have {n}). Reduce your selection to 10 or fewer files.")
            return

        st.markdown("---")
        run_btn = st.button("🚀 Run Analysis", key="run_analysis_btn", type="primary", use_container_width=True)

        if run_btn:
            _run_analysis(uploaded_files, year_overrides=year_overrides if 'year_overrides' in dir() else {})
    else:
        # Upload prompt
        st.markdown(
            """
            <div style='border: 2px dashed rgba(88,166,255,0.25); border-radius: 16px;
                        padding: 3rem 2rem; text-align: center; margin: 1.5rem 0;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>📂</div>
                <div style='font-size:1.05rem; font-weight:600; color:#e6edf3; margin-bottom:0.5rem;'>
                    Drop your research PDFs here
                </div>
                <div style='color:#8b949e; font-size:0.88rem;'>
                    Upload between 5 and 10 PDF files to begin analysis<br>
                    Works with standard research papers — no scanned/image PDFs
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show previous result if exists in session
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
        st.markdown("---")
        st.markdown("### 📊 Previous Analysis Result")
        _render_result(st.session_state.analysis_result)


def _run_analysis(uploaded_files, year_overrides: dict = None):
    """Execute the pipeline with live progress updates."""
    from services.analysis_pipeline import run_pipeline

    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    def progress_callback(step: int, total: int, message: str):
        pct = int((step / total) * 100)
        progress_bar.progress(pct)
        progress_placeholder.markdown(
            f"<div class='step-indicator'><div class='step-dot'></div>Step {step}/{total} — {message}</div>",
            unsafe_allow_html=True,
        )

    try:
        result = run_pipeline(
            uploaded_files=uploaded_files,
            progress_callback=progress_callback,
            top_candidates=3,
            save_history=True,
            year_overrides=year_overrides or {},
        )
    except Exception as exc:
        progress_bar.empty()
        progress_placeholder.empty()
        st.error(f"❌ An unexpected error occurred: {exc}")
        return

    progress_bar.progress(100)
    progress_placeholder.empty()

    if not result.get("success"):
        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
        return

    # Show warnings
    for warning in result.get("warnings", []):
        st.warning(warning)

    st.success("✅ Analysis complete!")
    st.session_state.analysis_result = result
    st.rerun()


def _render_result(result: dict):
    """Render the full analysis result."""
    import plotly.graph_objects as go
    import networkx as nx

    # ── Pipeline visualization ────────────────────────────────
    tsum = result.get("temporal_summary", {})
    setgn_active = any(c.get("setgn_active") for c in result.get("candidates", []))
    eval_result = result.get("evaluation", {})

    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;
                    padding:0.75rem 1rem; background:rgba(88,166,255,0.04);
                    border:1px solid rgba(88,166,255,0.15); border-radius:12px; margin-bottom:1rem;'>
            <span style='font-size:0.8rem; color:#8b949e;'>Pipeline:</span>
            <span style='font-size:0.82rem; color:#e6edf3;'>📄 PDFs</span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:#e6edf3;'>🔍 Concepts</span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:#e6edf3;'>⏱️ Temporal Events ({tsum.get('total_events', 0)})</span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:{'#3fb950' if setgn_active else '#6e7681'};'>
                🧬 SE-TGN {'✅' if setgn_active else '⚫'}
            </span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:#e6edf3;'>🔗 Candidates</span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:{'#3fb950' if result.get('llm_available') else '#6e7681'};'>
                🤖 CREF {'✅' if result.get('llm_available') else '⚫'}
            </span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:{'#3fb950' if result.get('llm_available') else '#6e7681'};'>
                💡 GIC {'✅' if result.get('llm_available') else '⚫'}
            </span>
            <span style='color:#444c56;'>→</span>
            <span style='font-size:0.82rem; color:{'#3fb950' if eval_result.get('sufficient_data') else '#6e7681'};'>
                📈 Eval {'✅' if eval_result.get('sufficient_data') else '⚫'}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary metrics ───────────────────────────────────────
    # ── Summary metrics ───────────────────────────────────────
    st.markdown(
        f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-value'>{result.get('paper_count', 0)}</div>
                <div class='metric-label'>Papers Analyzed</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{result.get('concept_count', 0)}</div>
                <div class='metric-label'>Concepts Found</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{result.get('relationship_count', 0)}</div>
                <div class='metric-label'>Relationships</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{tsum.get('total_events', 0)}</div>
                <div class='metric-label'>Temporal Events</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{len(result.get('candidates', []))}</div>
                <div class='metric-label'>Candidates</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Analysis mode badge ───────────────────────────────────
    mode = result.get("analysis_mode", "Lightweight Graph Analysis")
    llm_active = result.get("llm_available", False)
    st.markdown(
        f"""
        <div style='display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1rem;'>
            <span class='status-badge status-active'>🔬 {mode}</span>
            <span class='status-badge {"status-active" if llm_active else "status-warning"}'>
                {"🤖 LLM Evaluation Active" if llm_active else "🤖 LLM Evaluation Inactive"}
            </span>
            {"<span class='status-badge status-active'>💾 Saved to History</span>" if result.get('analysis_id') else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "💡 Top Insight",
        "🎯 Predictions & Feedback",
        "🔎 Research Gaps",
        "🌉 Cross-Domain",
        "📚 Evidence & Provenance",
        "💬 Research Assistant",
        "🕸️ Graph View",
        "⏱️ Temporal Graph",
        "📈 Model Evaluation",
        "📄 Papers",
    ])

    final = result.get("final_result", {})
    candidates = result.get("candidates", [])
    papers = result.get("papers", [])
    research_gaps = result.get("research_gaps", [])
    cross_domain_discoveries = result.get("cross_domain_discoveries", [])
    concept_domains = result.get("concept_domains", {})
    provenance_map = result.get("provenance", {})
    personalized_candidates = result.get("personalized_candidates", candidates)

    # ── Tab 1: Top Insight ────────────────────────────────────
    with tab1:
        if not final:
            st.info("No final result generated.")
        else:
            _render_insight(final, llm_active)

    # ── Tab 2: Predictions & Feedback ─────────────────────────
    with tab2:
        _render_candidates_and_personalization(
            candidates=candidates,
            personalized_candidates=personalized_candidates,
            mode=result.get("analysis_mode", ""),
            concept_domains=concept_domains,
        )

    # ── Tab 3: Research Gaps ──────────────────────────────────
    with tab3:
        _render_research_gaps(research_gaps)

    # ── Tab 4: Cross-Domain Discoveries ───────────────────────
    with tab4:
        _render_cross_domain(cross_domain_discoveries, concept_domains)

    # ── Tab 5: Evidence & Provenance ──────────────────────────
    with tab5:
        _render_provenance(provenance_map, candidates)

    # ── Tab 6: Research Assistant ─────────────────────────────
    with tab6:
        _render_research_assistant(result)

    # ── Tab 7: Graph View ─────────────────────────────────────
    with tab7:
        _render_graph_viz(result, candidates)

    # ── Tab 8: Temporal Graph ─────────────────────────────────
    with tab8:
        _render_temporal_graph(result)

    # ── Tab 9: Model Evaluation ───────────────────────────────
    with tab9:
        _render_evaluation(result)

    # ── Tab 10: Papers ────────────────────────────────────────
    with tab10:
        _render_papers(papers)

    # ── Technical Details expander ───────────────────────────
    with st.expander("🔧 Technical Details"):
        _render_technical_details(result)


def _render_insight(final: dict, llm_active: bool):
    """Render the top insight card."""
    connection = final.get("connection", "Unknown connection")

    st.markdown(
        f"""
        <div class='result-section'>
            <div class='result-section-title'>🔗 Top Candidate Connection</div>
            <div style='font-size:1.4rem; font-weight:700; color:#e6edf3; margin-bottom:0.5rem;'>
                {connection}
            </div>
            <div style='font-size:0.8rem; color:#6e7681;'>
                Candidate research direction identified through SE-TGN temporal link prediction
                {"and CREF/GIC LLM evaluation" if llm_active else "(LLM evaluation not available)"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # LLM Scores (if available)
    if llm_active and final.get("novelty") is not None:
        st.markdown("#### CREF Evaluation Scores")
        scores = [
            ("Novelty", final.get("novelty", 0), "score-fill-blue"),
            ("Impact", final.get("impact", 0), "score-fill-purple"),
            ("Plausibility", final.get("plausibility", 0), "score-fill-green"),
            ("Interdisciplinarity", final.get("interdisciplinarity", 0), "score-fill-orange"),
        ]
        for label, score, css_class in scores:
            pct = (score / 5) * 100
            st.markdown(
                f"""
                <div class='score-bar-container'>
                    <div class='score-label'>
                        <span class='score-label-name'>{label}</span>
                        <span class='score-label-value'>{score}/5</span>
                    </div>
                    <div class='score-track'>
                        <div class='score-fill {css_class}' style='width:{pct}%;'></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if final.get("reasoning"):
            st.markdown(
                f"<div class='insight-box'><strong>Reasoning:</strong> {final['reasoning']}</div>",
                unsafe_allow_html=True,
            )

    # Research direction
    if final.get("research_direction"):
        st.markdown("#### 🎯 Research Direction")
        st.markdown(
            f"<div class='insight-box'>{final['research_direction']}</div>",
            unsafe_allow_html=True,
        )

    # Explanation
    if final.get("explanation"):
        st.markdown("#### 📖 Explanation")
        st.markdown(
            f"<div class='result-section' style='padding:1rem;'>{final['explanation']}</div>",
            unsafe_allow_html=True,
        )

    # Research questions
    if final.get("research_questions"):
        st.markdown("#### ❓ Research Questions")
        for i, rq in enumerate(final["research_questions"], 1):
            st.markdown(f"**Q{i}.** {rq}")

    # Hypothesis
    if final.get("hypothesis"):
        st.markdown("#### 🧪 Hypothesis")
        st.markdown(
            f"<div class='hypothesis-box'>{final['hypothesis']}</div>",
            unsafe_allow_html=True,
        )

    # Limitations
    if final.get("limitations"):
        st.markdown("#### ⚠️ Limitations")
        st.markdown(
            f"<div class='warning-box'>{final['limitations']}</div>",
            unsafe_allow_html=True,
        )

    # Academic disclaimer
    st.markdown(
        """
        <div class='disclaimer-box'>
            ⚠️ <strong>Academic Disclaimer:</strong> This system identifies potentially interesting
            relationships within the uploaded research papers. It does not prove that a relationship
            is scientifically novel or that the generated research direction is experimentally validated.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_candidates_and_personalization(
    candidates: list,
    personalized_candidates: list,
    mode: str,
    concept_domains: dict,
):
    """Render candidate connections with interactive Human-in-the-Loop feedback and personalized reranking."""
    from services.user_feedback import record_feedback, get_user_profile_summary

    st.markdown("### 🎯 Candidate Concept Connections & Personalization")
    st.markdown(
        f"<div style='font-size:0.82rem; color:#8b949e; margin-bottom:1rem;'>Analysis Mode: <code>{mode}</code></div>",
        unsafe_allow_html=True,
    )

    ranking_view = st.radio(
        "Select Ranking View:",
        ["Original Ranking (Base SE-TGN Model)", "Personalized Ranking (Human-in-the-Loop)"],
        horizontal=True,
        key="ranking_view_toggle",
    )

    active_list = personalized_candidates if "Personalized" in ranking_view else candidates

    if not active_list:
        st.info("No candidate connections generated.")
        return

    # Profile summary metrics
    stats = get_user_profile_summary()
    if stats["total_feedbacks"] > 0 and "Personalized" in ranking_view:
        st.info(
            f"🧠 **Personalized Layer Active:** {stats['total_feedbacks']} ratings recorded. "
            f"Boosted concepts: {', '.join(list(stats['preferred_concepts'].keys())[:3]) or 'None'}"
        )

    for i, cand in enumerate(active_list[:12]):
        cand_id = cand.get("candidate_id", f"cand_{i+1:03d}")
        ca = cand.get("concept_a", "")
        cb = cand.get("concept_b", "")
        score = cand.get("personalized_score", cand.get("candidate_score", 0))
        orig_score = cand.get("candidate_score", 0)
        setgn_score = cand.get("setgn_score")
        sem_score = cand.get("semantic_similarity", 0)
        graph_score = cand.get("graph_score", 0)
        orig_rank = cand.get("original_rank", i + 1)
        pers_rank = cand.get("personalized_rank", i + 1)
        rank_delta = cand.get("rank_delta", 0)

        dom_a = concept_domains.get(ca, {}).get("domain", "")
        dom_b = concept_domains.get(cb, {}).get("domain", "")

        delta_badge = ""
        if "Personalized" in ranking_view and rank_delta != 0:
            if rank_delta > 0:
                delta_badge = f"<span style='color:#3fb950; font-weight:700;'>▲ +{rank_delta} (Orig #{orig_rank})</span>"
            else:
                delta_badge = f"<span style='color:#f85149; font-weight:700;'>▼ {rank_delta} (Orig #{orig_rank})</span>"

        st.markdown(
            f"""
            <div class='candidate-card {"selected" if i==0 else ""}'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div class='candidate-connection'>
                        {"🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"}&nbsp;
                        {cand.get('connection', f'{ca} + {cb}')}
                        &nbsp;&nbsp;{delta_badge}
                    </div>
                    <div style='font-size:0.8rem; color:#8b949e;'>
                        <span style='background:#21262d; padding:2px 8px; border-radius:12px; border:1px solid #30363d;'>{dom_a} ↔ {dom_b}</span>
                    </div>
                </div>
                <div class='candidate-score' style='margin-top:0.35rem;'>
                    Score: <strong style='color:#58a6ff;'>{score:.3f}</strong>
                    {" (Orig: " + f"{orig_score:.3f})" if 'Personalized' in ranking_view else ""}
                    &nbsp;·&nbsp; SE-TGN: {f"{setgn_score:.3f}" if setgn_score is not None else "N/A"}
                    &nbsp;·&nbsp; Semantic: {sem_score:.3f}
                    &nbsp;·&nbsp; Graph: {graph_score:.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Human feedback rating buttons
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])
        with c1:
            if st.button("👍 Useful", key=f"fb_int_{i}_{cand_id}"):
                record_feedback(cand_id, ca, cb, "interesting", dom_a, dom_b)
                st.success("Feedback recorded! Rerank will reflect this.")
                st.rerun()
        with c2:
            if st.button("⭐ High Value", key=f"fb_star_{i}_{cand_id}"):
                record_feedback(cand_id, ca, cb, "highly_relevant", dom_a, dom_b)
                st.success("Marked as High Value!")
                st.rerun()
        with c3:
            if st.button("👎 Irrelevant", key=f"fb_not_{i}_{cand_id}"):
                record_feedback(cand_id, ca, cb, "not_relevant", dom_a, dom_b)
                st.info("Downvoted candidate.")
                st.rerun()
        with c4:
            if st.button("🔖 Save", key=f"fb_save_{i}_{cand_id}"):
                record_feedback(cand_id, ca, cb, "saved", dom_a, dom_b)
                st.success("Saved to favorites.")
                st.rerun()
        with c5:
            if cand.get("personalization_reason"):
                st.caption(f"💡 {cand['personalization_reason']}")


def _render_research_gaps(research_gaps: list):
    """Render detected research gaps with metric breakdowns and rationales."""
    st.markdown("### 🔎 Detected Scientific Research Gaps")
    st.markdown(
        """
        <div style='font-size:0.85rem; color:#8b949e; margin-bottom:1.2rem;'>
            Identifies concept pairs with strong semantic compatibility and high individual importance,
            yet minimal direct co-occurrence in the current corpus.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not research_gaps:
        st.info("No prominent research gaps detected above the significance threshold.")
        return

    # Metric summaries
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Total Gaps Detected", len(research_gaps))
    with g2:
        avg_score = sum(g.get("gap_score", 0) for g in research_gaps) / len(research_gaps)
        st.metric("Avg Gap Score", f"{avg_score:.2f}")
    with g3:
        high_prom = sum(1 for g in research_gaps if g.get("status") == "highly_promising")
        st.metric("High-Potential Gaps", high_prom)

    st.markdown("---")

    for i, gap in enumerate(research_gaps[:10]):
        gap_score = gap.get("gap_score", 0.0)
        score_color = "#3fb950" if gap_score >= 0.75 else "#58a6ff"
        status_badge = (
            "<span style='background:rgba(63,185,80,0.15); color:#3fb950; padding:2px 8px; border-radius:12px; font-size:0.75rem; border:1px solid rgba(63,185,80,0.3);'>⭐ Highly Promising</span>"
            if gap.get("status") == "highly_promising"
            else "<span style='background:rgba(88,166,255,0.15); color:#58a6ff; padding:2px 8px; border-radius:12px; font-size:0.75rem; border:1px solid rgba(88,166,255,0.3);'>Underexplored</span>"
        )

        with st.expander(
            f"#{i+1} {gap.get('concept_a')} ↔ {gap.get('concept_b')} (Gap Score: {gap_score:.2f})",
            expanded=(i == 0),
        ):
            st.markdown(
                f"""
                <div style='margin-bottom:0.75rem;'>
                    {status_badge}
                    &nbsp;&nbsp;<strong>Domains:</strong> {gap.get('domain_a')} ↔ {gap.get('domain_b')}
                    &nbsp;&nbsp;|&nbsp;&nbsp;<strong>Existing Connections:</strong> {gap.get('existing_connections', 0)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(f"**Rationale:** {gap.get('explanation', '')}")

            # Component score breakdown
            st.markdown("##### Gap Metric Composition")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Semantic (30%)", f"{gap.get('semantic_score', 0):.2f}")
            c2.metric("Temporal (25%)", f"{gap.get('temporal_score', 0):.2f}")
            c3.metric("Structural (20%)", f"{gap.get('structural_score', 0):.2f}")
            c4.metric("Cross-Domain (15%)", f"{gap.get('cross_domain_score', 0):.2f}")
            c5.metric("Novelty (10%)", f"{gap.get('novelty_score', 0):.2f}")

            if gap.get("supporting_papers"):
                st.markdown("##### Contextual Papers in Corpus")
                for p in gap["supporting_papers"]:
                    st.markdown(f"- **{p.get('title', 'Untitled')}** ({p.get('year', 'N/A')}) — *mentions {', '.join(p.get('contains', []))}*")


def _render_cross_domain(cross_domain_discoveries: list, concept_domains: dict):
    """Render interdisciplinary cross-domain discoveries."""
    st.markdown("### 🌉 Cross-Domain Discoveries")
    st.markdown(
        """
        <div style='font-size:0.85rem; color:#8b949e; margin-bottom:1.2rem;'>
            Highlights emergent synergies bridging different scientific disciplines.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not cross_domain_discoveries:
        st.info("No strong cross-domain discoveries identified in this paper collection.")
        return

    # Filter by domain pair
    available_pairs = sorted(list(set(
        f"{cd['domain_a']} ↔ {cd['domain_b']}" for cd in cross_domain_discoveries
    )))
    selected_pair = st.selectbox(
        "Filter by Discipline Bridge:",
        ["All Domains"] + available_pairs,
        key="cross_domain_filter_select",
    )

    filtered = cross_domain_discoveries
    if selected_pair != "All Domains":
        filtered = [
            cd for cd in cross_domain_discoveries
            if f"{cd['domain_a']} ↔ {cd['domain_b']}" == selected_pair
        ]

    for cd in filtered:
        cd_score = cd.get("cross_domain_score", 0.0)
        st.markdown(
            f"""
            <div style='background:rgba(22,27,34,0.7); border:1px solid rgba(163,113,247,0.3); border-radius:12px; padding:1.2rem; margin-bottom:1rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='font-size:1.1rem; font-weight:700; color:#e6edf3;'>
                        {cd.get('concept_a')} <span style='color:#a371f7;'>↔</span> {cd.get('concept_b')}
                    </div>
                    <div style='font-size:1rem; font-weight:700; color:#a371f7;'>
                        Score: {cd_score:.3f}
                    </div>
                </div>
                <div style='margin-top:0.4rem; font-size:0.85rem; color:#8b949e;'>
                    <span style='color:#58a6ff;'>{cd.get('domain_a')}</span> &nbsp;⇄&nbsp; <span style='color:#3fb950;'>{cd.get('domain_b')}</span>
                    &nbsp;·&nbsp; Semantic Compatibility: {cd.get('semantic_compatibility', 0):.2f}
                    &nbsp;·&nbsp; Interdisciplinary Distance: {cd.get('domain_distance', 0):.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_provenance(provenance_map: dict, candidates: list):
    """Render transparent evidence and provenance trail."""
    st.markdown("### 📚 Recommendation Evidence & Provenance")
    st.markdown(
        """
        <div style='font-size:0.85rem; color:#8b949e; margin-bottom:1.2rem;'>
            Trace every recommendation directly back to supporting papers, graph topology, and model metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not provenance_map or not candidates:
        st.info("Provenance data is not available.")
        return

    candidate_options = [c.get("connection", f"{c.get('concept_a')} + {c.get('concept_b')}") for c in candidates[:10]]
    chosen_conn = st.selectbox("Select Candidate to Inspect Evidence:", candidate_options, key="provenance_select_box")

    prov = provenance_map.get(chosen_conn)
    if not prov:
        st.warning("No detailed provenance found for this candidate.")
        return

    strength = prov.get("evidence_strength", "EXPLORATORY")
    strength_color = "#3fb950" if strength == "HIGH" else ("#f0883e" if strength == "MEDIUM" else "#8b949e")

    st.markdown(
        f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.25rem; margin-bottom:1.5rem;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-size:1.2rem; font-weight:700; color:#e6edf3;'>Evidence Trail for {chosen_conn}</div>
                <div style='background:{strength_color}22; color:{strength_color}; border:1px solid {strength_color}; padding:2px 10px; border-radius:12px; font-weight:600;'>
                    Evidence Strength: {strength}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Supporting Papers")
        papers = prov.get("supporting_papers", [])
        if papers:
            for p in papers:
                ev_type = p.get("evidence_type", "").replace("_", " ").title()
                st.markdown(
                    f"- **{p.get('title', 'Untitled')}** ({p.get('year', 'N/A')})\n"
                    f"  - *Evidence:* `{ev_type}` | *Matched:* {', '.join(p.get('matched_concepts', []))}"
                )
        else:
            st.write("No direct paper co-occurrences (inferred via graph structure).")

    with col2:
        st.markdown("#### 🕸️ Graph & Temporal Evidence")
        ge = prov.get("graph_evidence", {})
        te = prov.get("temporal_evidence", {})
        st.write(f"- **Shortest Path Length:** `{ge.get('shortest_path_length', 'N/A')}`")
        st.write(f"- **Direct Co-occurrence Weight:** `{ge.get('cooccurrence_weight', 0)}`")
        if ge.get("common_neighbors"):
            st.write(f"- **Common Bridge Concepts:** {', '.join(ge.get('common_neighbors', []))}")
        st.write(f"- **First Observed Years:** Concept A ({te.get('first_seen_year_a', 'N/A')}) · Concept B ({te.get('first_seen_year_b', 'N/A')})")

    st.markdown("#### 🔍 Traceability Narrative")
    for np_item in prov.get("narrative_points", []):
        st.markdown(f"• {np_item}")


def _render_research_assistant(result: dict):
    """Render interactive context-grounded conversational assistant."""
    from services.research_assistant import query_research_assistant, PRESET_QUESTIONS

    st.markdown("### 💬 Interactive Research Assistant")
    st.markdown(
        """
        <div style='font-size:0.85rem; color:#8b949e; margin-bottom:1rem;'>
            Ask questions about the uploaded papers, graph metrics, research gaps, and hypotheses.
            Answers are strictly grounded in your corpus.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your PaperGraph Research Assistant. Ask me anything about your uploaded corpus, top recommendations, research gaps, or experimental methodologies.",
            }
        ]

    # Quick action prompt buttons
    st.markdown("##### ⚡ Quick Queries")
    q_cols = st.columns(4)
    with q_cols[0]:
        if st.button("Why recommended?", key="qa_why", use_container_width=True):
            _handle_assistant_query(PRESET_QUESTIONS["why_recommended"], result)
    with q_cols[1]:
        if st.button("Supporting papers?", key="qa_papers", use_container_width=True):
            _handle_assistant_query(PRESET_QUESTIONS["supporting_papers"], result)
    with q_cols[2]:
        if st.button("Why is this a gap?", key="qa_gap", use_container_width=True):
            _handle_assistant_query(PRESET_QUESTIONS["why_gap"], result)
    with q_cols[3]:
        if st.button("Research questions?", key="qa_rq", use_container_width=True):
            _handle_assistant_query(PRESET_QUESTIONS["research_questions"], result)

    # Render message history
    for msg in st.session_state.assistant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Ask a research question about this analysis…"):
        _handle_assistant_query(prompt, result)


def _handle_assistant_query(query_text: str, result: dict):
    """Process user prompt and update assistant message history."""
    from services.research_assistant import query_research_assistant

    st.session_state.assistant_messages.append({"role": "user", "content": query_text})
    with st.chat_message("user"):
        st.markdown(query_text)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing corpus context…"):
            reply = query_research_assistant(
                user_query=query_text,
                analysis_result=result,
                conversation_history=st.session_state.assistant_messages,
            )
            st.markdown(reply)
            st.session_state.assistant_messages.append({"role": "assistant", "content": reply})



def _render_graph_viz(result: dict, candidates: list):
    """Render an interactive Plotly graph visualization."""
    import plotly.graph_objects as go
    import networkx as nx

    papers = result.get("papers", [])

    if not papers:
        st.info("No papers to visualize.")
        return

    # Rebuild a small visualization graph from paper concept lists
    from services.graph_builder import build_graph

    G = build_graph(papers)

    if G.number_of_nodes() == 0:
        st.info("Graph is empty — no concepts to visualize.")
        return

    # Limit to top 40 nodes by degree for readability
    if G.number_of_nodes() > 40:
        top_nodes = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:40]
        top_node_names = {n for n, _ in top_nodes}
        G = G.subgraph(top_node_names).copy()

    # Layout
    try:
        pos = nx.spring_layout(G, seed=42, k=2.5)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # Highlight top candidate nodes
    highlight_nodes = set()
    if candidates:
        top_cand = candidates[0]
        highlight_nodes.add(top_cand.get("concept_a", ""))
        highlight_nodes.add(top_cand.get("concept_b", ""))

    # Build edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(88,166,255,0.2)"),
        hoverinfo="none",
        name="Co-occurrence edges",
    )

    # Build node traces
    node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        node_text.append(
            f"<b>{node}</b><br>Connections: {degree}<br>"
            f"Papers: {G.nodes[node].get('paper_count', '?')}"
        )
        node_sizes.append(max(10, min(35, 8 + degree * 3)))
        if node in highlight_nodes:
            node_colors.append("#f0883e")  # highlight top candidate nodes
        else:
            node_colors.append("#58a6ff")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color="#0d1117")),
        text=[n if len(n) < 20 else n[:17] + "…" for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=9, color="#8b949e"),
        hovertext=node_text,
        hoverinfo="text",
        name="Concepts",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Concept Co-occurrence Knowledge Graph",
                font=dict(color="#e6edf3", size=14),
            ),
            showlegend=False,
            hovermode="closest",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
            height=500,
            font=dict(family="Inter, sans-serif"),
            annotations=[
                dict(
                    text="🟠 Top candidate concepts  🔵 Other concepts",
                    xref="paper", yref="paper",
                    x=0.01, y=0.01, showarrow=False,
                    font=dict(color="#6e7681", size=10),
                )
            ],
        ),
    )
    st.plotly_chart(fig, width="stretch")

    # Graph stats
    gs = result.get("graph_summary", {})
    if gs:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", gs.get("node_count", 0))
        with col2:
            st.metric("Edges", gs.get("edge_count", 0))
        with col3:
            st.metric("Density", f"{gs.get('density', 0):.4f}")
        with col4:
            st.metric("Components", gs.get("connected_components", 0))


def _render_temporal_graph(result: dict):
    """Render the temporal event timeline and year distribution."""
    import plotly.graph_objects as go

    tsum = result.get("temporal_summary", {})
    if not tsum or tsum.get("total_events", 0) == 0:
        st.info(
            "ℹ️ No temporal event data available. "
            "This can happen if all papers lack publication years — use the "
            "Paper Metadata expander to correct years before running."
        )
        return

    st.markdown("### ⏱️ Temporal Graph Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", tsum.get("total_events", 0))
    with col2:
        st.metric("Unique Concepts", tsum.get("unique_concepts", 0))
    with col3:
        st.metric("Unique Pairs", tsum.get("unique_pairs", 0))
    with col4:
        y_min = tsum.get("year_min")
        y_max = tsum.get("year_max")
        st.metric("Year Range", f"{y_min}–{y_max}" if y_min and y_max else "N/A")

    # Per-year event count bar chart
    papers = result.get("papers", [])
    year_counts: dict = {}
    for paper in papers:
        yr = paper.get("year")
        if yr:
            concepts = paper.get("concepts", [])
            from itertools import combinations as _combs
            n_pairs = len(list(_combs(set(concepts), 2)))
            year_counts[yr] = year_counts.get(yr, 0) + n_pairs

    if year_counts:
        years_sorted = sorted(year_counts.keys())
        counts = [year_counts[y] for y in years_sorted]

        bar_fig = go.Figure(
            data=[
                go.Bar(
                    x=years_sorted,
                    y=counts,
                    marker_color="#58a6ff",
                    marker_line_color="#1f6feb",
                    marker_line_width=1,
                    opacity=0.85,
                )
            ],
            layout=go.Layout(
                title=dict(text="Concept Co-occurrence Events per Year", font=dict(color="#e6edf3", size=13)),
                xaxis=dict(
                    title="Publication Year", tickmode="array",
                    tickvals=years_sorted, color="#8b949e",
                    gridcolor="#21262d",
                ),
                yaxis=dict(title="Event Count", color="#8b949e", gridcolor="#21262d"),
                paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                margin=dict(l=20, r=20, t=50, b=20),
                height=280,
                font=dict(family="Inter, sans-serif"),
            ),
        )
        st.plotly_chart(bar_fig, width="stretch")

    # Temporal split info
    eval_result = result.get("evaluation", {})
    if eval_result.get("train_events") is not None:
        st.markdown("### 📊 Temporal Split")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            st.metric("Train Events", eval_result.get("train_events", 0))
        with tcol2:
            st.metric("Val Events", eval_result.get("val_events", 0))
        with tcol3:
            st.metric("Test Events", eval_result.get("test_events", 0))
    else:
        st.markdown(
            f"""
            <div class='warning-box'>
                ⚠️ Temporal split unavailable: {eval_result.get('message', 'Insufficient data.')
                if not eval_result.get('sufficient_data') else 'Split performed.'}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class='disclaimer-box' style='font-size:0.8rem;'>
            📖 <strong>How temporal events work:</strong> Each paper contributes one event per
            unique concept pair it contains, timestamped by its publication year. SE-TGN
            processes these chronologically to learn which concept pairs tend to appear
            together in <em>later</em> papers — predicting future research connections.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_evaluation(result: dict):
    """Render the Model Evaluation tab with AUC/AP/P@K/NDCG@K metrics."""
    eval_result = result.get("evaluation", {})

    st.markdown("### 📈 Model Evaluation")

    if not eval_result:
        st.info("No evaluation data available.")
        return

    if not eval_result.get("sklearn_available", True):
        st.warning(
            "**scikit-learn not installed.** Run `pip install scikit-learn` to enable "
            "AUC, Average Precision, P@K and NDCG@K evaluation metrics."
        )
        return

    if not eval_result.get("sufficient_data"):
        st.info(
            f"ℹ️ {eval_result.get('message', 'Insufficient temporal data for formal evaluation.')}\n\n"
            "**Tip:** Upload papers spanning at least 4 distinct publication years to enable "
            "temporal train/val/test split evaluation."
        )
        return

    # Evaluation split stats
    st.markdown(
        f"""
        <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem;'>
            <div class='metric-card'>
                <div class='metric-value'>{eval_result.get('train_events', 0)}</div>
                <div class='metric-label'>Train Events</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{eval_result.get('val_events', 0)}</div>
                <div class='metric-label'>Val Events</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{eval_result.get('test_events', 0)}</div>
                <div class='metric-label'>Test Events</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{eval_result.get('test_positives', 0)}</div>
                <div class='metric-label'>Test Positives</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Baseline comparison table
    baselines = eval_result.get("baselines", {})
    if not baselines:
        st.info("No baseline comparison data available.")
        return

    st.markdown("#### Baseline Comparison")

    # Build table data
    import pandas as pd

    table_rows = []
    for baseline_name, metrics in baselines.items():
        if not metrics.get("sufficient_data"):
            row = {
                "Method": baseline_name,
                "AUC": "–",
                "AP": "–",
                "P@10": "–",
                "NDCG@10": "–",
                "Note": metrics.get("message", "Insufficient data"),
            }
        else:
            pk = metrics.get("precision_at_k", {})
            nk = metrics.get("ndcg_at_k", {})
            is_setgn = baseline_name == "SE-TGN"
            row = {
                "Method": f"⭐ {baseline_name}" if is_setgn else baseline_name,
                "AUC": f"{metrics['auc']:.4f}",
                "AP": f"{metrics['ap']:.4f}",
                "P@10": f"{pk.get('p@10', 0):.4f}",
                "NDCG@10": f"{nk.get('ndcg@10', 0):.4f}",
                "Note": "",
            }
        table_rows.append(row)

    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        <div class='disclaimer-box' style='font-size:0.8rem;'>
            ⚠️ <strong>Evaluation honesty note:</strong> Metrics are computed on the
            held-out <em>test</em> papers from a year-based temporal split. With only 5–10
            uploaded PDFs, the test set is very small, so AUC and AP values have very high
            variance and should NOT be interpreted as stable performance estimates.
            This evaluation exists to show the methodology, not to claim validated performance.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_papers(papers: list):
    """Render the analyzed papers list."""
    if not papers:
        st.info("No papers processed.")
        return

    for paper in papers:
        with st.expander(
            f"📄 [{paper['paper_id']}] {paper.get('title', paper['filename'])[:70]}",
            expanded=False,
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                if paper.get("abstract"):
                    st.markdown("**Abstract:**")
                    st.markdown(
                        f"<div style='color:#8b949e; font-size:0.85rem; line-height:1.6;'>{paper['abstract'][:600]}{'...' if len(paper.get('abstract','')) > 600 else ''}</div>",
                        unsafe_allow_html=True,
                    )
            with col2:
                st.markdown(f"**Year:** {paper.get('year', 'Unknown')}")
                if paper.get("authors"):
                    st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}")
                st.markdown(f"**Concepts:** {len(paper.get('concepts', []))}")
                if paper.get("concepts"):
                    st.markdown(
                        " ".join(
                            f"<span style='background:rgba(88,166,255,0.1); border:1px solid rgba(88,166,255,0.2); border-radius:4px; padding:2px 6px; font-size:0.72rem; color:#58a6ff;'>{c}</span>"
                            for c in paper["concepts"][:10]
                        ),
                        unsafe_allow_html=True,
                    )


def _render_technical_details(result: dict):
    """Render the Technical Details expander content."""
    from services.gnn_model import GNN_AVAILABLE

    gnn_active = GNN_AVAILABLE and any(
        c.get("gnn_active", False) for c in result.get("candidates", [])
    )

    st.markdown("#### Candidate Score Formula")
    if gnn_active:
        st.code(
            "candidate_score = 0.40 × graph_score + 0.30 × semantic_similarity + 0.30 × gnn_score",
            language="text",
        )
    else:
        st.code(
            "candidate_score = 0.57 × graph_score + 0.43 × semantic_similarity\n"
            "(GNN term dropped; weights renormalized)",
            language="text",
        )

    st.markdown("#### Graph Score Components")
    st.markdown(
        """
        - **Degree centrality** (0–1): fraction of all concepts this concept connects to
        - **Betweenness centrality** (0–1): how often this concept acts as a bridge
        - **Novelty bonus**: inversely proportional to existing edge weight (strongly connected pairs score lower — they are already "known" co-occurrences)
        
        `graph_score = 0.45 × centrality + 0.30 × bridge_score + 0.25 × novelty_bonus`
        """
    )

    st.markdown("#### Component Status")
    st.markdown(f"- **Analysis mode:** {result.get('analysis_mode', 'N/A')}")
    st.markdown(f"- **LLM evaluation:** {'Active' if result.get('llm_available') else 'Inactive'}")
    st.markdown(f"- **GNN component:** {'Active' if gnn_active else 'Inactive'}")
    st.markdown(f"- **Analysis ID:** `{result.get('analysis_id', 'N/A')}`")

    gs = result.get("graph_summary", {})
    if gs:
        st.markdown("#### Graph Statistics")
        st.json(gs)

    st.markdown("#### Academic Honesty Note")
    st.markdown(
        """
        This project is a simplified B.Tech analogue of the SE-TGN/CREF/GIC framework
        from *"Uncovering novel scientific insights with a synergistic GNN-LLM framework"*
        (Knowledge-Based Systems, 2025). Key differences:
        
        | Base Paper | This System |
        |---|---|
        | Trained temporal GNN (SE-TGN) on thousands of papers | NetworkX graph metrics on 5–10 PDFs |
        | Multi-year event streams | Single snapshot co-occurrence graph |
        | Validated on AUC/AP/P@K/NDCG@K | No formal validation |
        | Top-N from a large corpus | Top-3 from a tiny local graph |
        
        Results here are **exploratory only** and have not been evaluated against real future links.
        """
    )


# ─────────────────────────────────────────────────────────────
# Page: History
# ─────────────────────────────────────────────────────────────

def page_history():
    st.markdown("## 📋 Analysis History")
    st.markdown(
        "<p style='color:#8b949e;'>Past analyses are stored as JSON files in <code>data/history/</code>.</p>",
        unsafe_allow_html=True,
    )

    from services.history_service import list_analyses, load_analysis, get_history_stats, delete_analysis

    stats = get_history_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{stats['total_analyses']}</div><div class='metric-label'>Total Analyses</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{stats['total_papers_analyzed']}</div><div class='metric-label'>Papers Analyzed</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{stats['total_concepts_found']}</div><div class='metric-label'>Concepts Found</div></div>", unsafe_allow_html=True)

    analyses = list_analyses()

    if not analyses:
        st.info("📭 No analyses saved yet. Run a New Analysis to get started.")
        if st.button("🔬 Start Analysis", key="hist_new"):
            st.session_state.page = "analysis"
            st.rerun()
        return

    st.markdown(f"---")

    # Check if a specific analysis is selected for viewing
    selected_id = st.session_state.get("viewing_analysis_id")
    if selected_id:
        record = load_analysis(selected_id)
        if record:
            if st.button("← Back to History List", key="back_to_list"):
                st.session_state.viewing_analysis_id = None
                st.rerun()
            st.markdown(f"### Analysis: `{selected_id}`")
            st.markdown(f"**Created:** {record.get('created_at', 'N/A')}")
            _render_result(record)
            return
        else:
            st.error(f"Could not load analysis '{selected_id}'.")
            st.session_state.viewing_analysis_id = None

    # List all analyses
    for analysis in analyses:
        aid = analysis["analysis_id"]
        created = analysis.get("created_at", "")[:19].replace("T", " ")
        connection = analysis.get("top_connection", "No connection found")
        mode = analysis.get("analysis_mode", "")
        llm = "🤖 LLM" if analysis.get("llm_available") else "📊 Graph only"

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f"""
                <div class='history-card'>
                    <div class='history-title'>🔗 {connection or aid}</div>
                    <div class='history-meta'>
                        📅 {created} &nbsp;·&nbsp;
                        📄 {analysis.get('paper_count', 0)} papers &nbsp;·&nbsp;
                        🔵 {analysis.get('concept_count', 0)} concepts &nbsp;·&nbsp;
                        {llm}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("View", key=f"view_{aid}", use_container_width=True):
                st.session_state.viewing_analysis_id = aid
                st.rerun()


# ─────────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────────

def page_about():
    st.markdown("## ℹ️ About PaperGraph")

    st.markdown(
        """
        <div class='disclaimer-box' style='font-size:0.92rem; padding:1.25rem;'>
            ⚠️ <strong>Academic Honesty Disclaimer</strong><br><br>
            This system identifies potentially interesting relationships within the uploaded
            research papers. <strong>It does not prove that a relationship is scientifically novel
            or that the generated research direction is experimentally validated.</strong><br><br>
            Results are exploratory and intended for ideation purposes only. All scores and
            insights should be reviewed by domain experts before informing any research decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 What This System Does")
        st.markdown(
            """
            PaperGraph is a simplified B.Tech research discovery tool that:
            
            - Accepts **5–10 uploaded research PDFs** as input
            - Extracts **concepts/keywords** from each paper
            - Builds a **co-occurrence knowledge graph** (NetworkX)
            - Scores candidate concept-pair connections using **graph metrics + semantic similarity**
            - Optionally adds a **lightweight GNN encoder** signal (PyTorch Geometric)
            - Uses an **LLM (Gemini)** to evaluate top candidates on 4 dimensions (CREF-style)
            - Generates **research insights** including questions and hypothesis (GIC-style)
            - Saves results to **JSON history files**
            """
        )

    with col2:
        st.markdown("### 🔬 What This System Does NOT Do")
        st.markdown(
            """
            PaperGraph is **not** the base paper's system. It does NOT:
            
            - Train a Temporal Graph Network on thousands of papers
            - Model multi-year co-occurrence event streams  
            - Predict future keyword links with validated AUC/AP/P@K/NDCG@K scores
            - Guarantee scientific novelty of any connection
            - Replace peer review or expert domain judgment
            - Prove that generated hypotheses are correct
            
            **Never interpret a score here as "proven novelty" or "scientific validation."**
            """
        )

    st.markdown("---")
    st.markdown("### 🏛️ Base Paper Methodology vs. Our 5 Academic Extensions")
    st.markdown(
        """
        | Component | Category | Base Paper (KBS 2025) | PaperGraph (Our Extension) |
        |---|---|---|---|
        | **SE-TGN & Temporal Graph** | Base Methodology | Trained on thousands of timestamped papers | Continuous GRU `NodeMemory` + `TimeEncode` on user corpus |
        | **CREF & GIC** | Base Methodology | Multi-dimension rubric + hypothesis generation | Gemini-powered CREF scores & GIC research roadmap |
        | **Quantitative Evaluation** | Base Methodology | AUC, AP, P@10, NDCG@10 metrics | `scikit-learn` baseline evaluator comparing Random, Graph, GCN, SE-TGN |
        | **1. Research Gap Detection** | 🌟 **Our Contribution** | Not implemented | Multi-factor gap scoring (semantic, structural, temporal, cross-domain) |
        | **2. Evidence / Provenance** | 🌟 **Our Contribution** | Not implemented | Traceable paper citations, graph topology evidence, and strength scores |
        | **3. Human-in-the-Loop Feedback**| 🌟 **Our Contribution** | Static ranking only | Interactive ratings (👍 ⭐ 👎 🔖) + real-time personalized reranking |
        | **4. Cross-Domain Discovery** | 🌟 **Our Contribution** | Not implemented | Multi-discipline domain classification & interdisciplinary synergy scoring |
        | **5. Interactive Research Assistant**| 🌟 **Our Contribution** | Not implemented | Context-grounded conversational agent with anti-hallucination guardrails |
        
        **Reference:** *"Uncovering novel scientific insights with a synergistic GNN-LLM framework"*, Knowledge-Based Systems, 2025.
        """
    )

    st.markdown("---")
    st.markdown("### ⚙️ Technology Stack")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            **Core Framework**
            - Python 3.11+
            - Streamlit (UI)
            - python-dotenv
            """
        )
    with col2:
        st.markdown(
            """
            **Graph & Machine Learning**
            - PyTorch (SE-TGN)
            - PyTorch Geometric (GCN)
            - sentence-transformers (`all-MiniLM-L6-v2`)
            - NetworkX & Plotly
            - scikit-learn (Metrics)
            """
        )
    with col3:
        st.markdown(
            """
            **LLM & Extraction**
            - PyMuPDF (PDF processing)
            - Google Gemini (`gemini-2.5-flash` / `gemini-3.6-flash`)
            - google-genai SDK
            """
        )

    st.markdown("---")
    st.markdown("### 📝 Language Requirements")
    st.markdown(
        """
        This system uses the following hedged language to avoid overstating results:
        
        ✅ **Use:** "potential research connection", "candidate research direction",
        "potentially underexplored relationship", "model-generated hypothesis"
        
        ❌ **Never use:** "guaranteed novel discovery", "guaranteed scientific breakthrough",
        "scientifically proven hypothesis", "definitely novel"
        """
    )

    st.markdown("---")
    with st.expander("⚖️ License & Attribution"):
        st.markdown(
            """
            PaperGraph is a B.Tech academic project. It is inspired by but not affiliated with or
            a reproduction of the base paper. All LLM-generated content is clearly labeled as
            model-generated. No commercial use intended.
            
            Sentence-transformers model: `all-MiniLM-L6-v2` (Apache 2.0 license).
            NetworkX: BSD license.
            """
        )


# ─────────────────────────────────────────────────────────────
# Main router
# ─────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    page = st.session_state.get("page", "home")

    if page == "home":
        page_home()
    elif page == "analysis":
        page_analysis()
    elif page == "history":
        page_history()
    elif page == "about":
        page_about()
    else:
        page_home()


if __name__ == "__main__":
    main()
