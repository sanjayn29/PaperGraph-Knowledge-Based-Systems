"""
analysis_pipeline.py
─────────────────────
Orchestrates the full PaperGraph analysis pipeline end-to-end.

Pipeline stages (mirrors the 10-step spec):
  1. PDF extraction         (pdf_processor)
  2. Concept extraction     (concept_extractor)
  3. Embeddings             (embeddings)
  4. Knowledge graph build  (graph_builder)
  5. Graph analysis         (graph_analyzer)
  6. Optional GNN scoring   (gnn_model)
  7. LLM CREF evaluation    (llm_service)
  8. LLM GIC insight gen    (llm_service)
  9. History save           (history_service)

The pipeline communicates progress via an optional callback:
    progress_callback(step: int, total: int, message: str)

Returns a complete result dict suitable for display and history storage.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TOTAL_STEPS = 6  # visible steps shown in the UI


def run_pipeline(
    uploaded_files: list,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    top_candidates: int = 3,
    save_history: bool = True,
) -> dict:
    """
    Run the full analysis pipeline on a list of Streamlit UploadedFile objects.

    Parameters
    ----------
    uploaded_files     : list of Streamlit UploadedFile objects
    progress_callback  : optional fn(step, total, message) for progress updates
    top_candidates     : number of top candidates to evaluate with the LLM
    save_history       : whether to persist the result to data/history/

    Returns
    -------
    dict with keys:
        success, error, paper_count, papers, concept_count, relationship_count,
        graph_summary, analysis_mode, candidates, final_result,
        llm_available, warnings, analysis_id, created_at
    """

    def _progress(step: int, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(step, _TOTAL_STEPS, msg)
            except Exception:
                pass

    result: dict = {
        "success": False,
        "error": None,
        "paper_count": 0,
        "papers": [],
        "concept_count": 0,
        "relationship_count": 0,
        "graph_summary": {},
        "analysis_mode": "Lightweight Graph Analysis",
        "candidates": [],
        "final_result": {},
        "llm_available": False,
        "warnings": [],
        "analysis_id": None,
        "created_at": datetime.now().isoformat(),
    }

    try:
        # ── Step 1: PDF Extraction ─────────────────────────────────
        _progress(1, "📄 Extracting text from PDFs…")

        from services.pdf_processor import process_pdfs

        papers, pdf_warnings = process_pdfs(uploaded_files)
        result["warnings"].extend(pdf_warnings)

        if not papers:
            result["error"] = (
                "No readable PDFs found. Please upload valid research papers "
                "and ensure they contain extractable text (not scanned images)."
            )
            return result

        if len(papers) < 2:
            result["warnings"].append(
                "⚠️ Only 1 readable PDF found. The graph needs at least 2 papers "
                "for co-occurrence analysis. Results will be limited."
            )

        result["paper_count"] = len(papers)
        result["papers"] = papers

        # ── Step 2: Concept Extraction ────────────────────────────
        _progress(2, "🔍 Extracting research concepts…")

        from services.concept_extractor import extract_all_concepts

        # Use min_paper_freq=1 for small uploads (5-10 papers)
        papers, concept_to_paper_ids = extract_all_concepts(papers, min_paper_freq=1, top_n=80)
        result["papers"] = papers
        result["concept_count"] = len(concept_to_paper_ids)

        if result["concept_count"] < 2:
            result["error"] = (
                "Could not extract enough concepts from the uploaded papers. "
                "Please upload papers with richer text content."
            )
            return result

        # ── Step 3: Embeddings ────────────────────────────────────
        _progress(3, "🧠 Computing semantic embeddings…")

        from services.embeddings import embed_concepts, is_available as emb_available

        all_concepts = list(concept_to_paper_ids.keys())
        concept_embeddings = embed_concepts(all_concepts) if emb_available() else {}

        if not concept_embeddings:
            result["warnings"].append(
                "ℹ️ Semantic embeddings unavailable (sentence-transformers not installed "
                "or model failed to load). Semantic similarity will be 0 for all pairs."
            )

        # ── Step 4: Knowledge Graph ───────────────────────────────
        _progress(4, "🕸️ Building knowledge graph…")

        from services.graph_builder import build_graph, graph_summary

        G = build_graph(papers)
        result["relationship_count"] = G.number_of_edges()
        result["graph_summary"] = graph_summary(G)

        if G.number_of_nodes() < 2:
            result["error"] = (
                "The knowledge graph has too few nodes to analyze. "
                "Please upload papers with more diverse content."
            )
            return result

        # ── Step 5: Graph Analysis + Optional GNN ─────────────────
        _progress(5, "📊 Analyzing graph and scoring candidates…")

        from services.gnn_model import GNN_AVAILABLE, compute_gnn_scores, gnn_status_label
        from services.graph_analyzer import rank_candidates, get_candidate_context

        # GNN scoring (optional)
        gnn_scores = None
        if GNN_AVAILABLE:
            gnn_scores = compute_gnn_scores(G, concept_embeddings if concept_embeddings else None)

        analysis_mode = gnn_status_label(G)
        result["analysis_mode"] = analysis_mode

        # Dynamic strong-edge threshold: scale with corpus size so same-domain
        # uploads (where most pairs co-occur in 3+ papers) aren't over-filtered.
        # Exclude pairs that appear together in more than 60% of papers.
        dynamic_threshold = max(3, int(len(papers) * 0.6))

        candidates = rank_candidates(
            G=G,
            concept_embeddings=concept_embeddings,
            gnn_scores=gnn_scores,
            top_k=20,
            exclude_strong_edges=True,
            strong_edge_threshold=dynamic_threshold,
        )

        # Fallback: if ALL pairs were filtered out (very uniform domain),
        # retry without the strong-edge exclusion so the analysis never silently fails.
        if not candidates:
            logger.info(
                "No candidates after strong-edge exclusion (threshold=%d). "
                "Retrying without filter.",
                dynamic_threshold,
            )
            candidates = rank_candidates(
                G=G,
                concept_embeddings=concept_embeddings,
                gnn_scores=gnn_scores,
                top_k=20,
                exclude_strong_edges=False,
            )

        if not candidates:
            result["error"] = (
                "No candidate concept pairs could be generated. "
                "Check that your PDFs contain extractable text (not scanned images) "
                "and that the papers discuss distinct research concepts."
            )
            return result

        result["candidates"] = candidates

        # ── Step 6: LLM Evaluation & Insight Generation ───────────
        _progress(6, "🤖 Running LLM evaluation (CREF + GIC)…")

        from services.llm_service import LLMService, llm_status_label

        llm = LLMService()
        result["llm_available"] = llm.is_available

        if not llm.is_available:
            result["warnings"].append(
                f"ℹ️ {llm_status_label(llm)} — LLM evaluation (CREF) and insight "
                "generation (GIC) stages are skipped. Set GEMINI_API_KEY in your .env "
                "file to enable them."
            )

        # Evaluate top candidates
        top_candidates_list = candidates[:top_candidates]
        evaluations: list[dict] = []

        for candidate in top_candidates_list:
            if not llm.is_available:
                break
            context = get_candidate_context(candidate, G, papers)
            evaluation = llm.evaluate_candidate(candidate, context, papers)
            if evaluation:
                evaluations.append({**candidate, **evaluation})

        # Select the best candidate for GIC insight generation
        # If LLM evaluated, pick highest total score; otherwise use graph score
        if evaluations:
            best_candidate_with_eval = max(
                evaluations,
                key=lambda x: x.get("novelty", 0) + x.get("impact", 0)
                + x.get("plausibility", 0) + x.get("interdisciplinarity", 0),
            )
            context = get_candidate_context(best_candidate_with_eval, G, papers)
            insight = llm.generate_insight(
                best_candidate_with_eval,
                best_candidate_with_eval,
                context,
                papers,
            )
            if insight:
                final_result = {**best_candidate_with_eval, **insight}
            else:
                final_result = _build_fallback_result(best_candidate_with_eval, G, papers)
        else:
            # No LLM — use the top graph-scored candidate
            best_candidate = candidates[0]
            final_result = _build_fallback_result(best_candidate, G, papers)

        result["final_result"] = final_result
        result["success"] = True

        # ── History Save ──────────────────────────────────────────
        if save_history:
            from services.history_service import save_analysis

            analysis_id = save_analysis(result)
            result["analysis_id"] = analysis_id

    except Exception as exc:
        logger.exception("Unexpected pipeline error: %s", exc)
        result["error"] = (
            f"An unexpected error occurred during analysis: {exc}. "
            "Please check the logs for details."
        )

    return result


def _build_fallback_result(candidate: dict, G, papers: list[dict]) -> dict:
    """
    Build a minimal final_result when LLM is not available.
    Clearly marks content as auto-generated graph analysis without LLM.
    """
    concept_a = candidate.get("concept_a", "")
    concept_b = candidate.get("concept_b", "")

    papers_a = [p.get("title") or p["filename"] for p in papers if concept_a in p.get("concepts", [])]
    papers_b = [p.get("title") or p["filename"] for p in papers if concept_b in p.get("concepts", [])]

    supporting_ids = [
        p["paper_id"]
        for p in papers
        if concept_a in p.get("concepts", []) or concept_b in p.get("concepts", [])
    ]

    return {
        "connection": candidate["connection"],
        "novelty": None,
        "impact": None,
        "plausibility": None,
        "interdisciplinarity": None,
        "reasoning": None,
        "research_direction": (
            f"Potential connection between '{concept_a}' and '{concept_b}' "
            "identified through graph co-occurrence analysis."
        ),
        "explanation": (
            f"The concepts '{concept_a}' and '{concept_b}' appear across the uploaded papers "
            f"with notable graph centrality and semantic proximity. "
            f"'{concept_a}' appears in: {', '.join(papers_a[:3])}. "
            f"'{concept_b}' appears in: {', '.join(papers_b[:3])}. "
            "Enable LLM evaluation (set GEMINI_API_KEY) for a richer, "
            "AI-generated explanation."
        ),
        "research_questions": [
            f"How can {concept_a} and {concept_b} be effectively combined in future research?",
            f"What existing methods from {concept_a} could be applied to problems in {concept_b}?",
            f"What are the key technical challenges in integrating {concept_a} with {concept_b}?",
        ],
        "hypothesis": (
            f"[Model-generated hypothesis — not LLM evaluated] "
            f"Integrating {concept_a} with {concept_b} may yield improvements "
            "in research outcomes compared to applying each approach independently."
        ),
        "supporting_papers": supporting_ids,
        "limitations": (
            "This insight was generated without LLM evaluation. It is based solely on "
            "graph co-occurrence metrics and semantic similarity from sentence embeddings. "
            "No scientific novelty or experimental validity is implied. "
            "Set GEMINI_API_KEY to enable AI-powered CREF evaluation and GIC insight generation."
        ),
    }
