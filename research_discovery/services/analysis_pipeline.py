"""
analysis_pipeline.py
─────────────────────
Orchestrates the full PaperGraph analysis pipeline end-to-end.

Pipeline stages (SE-TGN upgrade):
  1.  PDF extraction              (pdf_processor)
  2.  Concept extraction          (concept_extractor)
  3.  Embeddings                  (embeddings)
  4.  Knowledge graph             (graph_builder)
  5.  Temporal event graph        (temporal_graph)   ← NEW
  6.  SE-TGN training             (se_tgn)           ← NEW
  7.  Future link prediction      (se_tgn)           ← NEW
  8.  GCN baseline (optional)     (gnn_model)
  9.  Candidate ranking           (graph_analyzer)   ← SE-TGN score now primary
  10. LLM CREF evaluation         (llm_service)
  11. LLM GIC insight generation  (llm_service)
  12. Evaluation metrics           (evaluator)        ← NEW
  13. History save                 (history_service)

The pipeline communicates progress via an optional callback:
    progress_callback(step: int, total: int, message: str)

Returns a complete result dict suitable for display and history storage.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TOTAL_STEPS = 9  # visible steps shown in the UI progress bar


def run_pipeline(
    uploaded_files: list,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    top_candidates: int = 3,
    save_history: bool = True,
    year_overrides: Optional[dict[str, int]] = None,
) -> dict:
    """
    Run the full SE-TGN-enhanced analysis pipeline.

    Parameters
    ----------
    uploaded_files    : list of Streamlit UploadedFile objects
    progress_callback : optional fn(step, total, message) for UI progress
    top_candidates    : number of top candidates to send to CREF/GIC
    save_history      : whether to persist result to data/history/
    year_overrides    : optional {filename → int year} for manual year correction

    Returns
    -------
    dict with keys:
        success, error, paper_count, papers, concept_count, relationship_count,
        graph_summary, temporal_summary, analysis_mode, candidates, final_result,
        llm_available, evaluation, warnings, analysis_id, created_at
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
        "temporal_summary": {},
        "analysis_mode": "Lightweight Graph Analysis",
        "candidates": [],
        "final_result": {},
        "llm_available": False,
        "evaluation": {},
        "warnings": [],
        "analysis_id": None,
        "created_at": datetime.now().isoformat(),
    }

    try:
        # ── Step 1: PDF Extraction ─────────────────────────────────
        _progress(1, "📄 Extracting text from PDFs…")

        from services.pdf_processor import process_pdfs

        papers, pdf_warnings = process_pdfs(uploaded_files, year_overrides=year_overrides)
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

        # ── Step 5: Temporal Event Graph ──────────────────────────
        _progress(5, "⏱️ Building temporal concept event graph…")

        from services.temporal_graph import build_temporal_events

        temporal_graph, tg_warnings = build_temporal_events(papers, concept_embeddings)
        result["warnings"].extend(tg_warnings)
        result["temporal_summary"] = temporal_graph.summary()

        # ── Step 6 & 7: SE-TGN Training + Link Prediction ─────────
        _progress(6, "🧬 Training SE-TGN temporal model…")

        from services.se_tgn import compute_setgn_scores, setgn_status_label, SETGN_AVAILABLE

        setgn_scores = None
        prediction_time = None

        if SETGN_AVAILABLE:
            _, y_max = temporal_graph.year_range()
            if y_max:
                prediction_time = y_max + 1

            setgn_scores = compute_setgn_scores(
                temporal_graph,
                concept_embeddings,
                query_time=prediction_time,
            )

            if setgn_scores is None:
                n_events = len(temporal_graph)
                result["warnings"].append(
                    f"ℹ️ {setgn_status_label(n_events)} — "
                    "Falling back to graph-only scoring."
                )
        else:
            result["warnings"].append(
                "ℹ️ SE-TGN inactive (PyTorch not installed). "
                "Install with: pip install torch. Falling back to graph + semantic scoring."
            )

        # ── Step 8: GCN Baseline (optional) ──────────────────────
        from services.gnn_model import GNN_AVAILABLE, compute_gnn_scores, gnn_status_label

        gnn_scores = None
        if GNN_AVAILABLE and setgn_scores is None:
            # Only run GCN if SE-TGN is inactive (GCN is a fallback baseline)
            gnn_scores = compute_gnn_scores(G, concept_embeddings if concept_embeddings else None)

        # Determine analysis mode label
        if setgn_scores is not None:
            analysis_mode = f"SE-TGN Temporal Analysis (predicting links for {prediction_time})"
        elif gnn_scores is not None:
            analysis_mode = gnn_status_label(G)
        else:
            analysis_mode = "Lightweight Graph Analysis"
        result["analysis_mode"] = analysis_mode

        # ── Step 9: Candidate Ranking ─────────────────────────────
        _progress(7, "📊 Ranking candidate concept connections…")

        from services.graph_analyzer import rank_candidates, get_candidate_context

        dynamic_threshold = max(3, int(len(papers) * 0.6))

        candidates = rank_candidates(
            G=G,
            concept_embeddings=concept_embeddings,
            gnn_scores=gnn_scores,
            setgn_scores=setgn_scores,
            prediction_time=prediction_time,
            top_k=20,
            exclude_strong_edges=True,
            strong_edge_threshold=dynamic_threshold,
        )

        # Fallback: retry without strong-edge exclusion if no candidates found
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
                setgn_scores=setgn_scores,
                prediction_time=prediction_time,
                top_k=20,
                exclude_strong_edges=False,
            )

        if not candidates:
            result["error"] = (
                "No candidate concept pairs could be generated. "
                "Check that your PDFs contain extractable text and discuss distinct research concepts."
            )
            return result

        result["candidates"] = candidates

        # ── Step 10 & 11: LLM CREF + GIC ─────────────────────────
        _progress(8, "🤖 Running LLM evaluation (CREF + GIC)…")

        from services.llm_service import LLMService, llm_status_label

        llm = LLMService()
        result["llm_available"] = llm.is_available

        if not llm.is_available:
            result["warnings"].append(
                f"ℹ️ {llm_status_label(llm)} — LLM evaluation (CREF) and insight "
                "generation (GIC) stages are skipped. Set GEMINI_API_KEY in your .env "
                "file to enable them."
            )

        top_candidates_list = candidates[:top_candidates]
        evaluations: list[dict] = []

        for candidate in top_candidates_list:
            if not llm.is_available:
                break
            context = get_candidate_context(candidate, G, papers)
            evaluation = llm.evaluate_candidate(candidate, context, papers)
            if evaluation:
                evaluations.append({**candidate, **evaluation})

        # Select best candidate (highest CREF total or highest graph score)
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
            final_result = {**best_candidate_with_eval, **(insight or {})}
        else:
            best_candidate = candidates[0]
            final_result = _build_fallback_result(best_candidate, G, papers)

        result["final_result"] = final_result

        # ── Step 12: Evaluation Metrics ───────────────────────────
        _progress(9, "📈 Computing evaluation metrics…")

        evaluation_result = _run_evaluation(
            temporal_graph=temporal_graph,
            candidates=candidates,
            setgn_scores=setgn_scores,
            gnn_scores=gnn_scores,
            concept_embeddings=concept_embeddings,
            G=G,
        )
        result["evaluation"] = evaluation_result

        result["success"] = True

        # ── Step 13: History Save ─────────────────────────────────
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


def _run_evaluation(
    temporal_graph,
    candidates: list[dict],
    setgn_scores: Optional[dict],
    gnn_scores: Optional[dict],
    concept_embeddings: dict,
    G,
) -> dict:
    """
    Run temporal split evaluation if enough data exists.
    Returns an evaluation dict (may contain 'sufficient_data': False).
    """
    try:
        from services.evaluator import (
            SKLEARN_AVAILABLE,
            run_baseline_comparison,
            all_concept_pairs,
            evaluator_status,
        )
        from services.graph_analyzer import compute_centralities, _graph_score_for_pair

        if not SKLEARN_AVAILABLE:
            return {
                "sufficient_data": False,
                "message": "scikit-learn not installed — pip install scikit-learn",
                "sklearn_available": False,
            }

        # Use temporal split for evaluation
        train_events, val_events, test_events = temporal_graph.temporal_split()

        if not test_events:
            return {
                "sufficient_data": False,
                "message": (
                    "Insufficient temporal data for formal evaluation. "
                    "Upload papers spanning at least 4 distinct publication years."
                ),
                "sklearn_available": True,
            }

        # Test positives = pairs that co-occur in test-set papers
        test_positives = {e.pair for e in test_events}

        # All concept pairs as evaluation universe
        concepts = temporal_graph.unique_concepts()
        all_pairs = all_concept_pairs(concepts)

        # Build graph scores for all pairs
        import numpy as np
        centralities = compute_centralities(G)
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0

        graph_scores = {}
        for c_a, c_b in all_pairs:
            if G.has_node(c_a) and G.has_node(c_b):
                gs = _graph_score_for_pair(c_a, c_b, G, centralities, max_weight)
            else:
                gs = 0.0
            graph_scores[(c_a, c_b)] = gs

        baselines = run_baseline_comparison(
            all_pairs=all_pairs,
            test_positives=test_positives,
            graph_scores=graph_scores,
            gnn_scores=gnn_scores,
            setgn_scores=setgn_scores,
        )

        return {
            "sufficient_data": True,
            "sklearn_available": True,
            "train_events": len(train_events),
            "val_events": len(val_events),
            "test_events": len(test_events),
            "test_positives": len(test_positives),
            "baselines": baselines,
        }

    except Exception as exc:
        logger.warning("Evaluation step failed: %s", exc)
        return {
            "sufficient_data": False,
            "message": f"Evaluation error: {exc}",
        }


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

    setgn_note = ""
    if candidate.get("setgn_active") and candidate.get("setgn_score") is not None:
        setgn_note = (
            f" SE-TGN temporal link-prediction score: {candidate['setgn_score']:.3f}."
        )

    return {
        "connection": candidate["connection"],
        "novelty": None,
        "impact": None,
        "plausibility": None,
        "interdisciplinarity": None,
        "reasoning": None,
        "research_direction": (
            f"Potential temporal connection between '{concept_a}' and '{concept_b}' "
            "identified through SE-TGN temporal graph analysis."
        ),
        "explanation": (
            f"The concepts '{concept_a}' and '{concept_b}' show a potentially emerging "
            f"relationship in the uploaded papers based on their temporal co-occurrence patterns.{setgn_note} "
            f"'{concept_a}' appears in: {', '.join(papers_a[:3])}. "
            f"'{concept_b}' appears in: {', '.join(papers_b[:3])}. "
            "Enable LLM evaluation (set GEMINI_API_KEY) for a richer, AI-generated explanation."
        ),
        "research_questions": [
            f"How can {concept_a} and {concept_b} be effectively combined in future research?",
            f"What existing methods from {concept_a} could be applied to problems in {concept_b}?",
            f"What are the key technical challenges in integrating {concept_a} with {concept_b}?",
        ],
        "hypothesis": (
            f"[Model-generated — not LLM evaluated] "
            f"Integrating {concept_a} with {concept_b} may yield improvements "
            "in research outcomes compared to applying each approach independently."
        ),
        "supporting_papers": supporting_ids,
        "limitations": (
            "This insight was generated without LLM evaluation. It is based on "
            "SE-TGN temporal graph analysis and graph co-occurrence metrics. "
            "No scientific novelty or experimental validity is implied. "
            "Set GEMINI_API_KEY to enable AI-powered CREF evaluation and GIC insight generation."
        ),
    }
