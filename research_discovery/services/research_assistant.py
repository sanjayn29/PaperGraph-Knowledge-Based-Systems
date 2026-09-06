"""
research_assistant.py
─────────────────────
Interactive Research Assistant Engine for PaperGraph.

Provides context-grounded conversational reasoning over the current analysis session,
including papers, concepts, graph topology, SE-TGN predictions, research gaps,
evidence provenance, CREF scores, and GIC insights.

Anti-hallucination rule:
When a question cannot be supported by the current corpus, the assistant explicitly states:
"I could not find sufficient evidence in the uploaded research corpus."
"""

from __future__ import annotations

import logging
import os
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)

PRESET_QUESTIONS: dict[str, str] = {
    "why_recommended": "Why was this connection recommended by the system?",
    "supporting_papers": "Which papers in the corpus support this connection?",
    "why_gap": "Why is this considered an underexplored research gap?",
    "research_questions": "Give me three concrete research questions based on this discovery.",
    "experiments": "What experimental methodology could validate this hypothesis?",
    "limitations": "What are the limitations and potential bottlenecks of this research direction?",
    "compare_top2": "Compare the top two candidate connections in terms of novelty and feasibility.",
}


def build_assistant_context(
    analysis_result: dict,
    active_candidate: Optional[dict] = None,
) -> str:
    """
    Format the complete analysis state into a concise, token-efficient prompt context.
    """
    papers = analysis_result.get("papers", [])
    candidates = analysis_result.get("candidates", [])
    gaps = analysis_result.get("research_gaps", [])
    cross_domains = analysis_result.get("cross_domain_discoveries", [])
    final_res = analysis_result.get("final_result", {})
    provenance_map = analysis_result.get("provenance", {})

    lines: list[str] = []

    # 1. Corpus Summary
    lines.append("=== RESEARCH CORPUS ===")
    lines.append(f"Total Papers: {len(papers)}")
    for p in papers[:8]:
        lines.append(
            f"- [{p.get('paper_id', '')}] '{p.get('title', 'Untitled')}' ({p.get('year', 'Unknown')}) | Concepts: {', '.join(p.get('concepts', [])[:5])}"
        )

    # 2. Top Candidates
    lines.append("\n=== TOP CANDIDATE CONNECTIONS ===")
    for c in candidates[:5]:
        setgn = f"SE-TGN: {c['setgn_score']:.3f}" if c.get("setgn_score") is not None else ""
        lines.append(
            f"- {c.get('concept_a')} + {c.get('concept_b')} | Score: {c.get('candidate_score', 0):.3f} | {setgn}"
        )

    # 3. Top Research Gaps
    if gaps:
        lines.append("\n=== TOP RESEARCH GAPS ===")
        for g in gaps[:4]:
            lines.append(
                f"- {g.get('concept_a')} + {g.get('concept_b')} (Gap Score: {g.get('gap_score', 0):.2f}) | {g.get('explanation', '')}"
            )

    # 4. Cross-Domain Discoveries
    if cross_domains:
        lines.append("\n=== CROSS-DOMAIN DISCOVERIES ===")
        for cd in cross_domains[:3]:
            lines.append(
                f"- {cd.get('concept_a')} ({cd.get('domain_a')}) ↔ {cd.get('concept_b')} ({cd.get('domain_b')}) | Cross-Domain Score: {cd.get('cross_domain_score', 0):.2f}"
            )

    # 5. Active Final Result / GIC Insight
    if final_res:
        lines.append("\n=== GENERATED INSIGHT & CREF EVALUATION ===")
        lines.append(f"Top Connection: {final_res.get('connection', '')}")
        lines.append(
            f"CREF Scores — Novelty: {final_res.get('novelty', '?')}/5, "
            f"Impact: {final_res.get('impact', '?')}/5, "
            f"Plausibility: {final_res.get('plausibility', '?')}/5, "
            f"Interdisciplinarity: {final_res.get('interdisciplinarity', '?')}/5"
        )
        if final_res.get("research_direction"):
            lines.append(f"Research Direction: {final_res.get('research_direction')}")
        if final_res.get("hypothesis"):
            lines.append(f"Hypothesis: {final_res.get('hypothesis')}")
        if final_res.get("research_questions"):
            lines.append(f"Research Questions: {'; '.join(final_res.get('research_questions', []))}")

    # 6. Specific candidate provenance if focused
    if active_candidate:
        cand_key = f"{active_candidate.get('concept_a', '')} + {active_candidate.get('concept_b', '')}"
        prov = provenance_map.get(cand_key)
        if prov:
            lines.append(f"\n=== PROVENANCE FOR '{cand_key}' ===")
            lines.append(f"Evidence Strength: {prov.get('evidence_strength', 'UNKNOWN')}")
            for np_item in prov.get("narrative_points", []):
                lines.append(f"• {np_item}")

    return "\n".join(lines)


def query_research_assistant(
    user_query: str,
    analysis_result: dict,
    conversation_history: Optional[list[dict]] = None,
    active_candidate: Optional[dict] = None,
) -> str:
    """
    Generate an answer to a user's research inquiry using current session data.

    Returns
    -------
    Context-grounded assistant response string.
    """
    from services.llm_service import LLMService

    llm = LLMService()
    context = build_assistant_context(analysis_result, active_candidate)

    # Deterministic fallback when LLM is unavailable
    if not llm.is_available:
        return _fallback_assistant_response(user_query, analysis_result, active_candidate)

    system_prompt = f"""You are the PaperGraph Interactive Research Assistant, an academic AI tool grounded in the user's uploaded scientific research papers.

CORPUS CONTEXT:
{context}

STRICT INSTRUCTIONS:
1. Answer the user's query thoroughly, clearly, and concisely based ONLY on the provided corpus context, graph metrics, SE-TGN predictions, research gaps, and insights.
2. If the user asks about facts, connections, or papers NOT mentioned or implied in the corpus, respond explicitly with:
   "I could not find sufficient evidence in the uploaded research corpus."
3. Do NOT fabricate citations, paper titles, mathematical metrics, or authors.
4. Structure your responses with clean Markdown bullet points and bold headers where appropriate.
"""

    messages = []
    if conversation_history:
        for turn in conversation_history[-4:]:  # Include last 4 turns for context
            role = turn.get("role", "user")
            content = turn.get("content", "")
            messages.append(f"{role.capitalize()}: {content}")

    messages.append(f"User: {user_query}")
    full_prompt = f"{system_prompt}\n\nCONVERSATION:\n" + "\n".join(messages) + "\nAssistant:"

    try:
        if llm._client:
            response = llm._client.models.generate_content(
                model=llm.model_name,
                contents=full_prompt,
            )
            return (response.text or "").strip()
        else:
            return _fallback_assistant_response(user_query, analysis_result, active_candidate)
    except Exception as exc:
        logger.warning("Research Assistant LLM query failed: %s", exc)
        return _fallback_assistant_response(user_query, analysis_result, active_candidate)


def _fallback_assistant_response(
    query: str,
    analysis_result: dict,
    active_candidate: Optional[dict] = None,
) -> str:
    """
    Deterministic rule-based response when LLM service is offline.
    """
    q_lower = query.lower()
    final_res = analysis_result.get("final_result", {})
    candidates = analysis_result.get("candidates", [])
    gaps = analysis_result.get("research_gaps", [])

    if "why" in q_lower and ("recommend" in q_lower or "chosen" in q_lower):
        top = active_candidate or (candidates[0] if candidates else {})
        if top:
            return (
                f"### Recommendation Rationale\n\n"
                f"**Connection:** {top.get('concept_a')} + {top.get('concept_b')}\n"
                f"- **Candidate Score:** {top.get('candidate_score', 0):.3f}\n"
                f"- **Semantic Similarity:** {top.get('semantic_similarity', 0):.3f}\n"
                f"- **SE-TGN Future Link Probability:** {top.get('setgn_score', 'N/A')}\n"
                f"- **Graph Metric:** {top.get('graph_score', 0):.3f}\n\n"
                f"This pair was prioritized because it combines strong latent semantic affinity with a high probability of emerging co-occurrence in future research."
            )

    if "paper" in q_lower or "support" in q_lower:
        papers = _supporting_papers_for_fallback(analysis_result, active_candidate)
        return (
            "### Supporting Papers in Corpus\n\n"
            + "\n".join([f"- **{p.get('title', 'Untitled')}** ({p.get('year', 'Unknown')})" for p in papers[:5]])
            if papers
            else "### Supporting Papers in Corpus\n\nNo supporting papers were identified for this connection."
        )

    if "gap" in q_lower:
        if gaps:
            top_gap = gaps[0]
            return (
                f"### Top Research Gap\n\n"
                f"**{top_gap.get('concept_a')} ↔ {top_gap.get('concept_b')}** (Gap Score: {top_gap.get('gap_score', 0):.2f})\n\n"
                f"{top_gap.get('explanation', 'Identified as an underexplored connection.')}"
            )

    if "question" in q_lower:
        qs = final_res.get("research_questions", [])
        if qs:
            return "### Suggested Research Questions\n\n" + "\n".join([f"1. {q}" for q in qs])

    return (
        f"### Summary of Analysis\n\n"
        f"- Top Connection: **{final_res.get('connection', 'N/A')}**\n"
        f"- Total Concepts: {analysis_result.get('concept_count', 0)}\n"
        f"- Total Events: {analysis_result.get('temporal_summary', {}).get('total_events', 0)}\n"
        f"- Note: Connect a `GEMINI_API_KEY` for conversational depth."
    )


def _supporting_papers_for_fallback(
    analysis_result: dict,
    active_candidate: Optional[dict] = None,
) -> list[dict]:
    """Return only corpus papers with explicit or concept-level support."""
    papers = analysis_result.get("papers", [])
    candidates = analysis_result.get("candidates", [])
    candidate = active_candidate or (candidates[0] if candidates else None)
    if not candidate:
        return []

    paper_by_id = {
        paper.get("paper_id"): paper
        for paper in papers
        if paper.get("paper_id")
    }
    candidate_key = f"{candidate.get('concept_a', '')} + {candidate.get('concept_b', '')}"
    provenance = analysis_result.get("provenance", {}).get(candidate_key, {})
    explicit_ids = {
        paper.get("paper_id")
        for paper in provenance.get("supporting_papers", [])
        if paper.get("paper_id")
    }
    if explicit_ids:
        return [paper_by_id[paper_id] for paper_id in explicit_ids if paper_id in paper_by_id]

    concepts = {
        candidate.get("concept_a"),
        candidate.get("concept_b"),
    } - {None, ""}
    return [
        paper
        for paper in papers
        if concepts.intersection(paper.get("concepts", []))
    ]
