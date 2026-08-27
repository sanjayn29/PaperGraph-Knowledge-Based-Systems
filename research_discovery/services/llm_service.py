"""
llm_service.py
──────────────
LLM abstraction layer for PaperGraph.
Uses the new `google-genai` SDK (google.genai), not the deprecated google-generativeai.

Implements two pipeline stages from the base paper in simplified form:

CREF stage (evaluate_candidate)
  - Scores a candidate concept-pair on 4 dimensions (1–5 each):
      novelty, impact, plausibility, interdisciplinarity
  - Returns strict JSON output
  - Wording is always hedged: "potentially underexplored within the uploaded
    papers" — NEVER "guaranteed novel" or "scientifically proven"

GIC stage (generate_insight)
  - For the top-ranked candidate, generates:
      research_direction, explanation, research_questions (3×),
      hypothesis, supporting_papers, limitations
  - Clearly separates evidence from the uploaded papers vs. model-generated
    speculation

Graceful degradation
─────────────────────
If GEMINI_API_KEY is not set or the API call fails, both functions return
None and the caller must handle the None case (show a friendly UI message).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

from utils.json_utils import extract_json_from_llm_response

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Optional google-genai import (new SDK)
# ─────────────────────────────────────────────────────────────
try:
    from google import genai as _genai  # type: ignore[import]

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    _genai = None  # type: ignore[assignment]
    logger.warning(
        "google-genai not installed — LLM stages will be unavailable. "
        "Install with: pip install google-genai"
    )

_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")

# ─────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────

_CREF_SYSTEM_PROMPT = """You are a research methodology assistant helping to identify potentially interesting relationships between research concepts.

IMPORTANT CONSTRAINTS:
- You are evaluating concept pairs found in a small set of uploaded research papers (5-10 PDFs).
- You CANNOT claim that a relationship is scientifically novel or experimentally validated.
- Use hedged language: "potentially underexplored", "may represent", "could suggest".
- Never use: "guaranteed novel", "scientifically proven", "definitely novel".
- Your evaluation is exploratory, not a peer-reviewed assessment.

You MUST respond with ONLY a valid JSON object — no markdown, no prose, no code fences.
"""

_CREF_USER_TEMPLATE = """Evaluate the following candidate research connection found in a small set of uploaded research papers.

Candidate Connection: {connection}

Context from the uploaded papers:
{context}

Paper abstracts for reference:
{abstracts}

Evaluate this connection on four dimensions (score 1–5, where 5 is highest):
- novelty: How potentially underexplored is this connection within the uploaded papers? (Not a claim of global novelty)
- impact: How significant could this connection be for advancing research if explored?
- plausibility: How technically feasible and scientifically grounded does this connection seem?
- interdisciplinarity: How much does this connection bridge different research domains?

Respond with ONLY this JSON (no other text):
{{
  "candidate": "{connection}",
  "novelty": <1-5>,
  "impact": <1-5>,
  "plausibility": <1-5>,
  "interdisciplinarity": <1-5>,
  "reasoning": "<2-3 sentence explanation using hedged language>"
}}"""

_GIC_SYSTEM_PROMPT = """You are a research ideation assistant helping researchers identify potentially promising directions based on a small set of uploaded research papers.

IMPORTANT CONSTRAINTS:
- Base your response ONLY on the provided paper abstracts and context.
- Clearly distinguish between: (a) evidence from the uploaded papers, and (b) your model-generated speculation.
- Use hedged language throughout: "could explore", "may suggest", "potentially", "if validated".
- Never claim: "this will work", "this is proven", "this is definitely novel".
- This system has NOT been validated against future research outcomes.

You MUST respond with ONLY a valid JSON object — no markdown, no prose, no code fences.
"""

_GIC_USER_TEMPLATE = """Based on the uploaded research papers, generate a research insight report for this candidate connection.

Candidate Connection: {connection}
Concept A: {concept_a}
Concept B: {concept_b}

CREF Evaluation Scores:
- Novelty: {novelty}/5
- Impact: {impact}/5
- Plausibility: {plausibility}/5
- Interdisciplinarity: {interdisciplinarity}/5
- Reasoning: {reasoning}

Context from the uploaded papers:
{context}

Paper titles and abstracts:
{abstracts}

Supporting papers (by ID): {supporting_paper_ids}

Generate a research insight report. Respond with ONLY this JSON:
{{
  "connection": "{connection}",
  "research_direction": "<A concise (1-2 sentence) research direction title>",
  "explanation": "<3-4 sentence explanation of why this connection is potentially interesting, citing specific evidence from the uploaded papers>",
  "research_questions": [
    "<Research question 1 — specific and testable>",
    "<Research question 2 — specific and testable>",
    "<Research question 3 — specific and testable>"
  ],
  "hypothesis": "<A single falsifiable hypothesis that could be experimentally tested, clearly labeled as model-generated speculation>",
  "supporting_papers": {supporting_paper_ids_json},
  "limitations": "<2-3 sentences about the limitations of this suggestion: small paper set, no temporal validation, model-generated speculation, etc.>"
}}"""


# ─────────────────────────────────────────────────────────────
# LLM Service
# ─────────────────────────────────────────────────────────────


class LLMService:
    """
    Provider-agnostic LLM service for PaperGraph.
    Uses google-genai (new SDK). Add new providers by implementing _call_llm().
    """

    def __init__(self):
        self.api_key = _GEMINI_API_KEY
        self.model_name = _DEFAULT_MODEL
        self._client = None

        if GENAI_AVAILABLE and self.api_key:
            try:
                self._client = _genai.Client(api_key=self.api_key)
                logger.info("LLM service initialized with model '%s'.", self.model_name)
            except Exception as exc:
                logger.warning("Failed to initialize google-genai client: %s", exc)
                self._client = None
        elif not self.api_key:
            logger.info("GEMINI_API_KEY not set — LLM stages will be skipped.")
        elif not GENAI_AVAILABLE:
            logger.info("google-genai not installed — LLM stages will be skipped.")

    @property
    def is_available(self) -> bool:
        """Return True if the LLM client is ready."""
        return self._client is not None

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ) -> Optional[str]:
        """
        Call the LLM and return the raw text response.
        Returns None on failure.
        """
        if not self.is_available:
            return None

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Model cascade to survive per-model free tier quota limits
        model_candidates = [self.model_name]
        for fallback in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-3.6-flash"]:
            if fallback not in model_candidates:
                model_candidates.append(fallback)

        for model_to_try in model_candidates:
            for attempt in range(max_retries + 1):
                try:
                    response = self._client.models.generate_content(  # type: ignore[union-attr]
                        model=model_to_try,
                        contents=combined_prompt,
                        config={
                            "automatic_function_calling": {"disable": True},
                        },
                    )
                    return response.text
                except Exception as exc:
                    exc_str = str(exc)
                    if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                        logger.warning(
                            "Model %s hit quota limit (429). Trying fallback model...",
                            model_to_try,
                        )
                        break  # Break retry loop to immediately try next model candidate
                    logger.warning(
                        "LLM call failed for %s (attempt %d/%d): %s",
                        model_to_try,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay)

        return None

    def evaluate_candidate(
        self,
        candidate: dict,
        context: str,
        papers: list[dict],
    ) -> Optional[dict]:
        """
        CREF-style evaluation: score a candidate concept-pair on 4 dimensions.

        Returns a dict matching the LLM evaluation output schema, or None if
        the LLM is unavailable or the response cannot be parsed.
        """
        if not self.is_available:
            return None

        abstracts = "\n\n".join(
            f"[{p['paper_id']}] {p.get('title', p['filename'])}: "
            f"{p.get('abstract', '')[:400]}"
            for p in papers[:8]
        )

        user_prompt = _CREF_USER_TEMPLATE.format(
            connection=candidate["connection"],
            context=context,
            abstracts=abstracts,
        )

        raw = self._call_llm(_CREF_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            logger.warning("CREF evaluation returned no response for '%s'.", candidate["connection"])
            return None

        parsed = extract_json_from_llm_response(raw)
        if parsed is None:
            logger.warning("Could not parse CREF JSON for '%s'. Raw: %s", candidate["connection"], raw[:200])
            return None

        return {
            "candidate": candidate["connection"],
            "novelty": int(max(1, min(5, parsed.get("novelty", 3)))),
            "impact": int(max(1, min(5, parsed.get("impact", 3)))),
            "plausibility": int(max(1, min(5, parsed.get("plausibility", 3)))),
            "interdisciplinarity": int(max(1, min(5, parsed.get("interdisciplinarity", 3)))),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
        }

    def generate_insight(
        self,
        candidate: dict,
        evaluation: dict,
        context: str,
        papers: list[dict],
    ) -> Optional[dict]:
        """
        GIC-style insight generation: produce a structured research insight report.

        Returns a dict matching the GIC insight schema, or None if the LLM is
        unavailable or the response cannot be parsed.
        """
        if not self.is_available:
            return None

        concept_a = candidate.get("concept_a", "")
        concept_b = candidate.get("concept_b", "")
        supporting_ids = [
            p["paper_id"]
            for p in papers
            if concept_a in p.get("concepts", []) or concept_b in p.get("concepts", [])
        ][:6]

        abstracts = "\n\n".join(
            f"[{p['paper_id']}] {p.get('title', p['filename'])}: "
            f"{p.get('abstract', '')[:400]}"
            for p in papers[:8]
        )

        import json as _json

        user_prompt = _GIC_USER_TEMPLATE.format(
            connection=candidate["connection"],
            concept_a=concept_a,
            concept_b=concept_b,
            novelty=evaluation.get("novelty", "N/A"),
            impact=evaluation.get("impact", "N/A"),
            plausibility=evaluation.get("plausibility", "N/A"),
            interdisciplinarity=evaluation.get("interdisciplinarity", "N/A"),
            reasoning=evaluation.get("reasoning", ""),
            context=context,
            abstracts=abstracts,
            supporting_paper_ids=", ".join(supporting_ids) if supporting_ids else "None",
            supporting_paper_ids_json=_json.dumps(supporting_ids),
        )

        raw = self._call_llm(_GIC_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            logger.warning("GIC insight generation returned no response.")
            return None

        parsed = extract_json_from_llm_response(raw)
        if parsed is None:
            logger.warning("Could not parse GIC JSON. Raw: %s", raw[:200])
            return None

        return {
            "connection": candidate["connection"],
            "research_direction": str(parsed.get("research_direction", "Research direction not generated.")),
            "explanation": str(parsed.get("explanation", "Explanation not generated.")),
            "research_questions": list(parsed.get("research_questions", [])),
            "hypothesis": str(parsed.get("hypothesis", "Hypothesis not generated.")),
            "supporting_papers": list(parsed.get("supporting_papers", supporting_ids)),
            "limitations": str(parsed.get("limitations", "Results are exploratory and based on a small set of uploaded papers. No formal validation has been performed.")),
        }


def llm_status_label(service: Optional[LLMService] = None) -> str:
    """Return a human-readable label for the current LLM status."""
    if not GENAI_AVAILABLE:
        return "LLM unavailable (google-genai not installed — run: pip install google-genai)"
    if not _GEMINI_API_KEY:
        return "LLM unavailable (GEMINI_API_KEY not set in .env)"
    if service is not None and not service.is_available:
        return "LLM unavailable (client initialization failed — check API key)"
    return f"LLM active ({_DEFAULT_MODEL})"
