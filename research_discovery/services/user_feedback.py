"""
user_feedback.py
────────────────
Human-in-the-Loop Feedback & Personalized Ranking Engine for PaperGraph.

Allows researchers to interactively rate recommendations (👍 Interesting, 👎 Not Relevant,
⭐ Highly Relevant, 🔖 Saved, etc.) and personalizes future candidate rankings without
retraining the underlying SE-TGN model.

Personalization Formula:
PersonalizedScore =
    0.70 * OriginalCandidateScore
  + 0.15 * UserPreferenceScore
  + 0.15 * FeedbackSimilarityScore
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FEEDBACK_FILE = _DATA_DIR / "user_feedback.json"

WEIGHT_ORIGINAL = 0.70
WEIGHT_PREFERENCE = 0.15
WEIGHT_SIMILARITY = 0.15

FEEDBACK_WEIGHTS: dict[str, float] = {
    "highly_relevant": 1.0,
    "interesting": 0.6,
    "saved": 0.4,
    "already_known": -0.2,
    "not_my_area": -0.5,
    "not_relevant": -0.8,
}


def _empty_profile() -> dict:
    """Return an empty feedback profile."""
    return {
        "preferred_concepts": {},
        "preferred_domains": {},
        "disliked_concepts": {},
        "feedback_history": [],
        "last_updated": datetime.now().isoformat(),
    }


def _load_feedback_store() -> dict:
    """Load the feedback store, preserving legacy top-level data."""
    if not _FEEDBACK_FILE.exists():
        return _empty_profile()

    try:
        with open(_FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load user feedback profile: %s", exc)
        return _empty_profile()


def _load_user_profile(context_id: Optional[str] = None) -> dict:
    """Load a context-scoped profile without assigning legacy data to it."""
    store = _load_feedback_store()
    if context_id:
        contexts = store.get("contexts", {})
        profile = contexts.get(context_id)
        return profile if isinstance(profile, dict) else _empty_profile()
    return store


def _save_user_profile(profile: dict, context_id: Optional[str] = None) -> None:
    """Save a legacy or context-scoped profile to data/user_feedback.json."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        profile["last_updated"] = datetime.now().isoformat()
        if context_id:
            store = _load_feedback_store()
            contexts = store.setdefault("contexts", {})
            contexts[context_id] = profile
            profile = store
        with open(_FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to save user feedback profile: %s", exc)


def record_feedback(
    candidate_id: str,
    concept_a: str,
    concept_b: str,
    feedback_type: str,
    domain_a: str = "",
    domain_b: str = "",
    note: str = "",
    context_id: Optional[str] = None,
) -> dict:
    """
    Record user feedback for a candidate pair and update the user preference profile.

    Parameters
    ----------
    candidate_id  : unique candidate identifier
    concept_a     : first concept
    concept_b     : second concept
    feedback_type : one of ('highly_relevant', 'interesting', 'saved', 'already_known', 'not_my_area', 'not_relevant')
    domain_a      : optional domain for concept_a
    domain_b      : optional domain for concept_b
    note          : optional user note
    context_id    : analysis/session context for isolating feedback

    Returns
    -------
    Updated profile dictionary.
    """
    profile = _load_user_profile(context_id)
    weight = FEEDBACK_WEIGHTS.get(feedback_type, 0.0)

    # 1. Update concept weights
    for c in (concept_a, concept_b):
        if not c:
            continue
        if weight > 0:
            profile["preferred_concepts"][c] = profile["preferred_concepts"].get(c, 0.0) + weight
        elif weight < 0:
            profile["disliked_concepts"][c] = profile["disliked_concepts"].get(c, 0.0) + abs(weight)

    # 2. Update domain weights
    for d in (domain_a, domain_b):
        if d and d != "Unknown":
            profile["preferred_domains"][d] = profile["preferred_domains"].get(d, 0.0) + weight

    # 3. Add to feedback history
    profile["feedback_history"].append({
        "candidate_id": candidate_id,
        "concept_a": concept_a,
        "concept_b": concept_b,
        "feedback_type": feedback_type,
        "weight": weight,
        "domain_a": domain_a,
        "domain_b": domain_b,
        "note": note,
        "timestamp": datetime.now().isoformat(),
        "context_id": context_id,
    })

    _save_user_profile(profile, context_id)
    return profile


def clear_user_profile(context_id: Optional[str] = None) -> None:
    """Reset legacy or one context-scoped feedback profile."""
    _save_user_profile(_empty_profile(), context_id)


def get_user_profile_summary(context_id: Optional[str] = None) -> dict:
    """Get context-scoped profile statistics for display in UI."""
    profile = _load_user_profile(context_id)
    return {
        "total_feedbacks": len(profile.get("feedback_history", [])),
        "preferred_concepts": profile.get("preferred_concepts", {}),
        "preferred_domains": profile.get("preferred_domains", {}),
        "disliked_concepts": profile.get("disliked_concepts", {}),
    }


def compute_personalized_rankings(
    candidates: list[dict],
    concept_domains: Optional[dict[str, dict]] = None,
    concept_embeddings: Optional[dict[str, np.ndarray]] = None,
    custom_profile: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> list[dict]:
    """
    Apply human-in-the-loop personalized reranking to candidates.

    Returns
    -------
    List of candidates augmented with 'personalized_score', 'original_rank',
    'personalized_rank', 'rank_delta', and 'personalization_reason'.
    """
    profile = custom_profile or _load_user_profile(context_id)
    concept_domains = concept_domains or {}

    pref_concepts = profile.get("preferred_concepts", {})
    disl_concepts = profile.get("disliked_concepts", {})
    pref_domains = profile.get("preferred_domains", {})

    has_feedback = bool(pref_concepts or disl_concepts or pref_domains)

    ranked_candidates: list[dict] = []

    for orig_idx, cand in enumerate(candidates):
        ca = cand.get("concept_a", "")
        cb = cand.get("concept_b", "")
        dom_a = concept_domains.get(ca, {}).get("domain", "")
        dom_b = concept_domains.get(cb, {}).get("domain", "")

        orig_score = float(cand.get("candidate_score", 0.5))

        if not has_feedback:
            # No feedback yet — original rank equals personalized rank
            cand_copy = dict(cand)
            cand_copy["original_rank"] = orig_idx + 1
            cand_copy["personalized_rank"] = orig_idx + 1
            cand_copy["personalized_score"] = orig_score
            cand_copy["rank_delta"] = 0
            cand_copy["personalization_reason"] = "Standard base-paper ranking (No feedback recorded yet)"
            ranked_candidates.append(cand_copy)
            continue

        # 1. User Preference Score based on explicit concept & domain boosts
        pref_score = 0.5  # Neutral baseline
        reasons: list[str] = []

        # Check positive concept boosts
        boost_a = pref_concepts.get(ca, 0.0)
        boost_b = pref_concepts.get(cb, 0.0)
        if boost_a > 0 or boost_b > 0:
            pref_score += min(0.4, 0.2 * (boost_a + boost_b))
            reasons.append("Contains previously favored concept")

        # Check negative concept penalties
        pen_a = disl_concepts.get(ca, 0.0)
        pen_b = disl_concepts.get(cb, 0.0)
        if pen_a > 0 or pen_b > 0:
            pref_score -= min(0.5, 0.25 * (pen_a + pen_b))
            reasons.append("Contains previously downvoted concept")

        # Check domain alignment
        dom_boost_a = pref_domains.get(dom_a, 0.0)
        dom_boost_b = pref_domains.get(dom_b, 0.0)
        if dom_boost_a != 0 or dom_boost_b != 0:
            pref_score += np.clip(0.1 * (dom_boost_a + dom_boost_b), -0.3, 0.3)
            if dom_boost_a > 0 or dom_boost_b > 0:
                reasons.append("Aligned with preferred research domain")

        pref_score = float(np.clip(pref_score, 0.0, 1.0))

        # 2. Feedback Similarity Score: semantic proximity to favored concepts
        sim_score = 0.5
        if concept_embeddings and pref_concepts:
            max_sim = 0.0
            for favored_concept in pref_concepts:
                if favored_concept in concept_embeddings:
                    fav_emb = concept_embeddings[favored_concept]
                    norm_f = np.linalg.norm(fav_emb)
                    for test_concept in (ca, cb):
                        if test_concept in concept_embeddings and norm_f > 1e-8:
                            t_emb = concept_embeddings[test_concept]
                            norm_t = np.linalg.norm(t_emb)
                            if norm_t > 1e-8:
                                cos_sim = float(np.dot(fav_emb, t_emb) / (norm_f * norm_t))
                                if cos_sim > max_sim:
                                    max_sim = cos_sim
            if max_sim > 0.65:
                sim_score = max_sim
                reasons.append(f"Semantically close to favored topic ({max_sim:.2f})")

        # 3. Combine scores
        personalized_score = (
            WEIGHT_ORIGINAL * orig_score
            + WEIGHT_PREFERENCE * pref_score
            + WEIGHT_SIMILARITY * sim_score
        )
        personalized_score = round(float(np.clip(personalized_score, 0.0, 1.0)), 4)

        reason_str = "; ".join(reasons) if reasons else "Aligned with research interests"

        cand_copy = dict(cand)
        cand_copy["original_rank"] = orig_idx + 1
        cand_copy["personalized_score"] = personalized_score
        cand_copy["personalization_reason"] = reason_str
        ranked_candidates.append(cand_copy)

    # Sort by personalized_score descending
    ranked_candidates.sort(key=lambda x: x["personalized_score"], reverse=True)

    # Assign personalized rank and calculate delta
    for p_idx, cand in enumerate(ranked_candidates):
        cand["personalized_rank"] = p_idx + 1
        cand["rank_delta"] = cand["original_rank"] - cand["personalized_rank"]  # Positive = moved up

    return ranked_candidates
