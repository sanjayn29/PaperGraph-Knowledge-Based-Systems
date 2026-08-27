"""
history_service.py
──────────────────
Persist and retrieve analysis results as JSON files.

Storage location: data/history/analysis_<YYYYMMDD_HHMMSS>.json
(relative to the project root — resolved at runtime)

IMPORTANT: full PDF text and binary PDF data are NEVER stored.
Only metadata, extracted concepts, candidate scores, and the
generated insight are persisted.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.json_utils import safe_json_dump, safe_json_load

logger = logging.getLogger(__name__)

# Resolve the history directory relative to this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HISTORY_DIR = _PROJECT_ROOT / "data" / "history"


def _ensure_history_dir() -> Path:
    """Create data/history/ if it doesn't already exist."""
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return _HISTORY_DIR


def save_analysis(result: dict) -> Optional[str]:
    """
    Save a completed analysis result to a JSON history file.

    Strips 'full_text' from each paper before saving to avoid persisting
    extracted text beyond the analysis session.

    Parameters
    ----------
    result : the complete analysis result dict from analysis_pipeline.py

    Returns
    -------
    The analysis_id string (e.g. "analysis_20260821_101530") or None on failure.
    """
    history_dir = _ensure_history_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    analysis_id = f"analysis_{timestamp}"

    # Sanitize: remove full_text from papers
    clean_papers = []
    for paper in result.get("papers", []):
        clean_paper = {k: v for k, v in paper.items() if k != "full_text"}
        clean_papers.append(clean_paper)

    history_record = {
        "analysis_id": analysis_id,
        "created_at": datetime.now().isoformat(),
        "paper_count": result.get("paper_count", len(clean_papers)),
        "papers": clean_papers,
        "concept_count": result.get("concept_count", 0),
        "relationship_count": result.get("relationship_count", 0),
        "graph_summary": result.get("graph_summary", {}),
        "temporal_summary": result.get("temporal_summary", {}),
        "analysis_mode": result.get("analysis_mode", "Lightweight Graph Analysis"),
        "candidates": result.get("candidates", []),
        "final_result": result.get("final_result", {}),
        "llm_available": result.get("llm_available", False),
        "evaluation": result.get("evaluation", {}),
        "warnings": result.get("warnings", []),
    }

    filepath = history_dir / f"{analysis_id}.json"
    success = safe_json_dump(history_record, filepath)

    if success:
        logger.info("Analysis saved: %s", filepath)
        return analysis_id
    else:
        logger.error("Failed to save analysis to %s", filepath)
        return None


def list_analyses() -> list[dict]:
    """
    Return a list of summary dicts for all stored analyses, sorted newest first.

    Each summary contains: analysis_id, created_at, paper_count,
    concept_count, relationship_count, top_connection.
    """
    history_dir = _ensure_history_dir()
    summaries: list[dict] = []

    for filepath in sorted(history_dir.glob("analysis_*.json"), reverse=True):
        record = safe_json_load(filepath)
        if record is None:
            logger.warning("Skipping unreadable history file: %s", filepath)
            continue

        # Extract the top connection from final_result or first candidate
        top_connection = ""
        final = record.get("final_result", {})
        if final:
            top_connection = final.get("connection", "")
        if not top_connection and record.get("candidates"):
            top_connection = record["candidates"][0].get("connection", "")

        summaries.append(
            {
                "analysis_id": record.get("analysis_id", filepath.stem),
                "created_at": record.get("created_at", ""),
                "paper_count": record.get("paper_count", 0),
                "concept_count": record.get("concept_count", 0),
                "relationship_count": record.get("relationship_count", 0),
                "top_connection": top_connection,
                "analysis_mode": record.get("analysis_mode", ""),
                "llm_available": record.get("llm_available", False),
                "filepath": str(filepath),
            }
        )

    return summaries


def load_analysis(analysis_id: str) -> Optional[dict]:
    """
    Load a complete analysis record by its analysis_id.

    Returns the full dict or None if not found / unreadable.
    """
    history_dir = _ensure_history_dir()
    filepath = history_dir / f"{analysis_id}.json"

    if not filepath.exists():
        # Try a partial match (the caller might pass a display-safe id)
        matches = list(history_dir.glob(f"{analysis_id}*.json"))
        if matches:
            filepath = matches[0]
        else:
            logger.warning("History file not found for id: %s", analysis_id)
            return None

    return safe_json_load(filepath)


def delete_analysis(analysis_id: str) -> bool:
    """
    Delete a history file by analysis_id.
    Returns True if deleted, False if not found or error.
    """
    history_dir = _ensure_history_dir()
    filepath = history_dir / f"{analysis_id}.json"
    try:
        filepath.unlink()
        logger.info("Deleted history file: %s", filepath)
        return True
    except FileNotFoundError:
        logger.warning("History file not found for deletion: %s", filepath)
        return False
    except OSError as exc:
        logger.error("Failed to delete history file %s: %s", filepath, exc)
        return False


def get_history_stats() -> dict:
    """Return aggregate stats over all history files."""
    analyses = list_analyses()
    return {
        "total_analyses": len(analyses),
        "total_papers_analyzed": sum(a.get("paper_count", 0) for a in analyses),
        "total_concepts_found": sum(a.get("concept_count", 0) for a in analyses),
    }
