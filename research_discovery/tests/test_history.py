"""
test_history.py
───────────────
Unit tests for services/history_service.py

Tests save/load/list/delete round-trips using a temporary directory.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_sample_result(analysis_id: str = "analysis_20260821_101530") -> dict:
    return {
        "analysis_id": analysis_id,
        "paper_count": 5,
        "papers": [
            {
                "paper_id": "paper_001",
                "filename": "paper1.pdf",
                "title": "GNN for Drug Discovery",
                "abstract": "This paper proposes...",
                "year": 2023,
                "authors": ["Alice"],
                "full_text": "LARGE TEXT THAT SHOULD NOT BE SAVED",
                "concepts": ["Graph Neural Network", "Drug Discovery"],
            }
        ],
        "concept_count": 10,
        "relationship_count": 15,
        "graph_summary": {"node_count": 10, "edge_count": 15, "density": 0.3},
        "analysis_mode": "Lightweight Graph Analysis",
        "candidates": [{"connection": "GNN + LLM", "candidate_score": 0.87}],
        "final_result": {
            "connection": "GNN + LLM",
            "novelty": 4,
            "impact": 5,
            "plausibility": 4,
            "interdisciplinarity": 5,
            "research_direction": "Potential integration of GNN and LLM.",
            "explanation": "These concepts appear across multiple papers...",
            "research_questions": ["Q1?", "Q2?", "Q3?"],
            "hypothesis": "GNN + LLM may yield improvements...",
            "supporting_papers": ["paper_001"],
            "limitations": "Exploratory analysis only.",
        },
        "llm_available": True,
        "warnings": [],
    }


class TestHistoryService:
    def test_save_and_load_roundtrip(self, tmp_path):
        """Save a result and verify it loads back correctly."""
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis, load_analysis

            result = _make_sample_result()
            analysis_id = save_analysis(result)

            assert analysis_id is not None
            assert analysis_id.startswith("analysis_")

            loaded = load_analysis(analysis_id)
            assert loaded is not None
            assert loaded["paper_count"] == 5
            assert loaded["concept_count"] == 10

    def test_full_text_not_persisted(self, tmp_path):
        """Verify full_text is stripped before saving."""
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis, load_analysis

            result = _make_sample_result()
            analysis_id = save_analysis(result)

            loaded = load_analysis(analysis_id)
            assert loaded is not None
            for paper in loaded.get("papers", []):
                assert "full_text" not in paper, "full_text should not be persisted"

    def test_list_analyses_returns_summaries(self, tmp_path):
        """Save two analyses and verify list_analyses returns both."""
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis, list_analyses
            import time

            save_analysis(_make_sample_result("analysis_20260821_100000"))
            time.sleep(0.01)
            save_analysis(_make_sample_result("analysis_20260821_100001"))

            analyses = list_analyses()
            assert len(analyses) == 2
            for a in analyses:
                assert "analysis_id" in a
                assert "paper_count" in a
                assert "top_connection" in a

    def test_list_analyses_sorted_newest_first(self, tmp_path):
        """Verify list_analyses returns newest first."""
        import time
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis, list_analyses

            save_analysis(_make_sample_result())
            time.sleep(0.05)
            save_analysis(_make_sample_result())

            analyses = list_analyses()
            if len(analyses) >= 2:
                assert analyses[0]["analysis_id"] >= analyses[1]["analysis_id"]

    def test_load_nonexistent_analysis_returns_none(self, tmp_path):
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import load_analysis

            result = load_analysis("nonexistent_analysis_id")
            assert result is None

    def test_invalid_analysis_id_is_rejected(self, tmp_path):
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import delete_analysis, load_analysis

            assert load_analysis("..\\outside") is None
            assert delete_analysis("..\\outside") is False

    def test_empty_history_returns_empty_list(self, tmp_path):
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import list_analyses

            analyses = list_analyses()
            assert analyses == []

    def test_delete_analysis(self, tmp_path):
        """Save then delete an analysis."""
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis, load_analysis, delete_analysis

            result = _make_sample_result()
            analysis_id = save_analysis(result)
            assert analysis_id is not None

            success = delete_analysis(analysis_id)
            assert success is True

            loaded = load_analysis(analysis_id)
            assert loaded is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import delete_analysis

            result = delete_analysis("analysis_does_not_exist")
            assert result is False

    def test_history_stats(self, tmp_path):
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis, get_history_stats

            save_analysis(_make_sample_result())
            save_analysis(_make_sample_result())

            stats = get_history_stats()
            assert stats["total_analyses"] == 2
            assert stats["total_papers_analyzed"] == 10  # 5 papers × 2 analyses

    def test_save_analysis_avoids_timestamp_collision(self, tmp_path):
        with patch("services.history_service._HISTORY_DIR", tmp_path), patch(
            "services.history_service.datetime"
        ) as datetime_mock:
            datetime_mock.now.return_value = datetime(2026, 9, 6, 12, 0, 0, 123456)

            from services.history_service import save_analysis

            first_id = save_analysis(_make_sample_result())
            second_id = save_analysis(_make_sample_result())

            assert first_id != second_id
            assert len(list(tmp_path.glob("analysis_*.json"))) == 2

    def test_saved_json_is_valid(self, tmp_path):
        """The saved file should be valid JSON with expected keys."""
        with patch("services.history_service._HISTORY_DIR", tmp_path):
            from services.history_service import save_analysis

            result = _make_sample_result()
            analysis_id = save_analysis(result)

            filepath = tmp_path / f"{analysis_id}.json"
            assert filepath.exists()

            with open(filepath, encoding="utf-8") as fh:
                data = json.load(fh)

            assert "analysis_id" in data
            assert "created_at" in data
            assert "paper_count" in data
            assert "final_result" in data
            assert "candidates" in data
