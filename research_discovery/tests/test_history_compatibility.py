"""
test_history_compatibility.py
──────────────────────────────
Ensure new history serialization works and old legacy history files load seamlessly.
"""

import json
import pytest
from pathlib import Path
from services.history_service import save_analysis, load_analysis, list_analyses


def test_history_save_and_load_extensions(tmp_path, monkeypatch):
    test_history_dir = tmp_path / "history"
    test_history_dir.mkdir(parents=True)
    monkeypatch.setattr("services.history_service._HISTORY_DIR", test_history_dir)

    # Mock new rich analysis
    mock_result = {
        "papers": [{"paper_id": "p1", "title": "Paper 1", "concepts": ["GNN"]}],
        "concept_count": 1,
        "relationship_count": 0,
        "analysis_mode": "SE-TGN Temporal Analysis",
        "candidates": [{"concept_a": "GNN", "concept_b": "Bio", "candidate_score": 0.88}],
        "personalized_candidates": [{"concept_a": "GNN", "concept_b": "Bio", "personalized_score": 0.92}],
        "research_gaps": [{"concept_a": "GNN", "concept_b": "Bio", "gap_score": 0.85}],
        "cross_domain_discoveries": [{"concept_a": "GNN", "concept_b": "Bio", "cross_domain_score": 0.89}],
        "provenance": {"GNN + Bio": {"evidence_strength": "HIGH"}},
        "final_result": {"connection": "GNN + Bio"},
    }

    aid = save_analysis(mock_result)
    assert aid is not None

    loaded = load_analysis(aid)
    assert loaded is not None
    assert "research_gaps" in loaded
    assert "cross_domain_discoveries" in loaded
    assert "provenance" in loaded
    assert loaded["final_result"]["connection"] == "GNN + Bio"


def test_legacy_history_file_backward_compatibility(tmp_path, monkeypatch):
    test_history_dir = tmp_path / "history"
    test_history_dir.mkdir(parents=True)
    monkeypatch.setattr("services.history_service._HISTORY_DIR", test_history_dir)

    # Legacy JSON file without new fields
    legacy_record = {
        "analysis_id": "analysis_legacy_001",
        "created_at": "2025-01-01T00:00:00",
        "paper_count": 5,
        "papers": [{"paper_id": "p1", "title": "Old Paper"}],
        "concept_count": 20,
        "relationship_count": 15,
        "candidates": [{"concept_a": "A", "concept_b": "B", "candidate_score": 0.75}],
        "final_result": {"connection": "A + B"},
    }

    legacy_path = test_history_dir / "analysis_legacy_001.json"
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(legacy_record, f)

    # List & load legacy file without crashing
    summaries = list_analyses()
    assert len(summaries) == 1
    assert summaries[0]["analysis_id"] == "analysis_legacy_001"

    loaded = load_analysis("analysis_legacy_001")
    assert loaded is not None
    assert loaded.get("research_gaps", []) == []
    assert loaded.get("provenance", {}) == {}
