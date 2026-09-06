"""Focused tests for chronological evaluation orchestration."""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import numpy as np

from services.analysis_pipeline import _run_evaluation
from services.temporal_graph import TemporalEvent, TemporalGraph


def _make_four_year_graph() -> TemporalGraph:
    temporal_graph = TemporalGraph()
    pairs = [("A", "B"), ("A", "C"), ("A", "D")]
    for year in range(2020, 2024):
        for source, target in pairs:
            temporal_graph.add_event(
                TemporalEvent(source, target, f"paper_{year}", year, np.zeros(384))
            )
    temporal_graph.sort()
    return temporal_graph


def test_evaluation_trains_setgn_on_train_events_only_and_returns_metrics():
    temporal_graph = _make_four_year_graph()
    all_pairs = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]
    captured = {}

    def fake_compute_setgn_scores(*args, **kwargs):
        captured["training_events"] = kwargs["training_events"]
        captured["concept_names"] = kwargs["concept_names"]
        return {pair: 0.5 for pair in all_pairs}

    with patch(
        "services.se_tgn.compute_setgn_scores", side_effect=fake_compute_setgn_scores
    ):
        result = _run_evaluation(
            temporal_graph=temporal_graph,
            candidates=[],
            setgn_scores={pair: 0.1 for pair in all_pairs},
            gnn_scores=None,
            concept_embeddings={},
            G=nx.Graph(),
        )

    assert [event.timestamp for event in captured["training_events"]] == [
        2020,
        2020,
        2020,
        2021,
        2021,
        2021,
    ]
    assert 2023 not in [event.timestamp for event in captured["training_events"]]
    assert captured["concept_names"] == ["A", "B", "C", "D"]
    assert result["sufficient_data"] is True
    assert result["test_events"] == 3
    assert result["baselines"]["SE-TGN"]["sufficient_data"] is True


def test_evaluation_ablations_use_same_test_universe():
    temporal_graph = _make_four_year_graph()
    all_pairs = [
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    ]
    embeddings = {
        concept: np.array([float(index + 1), 1.0])
        for index, concept in enumerate(("A", "B", "C", "D"))
    }

    def fake_compute_setgn_scores(*args, **kwargs):
        return {pair: 0.5 for pair in all_pairs}

    with patch(
        "services.se_tgn.compute_setgn_scores", side_effect=fake_compute_setgn_scores
    ):
        result = _run_evaluation(
            temporal_graph=temporal_graph,
            candidates=[],
            setgn_scores=None,
            gnn_scores=None,
            concept_embeddings=embeddings,
            G=nx.Graph(),
        )

    ablations = result["ablations"]
    assert set(ablations) == {"full", "graph_only", "semantic_only"}
    assert all(metrics["sufficient_data"] for metrics in ablations.values())
    assert {metrics["n_positive"] for metrics in ablations.values()} == {3}
    assert {metrics["n_negative"] for metrics in ablations.values()} == {3}
    assert {metrics["n_total"] for metrics in ablations.values()} == {6}