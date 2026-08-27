"""
temporal_graph.py
─────────────────
Temporal event representation for PaperGraph's SE-TGN component.

Each concept co-occurrence that appeared in a specific paper/year is
recorded as a TemporalEvent. The TemporalGraph collects these events
chronologically and supports the temporal train/val/test split needed
for meaningful future-link evaluation.

Role in the pipeline
--------------------
- Built from paper dicts after concept extraction + embedding steps.
- Passed to se_tgn.py for model training and link prediction.
- Kept alongside the static NetworkX graph (which is used for
  visualization and graph-metric scoring).

Key data structures
-------------------
TemporalEvent:
    (source_concept, target_concept, paper_id, timestamp, paper_embedding)

TemporalGraph:
    Ordered list of TemporalEvents + helpers for temporal split /
    unique concept enumeration / event filtering by time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TemporalEvent
# ─────────────────────────────────────────────────────────────

@dataclass
class TemporalEvent:
    """
    A single directed concept co-occurrence event.

    Attributes
    ----------
    source_concept  : canonical concept string (lexicographically smaller)
    target_concept  : canonical concept string (lexicographically larger)
    paper_id        : originating paper identifier
    timestamp       : publication year (int); estimated years accepted
    paper_embedding : 384-dim sentence-transformer embedding of the paper;
                      used as the event's semantic feature in SE-TGN messages.
    year_estimated  : True if the year was not reliably extracted from the PDF
    """

    source_concept: str
    target_concept: str
    paper_id: str
    timestamp: int
    paper_embedding: np.ndarray
    year_estimated: bool = False

    def __post_init__(self) -> None:
        # Canonical ordering: always (smaller, larger) so (A,B) == (B,A)
        if self.source_concept > self.target_concept:
            self.source_concept, self.target_concept = (
                self.target_concept,
                self.source_concept,
            )

    @property
    def pair(self) -> tuple[str, str]:
        """Return the canonical (source, target) pair tuple."""
        return (self.source_concept, self.target_concept)


# ─────────────────────────────────────────────────────────────
# TemporalGraph
# ─────────────────────────────────────────────────────────────

class TemporalGraph:
    """
    Ordered collection of temporal concept co-occurrence events.

    The list is kept sorted by timestamp after every bulk add so
    SE-TGN can process events in chronological order without leaking
    future information into past message-passing steps.
    """

    def __init__(self) -> None:
        self._events: list[TemporalEvent] = []
        self._sorted: bool = True  # tracks whether a sort is needed

    # ── Mutation ─────────────────────────────────────────────

    def add_event(self, event: TemporalEvent) -> None:
        """Append a single event. Call sort() after bulk adds."""
        self._events.append(event)
        self._sorted = False

    def sort(self) -> None:
        """Sort events chronologically (stable sort by timestamp)."""
        self._events.sort(key=lambda e: e.timestamp)
        self._sorted = True

    # ── Read-only access ─────────────────────────────────────

    @property
    def events(self) -> list[TemporalEvent]:
        """Return all events (sorted if sort() has been called)."""
        return self._events

    def __len__(self) -> int:
        return len(self._events)

    def is_empty(self) -> bool:
        return len(self._events) == 0

    # ── Concept enumeration ──────────────────────────────────

    def unique_concepts(self) -> list[str]:
        """Return a sorted list of all unique concept strings."""
        seen: set[str] = set()
        for e in self._events:
            seen.add(e.source_concept)
            seen.add(e.target_concept)
        return sorted(seen)

    def concept_index(self) -> dict[str, int]:
        """Return {concept: integer_index} for all unique concepts."""
        return {c: i for i, c in enumerate(self.unique_concepts())}

    # ── Temporal filtering ───────────────────────────────────

    def events_before(self, timestamp: int) -> list[TemporalEvent]:
        """Return all events with timestamp strictly less than `timestamp`."""
        return [e for e in self._events if e.timestamp < timestamp]

    def events_in_range(self, t_start: int, t_end: int) -> list[TemporalEvent]:
        """Return events where t_start <= timestamp < t_end."""
        return [e for e in self._events if t_start <= e.timestamp < t_end]

    # ── Temporal statistics ──────────────────────────────────

    def year_range(self) -> tuple[Optional[int], Optional[int]]:
        """Return (min_year, max_year) of all events, or (None, None)."""
        if not self._events:
            return None, None
        years = [e.timestamp for e in self._events]
        return min(years), max(years)

    def unique_years(self) -> list[int]:
        """Return sorted list of unique years in the event list."""
        return sorted(set(e.timestamp for e in self._events))

    def positive_pairs(self) -> set[tuple[str, str]]:
        """Return the set of all (source, target) concept pairs that co-occur."""
        return {e.pair for e in self._events}

    # ── Temporal split ───────────────────────────────────────

    def temporal_split(
        self,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
    ) -> tuple[list[TemporalEvent], list[TemporalEvent], list[TemporalEvent]]:
        """
        Split events into train / validation / test sets based on timestamp.

        Uses a year-boundary split so no future paper's events leak into
        the training set — matching the base paper's evaluation protocol.

        Parameters
        ----------
        val_ratio  : fraction of events (by time) to hold out for validation
        test_ratio : fraction of events (by time) to hold out for testing

        Returns
        -------
        (train_events, val_events, test_events)
        train_events : earliest events (1 - val_ratio - test_ratio of total)
        val_events   : middle events
        test_events  : latest events

        Notes
        -----
        - If there are fewer than 4 distinct years, returns (all, [], [])
          and the caller must report "Insufficient temporal data."
        - Splits are year-aligned, not sample-aligned, to avoid leaking.
        """
        if not self._events:
            return [], [], []

        years = self.unique_years()
        if len(years) < 4:
            logger.info(
                "TemporalGraph has only %d distinct year(s) — "
                "temporal split not possible; using all events for training.",
                len(years),
            )
            return list(self._events), [], []

        n = len(years)
        train_end_idx = max(1, int(n * (1.0 - val_ratio - test_ratio)))
        val_end_idx = max(train_end_idx + 1, int(n * (1.0 - test_ratio)))

        train_year = years[train_end_idx - 1]
        val_year = years[val_end_idx - 1]

        train = [e for e in self._events if e.timestamp <= train_year]
        val = [e for e in self._events if train_year < e.timestamp <= val_year]
        test = [e for e in self._events if e.timestamp > val_year]

        logger.info(
            "Temporal split: train=%d events (≤%d), val=%d events (%d–%d), test=%d events (>%d)",
            len(train), train_year,
            len(val), train_year + 1, val_year,
            len(test), val_year,
        )
        return train, val, test

    # ── Summary ──────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary dict for display in the UI."""
        y_min, y_max = self.year_range()
        return {
            "total_events": len(self._events),
            "unique_concepts": len(self.unique_concepts()),
            "unique_pairs": len(self.positive_pairs()),
            "year_min": y_min,
            "year_max": y_max,
            "unique_years": len(self.unique_years()),
        }


# ─────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────

_FALLBACK_YEAR = 2020  # Used when a paper has no year; displayed with warning


def build_temporal_events(
    papers: list[dict],
    concept_embeddings: dict[str, np.ndarray],
    paper_embeddings: Optional[dict[str, np.ndarray]] = None,
) -> tuple[TemporalGraph, list[str]]:
    """
    Build a TemporalGraph from the processed paper list.

    For each paper, a TemporalEvent is created for every unique pair of
    concepts that co-occur in that paper. The event's timestamp is the
    paper's publication year (or a fallback value with a warning).

    The paper's semantic embedding is either taken from `paper_embeddings`
    (if provided and populated with paper-level embeddings) or approximated
    by averaging the embeddings of its concepts.

    Parameters
    ----------
    papers             : list of paper dicts (must have 'concepts', 'year', 'paper_id')
    concept_embeddings : {concept_str → np.ndarray(384)} from embeddings.py
    paper_embeddings   : optional {paper_id → np.ndarray(384)}; if None, approximated

    Returns
    -------
    (temporal_graph, warnings)
    temporal_graph : populated and sorted TemporalGraph
    warnings       : list of human-readable warning strings
    """
    tg = TemporalGraph()
    warnings: list[str] = []
    emb_dim = 384

    # Determine a fallback year from available paper years
    known_years = [p["year"] for p in papers if isinstance(p.get("year"), int)]
    fallback_year = int(round(sum(known_years) / len(known_years))) if known_years else _FALLBACK_YEAR

    for paper in papers:
        pid = paper["paper_id"]
        concepts = paper.get("concepts", [])
        year = paper.get("year")
        year_estimated = paper.get("year_estimated", False)

        if not isinstance(year, int) or year < 1900 or year > 2100:
            warnings.append(
                f"⚠️ Paper '{paper.get('title') or pid}' has no reliable year "
                f"— using estimated year {fallback_year}."
            )
            year = fallback_year
            year_estimated = True

        # Compute paper embedding: average of its concept embeddings
        if paper_embeddings and pid in paper_embeddings:
            paper_emb = paper_embeddings[pid]
        else:
            concept_embs = [
                concept_embeddings[c]
                for c in concepts
                if c in concept_embeddings
            ]
            if concept_embs:
                paper_emb = np.mean(concept_embs, axis=0).astype(np.float32)
            else:
                paper_emb = np.zeros(emb_dim, dtype=np.float32)

        # Create one TemporalEvent per unique concept pair in this paper
        unique_concepts = list(dict.fromkeys(concepts))  # deduplicate, preserve order
        for c_a, c_b in combinations(unique_concepts, 2):
            event = TemporalEvent(
                source_concept=c_a,
                target_concept=c_b,
                paper_id=pid,
                timestamp=year,
                paper_embedding=paper_emb,
                year_estimated=year_estimated,
            )
            tg.add_event(event)

    tg.sort()

    logger.info(
        "TemporalGraph built: %d events, %d unique concepts, %d unique pairs, years %s–%s",
        len(tg),
        len(tg.unique_concepts()),
        len(tg.positive_pairs()),
        *tg.year_range(),
    )
    return tg, warnings
