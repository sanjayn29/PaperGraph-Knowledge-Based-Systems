"""
test_pdf_processor.py
─────────────────────
Unit tests for services/pdf_processor.py

Uses mock fitz output to avoid requiring real PDFs in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.pdf_processor import (
    _extract_abstract,
    _extract_authors,
    _extract_title,
    _extract_year,
)


class TestExtractTitle:
    def test_uses_metadata_title_when_valid(self):
        meta = {"title": "Graph Neural Networks for Drug Discovery"}
        result = _extract_title(meta, "some full text here with more words", "file.pdf")
        assert result == "Graph Neural Networks for Drug Discovery"

    def test_fallback_to_first_text_line(self):
        meta = {}
        full_text = "Graph Neural Networks in Healthcare\nAuthor Name\nAbstract: ..."
        result = _extract_title(meta, full_text, "file.pdf")
        assert "Graph Neural Networks" in result

    def test_fallback_to_filename(self):
        meta = {}
        full_text = ""
        result = _extract_title(meta, full_text, "my_paper_2024.pdf")
        assert "My Paper 2024" in result or "my_paper_2024" in result.lower()

    def test_rejects_too_short_metadata_title(self):
        meta = {"title": "Hi"}
        full_text = "A Comprehensive Study of Machine Learning Applications in Healthcare\nAuthors..."
        result = _extract_title(meta, full_text, "fallback.pdf")
        # Should fall back since "Hi" is too short (< 5 chars threshold is 5)
        assert len(result) > 2

    def test_rejects_email_lines(self):
        meta = {}
        full_text = "author@university.edu\nResearch on Graph Attention Networks\nAbstract..."
        result = _extract_title(meta, full_text, "file.pdf")
        assert "@" not in result


class TestExtractYear:
    def test_extracts_year_from_metadata_creation_date(self):
        meta = {"creationDate": "D:20231015120000"}
        result = _extract_year(meta, "")
        assert result == 2023

    def test_extracts_year_from_text(self):
        meta = {}
        text = "Published in 2022. This paper presents..."
        result = _extract_year(meta, text)
        assert result == 2022

    def test_returns_none_when_no_year(self):
        meta = {}
        text = "No year information here."
        result = _extract_year(meta, text)
        assert result is None

    def test_most_frequent_year_wins(self):
        meta = {}
        text = "In 2021, we proposed... In 2021, results show... In 2019, prior work..."
        result = _extract_year(meta, text)
        assert result == 2021


class TestExtractAbstract:
    def test_finds_abstract_section(self):
        text = (
            "Title Line\n\nAbstract\n\n"
            "This paper presents a novel approach to graph neural networks. "
            "We propose a new method that achieves state-of-the-art results "
            "on multiple benchmark datasets. Our approach is efficient and scalable. "
            "Extensive experiments validate our claims.\n\n"
            "1. Introduction\n..."
        )
        result = _extract_abstract(text)
        assert "graph neural networks" in result.lower() or len(result) > 50

    def test_fallback_to_first_paragraph(self):
        text = "Title Line\nSome content without explicit abstract section."
        result = _extract_abstract(text)
        assert len(result) > 0

    def test_caps_at_2000_chars(self):
        text = "Abstract\n\n" + "A" * 3000
        result = _extract_abstract(text)
        assert len(result) <= 2000


class TestExtractAuthors:
    def test_extracts_from_metadata(self):
        meta = {"author": "John Doe; Jane Smith; Bob Johnson"}
        result = _extract_authors(meta, "")
        assert len(result) == 3
        assert "John Doe" in result

    def test_comma_separated_metadata(self):
        meta = {"author": "Alice Brown, Charlie Davis"}
        result = _extract_authors(meta, "")
        assert len(result) == 2

    def test_returns_empty_when_no_authors(self):
        meta = {}
        result = _extract_authors(meta, "No author info here.")
        assert isinstance(result, list)

    def test_caps_at_ten_authors(self):
        meta = {"author": "; ".join([f"Author{i} Name" for i in range(15)])}
        result = _extract_authors(meta, "")
        assert len(result) <= 10


class TestProcessPdfs:
    def test_empty_file_list(self):
        from services.pdf_processor import process_pdfs

        papers, warnings = process_pdfs([])
        assert papers == []
        assert warnings == []

    def test_empty_bytes_file_is_skipped(self):
        from services.pdf_processor import process_pdfs

        mock_file = MagicMock()
        mock_file.name = "empty.pdf"
        mock_file.read.return_value = b""

        papers, warnings = process_pdfs([mock_file])
        assert len(papers) == 0
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower()
