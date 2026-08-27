"""
pdf_processor.py
────────────────
Extracts structured metadata and full text from research PDFs using PyMuPDF.

Analogue of the "corpus ingestion" step in the base paper, but operating on
a tiny batch of 5–10 user-uploaded PDFs rather than a large historical corpus.

Output per paper follows the PaperGraph paper schema:
{
    "paper_id": "paper_001",
    "filename": "paper1.pdf",
    "title": "...",
    "abstract": "...",
    "year": 2024,             ← int or None
    "year_source": "metadata",← 'metadata'|'text_regex'|'estimated'|'manual'
    "year_estimated": False,  ← True if year is not reliably known
    "authors": [...],
    "full_text": "...",
    "concepts": []            ← populated later by concept_extractor
}
"""

from __future__ import annotations

import logging
import re
import tempfile
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Optional PyMuPDF import — fail gracefully with a clear message
# ─────────────────────────────────────────────────────────────
try:
    import pymupdf as fitz  # PyMuPDF (new import style; 'import fitz' is deprecated)

    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.warning(
        "PyMuPDF (fitz) is not installed. PDF processing will be unavailable. "
        "Install with: pip install PyMuPDF"
    )


# ─────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(19[89]\d|20[012]\d)\b")  # 1980–2029

_ABSTRACT_PATTERNS = [
    re.compile(r"(?i)abstract[\s\n:—–-]+(.{200,2000}?)(?:\n{2,}|\Z)", re.DOTALL),
    re.compile(r"(?i)abstract\n(.+?)(?:\n\n|\Z)", re.DOTALL),
]

_AUTHOR_LINE_RE = re.compile(
    r"(?i)(?:by|authors?)[:\s]+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?(?:\s*,\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)*)"
)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def process_pdf(
    file_bytes: bytes,
    filename: str,
    paper_index: int,
    year_override: Optional[int] = None,
) -> dict:
    """
    Extract structured data from a single PDF.

    Parameters
    ----------
    file_bytes    : bytes         — raw bytes of the PDF file
    filename      : str           — original filename (used as title fallback)
    paper_index   : int           — 0-based index, used to generate paper_id
    year_override : Optional[int] — if provided, use this year instead of
                                    auto-detected value (manual user correction)

    Returns
    -------
    dict following the PaperGraph paper schema.
    Never raises — on any error, returns a partial dict with empty fields
    and a 'parse_error' key.
    """
    paper_id = f"paper_{paper_index + 1:03d}"
    base = {
        "paper_id": paper_id,
        "filename": filename,
        "title": "",
        "abstract": "",
        "year": None,
        "year_source": "estimated",
        "year_estimated": True,
        "authors": [],
        "full_text": "",
        "concepts": [],
    }

    if not FITZ_AVAILABLE:
        base["parse_error"] = "PyMuPDF not installed"
        return base

    # Write bytes to a temp file (fitz.open() needs a path or bytes stream)
    tmp_path: Optional[str] = None
    doc = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as fh:
            fh.write(file_bytes)

        doc = fitz.open(tmp_path)  # type: ignore[attr-defined]

        if doc.page_count == 0:
            base["parse_error"] = "PDF has 0 pages"
            return base

        # ── Full text extraction ──────────────────────────────
        pages_text: list[str] = []
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                pages_text.append(page.get_text("text"))  # type: ignore[attr-defined]
            except Exception as exc:
                logger.debug("Page %d extraction error in %s: %s", page_num, filename, exc)

        full_text = "\n".join(pages_text).strip()
        base["full_text"] = full_text

        if not full_text:
            base["parse_error"] = (
                "No extractable text — PDF may be a scanned image. "
                "OCR is not supported in this version."
            )
            return base

        # ── Metadata from fitz.Document.metadata ─────────────
        meta = doc.metadata or {}

        # Title
        base["title"] = _extract_title(meta, full_text, filename)

        # Authors
        base["authors"] = _extract_authors(meta, full_text)

        # Year (with source tracking)
        if year_override is not None:
            base["year"] = int(year_override)
            base["year_source"] = "manual"
            base["year_estimated"] = False
        else:
            year, source = _extract_year_with_source(meta, full_text)
            base["year"] = year
            base["year_source"] = source
            base["year_estimated"] = (source == "estimated" or year is None)

        # Abstract
        base["abstract"] = _extract_abstract(full_text)

        return base

    except Exception as exc:
        logger.warning("Failed to process PDF '%s': %s", filename, exc)
        base["parse_error"] = str(exc)
        return base
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def process_pdfs(
    uploaded_files: list,
    year_overrides: Optional[dict[str, int]] = None,
) -> tuple[list[dict], list[str]]:
    """
    Process a list of Streamlit UploadedFile objects.

    Parameters
    ----------
    uploaded_files : list of Streamlit UploadedFile objects
    year_overrides : optional {filename → year} mapping for manual year correction

    Returns
    -------
    (papers, warnings)
    papers   : list of paper dicts (may have 'parse_error' key for bad PDFs)
    warnings : list of human-readable warning strings
    """
    papers: list[dict] = []
    warnings: list[str] = []

    for idx, uploaded_file in enumerate(uploaded_files):
        filename = getattr(uploaded_file, "name", f"file_{idx}.pdf")
        try:
            file_bytes = uploaded_file.read()
            if not file_bytes:
                warnings.append(f"⚠️ '{filename}' appears to be empty and was skipped.")
                continue

            # Check for manual year overrides (passed as {filename: year} dict)
            year_ov = year_overrides.get(filename) if year_overrides else None
            paper = process_pdf(file_bytes, filename, len(papers), year_override=year_ov)

            if "parse_error" in paper:
                warnings.append(
                    f"⚠️ '{filename}': {paper['parse_error']}. "
                    "This file will be excluded from analysis."
                )
            else:
                # Warn about estimated years
                if paper.get("year_estimated") or paper.get("year") is None:
                    warnings.append(
                        f"📅 '{filename}': publication year could not be reliably extracted "
                        f"(source: {paper.get('year_source', 'unknown')}). "
                        "A fallback year will be assigned. You can correct it in Paper Metadata."
                    )
                papers.append(paper)

        except Exception as exc:
            logger.exception("Unexpected error processing '%s'", filename)
            warnings.append(
                f"⚠️ '{filename}' could not be read ({exc}). Skipping."
            )

    return papers, warnings


def peek_pdf_year(file_bytes: bytes, filename: str = "") -> tuple[Optional[int], str]:
    """
    Quickly detect the publication year of a PDF from its metadata or first page text.
    Used by the UI to pre-populate the year editor with the detected year.

    Returns
    -------
    (year, source) e.g. (2021, 'metadata') or (2023, 'text_regex') or (None, 'estimated')
    """
    if not FITZ_AVAILABLE or not file_bytes:
        return None, "estimated"

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        meta = doc.metadata or {}
        first_page_text = ""
        if len(doc) > 0:
            first_page_text = doc[0].get_text("text") or ""
        return _extract_year_with_source(meta, first_page_text)
    except Exception as exc:
        logger.debug("Failed to peek year in %s: %s", filename, exc)
        return None, "estimated"



# ─────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────


def _extract_title(meta: dict, full_text: str, filename: str) -> str:
    """
    Try (in order):
    1. fitz metadata 'title' field
    2. First non-empty line of full_text that looks like a title
    3. Filename without extension
    """
    # 1. Metadata
    title = (meta.get("title") or "").strip()
    if title and len(title) > 5 and len(title) < 300:
        return title

    # 2. First substantial line of text
    for line in full_text.splitlines():
        line = line.strip()
        # A title is likely 10–200 chars, not starting with digits or special chars
        if 10 < len(line) < 250 and not line[0].isdigit() and not line.startswith(("©", "http")):
            # Skip lines that look like author lines, emails, or URLs
            if "@" not in line and "http" not in line.lower():
                return line

    # 3. Filename fallback
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _extract_authors(meta: dict, full_text: str) -> list[str]:
    """
    Try fitz metadata 'author' field, then scan first ~500 chars of text.
    Returns a list of author name strings (may be empty).
    """
    # 1. Metadata
    meta_authors = (meta.get("author") or "").strip()
    if meta_authors:
        # Handle semicolon or comma-separated lists
        sep = ";" if ";" in meta_authors else ","
        parts = [p.strip() for p in meta_authors.split(sep) if p.strip()]
        if parts:
            return parts[:10]  # Cap at 10 authors

    # 2. Regex scan in first 500 characters
    snippet = full_text[:500]
    match = _AUTHOR_LINE_RE.search(snippet)
    if match:
        raw = match.group(1)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts[:10]

    return []


def _extract_year(meta: dict, full_text: str) -> Optional[int]:
    """
    Try fitz metadata dates, then regex scan of the full text.
    Returns an int year or None.

    Deprecated: use _extract_year_with_source() for richer output.
    """
    year, _ = _extract_year_with_source(meta, full_text)
    return year


def _extract_year_with_source(
    meta: dict, full_text: str
) -> tuple[Optional[int], str]:
    """
    Try fitz metadata dates, then regex scan of the full text.

    Returns
    -------
    (year, source) where source is one of:
        'metadata'   — extracted from PDF metadata date field
        'text_regex' — extracted from year pattern in first 2000 chars of text
        'estimated'  — could not determine; year is None
    """
    # 1. Metadata dates (format: D:YYYYMMDDHHmmSS)
    for field in ("creationDate", "modDate"):
        raw = (meta.get(field) or "").strip()
        if raw.startswith("D:") and len(raw) >= 6:
            try:
                return int(raw[2:6]), "metadata"
            except ValueError:
                pass

    # 2. Regex in first 2000 chars (where citation/copyright usually appears)
    snippet = full_text[:2000]
    years = _YEAR_RE.findall(snippet)
    if years:
        from collections import Counter
        year_counts = Counter(int(y) for y in years)
        return year_counts.most_common(1)[0][0], "text_regex"

    return None, "estimated"


def _extract_abstract(full_text: str) -> str:
    """
    Attempt to locate and extract the abstract section.
    Falls back to the first 500 characters of the text.
    """
    for pattern in _ABSTRACT_PATTERNS:
        match = pattern.search(full_text)
        if match:
            abstract = match.group(1).strip()
            # Trim to a reasonable length
            abstract = re.sub(r"\s+", " ", abstract)
            if len(abstract) > 100:
                return abstract[:2000]  # Cap at 2000 chars

    # Fallback: first paragraph after skipping the title line
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    fallback_lines = lines[1:6] if len(lines) > 2 else lines
    fallback = " ".join(fallback_lines)
    return re.sub(r"\s+", " ", fallback)[:1000]
