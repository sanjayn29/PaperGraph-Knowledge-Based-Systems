"""
json_utils.py
─────────────
Safe JSON helpers for PaperGraph.

Provides:
- safe_json_load()   : load a JSON file without crashing
- safe_json_dump()   : write a JSON file atomically
- extract_json_from_llm_response() : parse JSON out of an LLM text response
  that may be wrapped in markdown code fences or contain extra prose
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_json_load(path: str | Path) -> dict | list | None:
    """
    Load and parse a JSON file.
    Returns None (and logs a warning) on any error.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.warning("JSON file not found: %s", path)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in %s: %s", path, exc)
    except OSError as exc:
        logger.warning("OS error reading %s: %s", path, exc)
    return None


def safe_json_dump(data: Any, path: str | Path, indent: int = 2) -> bool:
    """
    Write *data* to *path* as pretty-printed JSON.

    Uses a temp-file + rename strategy for atomicity so a crash during
    writing never leaves a partial file.

    Returns True on success, False on failure.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file first
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
            # Atomic rename (on Windows this may overwrite the destination)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up the temp file if something went wrong
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:
        logger.error("Failed to write JSON to %s: %s", path, exc)
        return False


# ─────────────────────────────────────────────────────────────
# LLM response → JSON extraction
# ─────────────────────────────────────────────────────────────

# Matches ```json … ``` or ``` … ``` code fences
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)

# Matches the outermost { … } JSON object
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)

# Matches the outermost [ … ] JSON array
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]", re.DOTALL)


def extract_json_from_llm_response(response_text: str) -> dict | list | None:
    """
    Robustly extract a JSON object or array from an LLM response that may:
    - Be wrapped in markdown code fences (```json … ```)
    - Contain prose before/after the JSON
    - Have extra trailing commas or minor formatting issues

    Returns the parsed Python object, or None if extraction fails.
    """
    if not response_text:
        return None

    # Strategy 1: look inside code fences
    fence_match = _CODE_FENCE_RE.search(response_text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        result = _try_parse(candidate)
        if result is not None:
            return result

    # Strategy 2: find the outermost { … }
    obj_match = _JSON_OBJECT_RE.search(response_text)
    if obj_match:
        result = _try_parse(obj_match.group(0))
        if result is not None:
            return result

    # Strategy 3: find the outermost [ … ]
    arr_match = _JSON_ARRAY_RE.search(response_text)
    if arr_match:
        result = _try_parse(arr_match.group(0))
        if result is not None:
            return result

    logger.warning("Could not extract JSON from LLM response (length=%d)", len(response_text))
    return None


def _try_parse(text: str) -> dict | list | None:
    """Attempt to parse *text* as JSON; return None on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try stripping trailing commas before closing braces/brackets (common LLM error)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def merge_dicts(*dicts: dict) -> dict:
    """Shallow-merge multiple dicts left-to-right (later keys win)."""
    result: dict = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result
