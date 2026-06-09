"""
utils/metadata.py
─────────────────
Helpers for extracting and normalising paper metadata from a DoclingDocument.

  - DOI extraction   : regex scan over full markdown text
  - Title extraction : Docling label hierarchy with fallbacks
  - Author extraction: heuristic scan of early text blocks
  - Section labels   : heading → canonical IMRaD label
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from config import SECTION_LABEL_MAP

# ── DOI ───────────────────────────────────────────────────────────────────────

_DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[^\s\"\'<>\]\[}{,;]+)",
    re.IGNORECASE,
)


def extract_doi(text: str) -> Optional[str]:
    """Return the first DOI found in *text*, or None."""
    m = _DOI_RE.search(text)
    if m:
        return m.group(1).rstrip(".")
    return None


# ── Section label normalisation ───────────────────────────────────────────────


def _normalise_heading(raw: str) -> str:
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(c for c in raw if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", raw).strip().lower()


def canonical_section(heading: str) -> str:
    """
    Map a raw section heading to a canonical IMRaD label.
    Falls back to the normalised heading itself if no match found.
    """
    norm = _normalise_heading(heading)
    if norm in SECTION_LABEL_MAP:
        return SECTION_LABEL_MAP[norm]
    for key, label in SECTION_LABEL_MAP.items():
        if norm.startswith(key) or key.startswith(norm):
            return label
    return norm


# ── Title / author extraction ─────────────────────────────────────────────────


def extract_title_from_docling(doc) -> Optional[str]:
    """
    Recover the paper title from a DoclingDocument.

    Priority:
      1. Item with label == "title"
      2. First level-1 section header
      3. First text block longer than 20 chars
    """
    try:
        for item in doc.texts:
            label = str(getattr(item, "label", "")).lower()
            if label == "title":
                text = item.text.strip()
                if text:
                    return text

        for item in doc.texts:
            label = str(getattr(item, "label", "")).lower()
            if "header" in label or "heading" in label:
                text = item.text.strip()
                if len(text) > 10:
                    return text

        for item in doc.texts:
            text = item.text.strip()
            if len(text) > 20:
                return text

    except Exception:
        pass
    return None


def extract_authors_from_docling(doc) -> list[str]:
    """
    Best-effort author list from the first 20 text items.
    Looks for comma/semicolon-separated short tokens in the header area.
    Returns raw strings — may be empty for unusual layouts.
    """
    try:
        for item in list(doc.texts)[:20]:
            text = item.text.strip()
            if not text or len(text) > 400:
                continue
            if any(
                kw in text.lower()
                for kw in (
                    "abstract",
                    "introduc",
                    "method",
                    "doi",
                    "http",
                    "©",
                    "received",
                )
            ):
                continue
            parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
            if len(parts) >= 2 and all(len(p.split()) <= 5 for p in parts):
                return parts
    except Exception:
        pass
    return []
