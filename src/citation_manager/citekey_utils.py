"""CiteKey generation and metadata extraction for PDFs.

Generates inline citation keys (e.g., @avery2024) from:
  1. Filename parsing (e.g., "Avery et al. - 2024 - Title.pdf")
  2. Zotero API lookup when credentials are available (fallback)
  3. DOI extraction from Docling metadata

Used during PDF ingest and propagated through extraction → synthesis.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Map filename stems to pre-generated cite keys for papers in the corpus
_FILENAME_CITEKEY_MAP: Dict[str, str] = {}


def parse_paper_metadata(filename: str) -> Dict[str, Any]:
    """Extract paper metadata from a PDF filename.

    Handles: "Avery et al. - 2024 - CD4+ and CD8+ T cells reduce...pdf"
    Returns dict with: title, date, surname, creators list.

    The returned metadata can be passed to try_zotero_add() to create
    a Zotero item.
    """
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE).strip()
    metadata: Dict[str, Any] = {"original_filename": filename}

    # Pattern: "Surname et al. - YYYY - Title text"
    m = re.match(
        r"([A-Za-z][\w'-]+)(?:\s+et\s+al\.?)?\s*[-–]\s*(\d{4})\s*[-–]\s*(.+)",
        stem,
    )
    if m:
        surname = m.group(1)
        year = m.group(2)
        title = m.group(3).strip().rstrip(".")
        metadata["surname"] = surname
        metadata["date"] = year
        metadata["title"] = title[:500]

        # Create simple creator entry
        metadata["creators"] = [
            {"creatorType": "author", "lastName": surname, "firstName": ""},
        ]
        return metadata

    # Fallback: just use the stem as title
    metadata["title"] = stem[:500]
    return metadata


def parse_cite_key_from_filename(filename: str) -> str:
    """Generate a cite key from a PDF filename.

    Handles formats like:
      - "Avery et al. - 2024 - CD4+ and CD8+ T cells reduce..."
      - "Morandini et al. - 2024 - Adoptive Transfer of..."

    Falls back to a hash-based key if parsing fails.
    """
    cache_key = filename.lower().replace(".pdf", "").strip()
    if cache_key in _FILENAME_CITEKEY_MAP:
        return _FILENAME_CITEKEY_MAP[cache_key]

    cite_key = _parse_filename(cache_key)
    _FILENAME_CITEKEY_MAP[cache_key] = cite_key
    logger.info("CiteKey: '%s' → %s", filename[:50], cite_key)
    return cite_key


def _parse_filename(stem: str) -> str:
    """Extract author-year from a filename stem."""
    # Strip extension
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE).strip()

    # Pattern: "Lastname et al. - YYYY - ..." or "Lastname - YYYY - ..."
    m = re.match(r"([A-Za-z][\w'-]+)\s+(?:et al\.?\s*[-–]\s*)?(\d{4})", stem)
    if m:
        surname = m.group(1).lower().replace(" ", "").replace("-", "")
        year = m.group(2)
        return f"@{surname}{year}"

    # Pattern: "Lastname, Firstname et al. - YYYY - ..."
    m = re.match(r"([A-Za-z][\w'-]+),\s*[A-Z].*?[-–]\s*(\d{4})", stem)
    if m:
        surname = m.group(1).lower().replace(" ", "").replace("-", "")
        year = m.group(2)
        return f"@{surname}{year}"

    # Pattern: numeric prefix → use hash
    if re.match(r"^\d+", stem):
        import hashlib
        return f"@ref_{hashlib.sha256(stem.encode()).hexdigest()[:8]}"

    # Fallback: first word + year-like number
    parts = re.split(r"[_\s-]+", stem)
    first_word = parts[0].lower().replace(" ", "") if parts else "paper"
    for part in parts:
        if re.match(r"^\d{4}$", part):
            return f"@{first_word}{part}"

    # Last resort: first meaningful word
    return f"@{first_word}"


# ── Zotero / external API helpers ────────────────────────────────────────────
def try_zotero_add(
    metadata: Dict[str, Any],
    library_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Try to create a Zotero item via the real API.

    Returns the item key on success, None on failure (no credentials,
    API error, etc.).
    """
    library_id = library_id or os.getenv("ZOTERO_LIBRARY_ID", "").strip()
    api_key = api_key or os.getenv("ZOTERO_API_KEY", "").strip()
    if not library_id or not api_key:
        return None

    try:
        from pyzotero import zotero
        zot = zotero.Zotero(library_id, "user", api_key)

        # Build Zotero item template
        item_type = "journalArticle"
        if metadata.get("item_type"):
            item_type = metadata["item_type"]

        template = zot.item_template(item_type)
        template["title"] = metadata.get("title", "Unknown Title")[:500]
        if metadata.get("creators"):
            template["creators"] = metadata["creators"]
        if metadata.get("date"):
            template["date"] = str(metadata["date"])
        if metadata.get("DOI"):
            template["DOI"] = str(metadata["DOI"])
        if metadata.get("url"):
            template["url"] = str(metadata["url"])
        if metadata.get("publicationTitle"):
            template["publicationTitle"] = str(metadata["publicationTitle"])[:200]
        if metadata.get("abstractNote"):
            template["abstractNote"] = str(metadata["abstractNote"])[:1000]

        resp = zot.create_items([template])
        if resp and "success" in resp and resp["success"]:
            item_key = resp["success"].get("0", metadata.get("cite_key", ""))
            logger.info("Zotero: created item %s", item_key)
            return str(item_key)
    except Exception as e:
        logger.debug("Zotero API call failed (credentials may be missing): %s", e)

    return None


def extract_doi_from_docling(doc_metadata: Dict[str, Any]) -> Optional[str]:
    """Try to extract a DOI from Docling document metadata."""
    # Docling may store DOI in various fields
    for key in ("doi", "DOI", "identifier", "dc:identifier"):
        val = doc_metadata.get(key, "")
        if val and isinstance(val, str) and "10." in val:
            # Extract DOI from possible URL format
            m = re.search(r"(10\.\d{4,}/[^\s\"']+)", val)
            if m:
                return m.group(1)
    return None
