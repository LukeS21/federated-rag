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
    """Create a Zotero item, or return existing item if already present.

    Checks Zotero for an existing item by title before creating a new one.
    This prevents duplicates when the same paper is ingested under
    different filenames.

    Returns the item key on success, None on failure.
    """
    library_id = library_id or os.getenv("ZOTERO_LIBRARY_ID", "").strip()
    api_key = api_key or os.getenv("ZOTERO_API_KEY", "").strip()
    if not library_id or not api_key:
        return None

    title = metadata.get("title", "") or ""

    # ── Check for existing item first ──
    if len(title) >= 20:
        existing = search_zotero_by_title(title)
        if existing:
            logger.info("Zotero: found existing item %s (skipping create)", existing)
            return existing

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
            logger.info("Zotero: created item %s for '%s...'", item_key, title[:60])
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


# ── Zotero search & dedup ────────────────────────────────────────────────────
def _get_zotero_client() -> Optional[Any]:
    """Return a pyzotero client if credentials are available."""
    library_id = os.getenv("ZOTERO_LIBRARY_ID", "").strip()
    api_key = os.getenv("ZOTERO_API_KEY", "").strip()
    if not library_id or not api_key:
        return None
    try:
        from pyzotero import zotero
        return zotero.Zotero(library_id, "user", api_key)
    except Exception:
        return None


def search_zotero_by_title(title: str) -> Optional[str]:
    """Search Zotero for an existing item by title.

    Returns the Zotero item key if a match is found, None otherwise.
    Also returns None if Zotero credentials are not configured.
    """
    if not title or len(title) < 20:
        return None
    zot = _get_zotero_client()
    if zot is None:
        return None
    try:
        # Truncate to avoid overly long queries
        q = title[:150].strip()
        items = zot.top(q=q, limit=5)
        for item in items:
            data = item.get("data", {})
            item_title = (data.get("title", "") or "").lower()
            q_lower = q.lower()
            # Partial match: query substring in item title, or vice versa
            if q_lower[:50] in item_title or item_title[:50] in q_lower:
                logger.info("Zotero: found existing item %s for title '%s...'",
                            item.get("key"), q[:60])
                return item.get("key")
    except Exception as e:
        logger.debug("Zotero search failed: %s", e)
    return None


def get_zotero_cite_key(item_key: str) -> Optional[str]:
    """Generate a cite key from a Zotero item's metadata.

    Returns e.g. @morandini2024 or None if metadata is insufficient.
    """
    zot = _get_zotero_client()
    if zot is None:
        return None
    try:
        item = zot.item(item_key)
        data = item.get("data", {})
        # Get first author's surname
        creators = data.get("creators", [])
        surname = ""
        if creators:
            surname = creators[0].get("lastName", "")
        else:
            # Fallback: try parsed first author from filename
            pass
        year = str(data.get("date", "") or "")[:4]
        if surname and year:
            cite_key = f"@{surname.lower().replace(' ', '').replace('-', '')}{year}"
            logger.info("Zotero: cite key %s from item %s", cite_key, item_key)
            return cite_key
    except Exception as e:
        logger.debug("Zotero cite key lookup failed: %s", e)
    return None


# ── Three-tier cite key resolution ────────────────────────────────────────────
def resolve_cite_key(filename: str, chunks_text: str = "") -> str:
    """Resolve a cite key using three tiers of information.

    Tier 1: Zotero search by extracted title → get author/year → @surnameYYYY
    Tier 2: Filename parsing (e.g., "Avery et al. - 2024 - Title.pdf")
    Tier 3: Content hash fallback (@ref_XXXXXXXX)

    Args:
        filename: The PDF filename.
        chunks_text: First 500 chars of body text (for title extraction).

    Returns a cite key like @avery2024 or @ref_a3f2b1c4.
    """
    # ── Tier 1: Zotero search ──
    title = chunks_text[:200] if chunks_text else ""
    if title:
        item_key = search_zotero_by_title(title)
        if item_key:
            cite_key = get_zotero_cite_key(item_key)
            if cite_key:
                logger.info("CiteKey (zotero): '%s' → %s", filename[:40], cite_key)
                return cite_key

    # ── Tier 2: Filename parsing ──
    cite_key = parse_cite_key_from_filename(filename)
    # Check if the parsed key looks meaningful (not a raw filename fallback)
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE).strip().lower()
    if cite_key.startswith("@test") and stem.startswith("test"):
        # "test.pdf" → fallback to Tier 3
        pass
    elif not cite_key.startswith("@ref_") and not cite_key.startswith("@paper"):
        logger.info("CiteKey (filename): '%s' → %s", filename[:40], cite_key)
        return cite_key

    # ── Tier 3: Content hash fallback ──
    if chunks_text:
        import hashlib
        hash_key = f"@ref_{hashlib.sha256(chunks_text[:2000].encode()).hexdigest()[:8]}"
        logger.info("CiteKey (hash): '%s' → %s", filename[:40], hash_key)
        return hash_key

    return cite_key  # last resort
