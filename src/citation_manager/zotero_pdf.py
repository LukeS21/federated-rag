"""
Zotero PDF attachment utilities — automated PDF finding and downloading.

Provides functions to:
  - Check if a Zotero item has a PDF attachment
  - Download PDF attachments from Zotero to local storage
  - Attach locally-downloaded PDFs back to Zotero items
  - Batch-create Zotero items from metadata
  - Manage the "external" Zotero collection for discovered papers
  - Try direct PDF downloads via Unpaywall/PMC OA/S2
  - Sync Zotero collection state with data/external/

Used by `scripts/zotero_sync.py` for automated corpus expansion.

Requires Zotero credentials in `.env`:
  ZOTERO_LIBRARY_ID=...
  ZOTERO_API_KEY=...
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("data/external")
EXTERNAL_COLLECTION = "external"
_external_collection_key: Optional[str] = None


def _get_zotero():
    """Return a pyzotero client or None if credentials aren't configured."""
    from pyzotero import zotero
    library_id = os.getenv("ZOTERO_LIBRARY_ID", "").strip()
    api_key = os.getenv("ZOTERO_API_KEY", "").strip()
    if not library_id or not api_key:
        return None
    return zotero.Zotero(library_id, "user", api_key)


def ensure_external_collection(zot) -> str:
    """Find or create the 'external' collection in Zotero.

    Returns the collection key. Cached after first call.
    """
    global _external_collection_key
    if _external_collection_key:
        return _external_collection_key
    try:
        collections = zot.collections()
        for col in collections:
            data = col.get("data", {})
            if data.get("name", "").lower() == EXTERNAL_COLLECTION:
                _external_collection_key = data.get("key", "")
                return _external_collection_key
        # Create it
        resp = zot.create_collection(EXTERNAL_COLLECTION)
        if isinstance(resp, dict):
            for v in resp.values():
                if isinstance(v, str) and len(v) == 8:
                    _external_collection_key = v
                    return _external_collection_key
    except Exception as e:
        logger.warning("Collection management failed: %s", e)
    return ""


def add_to_external_collection(zot, item_key: str) -> bool:
    """Add a Zotero item to the 'external' collection. Dedup-safe.
    
    Fetches the full item dict from Zotero (needed for version tracking),
    then patches the collections array to include the external collection.
    """
    collection_key = ensure_external_collection(zot)
    if not collection_key:
        return False
    try:
        # Fetch full item (needed for key, version, collections)
        full_item = zot.item(item_key)
        if full_item and "data" in full_item:
            full_item["key"] = item_key
            zot.addto_collection(collection_key, full_item)
            logger.debug("Added item %s to 'external' collection", item_key)
            return True
    except Exception as e:
        if "already" in str(e).lower() or "exists" in str(e).lower():
            return True
        if "version" not in str(e).lower():  # version errors are real failures
            logger.debug("Failed to add item %s to collection: %s", item_key, e)
    return False


def get_external_collection_items(zot) -> List[Dict[str, Any]]:
    """Return all items in the 'external' collection."""
    collection_key = ensure_external_collection(zot)
    if not collection_key:
        return []
    try:
        items = zot.collection_items(collection_key)
        return items
    except Exception as e:
        logger.debug("Failed to list external collection: %s", e)
        return []


def sync_external_pdfs_with_zotero(
    dest_dir: str | Path = EXTERNAL_DIR,
) -> Dict[str, int]:
    """Sync: ensure every PDF in data/external/ is attached to its Zotero item.

    Scans data/external/ for PDFs, looks up the corresponding Zotero item
    by DOI/title, and attaches the PDF if it hasn't been attached yet.

    Returns:
        {"synced": N, "skipped": N, "errors": N}
    """
    dest = Path(dest_dir)
    zot = _get_zotero()
    if zot is None:
        return {"synced": 0, "skipped": 0, "errors": 0}

    pdfs = sorted(dest.glob("*.pdf"))
    result = {"synced": 0, "skipped": 0, "errors": 0}

    for pdf_path in pdfs:
        try:
            # Check if already attached — search for items with this filename
            existing = _find_zotero_item_for_pdf(zot, pdf_path)
            if existing and has_pdf_attachment(zot, existing)[0]:
                result["skipped"] += 1
                continue

            # If we have a Zotero key cached in the filename or we find one
            if existing:
                attach_pdf_to_item(zot, existing, pdf_path)
                add_to_external_collection(zot, existing)
                result["synced"] += 1
            else:
                result["errors"] += 1
        except Exception as e:
            result["errors"] += 1
            logger.debug("Sync error for %s: %s", pdf_path.name, e)

    logger.info("Sync complete: %d synced, %d skipped, %d errors",
                 result["synced"], result["skipped"], result["errors"])
    return result


def _find_zotero_item_for_pdf(zot, pdf_path: Path) -> Optional[str]:
    """Find the Zotero item key for a downloaded PDF by searching titles/DOIs."""
    # Strategy 1: DOI hash in filename
    import re
    doi_match = re.search(r"_([a-f0-9]{8})\.pdf$", pdf_path.name)
    if doi_match:
        doi_hash = doi_match.group(1)
        # Search through external collection items
        items = get_external_collection_items(zot)
        for item in items:
            data = item.get("data", {})
            item_doi = data.get("DOI", "") or ""
            import hashlib
            if hashlib.sha256(item_doi.encode()).hexdigest()[:8] == doi_hash:
                return data.get("key", "")

    # Strategy 2: Title in filename
    stem = pdf_path.stem.lower()
    # Remove year prefix and hash suffix
    stem_clean = re.sub(r"^\d{4}_", "", stem)
    stem_clean = re.sub(r"_[a-f0-9]{6,}$", "", stem_clean)

    if len(stem_clean) > 20:
        try:
            # Search Zotero by title fragment
            search_key = stem_clean[:80]
            results = zot.top(q=search_key, limit=3, itemType="journalArticle")
            for item in results:
                data = item.get("data", {})
                return data.get("key", "")
        except Exception:
            pass

    return None


def get_item_attachments(zot, item_key: str) -> List[Dict[str, Any]]:
    """Return all child attachment items for a Zotero item."""
    try:
        children = zot.children(item_key)
        attachments = []
        for child in children:
            data = child.get("data", {})
            if data.get("itemType") == "attachment":
                attachments.append(data)
        return attachments
    except Exception as e:
        logger.debug("Failed to get attachments for %s: %s", item_key, e)
        return []


def has_pdf_attachment(zot, item_key: str) -> Tuple[bool, Optional[str]]:
    """Check if a Zotero item has a PDF child attachment.

    Returns:
        (has_pdf, attachment_key_or_None)
    """
    attachments = get_item_attachments(zot, item_key)
    for att in attachments:
        content_type = att.get("contentType", "")
        filename = (att.get("filename", "") or "").lower()
        if content_type == "application/pdf" or filename.endswith(".pdf"):
            return True, att.get("key")
    return False, None


def download_pdf_attachment(
    zot,
    attachment_key: str,
    dest_dir: str | Path = EXTERNAL_DIR,
    filename: str | None = None,
) -> Optional[Path]:
    """Download a PDF attachment from Zotero to local storage.

    Args:
        zot: Zotero client.
        attachment_key: The attachment's item key.
        dest_dir: Where to save the PDF.
        filename: Optional custom filename (without extension).

    Returns:
        Path to the downloaded file, or None on failure.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        # Get the attachment's metadata for filename
        att = zot.item(attachment_key)
        att_data = att.get("data", {})
        att_filename = att_data.get("filename", f"{attachment_key}.pdf")

        # Build output path
        if filename:
            out_path = dest / f"{filename}.pdf"
        else:
            out_path = dest / att_filename
            if not str(out_path).endswith(".pdf"):
                out_path = dest / f"{att_filename}.pdf"

        # Download the file
        file_content = zot.file(attachment_key)
        if file_content is None:
            logger.warning("Zotero returned None for file %s", attachment_key)
            return None

        out_path.write_bytes(file_content)
        logger.info("Downloaded: %s (%d KB)", out_path.name, len(file_content) // 1024)
        return out_path

    except Exception as e:
        logger.warning("Failed to download Zotero attachment %s: %s", attachment_key, e)
        return None


def attach_pdf_to_item(
    zot,
    parent_key: str,
    pdf_path: Path,
) -> Optional[str]:
    """Upload a local PDF as a child attachment of a Zotero item.

    Args:
        zot: Zotero client.
        parent_key: The parent item's key.
        pdf_path: Path to the PDF file.

    Returns:
        Attachment item key on success, None on failure.
    """
    if not pdf_path.exists():
        return None

    try:
        # pyzotero's attachment_simple: files list + optional parentid string
        result = zot.attachment_simple([str(pdf_path)], str(parent_key))
        if result:
            if isinstance(result, dict) and result.get("success"):
                return str(result["success"].get("0", ""))
            return str(parent_key)
        logger.warning("Zotero attachment upload returned None for %s", pdf_path.name)
        return None
    except Exception as e:
        logger.warning("Failed to attach PDF %s to item %s: %s",
                       pdf_path.name, parent_key, e)
        return None


def create_zotero_items_from_missing(
    missing_json_path: str | Path,
) -> Dict[str, Dict[str, Any]]:
    """Batch-create Zotero items for all papers in missing.json.

    Uses the existing ``try_zotero_add()`` function for dedup-safe creation.

    Args:
        missing_json_path: Path to data/external/missing.json.

    Returns:
        {doi_or_pmid: {"zotero_key": ..., "title": ..., "status": ...}}
    """
    from src.citation_manager.citekey_utils import try_zotero_add

    src = Path(missing_json_path)
    if not src.exists():
        logger.error("Missing file not found: %s", src)
        return {}

    data = json.loads(src.read_text(encoding="utf-8"))
    items: List[Dict] = data if isinstance(data, list) else data.get("papers", [])

    results: Dict[str, Dict] = {}
    zot = _get_zotero()
    if zot is None:
        logger.error("Zotero credentials not configured")
        return {}

    collection_key = ensure_external_collection(zot)

    for i, paper in enumerate(items):
        title = paper.get("title", "")[:200]
        doi = paper.get("doi", "")
        pmid = paper.get("pmid", "")
        key = doi or pmid or title[:60]

        if i > 0 and i % 20 == 0:
            logger.info("Zotero item creation: %d/%d", i, len(items))
            time.sleep(1.0)  # avoid rate limit

        try:
            metadata = {
                "title": title,
                "DOI": doi,
                "date": str(paper.get("year", ""))[:4],
                "url": f"https://doi.org/{doi}" if doi else "",
                "creators": [
                    {"creatorType": "author", "lastName": a.split()[-1] if " " in a else a, "firstName": " ".join(a.split()[:-1])}
                    for a in (paper.get("authors", [])[:3] or [])
                ],
                "item_type": "journalArticle",
            }

            item_key = try_zotero_add(metadata)
            if item_key:
                results[key] = {
                    "zotero_key": item_key,
                    "title": title,
                    "doi": doi,
                    "pmid": pmid,
                    "status": "zotero_created",
                    "by": "new" if title not in str(results) else "existing",
                }
                # Add to 'external' collection
                if collection_key:
                    add_to_external_collection(zot, item_key)
            else:
                results[key] = {
                    "title": title,
                    "doi": doi,
                    "pmid": pmid,
                    "status": "zotero_failed",
                    "error": "try_zotero_add returned None",
                }
        except Exception as e:
            logger.warning("Zotero create failed for '%s': %s", title[:60], e)
            results[key] = {
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "status": "zotero_error",
                "error": str(e)[:200],
            }

    logger.info("Zotero item creation complete: %d items processed", len(results))
    return results


def try_direct_pdf_download(
    paper: Dict[str, Any],
    dest_dir: str | Path = EXTERNAL_DIR,
) -> Optional[Path]:
    """Attempt direct PDF download via Unpaywall, PMC OA, and Semantic Scholar.

    This is the fast path — doesn't require Zotero Desktop to be open.

    Returns:
        Path to downloaded PDF, or None.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    doi = paper.get("doi", "")
    pmid = paper.get("pmid", "")
    title = paper.get("title", "untitled")[:60]
    year = paper.get("year", "")

    import hashlib
    import re
    import requests

    # Safe filename
    safe_title = re.sub(r"[^a-zA-Z0-9_\-\s]", "", title)[:50].strip()
    if not safe_title:
        safe_title = "paper"

    # Try multiple PDF URLs
    urls_to_try = []

    # 1. PMC OA Service (by PMID)
    if pmid:
        try:
            pmc_resp = requests.get(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi",
                params={"id": pmid, "format": "pdf"},
                timeout=15,
            )
            pmc_links = re.findall(r'<link\s+format="([^"]+)"\s+href="([^"]+)"', pmc_resp.text)
            for fmt, href in pmc_links:
                if "pdf" in fmt.lower():
                    urls_to_try.append(("pmc_oa", href))
        except Exception:
            pass

    # 2. European PMC
    if pmid:
        urls_to_try.append(("europe_pmc", f"https://europepmc.org/articles/PMC{pmid}/pdf?pdf=render"))

    # 3. Unpaywall
    if doi:
        try:
            email = os.getenv("UNPAYWALL_EMAIL", "")
            uw_resp = requests.get(
                f"https://api.unpaywall.org/v2/{doi.strip()}",
                params={"email": email} if email else {},
                timeout=15,
            )
            if uw_resp.status_code == 200:
                uw_data = uw_resp.json()
                best = uw_data.get("best_oa_location") or {}
                pdf_url = best.get("url_for_pdf", "")
                if pdf_url:
                    urls_to_try.append(("unpaywall", pdf_url))
        except Exception:
            pass

    # 4. Semantic Scholar (if we have an OA PDF URL from the search)
    s2_oa = paper.get("open_access_pdf", "")
    if s2_oa:
        urls_to_try.append(("semantic_scholar", s2_oa))

    # Try each URL
    for source, url in urls_to_try:
        if not url:
            continue
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; FederatedRAG/1.0)"},
                timeout=60,
                stream=True,
            )
            if resp.status_code == 200:
                content = b""
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk

                if len(content) > 1000:
                    # Check if it's actually PDF
                    if content[:4] == b"%PDF":
                        doi_hash = hashlib.sha256(doi.encode()).hexdigest()[:8] if doi else "nop"
                        fname = dest / f"{year}_{safe_title}_{doi_hash}.pdf"
                        fname.write_bytes(content)
                        logger.info("Direct download [%s]: %s (%d KB)",
                                     source, fname.name, len(content) // 1024)
                        return fname
                    else:
                        logger.debug("URL from %s returned non-PDF content for %s", source, title[:40])
        except Exception:
            pass

    return None
