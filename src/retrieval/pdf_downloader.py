"""
PDF downloader with multi-source resolution chain for corpus acquisition.

Resolution order:
  1. Semantic Scholar openAccessPdf URL (already in search results)
  2. PMC OA Service (by PMID — free, no key)
  3. Unpaywall API (by DOI — covers OA versions of paywalled articles)
  4. Log unfetchable papers to external/missing.json

Usage::

    from src.retrieval.pdf_downloader import PDFDownloader

    dl = PDFDownloader(output_dir="data/external")
    result = dl.download(paper_metadata)
    # result["status"] is "downloaded", "skipped_exists", or "unfetchable"
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/external")
MISSING_LOG = DEFAULT_OUTPUT / "missing.json"


class PDFDownloader:
    """Downloads PDFs via a multi-source resolution chain."""

    def __init__(self, output_dir: str | Path = DEFAULT_OUTPUT):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"downloaded": 0, "skipped_exists": 0, "unfetchable": 0, "errors": 0}

    def _sanitize_filename(self, text: str, max_len: int = 80) -> str:
        """Create a safe filename from paper title/authors."""
        text = text.replace("/", "-").replace(":", "-").replace("?", "")
        text = " ".join(text.split())[:max_len]
        safe = "".join(c if c.isalnum() or c in " -_.,()[]" else "" for c in text)
        return safe.strip().rstrip(".") if safe else "paper"

    def _already_exists(self, paper: Dict) -> Optional[Path]:
        """Check if paper already exists by title or DOI."""
        title = paper.get("title", "")
        doi = paper.get("doi", "")
        year = paper.get("year", "")

        if title:
            fname = self._sanitize_filename(title, max_len=60)
            if year:
                fname = f"{year}_{fname}"
            path = self.output_dir / f"{fname}.pdf"
            if path.exists() and path.stat().st_size > 1000:
                return path

        if doi:
            doi_hash = hashlib.sha256(doi.encode()).hexdigest()[:12]
            for f in self.output_dir.glob(f"*_{doi_hash}.pdf"):
                if f.stat().st_size > 1000:
                    return f

        return None

    def _fetch_pdf(self, url: str, paper: Dict) -> Optional[Path]:
        """Download a PDF from a URL. Returns path on success, None on failure."""
        if not url:
            return None

        title = paper.get("title", "untitled")
        year = paper.get("year", "")
        doi = paper.get("doi", "")

        fname = self._sanitize_filename(title, max_len=50)
        if year:
            fname = f"{year}_{fname}"
        if doi:
            doi_hash = hashlib.sha256(doi.encode()).hexdigest()[:8]
            fname = f"{fname}_{doi_hash}"

        path = self.output_dir / f"{fname}.pdf"
        if path.exists():
            return path if path.stat().st_size > 1000 else None

        try:
            headers = {
                "User-Agent": "FederatedRAG/1.0 (mailto:luke.sheakley@gmail.com)",
            }
            resp = requests.get(url, headers=headers, timeout=60, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower() and "pdf" not in content_type.lower():
                logger.debug("URL returned HTML (not PDF) for %s", paper.get("title", "")[:50])
                return None

            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if path.stat().st_size < 1000:
                path.unlink()
                return None

            logger.info("Downloaded: %s (%d KB)", path.name, path.stat().st_size // 1024)
            return path

        except Exception as e:
            logger.warning("Download failed for %s: %s", paper.get("title", "")[:50], e)
            if path.exists():
                path.unlink()
            return None

    def download(
        self,
        paper: Dict,
        pmc_client=None,
        unpaywall_client=None,
    ) -> Dict[str, Any]:
        """Attempt to download a paper's PDF through the resolution chain.

        Args:
            paper: Dict with title, doi, pmid, year, open_access_pdf (from S2),
                   and any other metadata.
            pmc_client: Optional PMCOAClient instance.
            unpaywall_client: Optional UnpaywallClient instance.

        Returns:
            Dict with: status (downloaded|skipped_exists|unfetchable),
                       path (if downloaded), paper metadata.
        """
        title = paper.get("title", "Unknown")[:60]
        result = {
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "doi": paper.get("doi", ""),
            "pmid": paper.get("pmid", ""),
            "status": "unfetchable",
            "source": None,
            "path": None,
        }

        # Check if already downloaded
        existing = self._already_exists(paper)
        if existing:
            self.stats["skipped_exists"] += 1
            result["status"] = "skipped_exists"
            result["path"] = str(existing)
            return result

        # Layer 1: Semantic Scholar openAccessPdf URL
        s2_oa_url = paper.get("open_access_pdf", "")
        if s2_oa_url:
            path = self._fetch_pdf(s2_oa_url, paper)
            if path:
                self.stats["downloaded"] += 1
                result["status"] = "downloaded"
                result["source"] = "semantic_scholar"
                result["path"] = str(path)
                return result

        # Layer 2: PMC OA Service (by PMID)
        pmid = paper.get("pmid", "")
        if pmid and pmc_client:
            try:
                pmc_result = pmc_client.lookup(pmid)
                if pmc_result.get("has_oa") and pmc_result.get("pdf_url"):
                    path = self._fetch_pdf(pmc_result["pdf_url"], paper)
                    if path:
                        self.stats["downloaded"] += 1
                        result["status"] = "downloaded"
                        result["source"] = "pmc_oa"
                        result["path"] = str(path)
                        return result
            except Exception as e:
                logger.debug("PMC OA failed for %s: %s", pmid, e)

        # Layer 3: Unpaywall (by DOI)
        doi = paper.get("doi", "")
        if doi and unpaywall_client:
            try:
                uw_result = unpaywall_client.lookup(doi)
                pdf_url = uw_result.get("best_pdf_url", "")
                if pdf_url:
                    path = self._fetch_pdf(pdf_url, paper)
                    if path:
                        self.stats["downloaded"] += 1
                        result["status"] = "downloaded"
                        result["source"] = "unpaywall"
                        result["path"] = str(path)
                        return result
            except Exception as e:
                logger.debug("Unpaywall failed for %s: %s", doi, e)

        # Layer 4: Unfetchable — log for manual retrieval
        self.stats["unfetchable"] += 1
        result["status"] = "unfetchable"
        _append_missing(result)
        return result


def _append_missing(paper: Dict) -> None:
    """Log an unfetchable paper to missing.json."""
    log = []
    if MISSING_LOG.exists():
        try:
            log = json.loads(MISSING_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    log.append({
        "title": paper.get("title", ""),
        "doi": paper.get("doi", ""),
        "pmid": paper.get("pmid", ""),
        "year": paper.get("year"),
        "timestamp": time.time(),
    })
    MISSING_LOG.parent.mkdir(parents=True, exist_ok=True)
    MISSING_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
