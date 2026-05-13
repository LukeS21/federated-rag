#!/usr/bin/env python3
"""
Headless PDF downloader via VCU EZProxy + Playwright route interception.

Generic approach:
  1. Navigate EZProxy → DOI to establish institutional session
  2. Targeted route interception catches PDF bytes before Chrome's viewer
  3. Falls back to clicking PDF links and URL construction if needed

Routes intercepted: **/*.pdf, **/pdf/**, **/pdfft, **/pdfdirect/**

Usage:
    python scripts/headless_download.py                # Batch all remaining
    python scripts/headless_download.py --doi "..."    # Single paper
    python scripts/headless_download.py --limit 10     # First 10 papers
    python scripts/headless_download.py --visible      # Show browser (debug)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("headless_download")

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("pip install playwright && python -m playwright install chromium")
    sys.exit(1)

import requests as http

EXTERNAL_DIR = Path("data/external")
STATUS_PATH = EXTERNAL_DIR / "zotero_sync_status.json"
AUTH_PATH = EXTERNAL_DIR / "vcu_auth.json"
EZPROXY = os.getenv("INSTITUTIONAL_PROXY_URL", "https://proxy.library.vcu.edu/login?url=")


def _get_pmid(doi: str) -> str | None:
    """Get PMID for a DOI via PubMed (fast, 10 req/s)."""
    try:
        r = http.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json",
                    "retmax": "1", "api_key": os.getenv("PUBMED_API_KEY", "")},
            timeout=10,
        )
        if r.status_code == 200:
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                return ids[0]
    except Exception:
        pass
    return None


def _try_fast_download(doi: str, pmid: str | None) -> bytes | None:
    """Try fast direct URLs (European PMC by PMID, no EZProxy needed)."""
    candidates = []
    if pmid:
        candidates.append(f"https://europepmc.org/articles/PMC{pmid}/pdf?pdf=render")
    candidates.append(f"https://europepmc.org/search?query=doi:{doi}&format=pdf")

    for url in candidates:
        try:
            r = http.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            }, timeout=30, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1000 and r.content[:4] == b"%PDF":
                logger.info("  [fast] %d KB via European PMC", len(r.content)//1024)
                return r.content
        except Exception:
            continue
    return None


def _build_requests_session() -> http.Session:
    """Build requests.Session with EZProxy auth cookies."""
    s = http.Session()
    if AUTH_PATH.exists():
        auth = json.loads(AUTH_PATH.read_text())
        for c in auth.get("cookies", []):
            s.cookies.set(c.get("name", ""), c.get("value", ""),
                         domain=c.get("domain", ""), path=c.get("path", "/"))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*",
    })
    return s


def _is_error_page(pdf_bytes: bytes) -> bool:
    """Check if PDF content is an error/access-denied page."""
    if len(pdf_bytes) < 50000:
        try:
            import re
            title_match = re.search(rb'/Title\s*\((.*?)\)', pdf_bytes[:5000])
            title = title_match.group(1).decode('latin-1', errors='replace') if title_match else ''
            if any(phrase in title.lower() for phrase in ('access denied', 'error', 'login', 'forbidden')):
                return True
        except Exception:
            pass
    text = pdf_bytes[:10000].decode('latin-1', errors='replace').lower()
    if any(phrase in text for phrase in ('please log in', 'sign in to access', 'authentication required', 'captcha')):
        return True
    return False


_OA_DOI_PREFIXES = (
    '10.3390/',  # MDPI
    '10.3389/',  # Frontiers
    '10.1371/',  # PLOS ONE
    '10.4103/',  # Medknow / open access
    '10.1186/',  # BioMed Central (also handled by European PMC)
    '10.7150/',  # Ivyspring / open access
    '10.7717/',  # PeerJ
    '10.11604/',  # Pan African Medical Journal
)

def _is_oa_prefix(doi_clean: str) -> bool:
    """Detect OA journals by DOI prefix to avoid EZProxy overhead."""
    return doi_clean.startswith(_OA_DOI_PREFIXES)


def _detect_error_page(page) -> bool:
    """Check if the current page is an error/access-denied page."""
    try:
        title = page.title()
        if any(phrase in title.lower() for phrase in ('access denied', 'forbidden', 'blocked', 'error')):
            return True
    except Exception:
        pass
    try:
        content = page.content()[:2000].lower()
        if any(phrase in content for phrase in ('access denied', 'you have been blocked', 'forbidden', 'captcha')):
            return True
    except Exception:
        pass
    return False


def _dismiss_overlays(page):
    """Dismiss cookie banners, popups, and modals before rendering PDF."""
    # Click common cookie consent buttons
    for selector in [
        'button:has-text("Accept All")',
        'button:has-text("Accept all")',
        'button:has-text("Accept Cookies")',
        'button:has-text("Accept")',
        'button:has-text("I Accept")',
        'button:has-text("OK")',
        'a:has-text("Accept")',
        '#onetrust-accept-btn-handler',
        '.accept-cookies',
        '[data-testid="cookie-policy-accept"]',
        '[aria-label="Accept cookies"]',
    ]:
        try:
            page.click(selector, timeout=1000)
            break
        except Exception:
            continue

    # Close modal dialogs / popups
    for selector in [
        '[aria-label="Close"]',
        '[aria-label="Close dialog"]',
        '.close',
        '.modal-close',
        'button.close',
        '.popup-close',
        '[data-dismiss="modal"]',
    ]:
        try:
            page.click(selector, timeout=500)
        except Exception:
            continue

    # Remove remaining overlay/banner elements via JavaScript
    try:
        page.evaluate("""
            document.querySelectorAll(
                '.cookie-banner, .cookie-consent, .cc-banner, ' +
                '.modal-backdrop, .modal-overlay, .overlay, ' +
                '.popup, .notification, .alert-banner, ' +
                '[class*="Cookie"], [id*="cookie"], [class*="popup"]'
            ).forEach(function(el) { el.remove(); });
            document.body.style.overflow = 'visible';
        """)
    except Exception:
        pass


_MIN_PAGE_WORDS = 1500   # below this, the page is likely a landing page, not full text

def _page_word_count(page) -> int:
    """Count visible words on the rendered page. Generic quality gate."""
    try:
        return len(page.evaluate("document.body.innerText").split())
    except Exception:
        return 0

def _has_download_link(page) -> bool:
    """Detect if page has a PDF/download link — signals content is elsewhere."""
    try:
        return page.evaluate(r"""() => {
            const links = document.querySelectorAll('a, button');
            for (const el of links) {
                const text = (el.textContent || '').toLowerCase();
                const href = (el.getAttribute('href') || '').toLowerCase();
                if (/(view|download)\s*(pdf|full\s*text|article)/.test(text)) return true;
                if (/\.pdf|pdfft|pdfdirect/.test(href)) return true;
            }
            return false;
        }""")
    except Exception:
        return False


def download_paper(doi: str, visible: bool = False, browser=None) -> str | None:
    """Download a single paper's PDF. Returns path, 'skipped', 'unavailable', or None.
    
    If browser is provided (from download_batch), reuses it for performance."""
    doi_clean = doi.strip().lower().replace("https://doi.org/", "")
    doi_hash = hashlib.sha256(doi_clean.encode()).hexdigest()[:8]

    existing = list(EXTERNAL_DIR.glob(f"*_{doi_hash}.pdf"))
    if existing:
        return "skipped"

    # — Tier 1: Fast direct (European PMC) —
    pmid = _get_pmid(doi_clean)
    time.sleep(0.3)   # polite rate limit for PubMed API (10 req/s with key)
    pdf_bytes = _try_fast_download(doi_clean, pmid)
    if pdf_bytes:
        fname = EXTERNAL_DIR / f"{doi_clean.replace('/', '_')[:50]}_{doi_hash}.pdf"
        fname.write_bytes(pdf_bytes)
        return str(fname)

    # — Tier 2: Playwright route interception + word-count gated page.pdf() —
    is_oa = _is_oa_prefix(doi_clean)
    target_url = f"https://doi.org/{doi_clean}" if is_oa else EZPROXY + f"https://doi.org/{doi_clean}"
    tried_direct = is_oa
    pdf_content: bytes | None = None
    page_word_count = 0

    own_browser = browser is None
    pw_ctx = None
    if own_browser:
        pw_ctx = sync_playwright()
        pw = pw_ctx.__enter__()
        browser = pw.chromium.launch(
            headless=not visible,
            args=["--disable-blink-features=AutomationControlled"],
        )

    try:
        ctx_kwargs = {
            "viewport": {"width": 1280, "height": 900},
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "accept_downloads": True,
        }
        if AUTH_PATH.exists():
            ctx_kwargs["storage_state"] = str(AUTH_PATH)
        context = browser.new_context(**ctx_kwargs)
        page = context.new_page()

        def handle_pdf_route(route):
            """Intercept PDF responses before Chrome's built-in viewer."""
            nonlocal pdf_content
            if pdf_content:
                route.continue_()
                return
            try:
                # Preserve original request params (critical for signed URLs)
                req = route.request
                resp = route.fetch(
                    url=req.url,
                    method=req.method,
                    headers=req.headers,
                    post_data=req.post_data_buffer(),
                )
                body = resp.body()
                if len(body) > 1000 and body[:4] == b"%PDF":
                    if len(body) > 5_000_000:
                        logger.debug("  Native PDF too large (%d KB), will use page.pdf()", len(body) // 1024)
                    else:
                        pdf_content = body
                        logger.debug("  Captured %d KB PDF via route interception", len(body) // 1024)
                route.fulfill(response=resp)
            except Exception:
                route.continue_()

        # Register routes on CONTEXT (applies to all pages, including new tabs)
        for pattern in ["**/*.pdf", "**/pdf/**", "**/pdfft", "**/pdfdirect/**"]:
            context.route(pattern, handle_pdf_route)

        # — Navigate (fast: commit fires on HTML response, not full DOM) —
        try:
            page.goto(target_url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)  # let JS render
            landing_url = page.url
            page_word_count = _page_word_count(page)

            # If error page and we went through EZProxy, retry without it
            if not tried_direct and _detect_error_page(page):
                logger.debug("  Error page via EZProxy — retrying directly")
                page.goto(f"https://doi.org/{doi_clean}", timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                landing_url = page.url
                page_word_count = _page_word_count(page)
                tried_direct = True

            if not landing_url or (not tried_direct and "login" in landing_url.lower() and "doi.org" not in landing_url):
                context.close()
                return None
            if _detect_error_page(page):
                logger.debug("  Error page — skipping")
                context.close()
                return None

        except Exception as e:
            logger.debug("Navigation error: %s", str(e)[:80])
            context.close()
            return None

        # — Native PDF captured during navigation —
        if pdf_content:
            context.close()
            fname = EXTERNAL_DIR / f"{doi_clean.replace('/', '_')[:50]}_{doi_hash}.pdf"
            fname.write_bytes(pdf_content)
            logger.info("  OK native PDF from navigation (%d KB)", len(pdf_content) // 1024)
            return str(fname)

        # — Phase 1: Click PDF links (works for same-tab PDF navigation) —
        click_selectors = [
            'a:has-text("Download PDF")',
            'a:has-text("PDF")',
            'a:has-text("Full Text")',
            'a:has-text("View PDF")',
            'a:has-text("Article PDF")',
            'button:has-text("View PDF")',
            'button:has-text("Download PDF")',
            'button:has-text("PDF")',
            'a[href*=".pdf"]',
            'a[href*="/pdf/"]',
            'a[href*="/pdfft"]',
            'a[href*="download"]',
            '.pdf-download-btn',
        ]
        for selector in click_selectors:
            if pdf_content:
                break
            try:
                page.click(selector, timeout=3000)
                page.wait_for_timeout(2000)
            except Exception:
                continue

        # — Phase 2: Try clicking PDF links (works for same-tab) —
        if not pdf_content:
            from urllib.parse import urljoin as _urljoin

            # Collect potential PDF URLs first (before navigation changes the page)
            pdf_candidate_urls = []
            for el in page.query_selector_all('a[href*="pdfft"], a[href*=".pdf"], a[href*="pdfdirect"], a[href*="/pdf/"], a:has-text("PDF"), a:has-text("View PDF"), a:has-text("Download PDF")'):
                try:
                    href = el.get_attribute("href")
                    if href and not href.startswith("#") and not href.startswith("javascript:"):
                        pdf_url = _urljoin(page.url, href)
                        pdf_candidate_urls.append(pdf_url)
                except Exception:
                    continue

            # Try navigating to each candidate URL in the current page
            for pdf_url in pdf_candidate_urls:
                if pdf_content:
                    break
                try:
                    logger.debug("  Navigating to PDF: %s", pdf_url[:80])
                    page.goto(pdf_url, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_timeout(2000)
                except Exception:
                    continue

            # — Phase 3: Download via requests (bypasses Chrome PDF viewer, works for signed URLs) —
            if not pdf_content and pdf_candidate_urls:
                try:
                    session = _build_requests_session()
                    session.headers["Referer"] = landing_url
                    r = session.get(pdf_candidate_urls[0], timeout=45, allow_redirects=True)
                    if r.status_code == 200 and len(r.content) > 20000 and r.content[:4] == b"%PDF":
                        pdf_content = r.content
                        logger.info("  Downloaded native PDF via requests (%d KB)", len(pdf_content) // 1024)
                except Exception as e:
                    logger.debug("  Phase 3 download failed: %s", str(e)[:80])

        # — Word-count gated fallback: only page.pdf() if page has real content —
        if not pdf_content:
            # Re-capture word count after possible Phase 2 navigation
            page_word_count = _page_word_count(page)
            has_pdf_link = _has_download_link(page)

            if page_word_count < _MIN_PAGE_WORDS:
                logger.debug("  Page too sparse (%d words < %d) — marking unavailable",
                             page_word_count, _MIN_PAGE_WORDS)
                context.close()
                return "unavailable"

            if has_pdf_link and page_word_count < 5000:
                logger.debug("  Landing page (%d words, has PDF link) — marking unavailable",
                             page_word_count)
                context.close()
                return "unavailable"
            logger.debug("  No native PDF — generating from article page (%d words)", page_word_count)
            try:
                _dismiss_overlays(page)
                page.wait_for_timeout(500)
                tmp_path = EXTERNAL_DIR / f"temp_{doi_hash}.pdf"
                page.pdf(path=str(tmp_path))
                if tmp_path.exists() and tmp_path.stat().st_size > 1000:
                    pdf_bytes = tmp_path.read_bytes()
                    tmp_path.unlink()
                    if _is_error_page(pdf_bytes):
                        logger.debug("  page.pdf() produced error page — discarding")
                        context.close()
                        return "unavailable"
                    pdf_content = pdf_bytes
                    logger.info("  Generated PDF from page (%d KB)", len(pdf_content) // 1024)
            except Exception as e:
                logger.debug("  page.pdf() failed: %s", str(e)[:80])

        context.close()

    finally:
        if own_browser and pw_ctx:
            pw_ctx.__exit__(None, None, None)

    if pdf_content and len(pdf_content) > 1000 and pdf_content[:4] == b"%PDF":
        fname = EXTERNAL_DIR / f"{doi_clean.replace('/', '_')[:50]}_{doi_hash}.pdf"
        fname.write_bytes(pdf_content)
        logger.info("  OK %s (%d KB)", doi_clean[:40], len(pdf_content) // 1024)
        return str(fname)

    return None


def _attach_to_zotero(item_key: str, pdf_path: Path):
    """Attach downloaded PDF to Zotero item. Silent on failure."""
    MAX_SIZE_MB = 5
    if pdf_path.stat().st_size > MAX_SIZE_MB * 1_000_000:
        logger.debug("  PDF too large for Zotero API (%d KB)", pdf_path.stat().st_size // 1024)
        return
    try:
        from pyzotero import zotero as zt
        zot = zt.Zotero(
            os.getenv("ZOTERO_LIBRARY_ID", ""),
            "user",
            os.getenv("ZOTERO_API_KEY", ""),
        )
        zot.attachment_simple([str(pdf_path)], item_key)
        logger.info("  Attached to Zotero item %s", item_key)
    except Exception:
        logger.debug("  Zotero attach skipped")


def main():
    import random as _random

    parser = argparse.ArgumentParser(description="Headless PDF downloader")
    parser.add_argument("--doi", type=str)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.doi:
        result = download_paper(args.doi, visible=args.visible)
        print(f"Result: {result}")
        return

    if not STATUS_PATH.exists():
        logger.error("No sync status found")
        return

    status = json.loads(STATUS_PATH.read_text())
    to_download = []
    for key, item in status.items():
        if not isinstance(item, dict):
            continue
        if item.get("status") not in ("awaiting_zotero", "zotero_created"):
            continue
        doi = item.get("doi", "")
        if not doi:
            item["status"] = "unavailable"
            item["reason"] = "no_doi"
            continue
        to_download.append((key, item))

    if args.limit:
        to_download = to_download[:args.limit]

    logger.info("BATCH: %d papers (persistent browser, 1-2.5s pacing)", len(to_download))

    # — Persistent browser for batch mode —
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.visible,
            args=["--disable-blink-features=AutomationControlled"],
        )

        results = {"downloaded": 0, "skipped": 0, "unavailable": 0, "failed": 0}
        for i, (key, item) in enumerate(to_download):
            doi = item.get("doi", "")
            title = item.get("title", "")[:60]
            logger.info("[%d/%d] %s", i + 1, len(to_download), title)

            zotero_key = item.get("zotero_key", "")

            try:
                result = download_paper(doi, visible=args.visible, browser=browser)
                if result == "skipped":
                    results["skipped"] += 1
                    doi_clean = doi.strip().lower().replace("https://doi.org/", "")
                    doi_hash = hashlib.sha256(doi_clean.encode()).hexdigest()[:8]
                    existing = list(EXTERNAL_DIR.glob(f"*_{doi_hash}.pdf"))
                    if existing:
                        item["pdf_path"] = str(existing[0])
                        item["status"] = "headless_downloaded"
                        if zotero_key:
                            _attach_to_zotero(zotero_key, existing[0])
                elif result == "unavailable":
                    results["unavailable"] += 1
                    item["status"] = "unavailable"
                    item["reason"] = "content_sparse"
                elif result:
                    results["downloaded"] += 1
                    item["status"] = "headless_downloaded"
                    item["pdf_path"] = result
                    if zotero_key:
                        _attach_to_zotero(zotero_key, Path(result))
                else:
                    results["failed"] += 1
            except Exception as e:
                results["failed"] += 1
                logger.warning("FAIL: %s", str(e)[:80])

            # — Rate limit: polite pacing to avoid bot detection —
            if i + 1 < len(to_download):
                time.sleep(_random.uniform(1.0, 2.5))

            # Checkpoint every 10 papers
            if (i + 1) % 10 == 0:
                logger.info("  --- %d/%d: %d ok, %d una, %d skip, %d fail ---",
                             i + 1, len(to_download), results["downloaded"],
                             results["unavailable"], results["skipped"], results["failed"])
                status["_updated_at"] = time.time()
                STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False))

        browser.close()

    status["_updated_at"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    logger.info("DONE: %d downloaded, %d unavailable, %d skipped, %d failed",
                 results["downloaded"], results["unavailable"],
                 results["skipped"], results["failed"])


if __name__ == "__main__":
    sys.exit(main())
