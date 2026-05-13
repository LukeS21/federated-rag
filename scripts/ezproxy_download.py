#!/usr/bin/env python3
"""
Headless PDF downloader via VCU EZProxy + Playwright.

Resolves DOIs through Unpaywall, routes publisher PDF URLs through
VCU's EZProxy for institutional access, and downloads PDFs to
data/external/ using Playwright (headless Chromium).

Usage:
    # One-time setup: authenticate with VCU SSO in a browser window
    python scripts/ezproxy_download.py --setup
    
    # Download remaining unresolved PDFs (requires setup first)
    python scripts/ezproxy_download.py
    
    # Download a specific DOI
    python scripts/ezproxy_download.py --doi "10.1016/j.actbio.2013.06.027"
    
    # Download all 184 (retry all)
    python scripts/ezproxy_download.py --all

Auth state saved to: data/external/vcu_auth.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv; load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ezproxy_download")

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("playwright not installed. Run: pip install playwright && python -m playwright install chromium")
    sys.exit(1)

EXTERNAL_DIR = Path("data/external")
STATUS_PATH = EXTERNAL_DIR / "zotero_sync_status.json"
AUTH_PATH = EXTERNAL_DIR / "vcu_auth.json"
EZPROXY = os.getenv("INSTITUTIONAL_PROXY_URL", "https://proxy.library.vcu.edu/login?url=")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")

# Module-level S2 client (created once to avoid rate-limit issues)
_s2_client = None


def _get_s2():
    global _s2_client
    if _s2_client is None:
        from src.retrieval.semantic_scholar import SemanticScholarClient
        _s2_client = SemanticScholarClient()
    return _s2_client

# DOIs from papers Zotero couldn't find — we'll try EZProxy for these


async def setup_auth():
    """Open a browser for the user to authenticate with VCU SSO.
    
    Navigates to the EZProxy URL, user logs in manually, then auth state
    (cookies, localStorage) is saved for future headless sessions.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = await context.new_page()
        
        # Navigate to VCU EZProxy login
        logger.info("Opening VCU EZProxy login page...")
        await page.goto("https://proxy.library.vcu.edu/login", timeout=30000)
        
        logger.info("Log in and approve DUO 2FA in the browser window.")
        logger.info("After login, VCU redirects to guides.library.vcu.edu — that confirms auth.")
        
        # Wait for redirect to guides.library.vcu.edu (post-login destination)
        try:
            await page.wait_for_url("**/guides.library.vcu.edu/**", timeout=300000)
            logger.info("Auth confirmed — redirect to guides.library.vcu.edu detected")
        except Exception:
            logger.info("Timeout — saving state anyway (you can try again if downloads fail)")
        
        logger.info("Saving auth state...")
        await context.storage_state(path=str(AUTH_PATH))
        logger.info("Auth state saved to %s", AUTH_PATH)
        
        await browser.close()


async def download_via_ezproxy(
    doi: str,
    output_dir: Path = EXTERNAL_DIR,
    zotero_key: str | None = None,
    headless: bool = True,
) -> str | None:
    """Download a paper's PDF via EZProxy using Playwright.

    Args:
        doi: The DOI to resolve.
        output_dir: Where to save the PDF.
        zotero_key: Optional Zotero item key to attach PDF to.
        headless: If False, show browser window for debugging.

    Returns:
        Path to downloaded PDF, "skipped" if already exists, None if failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doi_clean = doi.strip().lower().replace("https://doi.org/", "")
    doi_hash = hashlib.sha256(doi_clean.encode()).hexdigest()[:8]
    
    # Check if already downloaded
    existing = list(output_dir.glob(f"*_{doi_hash}.pdf"))
    if existing:
        return "skipped"
    
    # Step 1: Get publisher PDF URL from Unpaywall + S2
    import requests as req
    pdf_url = ""
    
    # Try Unpaywall first
    try:
        resp = req.get(
            f"https://api.unpaywall.org/v2/{doi_clean}",
            params={"email": UNPAYWALL_EMAIL} if UNPAYWALL_EMAIL else {},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            best = data.get("best_oa_location", {})
            pdf_url = best.get("url_for_pdf", "")
            if not pdf_url:
                for loc in data.get("oa_locations", []):
                    u = loc.get("url_for_pdf", "")
                    if u:
                        pdf_url = u
                        break
    except Exception:
        pass
    
    # Try Semantic Scholar openAccessPdf as backup URL
    s2_pdf = ""
    if not pdf_url:
        try:
            from src.retrieval.semantic_scholar import SemanticScholarClient
            s2 = SemanticScholarClient()
            results = s2.search(doi_clean.split('/')[-1][:40], limit=3)
            for r in results:
                if r.get("open_access_pdf"):
                    s2_pdf = r["open_access_pdf"]
                    break
            if not s2_pdf and results:
                # Try DOI lookup
                r = s2.search_by_doi(doi_clean)
                if r and r.get("open_access_pdf"):
                    s2_pdf = r["open_access_pdf"]
            time.sleep(1.5)  # S2 rate limit
        except Exception:
            pass
    
    # Choose best URL: S2 (repository) preferred over publisher
    final_url = s2_pdf or pdf_url
    if not final_url:
        # Try European PMC as last resort
        pmid = _get_pmid_for_doi(doi_clean)
        if pmid:
            final_url = f"https://europepmc.org/articles/PMC{pmid}/pdf?pdf=render"
        else:
            final_url = f"https://doi.org/{doi_clean}"
    
    # Decide whether to route through EZProxy
    is_publisher = not any(domain in final_url.lower() for domain in [
        "europepmc.org", "ncbi.nlm.nih.gov", "pubmed",
        "arxiv.org", "biorxiv.org", "medrxiv.org",
    ])
    
    if is_publisher and "proxy.library.vcu.edu" not in final_url:
        target_url = EZPROXY + final_url
    else:
        target_url = final_url
    
    # Step 2: Download via Playwright
    auth_ctx = {}
    if AUTH_PATH.exists():
        auth_ctx = {"storage_state": str(AUTH_PATH)}
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            **auth_ctx,
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            accept_downloads=True,
        )
        page = await context.new_page()
        pdf_content: bytes | None = None
        
        try:
            # Set up download handler BEFORE navigation
            download_event = None
            
            async def handle_download(download):
                nonlocal download_event
                download_event = download
            
            page.on("download", handle_download)
            
            # Navigate — the download may trigger during navigation
            logger.debug("Navigating: %s", target_url[:120])
            try:
                resp = await page.goto(target_url, timeout=45000, wait_until="load")
                if resp:
                    ct = resp.headers.get("content-type", "").lower()
                    if "pdf" in ct:
                        pdf_content = await resp.body()
                        logger.debug("Got PDF directly (%d bytes)", len(pdf_content))
                
                # Wait for the page to fully render — downloads may trigger on load
                await page.wait_for_timeout(3000)
                
            except Exception as e:
                if "Download is starting" in str(e) or "net::ERR_ABORTED" in str(e):
                    logger.debug("Download triggered during navigation")
                elif "timeout" in str(e).lower():
                    logger.debug("Page load timeout — may still have triggered download")
                else:
                    logger.debug("Goto error: %s", str(e)[:80])
            
            # Check if download was captured during navigation
            if download_event:
                logger.debug("Processing captured download...")
                try:
                    tmp = output_dir / f"tmp_{doi_hash}.pdf"
                    await download_event.save_as(str(tmp))
                    if tmp.exists():
                        pdf_content = tmp.read_bytes()
                        tmp.unlink()
                        logger.debug("Saved download: %d bytes", len(pdf_content))
                except Exception as e2:
                    logger.debug("Save download failed: %s", str(e2)[:80])
            
            # If still no PDF, try clicking download links on the page
            if not pdf_content:
                logger.debug("No direct download — searching page for PDF links...")
                # Let page JS render
                await page.wait_for_timeout(2000)
                
                # Log the page title for debugging
                try:
                    page_title = await page.title()
                    logger.debug("Page title: %s", page_title[:80])
                except Exception:
                    pass
                for selector in [
                    'a[href*=".pdf"]',
                    'a[href*="download"]',
                    'button:has-text("PDF")',
                    'a:has-text("Download PDF")',
                    'a:has-text("Full Text")',
                ]:
                    try:
                        dl_event = None
                        async def _on_dl(d): nonlocal dl_event; dl_event = d
                        page.on("download", _on_dl)
                        await page.click(selector, timeout=5000)
                        await page.wait_for_timeout(2000)
                        if dl_event:
                            tmp = output_dir / f"tmp_{doi_hash}.pdf"
                            await dl_event.save_as(str(tmp))
                            if tmp.exists():
                                pdf_content = tmp.read_bytes()
                                tmp.unlink()
                                break
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("Page error for %s: %s", doi_clean, str(e)[:80])
        finally:
            await browser.close()
    
    if pdf_content and len(pdf_content) > 1000:
        if pdf_content[:4] == b"%PDF" or b"PDF" in pdf_content[:100]:
            title = doi_clean.replace("/", "_")[:50]
            fname = output_dir / f"{title}_{doi_hash}.pdf"
            fname.write_bytes(pdf_content)
            logger.info("OK: %s (%d KB)", fname.name, len(pdf_content) // 1024)
            
            if zotero_key:
                _attach_to_zotero(zotero_key, fname)
            
            return str(fname)
    
    return None


def _attach_to_zotero(item_key: str, pdf_path: Path):
    """Attach a downloaded PDF to a Zotero item. Errors are non-fatal."""
    try:
        from pyzotero import zotero as zt
        zot = zt.Zotero(
            os.getenv("ZOTERO_LIBRARY_ID", ""),
            "user",
            os.getenv("ZOTERO_API_KEY", ""),
        )
        if pdf_path.stat().st_size < 5_000_000:  # Only upload files < 5MB
            zot.attachment_simple([str(pdf_path)], item_key)
            logger.debug("Attached to Zotero item %s", item_key)
        else:
            logger.debug("PDF too large for Zotero API (%d KB) — saved locally only",
                        pdf_path.stat().st_size // 1024)
    except Exception as e:
        logger.debug("Zotero attach skipped: %s", str(e)[:80])


def download_remaining():
    """Download PDFs for all papers still awaiting resolution. Uses requests (fast)."""
    if not STATUS_PATH.exists():
        logger.error("No sync status found.")
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
            continue
        to_download.append((key, item))
    
    logger.info("DOWNLOADING %d remaining papers via requests + EZProxy", len(to_download))
    
    results = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    for i, (key, item) in enumerate(to_download):
        doi = item.get("doi", "")
        zk = item.get("zotero_key", "")
        title = item.get("title", "")[:60]
        
        logger.info("[%d/%d] %s", i + 1, len(to_download), title)
        
        try:
            result = download_pdf_direct(doi, zotero_key=zk)
            if result == "skipped":
                results["skipped"] += 1
            elif result:
                results["downloaded"] += 1
                item["status"] = "ezproxy_downloaded"
                item["pdf_path"] = result
            else:
                results["failed"] += 1
        except Exception as e:
            results["failed"] += 1
            logger.warning("FAIL: %s", str(e)[:80])
        
        if i > 0 and i % 10 == 0:
            logger.info("  --- %d/%d: %d ok, %d fail ---",
                         i + 1, len(to_download), results["downloaded"], results["failed"])
            status["_updated_at"] = time.time()
            STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False))
        
        time.sleep(1.0)
    
    status["_updated_at"] = time.time()
    STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    
    logger.info("DONE: %d downloaded, %d skipped, %d failed",
                 results["downloaded"], results["skipped"], results["failed"])


def _build_requests_session() -> requests.Session:
    """Build a requests.Session with VCU auth cookies from Playwright auth state."""
    import requests as req
    session = req.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    })
    if AUTH_PATH.exists():
        auth = json.loads(AUTH_PATH.read_text())
        for cookie in auth.get("cookies", []):
            session.cookies.set(
                cookie.get("name", ""),
                cookie.get("value", ""),
                domain=cookie.get("domain", ""),
                path=cookie.get("path", "/"),
            )
    return session


def download_pdf_direct(
    doi: str,
    output_dir: Path = EXTERNAL_DIR,
    zotero_key: str | None = None,
) -> str | None:
    """Download a paper's PDF using requests (no browser overhead).
    
    Resolution chain:
      1. Semantic Scholar openAccessPdf
      2. European PMC (by PMID, obtained via S2)
      3. PMC OA Service
      4. EZProxy + publisher (with VCU auth cookies)
      5. Direct Unpaywall PDF URL
      6. Direct DOI redirect
    
    Returns: Path to PDF, 'skipped', or None.
    """
    import requests as req
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doi_clean = doi.strip().lower().replace("https://doi.org/", "")
    doi_hash = hashlib.sha256(doi_clean.encode()).hexdigest()[:8]
    
    # Already downloaded?
    existing = list(output_dir.glob(f"*_{doi_hash}.pdf"))
    if existing:
        return "skipped"
    
    session = _build_requests_session()
    urls_to_try: list[tuple[str, str]] = []  # (source, url)
    
    # Layer 1: Semantic Scholar openAccessPdf + PMID lookup
    pmid = None
    try:
        s2 = _get_s2()
        r = s2.search_by_doi(doi_clean)
        time.sleep(3.0)  # S2 free tier: 1 req/s. 3s margin avoids 429.
        if r:
            s2_pdf = r.get("open_access_pdf", "")
            if s2_pdf:
                urls_to_try.append(("semantic_scholar", s2_pdf))
            pmid = r.get("pmid", "")
    except Exception:
        time.sleep(3.0)
    
    # Layer 1b: PubMed PMID lookup (fast, 10 req/s with API key)
    if not pmid:
        try:
            resp = req.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": f"{doi_clean}[doi]",
                    "retmode": "json",
                    "retmax": "1",
                    "api_key": os.getenv("PUBMED_API_KEY", ""),
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                if ids:
                    pmid = ids[0]
        except Exception:
            pass
    
    # Layer 2: European PMC / PMC OA by PMID
    if pmid:
        urls_to_try.append(("europe_pmc", f"https://europepmc.org/articles/PMC{pmid}/pdf?pdf=render"))
        urls_to_try.append(("pmc_oa", f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/pdf/"))
    
    # Layer 3: Unpaywall → publisher PDF URL
    try:
        resp = req.get(
            f"https://api.unpaywall.org/v2/{doi_clean}",
            params={"email": UNPAYWALL_EMAIL} if UNPAYWALL_EMAIL else {},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            best = data.get("best_oa_location", {})
            uw_pdf = best.get("url_for_pdf", "")
            if uw_pdf:
                # Try both direct and proxied
                is_repo = any(d in uw_pdf.lower() for d in [
                    "europepmc.org", "ncbi.nlm.nih.gov", "arxiv.org",
                    "biorxiv.org", "medrxiv.org", "pubmed",
                ])
                if is_repo:
                    urls_to_try.append(("unpaywall_direct", uw_pdf))
                else:
                    # Try direct first, then EZProxy
                    urls_to_try.append(("unpaywall_direct", uw_pdf))
                    urls_to_try.append(("ezproxy", EZPROXY + uw_pdf))
    except Exception:
        pass
    
    # Layer 4: EZProxy via DOI redirect
    urls_to_try.append(("ezproxy_doi", EZPROXY + f"https://doi.org/{doi_clean}"))
    
    # Try each URL
    for source, url in urls_to_try:
        if not url:
            continue
        try:
            r = session.get(url, timeout=45, allow_redirects=True, stream=True)
            if r.status_code == 200:
                content = b""
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                        if len(content) > 20_000_000:  # 20 MB cap
                            break
                
                if len(content) > 1000 and content[:4] == b"%PDF":
                    safe = doi_clean.replace("/", "_")[:50]
                    fname = output_dir / f"{safe}_{doi_hash}.pdf"
                    fname.write_bytes(content)
                    logger.info("  [%s] %s (%d KB)", source, fname.name, len(content) // 1024)
                    if zotero_key:
                        _attach_to_zotero(zotero_key, fname)
                    return str(fname)
        except Exception:
            continue
    
    return None


async def main_async():
    parser = argparse.ArgumentParser(description="EZProxy + Playwright PDF downloader")
    parser.add_argument("--setup", action="store_true", help="Authenticate with VCU SSO")
    parser.add_argument("--doi", type=str, help="Download a specific DOI")
    parser.add_argument("--all", action="store_true", help="Retry all papers (even already resolved)")
    parser.add_argument("--visible", action="store_true", help="Show browser window (setup only)")
    args = parser.parse_args()
    
    if args.setup:
        await setup_auth()
        return
    
    if args.doi:
        result = download_pdf_direct(args.doi)
        print(f"Result: {result}")
        return
    
    # download_remaining is synchronous now (uses requests, not Playwright)
    download_remaining()


def main():
    # Check if we need asyncio (only for setup)
    import sys as _sys
    if "--setup" in _sys.argv:
        asyncio.run(main_async())
    else:
        # Synchronous path — no asyncio needed for downloads
        if "--doi" in _sys.argv:
            parser = argparse.ArgumentParser()
            parser.add_argument("--doi", type=str)
            args, _ = parser.parse_known_args()
            if args.doi:
                print(f"Result: {download_pdf_direct(args.doi)}")
                return
        download_remaining()


if __name__ == "__main__":
    sys.exit(main())
