#!/usr/bin/env python
"""
Phase 8: Zotero Sync — Automated PDF acquisition via Zotero Desktop integration.

Batch-creates Zotero items from missing.json, tries direct PDF downloads
(Unpaywall/PMC/S2), polls Zotero API for Desktop-found PDFs, and downloads
everything to data/external/.

Usage:
    python scripts/zotero_sync.py              # Phase 1: create items + direct downloads
    python scripts/zotero_sync.py --poll       # Phase 2: check Zotero for new PDFs
    python scripts/zotero_sync.py --loop       # Phase 2b: poll in a loop (leave overnight)
    python scripts/zotero_sync.py --status     # View current sync status
    python scripts/zotero_sync.py --full-sync  # Phase 1 + 2 combined
    python scripts/zotero_sync.py --sync       # Cross-sync: data/external/ ↔ Zotero

Zotero Desktop Setup (one-time):
  1. Preferences → Advanced → General → OpenURL Resolver:
     https://vcu.primo.exlibrisgroup.com/openurl/01VCU_INST/01VCU_INST:VCUL
  2. Preferences → General: check "Automatically attach PDFs when saving items"
  3. Preferences → Proxies: ensure "Enable proxy redirection" is checked

After running, find all external papers in Zotero under the "external" collection.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.citation_manager.zotero_pdf import (
    _get_zotero,
    get_item_attachments,
    has_pdf_attachment,
    download_pdf_attachment,
    create_zotero_items_from_missing,
    try_direct_pdf_download,
    attach_pdf_to_item,
    ensure_external_collection,
    add_to_external_collection,
    sync_external_pdfs_with_zotero,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zotero_sync")

STATUS_PATH = Path("data/external/zotero_sync_status.json")
MISSING_PATH = Path("data/external/missing.json")
EXTERNAL_DIR = Path("data/external")

POLL_INTERVALS = [300, 900, 3600, 14400, 43200]  # 5m, 15m, 1h, 4h, 12h


def load_status() -> Dict:
    """Load the sync status file or return empty."""
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_status(status: Dict) -> None:
    """Persist the sync status to disk."""
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status["_updated_at"] = time.time()
    status["_updated_iso"] = time.ctime()
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def status_key(paper: Dict) -> str:
    """Generate a stable key from paper metadata."""
    return paper.get("doi", "") or paper.get("pmid", "") or paper.get("title", "unknown")[:60]


def phase1_create_and_download() -> Dict:
    """Phase 1: Create Zotero items and try direct PDF downloads.

    Returns updated status dict.
    """
    status = load_status()

    logger.info("=" * 60)
    logger.info("PHASE 1: Batch-create Zotero items + direct PDF downloads")
    logger.info("=" * 60)

    if not MISSING_PATH.exists():
        logger.error("Missing file not found: %s", MISSING_PATH)
        return status

    # Load missing papers
    data = json.loads(MISSING_PATH.read_text(encoding="utf-8"))
    papers: list = data if isinstance(data, list) else []
    logger.info("Loaded %d papers from missing.json", len(papers))

    if not papers:
        logger.info("Nothing to do — missing.json is empty")
        return status

    # Step A: Batch-create Zotero items
    logger.info("Step A: Creating Zotero items...")
    try:
        zotero_results = create_zotero_items_from_missing(MISSING_PATH)
    except Exception as e:
        logger.error("Zotero item creation failed: %s", e)
        return status

    # Merge Zotero keys into status
    updated = 0
    for key, zr in zotero_results.items():
        if key not in status:
            status[key] = {}
        status[key].update(zr)
        status[key]["_phase1_time"] = time.time()
        updated += 1

    logger.info("Step A complete: %d items in Zotero", updated)

    # Step B: Try direct PDF downloads
    logger.info("Step B: Trying direct PDF downloads (Unpaywall/PMC/S2)...")
    direct_count = 0
    zot = _get_zotero()

    for key, item in list(status.items()):
        if item.get("status") in ("direct_downloaded", "zotero_found", "manual_needed"):
            continue  # already resolved

        zotero_key = item.get("zotero_key", "")
        doi = item.get("doi", "")
        title = item.get("title", "")

        # Try direct download
        paper_info = {
            "doi": doi,
            "pmid": item.get("pmid", ""),
            "title": title,
            "year": item.get("year", ""),
        }
        pdf_path = try_direct_pdf_download(paper_info)

        if pdf_path:
            item["status"] = "direct_downloaded"
            item["pdf_path"] = str(pdf_path)
            item["source"] = "direct"
            direct_count += 1

            # Also attach to Zotero item
            if zot and zotero_key:
                try:
                    attach_pdf_to_item(zot, zotero_key, pdf_path)
                    logger.info("Attached PDF to Zotero item %s", zotero_key)
                except Exception as e:
                    logger.warning("Failed to attach PDF: %s", e)
        else:
            # No direct download — mark for Zotero Desktop
            if item.get("status") != "zotero_failed":
                item["status"] = "awaiting_zotero"
                item["attempts"] = item.get("attempts", 0)
                item["first_attempt"] = item.get("first_attempt", time.time())

        if direct_count > 0 and direct_count % 5 == 0:
            save_status(status)

    save_status(status)

    # Summary
    awaiting = sum(1 for v in status.values() if isinstance(v, dict) and v.get("status") == "awaiting_zotero")
    downloaded = sum(1 for v in status.values() if isinstance(v, dict) and v.get("status") == "direct_downloaded")
    logger.info("Phase 1 complete: %d direct downloads, %d awaiting Zotero Desktop",
                 downloaded, awaiting)
    return status


def phase2_poll_zotero() -> Dict:
    """Phase 2: Poll Zotero API for Desktop-found PDFs.

    Returns updated status dict.
    """
    status = load_status()
    zot = _get_zotero()
    if zot is None:
        logger.error("Zotero credentials not configured — cannot poll")
        return status

    logger.info("=" * 60)
    logger.info("PHASE 2: Polling Zotero for Desktop-found PDFs")
    logger.info("=" * 60)

    found_count = 0
    still_waiting = 0
    manual_count = 0

    for key, item in list(status.items()):
        if not isinstance(item, dict):
            continue
        if item.get("status") not in ("awaiting_zotero",):
            continue

        zotero_key = item.get("zotero_key", "")
        if not zotero_key:
            continue

        attempts = item.get("attempts", 0)
        first_attempt = item.get("first_attempt", time.time())
        elapsed = time.time() - first_attempt

        # Check if past timeout (24 hours = 86400 seconds)
        max_time = 86400  # 24 hours
        if elapsed > max_time and attempts > 5:
            item["status"] = "manual_needed"
            item["error"] = f"Timeout after {elapsed/3600:.1f}h"
            manual_count += 1
            continue

        # Check Zotero for PDF attachment
        has_pdf, att_key = has_pdf_attachment(zot, zotero_key)
        if has_pdf and att_key:
            pdf_path = download_pdf_attachment(
                zot, att_key, EXTERNAL_DIR,
                filename=key.replace("/", "_")[:80] if key else None,
            )
            if pdf_path:
                item["status"] = "zotero_found"
                item["pdf_path"] = str(pdf_path)
                item["attachment_key"] = att_key
                item["found_after_attempts"] = attempts + 1
                found_count += 1
                # Ensure item is in 'external' collection
                add_to_external_collection(zot, zotero_key)
                logger.info("Found PDF via Zotero: %s", item.get("title", "")[:60])
            else:
                item["attempts"] = attempts + 1
                still_waiting += 1
        else:
            item["attempts"] = attempts + 1
            still_waiting += 1

        if (found_count + still_waiting) % 10 == 0:
            save_status(status)

    save_status(status)
    logger.info("Phase 2: %d found, %d still waiting, %d timed out (manual)",
                 found_count, still_waiting, manual_count)
    return status


def loop_poll():
    """Phase 2b: Poll in a loop with exponential backoff."""
    logger.info("Starting poll loop with exponential backoff...")
    interval_idx = 0

    while True:
        status = phase2_poll_zotero()
        awaiting = sum(
            1 for v in status.values()
            if isinstance(v, dict) and v.get("status") == "awaiting_zotero"
        )
        if awaiting == 0:
            logger.info("All items resolved — exiting poll loop")
            break

        interval = POLL_INTERVALS[min(interval_idx, len(POLL_INTERVALS) - 1)]
        logger.info("%d items still awaiting Zotero Desktop. Next poll in %ds (%s)",
                     awaiting, interval,
                     time.ctime(time.time() + interval))
        time.sleep(interval)
        interval_idx = min(interval_idx + 1, len(POLL_INTERVALS) - 1)


def print_status():
    """Print a readable summary of the sync status."""
    status = load_status()
    if not status:
        print("No sync status found. Run `python scripts/zotero_sync.py` first.")
        return

    # Count by status
    counts = {
        "zotero_created": 0,
        "direct_downloaded": 0,
        "zotero_found": 0,
        "awaiting_zotero": 0,
        "manual_needed": 0,
        "zotero_failed": 0,
        "zotero_error": 0,
        "other": 0,
    }
    for key, item in status.items():
        if key.startswith("_"):
            continue
        st = item.get("status", "unknown") if isinstance(item, dict) else str(item)
        if st in counts:
            counts[st] += 1
        else:
            counts["other"] += 1

    total = sum(counts.values())
    resolved = counts["direct_downloaded"] + counts["zotero_found"]

    print(f"\n{'='*60}")
    print("  ZOTERO SYNC STATUS")
    print(f"{'='*60}")
    print(f"  Total papers:         {total}")
    print(f"  Direct downloaded:    {counts['direct_downloaded']}")
    print(f"  Found via Zotero:     {counts['zotero_found']}")
    print(f"  Awaiting Zotero:      {counts['awaiting_zotero']}")
    print(f"  Manual needed:        {counts['manual_needed']}")
    print(f"  Failed (Zotero API):  {counts['zotero_failed']}")
    print(f"  Resolution rate:      {resolved}/{total} ({resolved*100//max(total,1)}%)")
    if counts['awaiting_zotero']:
        print(f"\n  {counts['awaiting_zotero']} papers awaiting Zotero Desktop auto-find.")
        print(f"  Keep Zotero Desktop open and run `python scripts/zotero_sync.py --poll`.")
    if counts['manual_needed']:
        print(f"\n  {counts['manual_needed']} papers need manual retrieval.")
        print(f"  Check data/external/zotero_sync_status.json for details.")
    print(f"\n  Updated: {status.get('_updated_iso', 'unknown')}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Zotero Sync")
    parser.add_argument("--poll", action="store_true",
                        help="Phase 2: Poll Zotero for Desktop-found PDFs")
    parser.add_argument("--loop", action="store_true",
                        help="Phase 2b: Poll in a loop (leave overnight)")
    parser.add_argument("--status", action="store_true",
                        help="View current sync status")
    parser.add_argument("--full-sync", action="store_true",
                        help="Run Phase 1 + Phase 2 combined")
    parser.add_argument("--sync", action="store_true",
                        help="Cross-sync: ensure data/external/ PDFs are attached in Zotero")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.loop:
        loop_poll()
        return

    if args.poll:
        phase2_poll_zotero()
        print_status()
        return

    if args.full_sync:
        phase1_create_and_download()
        phase2_poll_zotero()
        print_status()
        return

    if args.sync:
        logger.info("Cross-syncing data/external/ ↔ Zotero 'external' collection...")
        result = sync_external_pdfs_with_zotero()
        print(f"\n  Sync result: {result['synced']} synced, "
              f"{result['skipped']} already attached, {result['errors']} errors")
        return

    # Default: Phase 1 only
    phase1_create_and_download()
    print_status()


if __name__ == "__main__":
    sys.exit(main())
