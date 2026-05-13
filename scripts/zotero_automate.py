#!/usr/bin/env python3
"""
Zotero Desktop automation — pure pyautogui, no AppleScript.

Triggers "Find Available PDFs" via keyboard shortcuts for all items
in the "external" collection.

Usage: python scripts/zotero_automate.py
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zotero_automate")

try:
    import pyautogui
    pyautogui.FAILSAFE = True
except ImportError:
    print("pyautogui not installed. Run: pip install pyautogui")
    sys.exit(1)

STATUS_PATH = Path("data/external/zotero_sync_status.json")
EXTERNAL_DIR = Path("data/external")
COLLECTION_KEY = "UHVYXG5W"


def activate_zotero() -> bool:
    """Bring Zotero to the front. Uses open -a."""
    subprocess.run(["open", "-a", "Zotero"], timeout=5)
    time.sleep(2)
    # Cmd+Tab to ensure it's frontmost
    pyautogui.hotkey("command", "tab")
    time.sleep(0.3)
    pyautogui.hotkey("command", "tab")
    time.sleep(0.3)
    logger.info("Zotero activated")
    return True


def select_external_collection() -> bool:
    """Navigate to 'external' collection via keyboard search."""
    logger.info("Selecting 'external' collection...")
    
    # Cmd+/ focuses the collections search bar
    pyautogui.hotkey("command", "/")
    time.sleep(0.5)
    
    # Clear existing search, type 'external'
    pyautogui.hotkey("command", "a")
    time.sleep(0.1)
    pyautogui.press("backspace")
    time.sleep(0.1)
    pyautogui.write("external", interval=0.04)
    time.sleep(0.6)
    
    # Press Down to select the collection, Enter to open it
    pyautogui.press("down")
    time.sleep(0.2)
    pyautogui.press("enter")
    time.sleep(1.0)
    
    # Press Tab to move focus to the items pane
    pyautogui.press("tab")
    time.sleep(0.3)
    
    logger.info("Collection selected")
    return True


def select_all_and_find_pdfs(batch_size: int = 0) -> bool:
    """Select items and trigger Find Available PDFs."""
    logger.info("Selecting items and triggering PDF find...")
    
    # Cmd+A to select all items in the center pane
    pyautogui.hotkey("command", "a")
    time.sleep(0.3)
    
    # Click once on the selected items to ensure right-click lands on them
    w, h = pyautogui.size()
    pyautogui.click(w // 2, h // 2)
    time.sleep(0.3)
    
    # Right-click to open context menu
    pyautogui.rightClick()
    time.sleep(0.8)
    
    # Context menu is open. "Find Available PDFs" starts with F.
    # Other F-items in Zotero's context menu: "Find Available PDF" (singular)
    # Strategy: type 'f' to highlight, then Enter
    pyautogui.press("f")
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(1.5)
    
    logger.info("'Find Available PDFs' triggered")
    return True


def poll_for_pdfs(max_checks: int = 120, interval: int = 10) -> int:
    """Poll Zotero API for new PDF attachments. Downloads found PDFs.
    
    Returns total number of PDFs found.
    """
    from dotenv import load_dotenv; load_dotenv(override=True)
    from pyzotero import zotero; import os
    
    zot = zotero.Zotero(os.getenv("ZOTERO_LIBRARY_ID"), "user", os.getenv("ZOTERO_API_KEY"))
    
    status = {}
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text())
    
    total_found = 0
    stable = 0
    
    logger.info("Polling for PDFs (up to %ds)...", max_checks * interval)
    
    for i in range(max_checks):
        time.sleep(interval)
        found_this_round = 0
        
        for key, item in list(status.items()):
            if not isinstance(item, dict):
                continue
            if item.get("status") not in ("awaiting_zotero", "zotero_created"):
                continue
            
            zk = item.get("zotero_key", "")
            if not zk:
                continue
            
            try:
                children = zot.children(zk)
                for child in children:
                    data = child.get("data", {})
                    ct = data.get("contentType", "")
                    fn = data.get("filename", "")
                    if ct == "application/pdf" or (fn and fn.endswith(".pdf")):
                        att_key = data.get("key", "")
                        if att_key:
                            try:
                                file_content = zot.file(att_key)
                                if file_content and len(file_content) > 1000:
                                    safe = key.replace("/", "_").replace(":", "_")[:80]
                                    fname = EXTERNAL_DIR / f"{safe}.pdf"
                                    fname.write_bytes(file_content)
                                    item["status"] = "zotero_found"
                                    item["pdf_path"] = str(fname)
                                    item["attachment_key"] = att_key
                                    found_this_round += 1
                                    total_found += 1
                                    logger.info("PDF: %s", item.get("title", "")[:70])
                                    break
                            except Exception:
                                pass
            except Exception:
                pass
        
        if found_this_round > 0:
            status["_updated_at"] = time.time()
            STATUS_PATH.write_text(json.dumps(status, indent=2, ensure_ascii=False))
            stable = 0
            logger.info("Round %d: +%d PDFs (total: %d)", i + 1, found_this_round, total_found)
        else:
            stable += 1
            if stable >= 3:
                logger.info("No new PDFs for %ds — Zotero likely done", stable * interval)
                break
        
        # Check progress in Zotero by looking at window title? Can't easily.
        # Just keep polling
    
    return total_found


def main():
    logger.info("=" * 50)
    logger.info("ZOTERO PDF AUTOMATION (pyautogui)")
    logger.info("=" * 50)
    
    activate_zotero()
    select_external_collection()
    select_all_and_find_pdfs()
    
    found = poll_for_pdfs(max_checks=120, interval=10)
    
    logger.info("=" * 50)
    logger.info("COMPLETE: %d PDFs found and downloaded", found)
    logger.info("Run `python scripts/zotero_sync.py --status` for report")
    logger.info("=" * 50)


if __name__ == "__main__":
    sys.exit(main())
