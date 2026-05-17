#!/usr/bin/env python
"""Diagnostic: test whether Ollama KV-cache accumulation causes mid-batch spam.

Hypothesis tested:
    ``keep_alive=0`` between batches does NOT fully free Metal GPU memory.
    KV-cache entries accumulate across batches, causing mid-batch corruption
    (token-level spam like ``Energy: Energy: Energy: …``) in later batches.

Test procedure:
    Run A (control):  extract paper with ``EXTRACTION_RESET_MODE=api``
                      (keep_alive=0 + /api/ps polling between batches).
    Run B (test):     extract paper with ``EXTRACTION_RESET_MODE=process``
                      (full ollama stop + serve restart between batches).

    If Run B produces ZERO token-spam errors while Run A produces any,
    cache accumulation is confirmed → enable ``process`` mode for daemon.

    If BOTH produce spam, the issue is inherent Metal KV-cache corruption
    on long generations — process restarts won't help → reduce batch size.

Usage:
    python scripts/diagnose_cache_accumulation.py PMC10571047
    python scripts/diagnose_cache_accumulation.py --search "piezoelectric PVDF energy harvesting"

Output:
    Side-by-side comparison table with batch counts, timings, and spam errors.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

# Inject project root so `src.` imports work regardless of invocation directory
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Load .env so OLLAMA_SMALL_MODEL / OLLAMA_LARGE_MODEL / LLM_PROVIDER are set
try:
    from dotenv import load_dotenv
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_cache")


# ── helpers ──────────────────────────────────────────────────────────────────


def _get_chunks_for_paper(pmcid: str) -> tuple[str, list[dict]] | tuple[None, None]:
    """Fetch and parse a paper from Europe PMC. Returns (title, chunks) or (None, None)."""
    from src.retrieval.europe_pmc import EuropePMCClient
    from src.ingestion.pmc_xml_parser import PMCXMLParser

    epmc = EuropePMCClient()
    results = epmc.search(f"PMCID:{pmcid}", oa_only=False, max_results=1)
    if not results:
        logger.error("No EPMC result for %s", pmcid)
        return None, None

    paper = results[0]
    pmc = paper.get("pmcid", pmcid)
    title = paper.get("title", pmcid)
    xml = epmc.full_text_xml(pmc)
    if not xml:
        logger.error("No full text XML for %s", pmcid)
        return None, None

    chunks = PMCXMLParser().parse(xml, pmcid=pmc, doi=paper.get("doi", ""))
    if not chunks:
        logger.error("No chunks parsed from %s", pmcid)
        return None, None

    logger.info("Fetched %s: %d chunks", title, len(chunks))
    return title, chunks


def _get_chunks_by_search(query: str) -> tuple[str, list[dict]] | tuple[None, None]:
    """Search EPMC and return title+chunks for the first OA result."""
    from src.retrieval.europe_pmc import EuropePMCClient
    from src.ingestion.pmc_xml_parser import PMCXMLParser

    epmc = EuropePMCClient()
    results = epmc.search(query, oa_only=True, max_results=3)
    if not results:
        logger.error("No OA results for query: %s", query)
        return None, None

    for paper in results:
        pmcid = paper.get("pmcid", "")
        if not pmcid:
            continue
        xml = epmc.full_text_xml(pmcid)
        if not xml:
            continue
        chunks = PMCXMLParser().parse(xml, pmcid=pmcid, doi=paper.get("doi", ""))
        if chunks:
            title = paper.get("title", query)
            logger.info("Fetched %s: %d chunks", title, len(chunks))
            return title, chunks

    logger.error("No parsable OA results for query: %s", query)
    return None, None


def _run_extraction(
    title: str,
    chunks: list[dict],
    reset_mode: str,
) -> dict:
    """Run batched extraction with a given reset mode. Returns result dict."""
    os.environ["EXTRACTION_RESET_MODE"] = reset_mode

    from src.agents.extraction_agent import ExtractionAgent
    from src.ingestion.pre_extractor import PreExtractor

    # Take ownership of Ollama process (disarm launchd watchdog once)
    PreExtractor._ensure_dedicated_ollama()

    # Clear stale LLM cache — prompt changed (DIRECTION→CLAIM),
    # old cached responses have wrong keys
    from src.cache.llm_cache import get_cache
    for p in get_cache()._cache_dir.glob("*.json"):
        try:
            p.unlink()
        except OSError:
            pass

    agent = ExtractionAgent(model="deepseek-chat")

    summary_chunks = []
    for ch in chunks:
        meta = ch.get("metadata", {}) or {}
        s = meta.get("chunk_summary", ch.get("text", "")[:200])
        summary_chunks.append({"text": s, "metadata": meta})

    query = (
        "What are the key materials, energy harvesting mechanisms, "
        "experimental methods, and results described in this paper?"
    )

    categories = agent.discover_categories(summary_chunks, query)

    # Capture log output during extraction to count spam errors.
    # _detect_token_spam raises RuntimeError which extract_entities_batched
    # catches silently, so we must intercept the warning-level log message.
    log_capture = io.StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    capture_handler.setLevel(logging.WARNING)
    capture_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(capture_handler)

    t0 = time.monotonic()
    entities = agent.extract_entities_batched(chunks, categories, query)
    elapsed = time.monotonic() - t0

    root_logger.removeHandler(capture_handler)
    captured_text = log_capture.getvalue()
    capture_handler.close()

    token_spam_errors = captured_text.count("Token-level spam detected")
    batch_failures = captured_text.count("Extraction batch") - captured_text.count(" done:")

    total_entities = sum(len(v) for v in entities.values())
    batches = (len(chunks) + 7) // 8

    # Unload model after extraction
    try:
        PreExtractor._reset_ollama(timeout=10.0)
    except Exception:
        pass

    return {
        "mode": reset_mode,
        "title": title,
        "chunks": len(chunks),
        "batches": batches,
        "entities": total_entities,
        "categories": len(categories.get("discovered_categories", [])) if categories else 0,
        "elapsed_s": round(elapsed, 1),
        "spam_errors": token_spam_errors,
        "batch_failures": max(0, batch_failures),
        "spam_detected": token_spam_errors > 0,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether Ollama KV-cache accumulation causes mid-batch token spam",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "pmcid", nargs="?", default=None,
        help="PMC ID to test (e.g. PMC10571047)",
    )
    source.add_argument(
        "--search", "-s", type=str, default=None,
        help="EPMC search query to find a test paper",
    )
    parser.add_argument(
        "--mode", "-m", choices=("process", "both"), default="process",
        help="process = only process restarts (default, since keep_alive is known broken). "
             "both = compare api vs process side-by-side.",
    )
    args = parser.parse_args()

    # ── Fetch paper ───────────────────────────────────────────────────────
    pmcid = args.pmcid
    search_query = args.search
    if pmcid and pmcid.startswith("PMC"):
        title, chunks = _get_chunks_for_paper(pmcid)
    elif search_query:
        title, chunks = _get_chunks_by_search(search_query)
    elif pmcid:
        title, chunks = _get_chunks_for_paper(pmcid)
    else:
        title, chunks = None, None

    if not title or not chunks:
        sys.exit(1)

    print(f"\nPaper: {title[:100]}")
    print(f"Chunks: {len(chunks)} ({(len(chunks) + 7) // 8} batches of 8)\n")

    if args.mode == "both":
        # ── Run A: API mode ──────────────────────────────────────────────
        print("=" * 65)
        print("Run A: EXTRACTION_RESET_MODE=api  (keep_alive=0 between batches)")
        print("=" * 65)
        result_a = _run_extraction(title, chunks, "api")
        print(json.dumps(result_a, indent=2))
        time.sleep(2)

        # ── Run B: Process mode ──────────────────────────────────────────
        print("\n" + "=" * 65)
        print("Run B: EXTRACTION_RESET_MODE=process  (ollama restart between batches)")
        print("=" * 65)
        result_b = _run_extraction(title, chunks, "process")
        print(json.dumps(result_b, indent=2))

        # ── Comparison ───────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("COMPARISON")
        print("=" * 65)
        print(f"{'':<25} {'Run A (api)':>18} {'Run B (process)':>18}")
        print(f"{'Entities extracted':<25} {result_a['entities']:>18} {result_b['entities']:>18}")
        print(f"{'Categories':<25} {result_a['categories']:>18} {result_b['categories']:>18}")
        print(f"{'Elapsed (s)':<25} {result_a['elapsed_s']:>18.1f} {result_b['elapsed_s']:>18.1f}")
        print(f"{'Spam detected':<25} {str(result_a['spam_detected']):>18} {str(result_b['spam_detected']):>18}")

        if result_a["spam_detected"] and not result_b["spam_detected"]:
            print("\n>>> RESULT: KV-cache accumulation CONFIRMED as root cause.")
            print("    Set EXTRACTION_RESET_MODE=process in .env for daemon extraction.")
        elif result_a["spam_detected"] and result_b["spam_detected"]:
            print("\n>>> RESULT: Both modes show spam — inherent Metal KV-cache corruption.")
            print("    Process restarts won't fix mid-batch degradation.")
            print("    Mitigation: reduce batch_size from 8 to 4 chunks.")
        else:
            print("\n>>> RESULT: No spam detected in either mode — cannot determine cause.")
            print("    Re-run with the exact paper/prompt that produced spam (e.g. PMC10571047).")

        if result_b["elapsed_s"] > 0:
            overhead = result_b["elapsed_s"] - result_a["elapsed_s"]
            print(f"\nProcess-restart overhead: +{overhead:.0f}s "
                  f"(+{overhead / result_b['batches']:.1f}s per batch)")
    else:
        # ── Process-restart only ─────────────────────────────────────────
        print("=" * 65)
        print("EXTRACTION_RESET_MODE=process  (ollama restart between batches)")
        print("=" * 65)
        result = _run_extraction(title, chunks, "process")
        print(json.dumps(result, indent=2))

        if result["spam_detected"]:
            print(f"\n>>> {result['spam_errors']} token-spam errors detected "
                  f"with process restarts.")
            print("    Mitigation: reduce batch_size from 8 to 4 chunks.")
        else:
            print(f"\n>>> NO spam detected. {result['entities']} entities "
                  f"extracted in {result['elapsed_s']:.0f}s, "
                  f"{result['batches']} batches.")

    # Clean up env var
    os.environ.pop("EXTRACTION_RESET_MODE", None)


if __name__ == "__main__":
    main()
