"""
Vision pipeline integration hook for PDF ingestion.

Coordinates the full vision pipeline during PDF ingestion:
  1. Extract figures from PDF via Docling
  2. Filter to keep only data-relevant figures (classification-based)
  3. Describe figures via gemma4:e4b (already loaded as fast-tier text model)
     or defer to a background queue with ``describe=False`` for scale.
  4. Embed descriptions into ChromaDB for cross-modal retrieval

Usage (during PDF ingest)::

    hybrid_retriever.ingest(text_chunks)
    vision_ingest_pdf(pdf_path, hybrid_retriever)
    # -> figures are now in ChromaDB alongside text chunks

Scale tip — at 100+ papers, use ``describe=False`` to embed captions
immediately (~0 LLM cost), then call ``describe_queued_figures()`` as a
background task to generate full descriptions later.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.vision.figure_extractor import FigureExtractor
from src.vision.figure_filter import FigureFilter
from src.vision.vision_descriptor import VisionDescriptor
from src.vision.figure_embedder import FigureEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

FIGURES_CACHE_DIR = Path("projects/default/figure_descriptions.json")
FIGURE_QUEUE_PATH = Path("projects/default/figure_description_queue.json")


def _load_queue() -> List[Dict]:
    """Load the deferred-description queue from disk."""
    if not FIGURE_QUEUE_PATH.exists():
        return []
    try:
        return json.loads(FIGURE_QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_queue(queue: List[Dict]) -> None:
    """Persist the deferred-description queue."""
    FIGURE_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_QUEUE_PATH.write_text(
        json.dumps(queue, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def describe_queued_figures(
    hybrid_retriever: HybridRetriever,
    max_figures: int | None = None,
) -> Dict[str, int]:
    """Process the deferred figure description queue.

    Reads figures that were ingested with ``describe=False``, runs the
    vision model to generate full descriptions, and re-embeds them into
    ChromaDB (replacing the caption-only entries).

    Args:
        hybrid_retriever: The HybridRetriever for ChromaDB updates.
        max_figures: Cap on number of figures to describe (None = all).

    Returns:
        Dict with {"processed": N, "described": N, "failed": N}.
    """
    queue = _load_queue()
    if not queue:
        logger.info("Figure description queue is empty")
        return {"processed": 0, "described": 0, "failed": 0}

    logger.info("Processing figure description queue: %d figures", len(queue))
    vd = VisionDescriptor()

    result = {"processed": 0, "described": 0, "failed": 0}
    remaining = []
    limit = max_figures or len(queue)

    for entry in queue[:limit]:
        result["processed"] += 1
        try:
            described = vd.describe_figures(
                entry["figures"],
                unload_first=None,
                reload_after=False,
                fallback_to_caption=True,
            )
            result["described"] += sum(
                1 for f in described if f.get("description", "").strip()
                and f.get("description") != f.get("caption", "")
            )
            # Re-embed described figures into ChromaDB
            embedder = FigureEmbedder(hybrid_retriever)
            embedder.embed(described, pdf_source=entry["pdf_name"])
            logger.info("Re-embedded described figures for %s", entry["pdf_name"])
        except Exception as e:
            logger.warning("Deferred description failed for %s: %s", entry["pdf_name"], e)
            result["failed"] += 1

    # Keep remaining entries for next run
    remaining = queue[limit:]
    _save_queue(remaining)

    logger.info("Figure description queue: %d processed, %d described, %d failed, %d remaining",
                 result["processed"], result["described"], result["failed"], len(remaining))
    return result


def vision_ingest_figure_url(
    image_url: str,
    caption: str,
    source: str,
    figure_index: int,
    hybrid_retriever: HybridRetriever,
    describe: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Download a figure image from an XML <graphic> URL and feed it into the
    vision pipeline (filter → describe → embed).

    Used by Phase 9 to process figures discovered in JATS XML <fig><graphic>
    elements during Europe PMC full-text ingestion.

    Args:
        image_url: The raw URL of the figure image (from <graphic xlink:href>).
        caption: The figure caption text (from <fig><caption>).
        source: Source identifier for metadata (e.g., ``"europe_pmc_xml_PMC12345"``).
        figure_index: Zero-based index within the paper.
        hybrid_retriever: The HybridRetriever for ChromaDB embedding.
        describe: If True, call the vision model to generate a description.
                  If False, embed caption as the description (zero LLM cost).
        timeout: Download timeout in seconds.

    Returns:
        Dict with keys: downloaded, described, embedded, error (str or None).
    """
    import io
    import requests as _requests
    from PIL import Image

    result: Dict[str, Any] = {
        "downloaded": False,
        "described": False,
        "embedded": False,
        "error": None,
    }

    if not image_url:
        result["error"] = "empty URL"
        return result

    # ── 1. Download image ──
    try:
        if image_url.startswith("file://"):
            local_path = Path(image_url[7:])
            img_bytes = local_path.read_bytes()
        elif image_url.startswith(("http://", "https://")):
            resp = _requests.get(image_url, timeout=timeout, stream=True)
            resp.raise_for_status()
            img_bytes = resp.content
        else:
            result["error"] = f"unsupported URL scheme: {image_url[:30]}"
            return result
        if len(img_bytes) < 100:
            result["error"] = f"image too small ({len(img_bytes)} bytes)"
            return result
        pil_image = Image.open(io.BytesIO(img_bytes))
        pil_image.load()
        result["downloaded"] = True
        w, h = pil_image.size
    except _requests.RequestException as e:
        result["error"] = f"download failed: {e}"
        logger.debug("Figure URL download failed: %s → %s", image_url[:80], e)
        return result
    except Exception as e:
        result["error"] = f"image decode failed: {e}"
        logger.debug("Figure image decode failed: %s → %s", image_url[:80], e)
        return result

    if w < 50 or h < 50:
        result["error"] = f"image too small ({w}×{h})"
        return result

    # ── 2. Describe (optional — caption is always the minimum) ──
    description = caption if caption else ""
    if describe:
        try:
            from src.vision.vision_descriptor import VisionDescriptor
            vd = VisionDescriptor()
            generated = vd.describe(pil_image)
            if generated.strip():
                description = generated.strip()
                result["described"] = True
        except Exception as e:
            logger.debug("Vision description failed for figure %d (%s): %s",
                         figure_index, source, e)

    # ── 3. Embed into ChromaDB ──
    try:
        from src.vision.figure_embedder import FigureEmbedder
        embedder = FigureEmbedder(hybrid_retriever)
        figure_dict = {
            "description": description,
            "caption": caption,
            "page_no": 0,
            "bbox": {},
            "file_path": str(image_url),
            "width": w,
            "height": h,
            "figure_index": figure_index,
        }
        embedder.embed([figure_dict], pdf_source=source)
        result["embedded"] = True
        logger.debug("Vision ingest: embedded figure %d for %s", figure_index, source)
    except Exception as e:
        result["error"] = result["error"] or f"embed failed: {e}"
        logger.debug("Figure embed failed for %s: %s", source, e)

    return result


def vision_ingest_xml_figures(
    chunks: list,
    hybrid_retriever: HybridRetriever,
    describe: bool = False,
) -> Dict[str, int]:
    """Scan parsed XML chunks for figure_image_url entries and feed them
    through the vision pipeline.

    Called during Phase 9 --ingest after PMCXMLParser.parse() to process
    figures discovered in JATS XML <fig><graphic> elements.

    Args:
        chunks: List of chunk dicts from PMCXMLParser.parse().
        hybrid_retriever: The HybridRetriever for ChromaDB embedding.
        describe: If True, call the vision model per figure.

    Returns:
        Dict with counts: found, downloaded, described, embedded, failed.
    """
    counts = {"found": 0, "downloaded": 0, "described": 0, "embedded": 0, "failed": 0}

    fig_index = 0
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if meta.get("chunk_type") != "figure":
            continue
        image_url = meta.get("figure_image_url", "")
        if not image_url:
            continue
        counts["found"] += 1

        caption = chunk.get("text", "")
        source = meta.get("source", "europe_pmc_xml")

        r = vision_ingest_figure_url(
            image_url=image_url,
            caption=caption,
            source=source,
            figure_index=fig_index,
            hybrid_retriever=hybrid_retriever,
            describe=describe,
        )
        fig_index += 1

        if r["downloaded"]:
            counts["downloaded"] += 1
        if r["described"]:
            counts["described"] += 1
        if r["embedded"]:
            counts["embedded"] += 1
        if r["error"]:
            counts["failed"] += 1

    return counts


def vision_ingest_pdf(
    pdf_path: Path,
    hybrid_retriever: HybridRetriever,
    describe: bool = True,
    threshold: float = 0.35,
) -> Dict[str, int]:
    """Run the full vision ingest pipeline for one PDF.

    Extracts figures, filters to keep only data-relevant ones, optionally
    describes them via a vision model, and embeds descriptions into the
    ChromaDB collection used by *hybrid_retriever*.

    Args:
        pdf_path: Path to the PDF file.
        hybrid_retriever: The HybridRetriever whose ChromaDB collection
                          will receive figure embeddings.
        describe: If True, call the vision model to generate descriptions.
                  If False, embed captions only (fast, zero LLM cost) and
                  queue for deferred description via ``describe_queued_figures()``.
        threshold: Relevance threshold for FigureFilter (default 0.35).

    Returns:
        Dict with counts: {"extracted": N, "kept": N, "embedded": N,
                           "described": N, "skipped_figures": N,
                           "queued": N (when describe=False)}.
    """
    result: Dict[str, int] = {
        "extracted": 0,
        "kept": 0,
        "embedded": 0,
        "described": 0,
        "skipped_figures": 0,
        "queued": 0,
    }

    try:
        # ── 1. Extract ──
        extractor = FigureExtractor()
        figures = extractor.extract(pdf_path)
        result["extracted"] = len(figures)
        if not figures:
            logger.info("Vision ingest: no figures in %s", pdf_path.name)
            return result
    except Exception as e:
        logger.error("Figure extraction failed for %s: %s", pdf_path.name, e)
        return result

    # ── 2. Filter ──
    ff = FigureFilter(threshold=threshold)
    kept_figures = ff.filter(figures)
    result["kept"] = len(kept_figures)
    result["skipped_figures"] = len(figures) - len(kept_figures)

    if not kept_figures:
        logger.info("Vision ingest: all %d figures filtered out of %s",
                     len(figures), pdf_path.name)
        return result

    # ── 3. Describe (or defer to queue) ──
    if describe:
        try:
            vd = VisionDescriptor()
            kept_figures = vd.describe_figures(
                kept_figures,
                unload_first=None,
                reload_after=False,
                fallback_to_caption=True,
            )
            result["described"] = sum(
                1 for f in kept_figures
                if f.get("description", "").strip()
            )
            logger.info("Vision ingest: described %d/%d figures for %s",
                         result["described"], len(kept_figures), pdf_path.name)
        except Exception as e:
            logger.warning("Vision description failed for %s: %s — using captions",
                           pdf_path.name, e)
            for fig in kept_figures:
                fig["description"] = fig.get("caption", "")
    else:
        # Fast path: embed captions now, queue for later description
        for fig in kept_figures:
            fig["description"] = fig.get("caption", "")
        # Queue for deferred processing
        queue = _load_queue()
        queue.append({
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "figures": [
                {k: v for k, v in f.items() if k != "description"}
                for f in kept_figures
            ],
        })
        _save_queue(queue)
        result["queued"] = len(kept_figures)
        logger.info("Vision ingest: queued %d figures for %s for deferred description",
                     len(kept_figures), pdf_path.name)

    # ── 4. Embed ──
    try:
        embedder = FigureEmbedder(hybrid_retriever)
        count = embedder.embed(kept_figures, pdf_source=pdf_path.name)
        result["embedded"] = count
        logger.info("Vision ingest: embedded %d figure descriptions for %s",
                      count, pdf_path.name)
    except Exception as e:
        logger.error("Figure embedding failed for %s: %s", pdf_path.name, e)

    return result
