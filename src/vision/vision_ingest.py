"""
Vision pipeline integration hook for PDF ingestion.

Coordinates the full vision pipeline during PDF ingestion:
  1. Extract figures from PDF via Docling
  2. Filter to keep only data-relevant figures (classification-based)
  3. Describe figures via gemma4:e4b (already loaded as fast-tier text model)
  4. Embed descriptions into ChromaDB for cross-modal retrieval

Usage (during PDF ingest)::

    hybrid_retriever.ingest(text_chunks)
    vision_ingest_pdf(pdf_path, hybrid_retriever)
    # -> figures are now in ChromaDB alongside text chunks
"""
from __future__ import annotations

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
                  If False, embed captions only (fast, zero LLM cost).
        threshold: Relevance threshold for FigureFilter (default 0.35).

    Returns:
        Dict with counts: {"extracted": N, "kept": N, "embedded": N,
                           "described": N, "skipped_figures": N}.
    """
    result: Dict[str, int] = {
        "extracted": 0,
        "kept": 0,
        "embedded": 0,
        "described": 0,
        "skipped_figures": 0,
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

    # ── 3. Describe ──
    if describe:
        try:
            vd = VisionDescriptor()
            # Don't unload the text model explicitly — gemma4:e4b is likely
            # already loaded for extraction/summarization tasks and we want
            # to keep it. The VisionDescriptor will use the same model.
            kept_figures = vd.describe_figures(
                kept_figures,
                unload_first=None,   # don't unload — gemma4 is already loaded
                reload_after=False,  # don't unload — it's needed for processing
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
        # Use captions as descriptions (fast path, zero LLM)
        for fig in kept_figures:
            fig["description"] = fig.get("caption", "")

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
