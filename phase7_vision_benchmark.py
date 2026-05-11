#!/usr/bin/env python
"""
Phase 7a Vision Pipeline — Benchmark & Quality Report

Extracts figures from all PDFs in data/, filters using classification-based
relevance scoring, and (optionally) generates descriptions via a multimodal
vision model.  Results are cached to disk for offline review.

Usage:
    # Benchmark figure extraction + filtering (no LLM, instant)
    python phase7_vision_benchmark.py

    # Full pipeline: extract + filter + describe via vision model
    python phase7_vision_benchmark.py --describe

    # Re-describe even if cached
    python phase7_vision_benchmark.py --describe --no-cache

    # Specify vision model (default: llava:7b)
    python phase7_vision_benchmark.py --describe --vision-model qwen2-vl:7b

    # Custom relevance threshold (default: 0.35)
    python phase7_vision_benchmark.py --threshold 0.5

The benchmark produces two output files:
    projects/default/vision_scorecard.json   — extraction + filtering stats
    projects/default/figure_descriptions.json — cached vision descriptions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vision.figure_extractor import FigureExtractor
from src.vision.figure_filter import FigureFilter
from src.vision.vision_descriptor import VisionDescriptor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase7_benchmark")

PROJECT_DIR = Path("projects/default")
SCORECARD_PATH = PROJECT_DIR / "vision_scorecard.json"
DESCRIPTIONS_CACHE = PROJECT_DIR / "figure_descriptions.json"
DATA_DIR = Path("data")


# ── Data-figure classes (from FigureFilter) ────────────────────────────────
DATA_CLASSES = {
    "bar_chart", "line_chart", "scatter_plot", "box_plot", "pie_chart",
    "flow_chart", "photograph", "chemistry_structure", "table",
}
EXTRANEOUS_CLASSES = {
    "logo", "icon", "stamp", "signature", "qr_code", "calendar",
    "page_thumbnail", "topographical_map", "crossword_puzzle", "bar_code", "music",
}


def find_pdfs(data_dir: Path) -> List[Path]:
    """Find all PDFs in the data directory (excluding hidden files)."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    # Filter out test PDFs if we have real biomedical papers
    test_pdfs = {"test.pdf", "test2.pdf", ".test"}
    return [p for p in pdfs if p.name not in test_pdfs or len(pdfs) <= 2]


def compute_stats(figures: List[Dict], filtered: List[Dict]) -> Dict:
    """Compute statistics about extracted and filtered figures."""
    n_total = len(figures)
    n_kept = len(filtered)
    n_discarded = n_total - n_kept

    # Classification breakdown of all figures
    top_classes: Dict[str, int] = {}
    for fig in figures:
        cls_list = fig.get("classification", [])
        if cls_list:
            top_cls = cls_list[0]["class_name"]
            top_classes[top_cls] = top_classes.get(top_cls, 0) + 1

    # Classification breakdown of filtered / discarded
    kept_classes: Dict[str, int] = {}
    discarded_classes: Dict[str, int] = {}
    for fig in filtered:
        cls_list = fig.get("classification", [])
        if cls_list:
            top_cls = cls_list[0]["class_name"]
            kept_classes[top_cls] = kept_classes.get(top_cls, 0) + 1
    for fig in figures:
        if fig not in filtered:
            cls_list = fig.get("classification", [])
            if cls_list:
                top_cls = cls_list[0]["class_name"]
                discarded_classes[top_cls] = discarded_classes.get(top_cls, 0) + 1

    # Average relevance scores
    all_scores = [f["relevance_score"] for f in filtered]

    # Per-PDF breakdown
    pdf_breakdown: Dict[str, Dict] = {}
    for fig in filtered:
        pdf = fig["pdf_source"]
        if pdf not in pdf_breakdown:
            pdf_breakdown[pdf] = {"total": 0, "kept": 0, "data_figs": 0, "extraneous": 0}
        bd = pdf_breakdown[pdf]
        bd["kept"] += 1
        comps = fig.get("relevance_components", {})
        top_cls = comps.get("classification_top", "unknown")
        if top_cls in DATA_CLASSES:
            bd["data_figs"] += 1
        elif top_cls in EXTRANEOUS_CLASSES:
            bd["extraneous"] += 1

    # Count data vs extraneous in all figures
    for fig in figures:
        pdf = fig["pdf_source"]
        if pdf not in pdf_breakdown:
            pdf_breakdown[pdf] = {"total": 0, "kept": 0, "data_figs": 0, "extraneous": 0}
        pdf_breakdown[pdf]["total"] += 1

    # Descriptions
    n_with_desc = sum(1 for f in filtered if f.get("description", "").strip())
    avg_desc_chars = (
        sum(len(f.get("description", "")) for f in filtered) / max(n_with_desc, 1)
    )

    return {
        "total_extracted": n_total,
        "total_kept": n_kept,
        "total_discarded": n_discarded,
        "keep_rate": round(n_kept / max(n_total, 1), 3),
        "avg_relevance_score": round(sum(all_scores) / max(len(all_scores), 1), 4) if all_scores else 0,
        "min_relevance_score": round(min(all_scores), 4) if all_scores else 0,
        "max_relevance_score": round(max(all_scores), 4) if all_scores else 0,
        "top_classes_all": top_classes,
        "top_classes_kept": kept_classes,
        "top_classes_discarded": discarded_classes,
        "per_pdf": pdf_breakdown,
        "figures_described": n_with_desc,
        "avg_description_chars": round(avg_desc_chars, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 7a Vision Pipeline Benchmark")
    parser.add_argument("--describe", action="store_true",
                        help="Generate figure descriptions using a vision model")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached descriptions and regenerate")
    parser.add_argument("--vision-model", type=str, default=None,
                        help="Vision model name (default: llava:7b)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Relevance threshold for filtering (default: 0.35)")
    parser.add_argument("--pdfs", type=str, nargs="*", default=None,
                        help="Specific PDFs to process (default: all in data/)")
    args = parser.parse_args()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Find PDFs ──
    if args.pdfs:
        pdf_paths = [Path(p) for p in args.pdfs]
    else:
        pdf_paths = find_pdfs(DATA_DIR)

    if not pdf_paths:
        logger.error("No PDFs found in %s", DATA_DIR)
        sys.exit(1)

    logger.info("Found %d PDFs to process", len(pdf_paths))

    # ── Phase 1: Extract figures ──
    t0 = time.monotonic()
    extractor = FigureExtractor()
    all_figures: List[Dict] = []
    pdf_name_counts: Dict[str, int] = {}

    for pdf_path in pdf_paths:
        logger.info("Extracting figures from %s...", pdf_path.name)
        try:
            figures = extractor.extract(pdf_path)
            all_figures.extend(figures)
            pdf_name_counts[pdf_path.name] = len(figures)
            logger.info("  -> %d figures extracted", len(figures))
        except Exception as e:
            logger.error("Failed to extract figures from %s: %s", pdf_path, e)

    extract_time = time.monotonic() - t0
    logger.info("Extraction complete: %d figures in %.1fs", len(all_figures), extract_time)

    # ── Phase 2: Filter figures ──
    t0 = time.monotonic()
    ff = FigureFilter(threshold=args.threshold)
    filtered = ff.score_all(all_figures)  # score all, then filter below
    kept = ff.filter(all_figures)
    filter_time = time.monotonic() - t0

    logger.info(
        "Filtering complete: %d/%d figures kept (%.1f%%) in %.1fs",
        len(kept), len(all_figures),
        100 * len(kept) / max(len(all_figures), 1),
        filter_time,
    )

    # ── Phase 3: Describe (optional) ──
    desc_time = 0.0
    if args.describe:
        # Check cache
        cached = {}
        if not args.no_cache and DESCRIPTIONS_CACHE.exists():
            cached = VisionDescriptor.load_descriptions(DESCRIPTIONS_CACHE)
            logger.info("Loaded %d cached descriptions", len(cached))

        # Apply cached descriptions where available
        for fig in kept:
            fp = fig["file_path"]
            if fp in cached:
                fig["description"] = cached[fp]

        # Describe remaining
        to_describe = [f for f in kept if not f.get("description", "").strip()]
        if to_describe:
            logger.info("Describing %d/%d figures with vision model...", len(to_describe), len(kept))
            t0 = time.monotonic()
            vd = VisionDescriptor(model=args.vision_model)

            try:
                described = vd.describe_figures(
                    to_describe,
                    unload_first=None,  # don't unload text model for benchmark
                    reload_after=True,
                    fallback_to_caption=True,
                )
                desc_time = time.monotonic() - t0
                logger.info("Description complete in %.1fs (%.1fs/fig)", desc_time, desc_time / max(len(to_describe), 1))

                # Update kept list with described figures
                desc_map = {d["file_path"]: d.get("description", "") for d in described}
                for fig in kept:
                    if fig["file_path"] in desc_map:
                        fig["description"] = desc_map[fig["file_path"]]
            except Exception as e:
                logger.error("Vision model failed: %s", e)
                logger.info("Skipping description phase. Figures saved without descriptions.")

        # Save cache
        desc_cache = {f["file_path"]: f.get("description", "") for f in kept}
        VisionDescriptor.save_descriptions(DESCRIPTIONS_CACHE, desc_cache)
    else:
        logger.info("Skipping description phase (use --describe to enable)")

    # ── Compute statistics ──
    stats = compute_stats(all_figures, kept)
    stats["extraction_time_s"] = round(extract_time, 2)
    stats["filter_time_s"] = round(filter_time, 2)
    stats["description_time_s"] = round(desc_time, 2)
    stats["vision_model"] = args.vision_model or "not_run"
    stats["threshold"] = args.threshold
    stats["pdfs_processed"] = [p.name for p in pdf_paths]
    stats["figures_per_pdf"] = pdf_name_counts

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  Phase 7a Vision Pipeline — Benchmark Summary")
    print("=" * 60)
    print(f"  PDFs processed:      {len(pdf_paths)}")
    print(f"  Figures extracted:   {stats['total_extracted']}")
    print(f"  Figures kept:        {stats['total_kept']} ({stats['keep_rate']*100:.1f}%)")
    print(f"  Figures discarded:   {stats['total_discarded']}")
    print(f"  Avg relevance score:  {stats['avg_relevance_score']:.3f}")
    print(f"  Score range:          [{stats['min_relevance_score']:.3f}, {stats['max_relevance_score']:.3f}]")
    print(f"  Figures described:   {stats['figures_described']}")
    print(f"  Avg desc length:     {stats['avg_description_chars']:.0f} chars")
    print()
    print("  Top classes (all extracted):")
    for cls, count in sorted(stats["top_classes_all"].items(), key=lambda x: -x[1])[:10]:
        data_tag = " [DATA]" if cls in DATA_CLASSES else (" [EXTRA]" if cls in EXTRANEOUS_CLASSES else "")
        print(f"    {cls}: {count}{data_tag}")
    print()
    print("  Discarded:")
    if stats["top_classes_discarded"]:
        for cls, count in sorted(stats["top_classes_discarded"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {cls}: {count}")
    else:
        print("    (none)")
    print()
    print(f"  ── Timing ──")
    print(f"  Extraction:  {stats['extraction_time_s']:.1f}s")
    print(f"  Filtering:   {stats['filter_time_s']:.2f}s")
    if desc_time > 0:
        print(f"  Description: {stats['description_time_s']:.1f}s (vision model)")
    print(f"  Total:       {stats['extraction_time_s'] + stats['filter_time_s'] + stats['description_time_s']:.1f}s")
    print()
    print(f"  Scorecard saved to:  {SCORECARD_PATH}")
    if args.describe:
        print(f"  Descriptions saved:  {DESCRIPTIONS_CACHE}")
    print("=" * 60)

    # ── Save scorecard ──
    SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    scorecard = {
        "phase": "7a_vision_pipeline",
        "threshold": stats["threshold"],
        "vision_model": stats["vision_model"],
        "timing": {
            "extraction_s": stats["extraction_time_s"],
            "filter_s": stats["filter_time_s"],
            "description_s": stats["description_time_s"],
        },
        "figures": {
            "total_extracted": stats["total_extracted"],
            "total_kept": stats["total_kept"],
            "total_discarded": stats["total_discarded"],
            "keep_rate": stats["keep_rate"],
            "avg_relevance_score": stats["avg_relevance_score"],
            "score_range": [stats["min_relevance_score"], stats["max_relevance_score"]],
            "figures_described": stats["figures_described"],
            "avg_description_chars": stats["avg_description_chars"],
        },
        "classification": {
            "all": stats["top_classes_all"],
            "kept": stats["top_classes_kept"],
            "discarded": stats["top_classes_discarded"],
        },
        "per_pdf": stats["per_pdf"],
        "pdfs_processed": stats["pdfs_processed"],
    }

    SCORECARD_PATH.write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Return exit code 0 if any figures were extracted
    return 0 if all_figures else 1


if __name__ == "__main__":
    sys.exit(main())
