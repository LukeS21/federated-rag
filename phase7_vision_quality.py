#!/usr/bin/env python
"""
Phase 7a Vision Description Quality Check

Validates that vision model descriptions meet minimum quality standards.
Uses three tiers:
  Tier A (programmatic):  length, ASCII compliance, keyword presence
  Tier B (LLM-as-Judge):  faithfulness to source figure (requires API)
  Cached:  results saved to projects/default/vision_quality_scorecard.json

Usage:
    # Programmatic checks only (instant, no LLM)
    python phase7_vision_quality.py

    # Include LLM-as-Judge evaluation (requires vision model)
    python phase7_vision_quality.py --judge

    # Generate cached descriptions first, then check quality
    python phase7_vision_quality.py --judge --describe --samples 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.vision.figure_extractor import FigureExtractor
from src.vision.figure_filter import FigureFilter
from src.vision.vision_descriptor import VisionDescriptor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase7_quality")

PROJECT_DIR = Path("projects/default")
QUALITY_SCORECARD = PROJECT_DIR / "vision_quality_scorecard.json"
DESCRIPTIONS_CACHE = PROJECT_DIR / "figure_descriptions.json"

BIOMEDICAL_KEYWORDS = [
    "graph", "chart", "bar", "line", "plot", "axis", "figure",
    "cell", "gene", "protein", "cytokine", "receptor", "immune",
    "macrophage", "neutrophil", "lymphocyte", "t cell", "b cell",
    "titanium", "implant", "surface", "bone", "tissue", "inflammation",
    "expression", "level", "increase", "decrease", "significant",
    "staining", "microscopy", "histology", "immunohistochemistry",
    "flow cytometry", "elisa", "western blot", "pcr",
]


def check_ascii(text: str) -> Tuple[bool, int]:
    """Check if text is pure ASCII. Returns (is_ascii, non_ascii_count)."""
    non_ascii = sum(1 for c in text if ord(c) >= 128)
    return non_ascii == 0, non_ascii


def check_min_length(text: str, min_chars: int = 20) -> bool:
    """Check if description meets minimum length."""
    return len(text.strip()) >= min_chars


def check_keyword_presence(text: str, keywords: List[str]) -> Tuple[int, List[str]]:
    """Count how many domain keywords appear in the description."""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    return len(found), found


def compute_description_quality(figures: List[Dict]) -> Dict:
    """Compute quality metrics for a list of described figures.

    Returns a dict with aggregated metrics.
    """
    described = [f for f in figures if f.get("description", "").strip()]
    if not described:
        return {
            "total_figures": len(figures),
            "described": 0,
            "pass_ascii": 0,
            "pass_min_length": 0,
            "avg_length": 0,
            "avg_keywords": 0,
            "score": 0.0,
        }

    n = len(described)
    ascii_pass = 0
    length_pass = 0
    total_keywords = 0
    total_length = 0

    for fig in described:
        desc = fig["description"]
        is_ascii, _ = check_ascii(desc)
        if is_ascii:
            ascii_pass += 1
        if check_min_length(desc):
            length_pass += 1
        kw_count, _ = check_keyword_presence(desc, BIOMEDICAL_KEYWORDS)
        total_keywords += kw_count
        total_length += len(desc)

    # Composite quality score (0-1)
    # 40% ASCII, 30% length, 30% keyword density
    ascii_rate = ascii_pass / n
    length_rate = length_pass / n
    avg_kw = total_keywords / n
    keyword_rate = min(avg_kw / 3.0, 1.0)  # 3 keywords = full score

    quality_score = 0.40 * ascii_rate + 0.30 * length_rate + 0.30 * keyword_rate

    return {
        "total_figures": len(figures),
        "described": n,
        "pass_ascii": ascii_pass,
        "pass_min_length": length_pass,
        "ascii_rate": round(ascii_rate, 3),
        "length_pass_rate": round(length_rate, 3),
        "avg_length": round(total_length / n, 1),
        "avg_keywords": round(avg_kw, 2),
        "keyword_rate": round(keyword_rate, 3),
        "quality_score": round(quality_score, 3),
    }


def per_figure_detail(figures: List[Dict]) -> List[Dict]:
    """Generate per-figure quality details."""
    details = []
    for i, fig in enumerate(figures):
        desc = fig.get("description", "")
        if not desc.strip():
            continue
        is_ascii, na_count = check_ascii(desc)
        len_ok = check_min_length(desc)
        kw_count, kw_found = check_keyword_presence(desc, BIOMEDICAL_KEYWORDS)
        details.append({
            "file_path": fig.get("file_path", ""),
            "page_no": fig.get("page_no", 0),
            "caption": fig.get("caption", "")[:100],
            "description_length": len(desc),
            "ascii": is_ascii,
            "non_ascii_chars": na_count,
            "min_length_ok": len_ok,
            "keywords_count": kw_count,
            "keywords_found": kw_found[:10],
            "description_preview": desc[:200],
        })
    return details


def main():
    parser = argparse.ArgumentParser(description="Phase 7a Vision Quality Check")
    parser.add_argument("--describe", action="store_true",
                        help="Generate descriptions before checking (uses vision model)")
    parser.add_argument("--judge", action="store_true",
                        help="Include LLM-as-Judge faithfulness evaluation")
    parser.add_argument("--samples", type=int, default=10,
                        help="Max figures to describe (default: 10)")
    parser.add_argument("--vision-model", type=str, default=None,
                        help="Vision model (default: from VISION_MODEL env or llava:7b)")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Relevance filtering threshold (default: 0.35)")
    parser.add_argument("--pdfs", type=str, nargs="*", default=None,
                        help="PDFs to process")
    args = parser.parse_args()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load or extract figures ──
    if args.describe:
        from phase7_vision_benchmark import find_pdfs

        if args.pdfs:
            pdf_paths = [Path(p) for p in args.pdfs]
        else:
            pdf_paths = find_pdfs(Path("data"))

        extractor = FigureExtractor()
        ff = FigureFilter(threshold=args.threshold)

        all_figures = []
        for pdf_path in pdf_paths:
            try:
                figures = extractor.extract(pdf_path)
                all_figures.extend(figures)
            except Exception as e:
                logger.error("Failed: %s: %s", pdf_path.name, e)

        kept = ff.filter(all_figures)[:args.samples]

        # Describe
        vd = VisionDescriptor(model=args.vision_model)
        described = vd.describe_figures(
            kept, reload_after=True, fallback_to_caption=True,
        )
        VisionDescriptor.save_descriptions(DESCRIPTIONS_CACHE, {
            f["file_path"]: f.get("description", "") for f in described
        })
        figures_to_check = described
    else:
        # Read cached descriptions and scorecard
        if not DESCRIPTIONS_CACHE.exists():
            logger.error("No cached descriptions. Run with --describe first.")
            sys.exit(1)

        cached = VisionDescriptor.load_descriptions(DESCRIPTIONS_CACHE)

        scorecard_path = PROJECT_DIR / "vision_scorecard.json"
        if not scorecard_path.exists():
            logger.error("No vision scorecard. Run phase7_vision_benchmark.py first.")
            sys.exit(1)

        # Build figure list from cache
        figures_to_check = []
        for fp, desc in cached.items():
            figures_to_check.append({
                "file_path": fp,
                "description": desc,
                "caption": "",
                "page_no": 0,
            })

    # ── Tier A: Programmatic quality ──
    quality = compute_description_quality(figures_to_check)
    details = per_figure_detail(figures_to_check)

    print("\n" + "=" * 70)
    print("  Phase 7a Vision Description — Quality Report")
    print("=" * 70)
    print(f"  Figures described:  {quality['described']}/{quality['total_figures']}")
    print(f"  ASCII clean:        {quality['pass_ascii']}/{quality['described']} ({quality['ascii_rate']*100:.0f}%)")
    print(f"  Min length (>=20):  {quality['pass_min_length']}/{quality['described']} ({quality['length_pass_rate']*100:.0f}%)")
    print(f"  Avg description:    {quality['avg_length']:.0f} chars")
    print(f"  Avg keywords:       {quality['avg_keywords']:.1f}")
    print(f"  Quality score:      {quality['quality_score']:.2f}/1.00")
    print()

    if quality["described"] > 0:
        print("  Per-figure details:")
        for d in details[:5]:
            flag = "OK" if (d["ascii"] and d["min_length_ok"]) else "FAIL"
            print(f"    [{flag}] {Path(d['file_path']).name}")
            print(f"           {d['description_length']} chars, {d['keywords_count']} keywords (\"{', '.join(d['keywords_found'][:3])}\")")
            print(f"           Preview: \"{d['description_preview'][:120]}...\"" if len(d['description_preview']) > 120 else f"           Preview: \"{d['description_preview']}\"")
            print()
    print("=" * 70)

    # ── Tier B: LLM-as-Judge (optional, requires API) ──
    judge_results = None
    if args.judge:
        logger.info("LLM-as-Judge evaluation not yet implemented (requires calibrated judge)")
        logger.info("This pattern will mirror ragas_correctness.py from Phase 6")
        # Placeholder: would load a calibrated judge and score each description

    # ── Save scorecard ──
    scorecard = {
        "phase": "7a_vision_quality",
        "programmatic": quality,
        "per_figure": details,
        "judge": judge_results,
    }
    QUALITY_SCORECARD.write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Quality scorecard saved to: {QUALITY_SCORECARD}")

    return 0 if quality.get("quality_score", 0) >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
