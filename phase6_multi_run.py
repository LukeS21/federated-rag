#!/usr/bin/env python3
"""Phase 6 Multi‑Run Variance Analysis.

Re‑runs the same survey query N times (default 3), each with a fresh
decomposition (L1 cache cleared), to measure anchoring‑score stability.

Variability between runs (observed 0.818 → 0.922) comes primarily from
the query decomposer producing different theme breakdowns on each call.
This script measures that variance so we know whether single‑run benchmark
scores are representative.

Outputs a JSON scorecard with per‑run metrics and a summary colour
report.  Requires running Ollama models (~5‑8 min per run).

Usage:
    python phase6_multi_run.py                  # 3 runs with default query
    python phase6_multi_run.py --runs 5         # 5 runs
    python phase6_multi_run.py --query "..."    # Custom query
    python phase6_multi_run.py --skip-run       # Just read cached survey
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_run")

# Default query (from cached survey result)
_DEFAULT_QUERY = (
    "How do T cells and macrophages coordinate bone healing around titanium "
    "implants? How does obesity impact this? what cytokines would be "
    "upregulated/downregulated for macrophages and what are targets for "
    "therapeutic intervention?"
)

QUERY_CACHE_DIR = Path("projects/default/query_cache")
SURVEY_RESULT = Path("projects/default/survey_result.json")
BENCHMARK_SCORECARD = Path("projects/default/benchmark_scorecard.json")


@dataclass
class RunMetrics:
    """Per‑run anchoring and synthesis metrics."""
    run_index: int
    elapsed_seconds: float
    anchoring_scores: Dict[str, float]  # theme_name → score
    anchoring_mean: float
    anchoring_min: float
    anchoring_max: float
    anchoring_std: float
    n_themes: int
    n_claims: int
    debate_invocations: int
    gap_novelty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_index": self.run_index,
            "elapsed_s": round(self.elapsed_seconds, 1),
            "anchoring_scores": {
                k: round(v, 4) for k, v in self.anchoring_scores.items()
            },
            "anchoring_mean": round(self.anchoring_mean, 4),
            "anchoring_min": round(self.anchoring_min, 4),
            "anchoring_max": round(self.anchoring_max, 4),
            "anchoring_std": round(self.anchoring_std, 4),
            "n_themes": self.n_themes,
            "n_claims": self.n_claims,
            "debate_invocations": self.debate_invocations,
        }


@dataclass
class VarianceScorecard:
    """Aggregate variance across all runs."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str = ""
    n_runs: int = 0
    total_elapsed_s: float = 0.0
    runs: List[RunMetrics] = field(default_factory=list)

    # Aggregate metrics
    anchoring_means: List[float] = field(default_factory=list)
    anchoring_mean_of_means: float = 0.0
    anchoring_std_of_means: float = 0.0
    anchoring_coefficient_of_variation: float = 0.0

    # Distribution
    anchor_range_min: float = 0.0
    anchor_range_max: float = 0.0
    anchor_range_spread: float = 0.0

    n_themes_range: tuple = (0, 0)
    n_claims_range: tuple = (0, 0)
    debate_rate: float = 0.0

    interpretation: str = ""

    def compute_aggregates(self) -> None:
        if not self.runs:
            return

        self.anchoring_means = [r.anchoring_mean for r in self.runs]
        self.anchoring_mean_of_means = statistics.mean(self.anchoring_means)
        self.anchoring_std_of_means = (
            statistics.stdev(self.anchoring_means) if len(self.anchoring_means) > 1 else 0.0
        )
        self.anchoring_coefficient_of_variation = (
            self.anchoring_std_of_means / self.anchoring_mean_of_means
            if self.anchoring_mean_of_means > 0
            else 0.0
        )

        self.anchor_range_min = min(self.anchoring_means)
        self.anchor_range_max = max(self.anchoring_means)
        self.anchor_range_spread = self.anchor_range_max - self.anchor_range_min

        self.n_themes_range = (
            min(r.n_themes for r in self.runs),
            max(r.n_themes for r in self.runs),
        )
        self.n_claims_range = (
            min(r.n_claims for r in self.runs),
            max(r.n_claims for r in self.runs),
        )

        total_debate = sum(r.debate_invocations for r in self.runs)
        total_themes = sum(r.n_themes for r in self.runs)
        self.debate_rate = total_debate / total_themes if total_themes > 0 else 0.0

        # Interpretation
        cv = self.anchoring_coefficient_of_variation
        if cv < 0.05:
            stability = "STABLE"
            self.interpretation = (
                f"Anchoring scores are stable (CoV={cv:.3f}). "
                f"Single‑run benchmarks are representative."
            )
        elif cv < 0.10:
            stability = "MICRO-VARIANCE"
            self.interpretation = (
                f"Minor variance in anchoring (CoV={cv:.3f}). "
                f"Single‑run benchmarks are acceptable; report mean of 3 runs for publication."
            )
        elif cv < 0.20:
            stability = "MODERATE-VARIANCE"
            self.interpretation = (
                f"Moderate variance detected (CoV={cv:.3f}). "
                f"Always report mean ± std from 3+ runs for benchmarks."
            )
        else:
            stability = "HIGH-VARIANCE"
            self.interpretation = (
                f"High variance (CoV={cv:.3f}). "
                f"Decomposition is unstable — investigate query decomposer prompt or model."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "n_runs": self.n_runs,
            "total_elapsed_s": round(self.total_elapsed_s, 1),
            "runs": [r.to_dict() for r in self.runs],
            "aggregate": {
                "anchoring_mean_of_means": round(self.anchoring_mean_of_means, 4),
                "anchoring_std_of_means": round(self.anchoring_std_of_means, 4),
                "anchoring_cv": round(self.anchoring_coefficient_of_variation, 4),
                "anchor_range": [round(self.anchor_range_min, 4), round(self.anchor_range_max, 4)],
                "anchor_spread": round(self.anchor_range_spread, 4),
                "n_themes_range": list(self.n_themes_range),
                "n_claims_range": list(self.n_claims_range),
                "debate_rate": round(self.debate_rate, 4),
            },
            "interpretation": self.interpretation,
        }


def _clear_decomposition_cache() -> int:
    """Remove L1 decomposition cache entries. Returns count removed."""
    if not QUERY_CACHE_DIR.exists():
        return 0
    removed = 0
    for p in sorted(QUERY_CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("type") == "decomposition":
                p.unlink()
                removed += 1
        except (json.JSONDecodeError, OSError):
            pass
    return removed


def _extract_metrics_from_survey() -> RunMetrics:
    """Parse the cached survey result and compute per‑theme anchoring scores."""
    if not SURVEY_RESULT.exists():
        raise FileNotFoundError(f"Survey result not found: {SURVEY_RESULT}")

    data = json.loads(SURVEY_RESULT.read_text(encoding="utf-8"))

    per_theme = data.get("per_theme_syntheses", {})
    anchoring_scores: Dict[str, float] = {}
    n_claims = 0
    debate_invocations = 0

    for theme_name, theme_data in per_theme.items():
        if isinstance(theme_data, dict):
            score = theme_data.get("anchoring_score", 0.0)
            anchoring_scores[theme_name] = float(score) if score else 0.0
        elif isinstance(theme_data, str):
            anchoring_scores[theme_name] = 0.0

    # Count claims from cross-theme synthesis
    cross = data.get("cross_theme_synthesis", "")
    if isinstance(cross, str):
        n_claims = len([l for l in cross.split("\n") if l.strip()])

    # Count debate invocations (themes with anchoring score < threshold)
    threshold = float(os.getenv("CONDITIONAL_CRITIC_THRESHOLD", "0.50"))
    for score in anchoring_scores.values():
        if 0 < score < threshold:
            debate_invocations += 1

    scores_list = list(anchoring_scores.values())
    return RunMetrics(
        run_index=0,
        elapsed_seconds=0.0,
        anchoring_scores=anchoring_scores,
        anchoring_mean=statistics.mean(scores_list) if scores_list else 0.0,
        anchoring_min=min(scores_list) if scores_list else 0.0,
        anchoring_max=max(scores_list) if scores_list else 0.0,
        anchoring_std=statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0,
        n_themes=len(per_theme),
        n_claims=n_claims,
        debate_invocations=debate_invocations,
    )


def run_single_survey_pipeline(query: str) -> float:
    """Run the Survey Mode pipeline once and return elapsed seconds.

    The pipeline writes to ``survey_result.json`` automatically.
    """
    from src.graph.survey_nodes import build_survey_graph
    from langgraph.graph import StateGraph
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_survey_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer, interrupt_before=["survey_scrub"])

    num_ctx = int(os.getenv("OLLAMA_CONTEXT_LENGTH", "32768"))
    timeout = int(os.getenv("LLM_TIMEOUT", "900"))
    client_kwargs = {
        "timeout": timeout,
        "num_ctx": num_ctx,
        "temperature": 0.0,
    }

    initial_state = {
        "user_query": query,
        "query_scope": "public",
        "mode": "survey",
        "num_ctx": num_ctx,
        "client_kwargs": client_kwargs,
    }

    config = {"configurable": {"thread_id": f"var-{int(time.time())}"}}

    t0 = time.time()
    final_state = app.invoke(initial_state, config)
    elapsed = time.time() - t0

    # Save survey result for later extraction
    result = {
        "decomposed_themes": final_state.get("decomposed_themes", []),
        "thematic_clusters": final_state.get("thematic_clusters", {}),
        "per_theme_syntheses": final_state.get("per_theme_syntheses", {}),
        "cross_theme_synthesis": final_state.get("cross_theme_synthesis", ""),
        "gap_analysis": final_state.get("gap_analysis", ""),
    }
    SURVEY_RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Auto‑approve for non‑interactive runs
    try:
        app.update_state(config, {"human_approved": True})
    except Exception:
        pass

    return elapsed


def run_variance_analysis(
    query: str = _DEFAULT_QUERY,
    n_runs: int = 3,
    skip_run: bool = False,
) -> VarianceScorecard:
    """Run the full multi‑run variance analysis.

    Args:
        query: The research query to evaluate.
        n_runs: Number of independent runs.
        skip_run: If True, read the existing ``survey_result.json``
            and report a single‑run summary (no live LLM calls).

    Returns:
        Aggregated ``VarianceScorecard``.
    """
    scorecard = VarianceScorecard(query=query, n_runs=n_runs)

    if skip_run:
        logger.info("Skipping live runs — reading cached survey_result.json")
        try:
            run = _extract_metrics_from_survey()
            run.run_index = 1
            scorecard.runs = [run]
            scorecard.n_runs = 1
        except FileNotFoundError:
            logger.error("No cached survey result — run without --skip-run first.")
        scorecard.compute_aggregates()
        return scorecard

    logger.info("Multi‑run variance analysis: %d runs | query: %s...", n_runs, query[:80])

    for i in range(n_runs):
        logger.info("─" * 60)
        logger.info("Run %d/%d", i + 1, n_runs)

        # Clear L1 cache to force fresh decomposition
        removed = _clear_decomposition_cache()
        if removed:
            logger.info("  Cleared %d L1 decomposition cache entries", removed)

        elapsed = run_single_survey_pipeline(query)
        logger.info("  Pipeline completed in %.1f s (%.1f min)", elapsed, elapsed / 60)

        try:
            run = _extract_metrics_from_survey()
        except FileNotFoundError:
            logger.error("  Survey result not written — skipping run %d", i + 1)
            continue

        run.run_index = i + 1
        run.elapsed_seconds = elapsed
        scorecard.runs.append(run)

        logger.info(
            "  Anchoring: mean=%.4f  min=%.4f  max=%.4f  std=%.4f  %d themes  %d claims",
            run.anchoring_mean, run.anchoring_min, run.anchoring_max,
            run.anchoring_std, run.n_themes, run.n_claims,
        )

    scorecard.total_elapsed_s = sum(r.elapsed_seconds for r in scorecard.runs)
    scorecard.compute_aggregates()

    logger.info("─" * 60)
    logger.info("Variance analysis complete (%d runs, %.1f min total)",
                len(scorecard.runs), scorecard.total_elapsed_s / 60)

    return scorecard


def _colour(label: str, condition: str) -> str:
    codes = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m", "INFO": "\033[96m"}
    return f"{codes.get(condition, '')}{label}\033[0m"


def print_report(scorecard: VarianceScorecard) -> None:
    """Print a colour‑coded variance report."""
    print()
    print("=" * 70)
    print("  PHASE 6 MULTI‑RUN VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"  Timestamp:  {scorecard.timestamp}")
    print(f"  Query:      {scorecard.query[:100]}{'...' if len(scorecard.query) > 100 else ''}")
    print(f"  Runs:       {scorecard.n_runs}")
    print(f"  Total time: {scorecard.total_elapsed_s:.0f}s ({scorecard.total_elapsed_s/60:.1f} min)")
    print()

    if not scorecard.runs:
        print("  No runs completed.")
        return

    # Per‑run table
    print(f"  {_colour('PER‑RUN ANCHORING', 'INFO')}")
    print(f"  {'Run':>4s}  {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}  {'Std':>8s}  {'Themes':>6s}  {'Claims':>6s}  {'Debate':>6s}")
    print(f"  {'─' * 68}")
    for run in scorecard.runs:
        print(
            f"  {run.run_index:>4d}  "
            f"{run.anchoring_mean:>8.4f}  "
            f"{run.anchoring_min:>8.4f}  "
            f"{run.anchoring_max:>8.4f}  "
            f"{run.anchoring_std:>8.4f}  "
            f"{run.n_themes:>6d}  "
            f"{run.n_claims:>6d}  "
            f"{run.debate_invocations:>6d}"
        )
    print()

    # Aggregate
    cv = scorecard.anchoring_coefficient_of_variation
    if cv < 0.05:
        stability = "STABLE"
    elif cv < 0.10:
        stability = "MICRO-VARIANCE"
    elif cv < 0.20:
        stability = "MODERATE-VARIANCE"
    else:
        stability = "HIGH-VARIANCE"

    cv_color = "PASS" if cv < 0.10 else ("WARN" if cv < 0.20 else "FAIL")
    spread = scorecard.anchor_range_spread
    spread_color = "PASS" if spread < 0.05 else ("WARN" if spread < 0.10 else "FAIL")

    print(f"  {_colour('AGGREGATE', 'INFO')}")
    print(f"  Anchoring mean of means:  {_colour(f'{scorecard.anchoring_mean_of_means:.4f}', 'INFO')}")
    print(f"  Std of means:             {_colour(f'{scorecard.anchoring_std_of_means:.4f}', 'INFO')}")
    print(f"  Coefficient of variation: {_colour(f'{cv:.4f}', cv_color)}")
    print(f"  Range (min–max):          {_colour(f'{scorecard.anchor_range_min:.4f} – {scorecard.anchor_range_max:.4f}', 'INFO')}")
    print(f"  Spread:                   {_colour(f'{spread:.4f}', spread_color)}")
    print(f"  Themes per run:           {scorecard.n_themes_range[0]}–{scorecard.n_themes_range[1]}")
    print(f"  Claims per run:           {scorecard.n_claims_range[0]}–{scorecard.n_claims_range[1]}")
    print(f"  Debate rate:              {scorecard.debate_rate:.1%}")
    print()
    print(f"  {_colour('INTERPRETATION', 'INFO')}")
    print(f"  {scorecard.interpretation}")
    print()
    print("=" * 70)


def save_scorecard(scorecard: VarianceScorecard, path: str = "projects/default/variance_scorecard.json") -> None:
    """Save variance scorecard to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Scorecard saved to {p}")


# ── Pytest integration ───────────────────────────────────────────────────

def test_variance_on_cached_survey() -> None:
    """Variance scorecard can be extracted from cached survey."""
    if not SURVEY_RESULT.exists():
        pytest.skip("No cached survey result available")
    scorecard = run_variance_analysis(skip_run=True)
    assert scorecard.n_runs >= 1
    assert len(scorecard.runs) == 1
    assert scorecard.anchoring_mean_of_means > 0
    run = scorecard.runs[0]
    assert run.n_themes > 0
    assert run.n_claims > 0


def test_anchoring_scores_in_range() -> None:
    """Anchoring scores are in valid range [0, 1]."""
    if not SURVEY_RESULT.exists():
        pytest.skip("No cached survey result available")
    scorecard = run_variance_analysis(skip_run=True)
    for run in scorecard.runs:
        assert 0.0 <= run.anchoring_mean <= 1.0
        for theme, score in run.anchoring_scores.items():
            assert 0.0 <= score <= 1.0, f"{theme} score {score} out of range"


# ── CLI entry ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 6 Multi‑Run Variance Analysis")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of independent runs (default: 3)")
    parser.add_argument("--query", type=str, default=_DEFAULT_QUERY,
                        help="Research query to evaluate")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip live runs — read cached survey_result.json")
    parser.add_argument("--output", type=str,
                        default="projects/default/variance_scorecard.json",
                        help="Output JSON path")
    args = parser.parse_args()

    print("Phase 6 Multi‑Run Variance Analysis")

    if not args.skip_run:
        print(f"  Running {args.runs} surveys (~{args.runs * 6}–{args.runs * 8} min on M3 Max)")
        print(f"  Press Ctrl+C to cancel.")
        yn = input("  Continue? [y/N] ").strip().lower()
        if yn != "y":
            print("  Cancelled.")
            exit(0)

    scorecard = run_variance_analysis(
        query=args.query,
        n_runs=args.runs,
        skip_run=args.skip_run,
    )

    print_report(scorecard)
    save_scorecard(scorecard, path=args.output)
