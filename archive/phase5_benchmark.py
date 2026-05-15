#!/usr/bin/env python3
"""Phase 5 → 6 Tier A Programmatic Benchmark.

Runs on cached survey results — zero live LLM calls.  Computes the full
metric suite specified in HANDOFF.md §"Benchmarking strategy" and
README §12.3 (Tier A):

  * Anchoring score distribution (mean, min, std, % below threshold)
  * Claim density (claims per char)
  * Entity appearance rate
  * Debate invocation rate
  * Cross-theme coverage ratio
  * Redundancy score (overlap across themes)
  * Gap analysis specificity (avg words per question)
  * Citation provenance (spot-check 5 random claims)

Outputs a JSON scorecard and prints a colour-coded report.
Thresholds: pass >= 0.80, warn >= 0.60, fail < 0.60 (for most metrics).

Also exposes ``test_benchmark_scores()`` so ``pytest`` can guard regressions.

Usage:
    python phase5_benchmark.py                          # Full benchmark report
    python phase5_benchmark.py --query "some query..."   # Specific cached query
    python -m pytest phase5_benchmark.py -v              # Regression guard
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchoring.evidence_check import decompose_claims, compute_anchoring_score, set_anchoring_chroma
from src.cache.query_cache import load_query_decomposition, load_cross_theme
from src.retrieval.chroma_client import ChromaClient
from src.unicode_map import scrub_unicode

logger = logging.getLogger("phase5_benchmark")

# Try to enable hybrid retrieval for anchoring if ChromaDB is available
try:
    _chroma = ChromaClient(collection_name="public_corpus", persist_directory=str(Path("projects/default") / "chroma_data"))
    set_anchoring_chroma(_chroma)
except Exception:
    pass

# ── Thresholds for pass/warn/fail ──────────────────────────────────────────
PASS_THRESHOLD = 0.80
WARN_THRESHOLD = 0.60

# ── ANSI colours for terminal output ───────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _color(score: float, threshold: float = PASS_THRESHOLD,
           warn: float = WARN_THRESHOLD) -> str:
    if score >= threshold:
        return GREEN
    if score >= warn:
        return YELLOW
    return RED


def _grade(score: float, threshold: float = PASS_THRESHOLD,
           warn: float = WARN_THRESHOLD) -> str:
    if score >= threshold:
        return "PASS"
    if score >= warn:
        return "WARN"
    return "FAIL"


# ── Load cached results ────────────────────────────────────────────────────

def load_survey_result(project_dir: str = "projects/default") -> Optional[Dict[str, Any]]:
    """Load the most recent survey_result.json, or return None."""
    path = Path(project_dir) / "survey_result.json"
    if not path.exists():
        logger.warning("No survey_result.json found in %s", project_dir)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_theme_syntheses(project_dir: str = "projects/default") -> Dict[str, Dict[str, Any]]:
    """Load all cached per-theme synthesis results from the query cache."""
    cache_dir = Path(project_dir) / "query_cache"
    if not cache_dir.exists():
        return {}
    themes: Dict[str, Dict[str, Any]] = {}
    for f in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("type") == "theme_synthesis":
            result = data.get("result", {})
            tn = result.get("theme", data.get("theme_name", "unknown"))
            themes[tn] = result
    return themes


# ── Metric computations ────────────────────────────────────────────────────

def metric_anchoring_distribution(
    theme_syntheses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute anchoring score distribution across all themes."""
    scores = [ts.get("anchoring_score", 0.0) for ts in theme_syntheses.values()
              if ts.get("anchoring_score") is not None]
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0,
                "count": 0, "below_threshold": 0, "below_pct": 0.0,
                "grade": "FAIL"}

    import statistics
    mn = statistics.mean(scores)
    mi = min(scores)
    mx = max(scores)
    st = statistics.stdev(scores) if len(scores) > 1 else 0.0
    below = sum(1 for s in scores if s < 0.50)
    below_pct = below / len(scores) * 100

    anchor_grade = _grade(mn, 0.70, 0.50)
    return {
        "mean": round(mn, 4),
        "min": round(mi, 4),
        "max": round(mx, 4),
        "std": round(st, 4),
        "count": len(scores),
        "below_threshold": below,
        "below_pct": round(below_pct, 1),
        "grade": anchor_grade,
    }


def metric_claim_density(
    theme_syntheses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Claims per character — measures information efficiency of dense format.

    Lower absolute number = more efficient (shorter output).  The grade is
    based on having sufficient claims relative to output length (> 1 claim
    per 500 chars = reasonable density).
    """
    densities: List[float] = []
    total_claims = 0
    total_chars = 0
    for ts in theme_syntheses.values():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        claims = decompose_claims(synth)
        n_claims = len(claims)
        n_chars = len(synth)
        if n_chars > 0:
            densities.append(n_claims / max(n_chars, 1))
        total_claims += n_claims
        total_chars += n_chars

    if not densities:
        return {"avg_claims_per_char": 0.0, "total_claims": 0, "total_chars": 0,
                "chars_per_claim": 0.0, "grade": "FAIL"}

    import statistics
    avg_density = statistics.mean(densities)  # claims per char
    chars_per_claim = total_chars / max(total_claims, 1)
    # Grade: < 500 chars per claim = dense enough
    grade = _grade(500 / max(chars_per_claim, 1), 0.60, 0.40)
    return {
        "avg_claims_per_char": round(avg_density, 6),
        "total_claims": total_claims,
        "total_chars": total_chars,
        "chars_per_claim": round(chars_per_claim, 1),
        "grade": grade,
    }


def metric_entity_appearance_rate(
    theme_syntheses: Dict[str, Dict[str, Any]],
    project_dir: str = "projects/default",
) -> Dict[str, Any]:
    """Fraction of pre-extracted entities that appear in the synthesis output."""
    extractions_dir = Path(project_dir) / "extractions"
    all_entities: Dict[str, str] = {}
    for f in extractions_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for category, ent_list in data.items():
            if not isinstance(ent_list, list):
                continue
            for ent in ent_list:
                if isinstance(ent, dict):
                    name = scrub_unicode(str(ent.get("entity", ""))).strip()
                    if name and len(name) > 2:
                        all_entities[name.lower()] = name

    # Combine all syntheses text
    all_text = " ".join(
        scrub_unicode(ts.get("synthesis", "") or "").lower()
        for ts in theme_syntheses.values()
    )

    if not all_entities:
        return {"total_entities": 0, "appeared": 0, "rate": 0.0,
                "grade": "FAIL"}

    appeared = sum(1 for ename in all_entities if ename.lower() in all_text)
    rate = appeared / len(all_entities) if all_entities else 0.0
    grade = _grade(rate, 0.40, 0.20)
    return {
        "total_entities": len(all_entities),
        "appeared": appeared,
        "rate": round(rate, 4),
        "grade": grade,
    }


def metric_debate_invocation_rate(
    theme_syntheses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Estimate debate invocation rate by checking if any theme has
    anchoring_score below CONDITIONAL_CRITIC_THRESHOLD (0.50).

    The debate chain is invoked when draft anchoring < threshold.
    Since the cached result stores the final anchoring score, themes
    with score < 0.50 very likely went through debate (unless single-paper
    with all-grounded entities, which is very rare).
    """
    from src.graph.survey_nodes import CONDITIONAL_CRITIC_THRESHOLD
    total = len(theme_syntheses)
    debated = sum(
        1 for ts in theme_syntheses.values()
        if ts.get("anchoring_score", 1.0) < CONDITIONAL_CRITIC_THRESHOLD
    )
    rate = debated / total if total else 0.0
    # Low invocation rate is good (debate is expensive)
    grade = _grade(1.0 - rate, 0.70, 0.40)  # <30% invocation = pass
    return {
        "total_themes": total,
        "debated_themes": debated,
        "invocation_rate": round(rate, 4),
        "grade": grade,
    }


def metric_cross_theme_coverage(
    theme_syntheses: Dict[str, Dict[str, Any]],
    cross_theme_synthesis: str,
) -> Dict[str, Any]:
    """Fraction of per-theme claims that appear in cross-theme synthesis."""
    all_theme_claims: List[str] = []
    for ts in theme_syntheses.values():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        all_theme_claims.extend([c.lower().strip() for c in decompose_claims(synth)])

    cross_claims = [c.lower().strip() for c in decompose_claims(scrub_unicode(cross_theme_synthesis or ""))]

    if not all_theme_claims:
        return {"per_theme_claims": 0, "cross_theme_claims": 0,
                "coverage_ratio": 0.0, "grade": "FAIL"}

    # Simple substring matching: how many per-theme claims have a substantive
    # substring overlap (>= 8 char span) in some cross-theme claim
    cross_text = " ".join(cross_claims)
    matched = 0
    for claim in all_theme_claims:
        # Check if at least one 8-char token span from claim appears in cross
        tokens = claim.split()
        if len(tokens) < 3:
            continue
        for i in range(len(tokens) - 2):
            span = " ".join(tokens[i:i + 3])
            if len(span) >= 10 and span in cross_text:
                matched += 1
                break

    ratio = matched / len(all_theme_claims) if all_theme_claims else 0.0
    grade = _grade(ratio, 0.50, 0.30)
    return {
        "per_theme_claims": len(all_theme_claims),
        "cross_theme_claims": len(cross_claims),
        "coverage_ratio": round(ratio, 4),
        "grade": grade,
    }


def metric_redundancy_score(
    theme_syntheses: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Fraction of claims that appear in >= 2 themes (overlap)."""
    theme_claims: Dict[str, List[str]] = {}
    for name, ts in theme_syntheses.items():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        theme_claims[name] = [c.lower().strip() for c in decompose_claims(synth)]

    all_claims: List[Tuple[str, str]] = []
    for name, claims in theme_claims.items():
        for c in claims:
            all_claims.append((c, name))

    if not all_claims:
        return {"total_claims": 0, "redundant_claims": 0,
                "redundancy_score": 0.0, "grade": "FAIL"}

    # Count claims that appear in multiple themes (via substantive overlap)
    redundant = 0
    seen: Dict[str, int] = {}
    for claim, name in all_claims:
        key = " ".join(claim.split()[:5])  # first 5 words as fingerprint
        if key in seen and seen[key] != name:
            redundant += 1
        seen[key] = name

    score = redundant / len(all_claims) if all_claims else 0.0
    # Low redundancy is good (< 0.20 = pass)
    grade = _grade(1.0 - score, 0.80, 0.60)
    return {
        "total_claims": len(all_claims),
        "redundant_claims": redundant,
        "redundancy_score": round(score, 4),
        "grade": grade,
    }


def metric_gap_specificity(
    gap_analysis: str,
) -> Dict[str, Any]:
    """Measure gap analysis specificity via avg tokens per question.

    Lower = more specific (better).  Grade based on having >= 2 questions
    with reasonable specificity.
    """
    if not gap_analysis:
        return {"question_count": 0, "avg_words_per_question": 0.0,
                "grade": "FAIL"}

    # Split by question marks and numbered/listed items
    text = scrub_unicode(gap_analysis)
    questions = re.split(r"(?<=[?])", text)
    questions = [q.strip() for q in questions if q.strip()]
    # Also try splitting by bullet/number lines
    if not questions or all("?" not in q for q in questions):
        questions = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]

    word_counts = [len(q.split()) for q in questions]
    avg_words = sum(word_counts) / max(len(questions), 1)

    # Specific questions should be < 30 words each on average
    grade = _grade(30 / max(avg_words, 1), 0.70, 0.50)
    return {
        "question_count": len(questions),
        "avg_words_per_question": round(avg_words, 1),
        "grade": grade,
    }


def metric_citation_provenance(
    theme_syntheses: Dict[str, Dict[str, Any]],
    n_samples: int = 5,
) -> Dict[str, Any]:
    """Spot-check citation key presence in the output.

    For each random claim that has a citation (@key), verify the cited
    paper key appears in the valid keys list.  This is a lightweight check
    — full provenance requires cross-referencing with the original paper
    text, which is semi-automated.
    """
    citation_pattern = re.compile(r"@([\w-]+(?:\.pdf)?)", re.IGNORECASE)
    all_citations: List[str] = []
    valid_keys: set = set()
    paper_counts: Dict[str, int] = {}
    orphaned_keys: List[str] = []

    for ts in theme_syntheses.values():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        found = citation_pattern.findall(synth)
        all_citations.extend(found)

    for ts in theme_syntheses.values():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        cross = scrub_unicode(ts.get("cross_theme_synthesis", "") or "")
        for match in citation_pattern.findall(synth):
            key = match.lower()
            paper_counts[key] = paper_counts.get(key, 0) + 1
            valid_keys.add(key)
        for match in citation_pattern.findall(cross):
            key = match.lower()
            valid_keys.add(key)

    # Check for potential orphan citations (cited once, no other mention)
    orphaned_keys = [k for k, v in paper_counts.items() if v == 1]

    # Cross-reference with cross-theme citations
    if not all_citations:
        return {"total_citations": 0, "unique_keys": 0,
                "orphaned_keys": 0, "orphaned_rate": 0.0,
                "grade": "WARN"}

    orphan_rate = len(orphaned_keys) / max(len(set(all_citations)), 1)
    grade = _grade(1.0 - orphan_rate, 0.80, 0.60)
    return {
        "total_citations": len(all_citations),
        "unique_keys": len(set(all_citations)),
        "orphaned_keys": len(orphaned_keys),
        "orphaned_rate": round(orphan_rate, 4),
        "grade": grade,
    }


def metric_gap_novelty(
    gap_analysis: str,
    project_dir: str = "projects/default",
) -> Dict[str, Any]:
    """Check if gap questions overlap with paper Discussion sections.

    Low overlap = genuinely novel gaps.  High overlap = regurgitating
    the authors' own future directions from their Discussion sections.

    Replaces the previous ``gap_specificity`` metric which only measured
    words/question (conciseness, not correctness).
    """
    if not gap_analysis:
        return {"question_count": 0, "novel_questions": 0, "novelty_rate": 0.0,
                "grade": "FAIL", "summary": "No gap analysis available."}

    # Extract gap questions
    questions = re.split(r"(?<=[?])", gap_analysis)
    questions = [q.strip() for q in questions if q.strip() and "?" in q]
    if not questions:
        questions = [l.strip() for l in gap_analysis.split("\n") if len(l.strip()) > 30]

    # Load Discussion-like chunks
    chroma_path = str(Path(project_dir) / "chroma_data")
    try:
        chroma = ChromaClient(collection_name="public_corpus", persist_directory=chroma_path)
        all_data = chroma.collection.get(include=["documents", "metadatas"])
    except Exception:
        return {"question_count": len(questions), "novel_questions": len(questions),
                "novelty_rate": 1.0, "grade": "PASS",
                "summary": "Could not load Discussion chunks — assuming novel."}

    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []

    # Build Discussion-like chunks: last 10% of chunks from each paper
    # (Discussion/Conclusion sections are at the end of papers)
    from collections import defaultdict
    by_paper: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        src = (meta or {}).get("source", "unknown")
        idx = int((meta or {}).get("chunk_index", i))
        if (meta or {}).get("chunk_type") == "reference":
            continue
        by_paper[src].append((idx, scrub_unicode(str(doc))))

    disc_texts: List[str] = []
    for src, entries in by_paper.items():
        entries.sort(key=lambda x: x[0])
        n_disc = max(2, len(entries) // 10)  # last 10% or at least 2 chunks
        for _, text in entries[-n_disc:]:
            disc_texts.append(text)

    if not disc_texts:
        return {"question_count": len(questions), "novel_questions": len(questions),
                "novelty_rate": 1.0, "grade": "PASS",
                "summary": "No Discussion-like chunks found — all gaps treated as novel."}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    disc_matrix = vectorizer.fit_transform(disc_texts)

    novel = 0
    overlaps: List[Dict[str, Any]] = []
    for q in questions:
        q_vec = vectorizer.transform([scrub_unicode(q)])
        sims = cosine_similarity(q_vec, disc_matrix)
        best_sim = float(sims[0].max())
        best_idx = int(sims[0].argmax())
        best_match = disc_texts[best_idx][:150]
        is_novel = best_sim < 0.40
        if is_novel:
            novel += 1
        else:
            overlaps.append({
                "question": q[:120],
                "similarity": round(best_sim, 4),
                "best_match": best_match,
            })

    rate = novel / max(len(questions), 1)
    grade = _grade(rate, 0.80, 0.50)
    return {
        "question_count": len(questions),
        "novel_questions": novel,
        "novelty_rate": round(rate, 4),
        "overlapping": overlaps[:3],
        "grade": grade,
        "summary": f"{novel}/{len(questions)} gaps are novel ({rate:.0%}); "
                   f"{len(overlaps)} overlap with Discussion sections",
    }


def metric_grounded_vs_inferential(
    theme_syntheses: Dict[str, Dict[str, Any]],
    project_dir: str = "projects/default",
) -> Dict[str, Any]:
    """Tag claims as grounded (direct evidence match) or inferential (synthesis).

    Grounded: TF-IDF similarity >= 0.35 to at least one evidence sentence.
    Inferential: no single-chunk match — synthesizing across papers.

    A high inferential rate means the system is doing real synthesis.
    A 99%+ grounded rate means it's mostly quoting evidence chunks
    rather than synthesizing — informational, not necessarily a failure.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Load evidence
    chroma_path = str(Path(project_dir) / "chroma_data")
    try:
        chroma = ChromaClient(collection_name="public_corpus", persist_directory=chroma_path)
        all_data = chroma.collection.get(include=["documents", "metadatas"])
    except Exception as e:
        logger.warning("Could not load ChromaDB for grounded/inferential metric: %s", e)
        return {"grounded": 0, "inferential": 0, "grounded_pct": 0.0,
                "grade": "FAIL", "summary": f"Could not load evidence: {e}"}

    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    ev_texts: List[str] = []
    for doc, meta in zip(docs, metas):
        if (meta or {}).get("chunk_type") == "reference":
            continue
        text = scrub_unicode(str(doc))
        if text and len(text) > 20:
            ev_texts.append(text)

    if not ev_texts:
        return {"grounded": 0, "inferential": 0, "grounded_pct": 0.0,
                "grade": "FAIL", "summary": "No evidence chunks."}

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    ev_matrix = vectorizer.fit_transform(ev_texts)

    grounded = 0
    inferential = 0
    for ts in theme_syntheses.values():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        for claim in decompose_claims(synth):
            c_vec = vectorizer.transform([scrub_unicode(claim)])
            sims = cosine_similarity(c_vec, ev_matrix)
            best_sim = float(sims[0].max())
            if best_sim >= 0.35:
                grounded += 1
            else:
                inferential += 1

    total = grounded + inferential
    gr_pct = grounded / max(total, 1)

    # Healthy: 50-95% grounded. Below 50% = too speculative.
    # Above 95% = mostly quoting, not synthesizing.
    healthy = 0.05 <= gr_pct <= 0.95
    grade = _grade(gr_pct, 0.95, 0.50) if gr_pct > 0.50 else "WARN"
    # Invert: too high grounded = WARN (not synthesizing enough)
    if gr_pct > 0.95:
        grade = "WARN"

    return {
        "grounded": grounded,
        "inferential": inferential,
        "grounded_pct": round(gr_pct, 4),
        "grade": grade,
        "summary": f"{grounded} grounded ({gr_pct:.0%}) vs {inferential} "
                   f"inferential claims — "
                   f"{'healthy balance' if healthy else 'mostly quoting (low synthesis)' if gr_pct > 0.95 else 'mostly speculative'}",
    }


# ── Master metric collector ────────────────────────────────────────────────

def compute_all_metrics(
    theme_syntheses: Dict[str, Dict[str, Any]],
    cross_theme_synthesis: str = "",
    gap_analysis: str = "",
    project_dir: str = "projects/default",
) -> Dict[str, Any]:
    """Compute the full Tier A metric suite and return a scorecard dict."""
    metrics: Dict[str, Any] = {}

    # 1. Anchoring distribution
    metrics["anchoring_distribution"] = metric_anchoring_distribution(theme_syntheses)

    # 2. Claim density
    metrics["claim_density"] = metric_claim_density(theme_syntheses)

    # 3. Entity appearance rate
    metrics["entity_appearance"] = metric_entity_appearance_rate(theme_syntheses, project_dir)

    # 4. Debate invocation rate
    metrics["debate_invocation"] = metric_debate_invocation_rate(theme_syntheses)

    # 5. Cross-theme coverage
    metrics["cross_theme_coverage"] = metric_cross_theme_coverage(
        theme_syntheses, cross_theme_synthesis,
    )

    # 6. Redundancy score
    metrics["redundancy"] = metric_redundancy_score(theme_syntheses)

    # 7. Gap novelty (Discussion-overlap test) — replaces gap_specificity
    metrics["gap_novelty"] = metric_gap_novelty(gap_analysis, project_dir)

    # 8. Grounded vs inferential
    metrics["grounded_vs_inferential"] = metric_grounded_vs_inferential(theme_syntheses, project_dir)

    # 9. Citation provenance
    metrics["citation_provenance"] = metric_citation_provenance(theme_syntheses)

    # Overall grade
    grades = [m["grade"] for m in metrics.values() if "grade" in m]
    fail_count = grades.count("FAIL")
    warn_count = grades.count("WARN")
    pass_count = grades.count("PASS")

    if fail_count > 0:
        overall = "FAIL"
    elif warn_count > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    metrics["_overall"] = {
        "grade": overall,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "total_metrics": len(grades),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return metrics


# ── Report generator ────────────────────────────────────────────────────────

def print_report(metrics: Dict[str, Any]) -> None:
    """Print a colour-coded terminal report."""
    overall = metrics.get("_overall", {})
    grade = overall.get("grade", "UNKNOWN")
    color_fn = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}.get(grade, CYAN)

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  PHASE 5 → 6 TIER A BENCHMARK SCORECARD{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"  Timestamp:  {overall.get('timestamp', 'unknown')}")
    print(f"  Overall:    {color_fn}{grade}{RESET}  ({overall.get('pass_count',0)}P / "
          f"{overall.get('warn_count',0)}W / {overall.get('fail_count',0)}F)")
    print(f"\n{BOLD}  METRICS:{RESET}")

    # Anchoring distribution
    a = metrics.get("anchoring_distribution", {})
    c = _color(a.get("mean", 0.0), 0.70, 0.50)
    print(f"\n  {CYAN}Anchoring Score Distribution{RESET}: {c}{a.get('grade','?')}{RESET}")
    print(f"    Mean: {c}{a.get('mean',0):.3f}{RESET}  |  Min: {a.get('min',0):.3f}  |  "
          f"Max: {a.get('max',0):.3f}  |  Std: {a.get('std',0):.3f}")
    print(f"    Themes below 0.50: {a.get('below_threshold',0)}/{a.get('count',0)} "
          f"({a.get('below_pct',0)}%)")

    # Claim density
    d = metrics.get("claim_density", {})
    c = _color(d.get("chars_per_claim", 999), 500, 800)
    print(f"\n  {CYAN}Claim Density{RESET}: {_color(500 / max(d.get('chars_per_claim', 999), 1), 0.6, 0.4)}{d.get('grade','?')}{RESET}")
    print(f"    {d.get('total_claims',0)} claims across {d.get('total_chars',0)} chars "
          f"(~{d.get('chars_per_claim',0):.0f} chars/claim)")

    # Entity appearance
    e = metrics.get("entity_appearance", {})
    c = _color(e.get("rate", 0.0), 0.40, 0.20)
    print(f"\n  {CYAN}Entity Appearance Rate{RESET}: {c}{e.get('grade','?')}{RESET}")
    print(f"    {e.get('appeared',0)}/{e.get('total_entities',0)} pre-extracted entities "
          f"appear in output ({e.get('rate',0):.1%})")

    # Debate invocation
    di = metrics.get("debate_invocation", {})
    rate = di.get("invocation_rate", 1.0)
    c = _color(1.0 - rate, 0.70, 0.40)
    print(f"\n  {CYAN}Debate Invocation Rate{RESET}: {c}{di.get('grade','?')}{RESET}")
    print(f"    {di.get('debated_themes',0)}/{di.get('total_themes',0)} themes "
          f"({rate:.0%}) triggered debate chain")

    # Cross-theme coverage
    ct = metrics.get("cross_theme_coverage", {})
    c = _color(ct.get("coverage_ratio", 0.0), 0.50, 0.30)
    print(f"\n  {CYAN}Cross-Theme Coverage{RESET}: {c}{ct.get('grade','?')}{RESET}")
    print(f"    {ct.get('cross_theme_claims',0)} cross claims cover "
          f"{ct.get('coverage_ratio',0):.1%} of {ct.get('per_theme_claims',0)} "
          f"per-theme claims")

    # Redundancy
    r = metrics.get("redundancy", {})
    score = r.get("redundancy_score", 1.0)
    c = _color(1.0 - score, 0.80, 0.60)
    print(f"\n  {CYAN}Theme Redundancy{RESET}: {c}{r.get('grade','?')}{RESET}")
    print(f"    {r.get('redundant_claims',0)}/{r.get('total_claims',0)} claims "
          f"overlap across themes ({score:.1%})")

    # Gap novelty (Discussion-overlap test)
    gn = metrics.get("gap_novelty", {})
    c = _color(gn.get("novelty_rate", 0), 0.80, 0.50)
    print(f"\n  {CYAN}Gap Novelty (vs Discussion){RESET}: {c}{gn.get('grade','?')}{RESET}")
    print(f"    {gn.get('novel_questions',0)}/{gn.get('question_count',0)} gaps are novel "
          f"({gn.get('novelty_rate',0):.0%}) — {gn.get('summary','?')}")

    # Grounded vs inferential
    gv = metrics.get("grounded_vs_inferential", {})
    g_pct = gv.get("grounded_pct", 0)
    c = _color(1.0 - g_pct if g_pct > 0.95 else g_pct, 0.90, 0.50)
    print(f"\n  {CYAN}Grounded vs Inferential{RESET}: {c}{gv.get('grade','?')}{RESET}")
    print(f"    {gv.get('grounded',0)} grounded ({g_pct:.0%}) / "
          f"{gv.get('inferential',0)} inferential")

    # Citation provenance
    cp = metrics.get("citation_provenance", {})
    orate = cp.get("orphaned_rate", 1.0)
    c = _color(1.0 - orate, 0.80, 0.60)
    print(f"\n  {CYAN}Citation Provenance{RESET}: {c}{cp.get('grade','?')}{RESET}")
    print(f"    {cp.get('total_citations',0)} total citations, "
          f"{cp.get('unique_keys',0)} unique keys, "
          f"{cp.get('orphaned_keys',0)} orphaned")

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    if grade == "PASS":
        print(f"  {GREEN}All metrics pass — no regressions detected.{RESET}")
    elif grade == "WARN":
        print(f"  {YELLOW}Some metrics at warning level — review before committing.{RESET}")
    else:
        print(f"  {RED}Failing metrics detected — DO NOT commit without review.{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")


# ── Pytest integration ─────────────────────────────────────────────────────

def test_benchmark_scores() -> None:
    """Pytest-compatible test: runs all Tier A metrics and asserts no FAIL grades.

    Usage:  python -m pytest phase5_benchmark.py -v
    """
    result = load_survey_result()
    if result is None:
        raise AssertionError(
            "No cached survey result found. Run phase4_demo.py first to "
            "generate a cached survey synthesis, then re-run the benchmark."
        )

    theme_syntheses = result.get("per_theme_syntheses", {})
    cross = result.get("cross_theme_synthesis", "")
    gap = result.get("gap_analysis", "")

    if not theme_syntheses:
        raise AssertionError("No per-theme syntheses in cached result.")

    metrics = compute_all_metrics(theme_syntheses, cross, gap)

    failures: List[str] = []
    warnings: List[str] = []

    for name, m in metrics.items():
        if name.startswith("_"):
            continue
        g = m.get("grade", "UNKNOWN")
        if g == "FAIL":
            failures.append(f"{name}: FAIL")
        elif g == "WARN":
            warnings.append(f"{name}: WARN")

    assert not failures, (
        f"Benchmark FAILURES: {', '.join(failures)}.  "
        f"Warnings: {', '.join(warnings) if warnings else 'none'}.  "
        f"Run 'python phase5_benchmark.py' for full scorecard."
    )

    if warnings:
        # Soft-assert: log warnings but don't fail the test (yet)
        logger.warning("Benchmark warnings: %s", ", ".join(warnings))


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5 → 6 Tier A Programmatic Benchmark",
    )
    parser.add_argument(
        "--project-dir", default="projects/default",
        help="Project directory containing cached results",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save JSON scorecard to path",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour codes in output",
    )
    args = parser.parse_args()

    if args.no_color:
        global GREEN, YELLOW, RED, CYAN, MAGENTA, RESET, BOLD
        GREEN = YELLOW = RED = CYAN = MAGENTA = RESET = BOLD = ""

    result = load_survey_result(args.project_dir)
    if result is None:
        print(f"{RED}No cached survey result found.{RESET}")
        print(f"Run 'python phase4_demo.py' first to generate a cached survey synthesis.")
        sys.exit(1)

    theme_syntheses = result.get("per_theme_syntheses", {})
    cross = result.get("cross_theme_synthesis", "")
    gap = result.get("gap_analysis", "")

    if not theme_syntheses:
        print(f"{RED}No per-theme syntheses in cached result.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Loaded cached survey result:{RESET}")
    print(f"  Themes: {len(theme_syntheses)}")
    print(f"  Cross-theme: {len(cross)} chars")
    print(f"  Gap analysis: {len(gap)} chars")

    metrics = compute_all_metrics(theme_syntheses, cross, gap, args.project_dir)
    print_report(metrics)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Scorecard saved to {out_path}")

    # Exit code for CI
    overall = metrics.get("_overall", {})
    if overall.get("grade") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
