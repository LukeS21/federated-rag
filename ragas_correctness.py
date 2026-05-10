#!/usr/bin/env python3
"""Phase 6 — LLM-as-Judge Correctness Evaluation.

Uses local Ollama models to evaluate synthesis quality via LLM-as-judge:
  1. Faithfulness — LLM checks each grounded claim against its evidence chunk
  2. Gap quality — LLM rates gap questions on novelty and actionability

Requires local Ollama models.  Zero cloud API cost.  Runs on cached syntheses.
Uses ragas-style prompting but does NOT depend on the ragas library directly
(ragas API is unstable across versions — direct prompting is more reliable).

Usage:
    python ragas_correctness.py                    # Full evaluation (~5-10 min)
    python ragas_correctness.py --sample 10        # Evaluate 10 claims only
    python ragas_correctness.py --skip-llm          # Dry run, show what'd be tested
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchoring.evidence_check import decompose_claims, compute_anchoring_score
from langchain_openai import ChatOpenAI

from src.llm import get_chat_model, resolve_model
from src.retrieval.chroma_client import ChromaClient
from src.unicode_map import sanitize_api_key, scrub_unicode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ragas_correctness")

PROJECT_DIR = Path("projects/default")
SURVEY_PATH = PROJECT_DIR / "survey_result.json"
OUTPUT_PATH = PROJECT_DIR / "correctness_scorecard.json"

# ── ANSI ───────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Evidence loader ────────────────────────────────────────────────────────

def _load_evidence() -> List[Dict[str, Any]]:
    chroma_path = str(PROJECT_DIR / "chroma_data")
    chroma = ChromaClient(collection_name="public_corpus", persist_directory=chroma_path)
    all_data = chroma.collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []
    chunks = []
    for doc, meta in zip(docs, metas):
        if (meta or {}).get("chunk_type") == "reference":
            continue
        chunks.append({"text": scrub_unicode(str(doc)), "metadata": meta or {}})
    return chunks


# ── LLM-as-Judge model factory ────────────────────────────────────────────

def _get_judge_llm(
    provider: str = "deepseek",
    model: str = "deepseek-v4-pro",
) -> ChatOpenAI:
    """Create a ChatOpenAI instance for LLM-as-judge evaluation.

    By default uses DeepSeek v4-pro API (fast, accurate, stronger judge than
    any local model).  Falls back to local Ollama if DEEPSEEK_API_KEY is
    missing or *provider* is set to 'ollama'.
    """
    if provider == "ollama" or not os.getenv("DEEPSEEK_API_KEY"):
        logger.info("Judge: using local Ollama (model=%s)", resolve_model(model))
        return get_chat_model(model=model, temperature=0.0, max_tokens=256)

    api_key = sanitize_api_key(os.getenv("DEEPSEEK_API_KEY", ""))
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    # Do NOT resolve_model() here — the global provider is Ollama, so
    # resolve_model("deepseek-v4-pro") would return "qwen3.6:35b".
    # DeepSeek API expects the literal model name.
    logger.info("Judge: using DeepSeek API (model=%s, base=%s)", model, base_url)
    return ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
        max_tokens=512,  # DeepSeek v4-pro is a reasoning model — needs headroom
        timeout=180,
        max_retries=2,
        default_headers={
            "User-Agent": "federated-rag-judge",
            "Accept": "application/json",
        },
    )


# ── LLM-as-Judge prompts ──────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = (
    "You are a rigorous biomedical evidence auditor. Your task is to verify "
    "whether a claim is factually supported by the provided evidence chunk. "
    "Be critical and objective — do not default to assuming the claim is correct. "
    "Answer ONLY with a JSON object: "
    '{"score": <1-5>, "reasoning": "<one sentence explaining what the evidence actually says vs the claim>"}\n\n'
    "Scoring (apply these thresholds strictly):\n"
    '  5 = Claim is VERBATIM or near-verbatim from the evidence. The evidence states the exact same finding. '
    'Reserve this for direct quotes and close paraphrases. Do NOT give 5 just because the topic matches.\n'
    '  4 = Evidence strongly supports the claim but uses different wording or different specific numbers. '
    'The direction/mechanism is clearly the same.\n'
    '  3 = Evidence discusses the same topic but the claim extrapolates, generalizes, or adds assertions '
    'not directly stated. The claim may be reasonable but is not directly supported.\n'
    '  2 = Evidence is weakly or tangentially related. The claim goes beyond what the evidence shows '
    'or makes assertions the evidence does not address.\n'
    '  1 = Claim directly contradicts the evidence, is factually impossible, or is completely fabricated '
    'with no basis in the evidence.\n\n'
    "CRITICAL: Default to 3 if unsure. Only give 5 for direct matches. "
    "If the evidence chunk does not mention the specific entity or mechanism in the claim, score 2 or below. "
    "Output ONLY the JSON object. No other text."
)

_FAITHFULNESS_USER = (
    "Claim: {claim}\n\n"
    "Evidence chunk: {evidence}\n\n"
    "Is this claim supported by the evidence? Be specific about what the evidence "
    "actually says versus what the claim asserts. Output JSON."
)

_GAP_JUDGE_SYSTEM = (
    "You are a biomedical research strategy analyst evaluating research gap questions. "
    "Be critical — rate gaps honestly, not generously. Rate each gap on two dimensions "
    "(1-5 scale) and output a JSON object:\n"
    '{"novelty": <1-5>, "actionability": <1-5>, "reasoning": "<one sentence>"}\n\n'
    "Novelty:\n"
    '  5 = The question is genuinely unanswered by ALL papers in the corpus. No paper addresses '
    'or even asks this question.\n'
    '  4 = The question touches on a topic the papers discuss but asks something specific they do not answer.\n'
    '  3 = Papers address related questions but this specific angle requires novel integration.\n'
    '  2 = The question largely overlaps with a question the papers themselves ask or answer.\n'
    '  1 = The question is directly answered or explicitly asked by one or more papers.\n\n'
    "Actionability:\n"
    '  5 = Immediately suggests a concrete, feasible experiment with clear variables and methods.\n'
    '  4 = Points to a specific experiment but the methods would need some development.\n'
    '  3 = Suggests a research direction but lacks a clear experimental hook.\n'
    '  2 = Vague research direction with no clear path to testing.\n'
    '  1 = Untestable, purely theoretical, or a question that cannot be answered experimentally.\n\n'
    "Default to 3 if unsure. Reserve 5 for clearly novel, clearly actionable gaps. "
    "Output ONLY the JSON object. No other text."
)

_GAP_JUDGE_USER = (
    "Research gap question: {gap}\n\n"
    "Context from the literature survey:\n{context}\n\n"
    "Rate this gap honestly. Output JSON."
)

# Modified faithfulness prompt for inferential claims (cross-paper synthesis).
# These claims combine evidence from multiple papers — they won't appear
# verbatim in any single chunk. Judge based on directional support.
_INFERENTIAL_FAITHFULNESS_SYSTEM = (
    "You are a rigorous biomedical evidence auditor. This claim SYNTHESIZES "
    "findings across multiple papers — it will NOT appear verbatim in any single "
    "evidence chunk. Judge whether the evidence DIRECTIONALLY supports or "
    "contradicts the claim.\n\n"
    "Answer ONLY with a JSON object: "
    '{"score": <1-5>, "reasoning": "<one sentence>"}\n\n'
    "Scoring for inferential claims:\n"
    '  5 = Multiple evidence chunks directionally support this claim. The '
    'synthesis is well-justified by the evidence provided.\n'
    '  4 = Evidence mostly supports the direction but some nuance is lost '
    'or one part of the claim is less supported.\n'
    '  3 = Evidence provides partial support. The claim extrapolates in a '
    'reasonable but unverified way.\n'
    '  2 = Evidence is too weak to support the claim. The synthesis goes '
    'well beyond what the data shows.\n'
    '  1 = Evidence directly contradicts the claim or the claim is fabricated.\n\n'
    "Default to 3 if unsure. Do NOT give 5 unless evidence clearly supports "
    "every component of the claim. Output ONLY the JSON object."
)


# ── Judging functions ─────────────────────────────────────────────────────

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from LLM output."""
    text = scrub_unicode(text).strip()
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fences
    if "```" in text:
        for segment in text.split("```"):
            seg = segment.strip()
            if seg.lower().startswith("json"):
                seg = seg[4:].lstrip()
            try:
                if seg.startswith("{"):
                    return json.loads(seg)
            except json.JSONDecodeError:
                continue
    # Brace extraction
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(text[l:r + 1])
        except json.JSONDecodeError:
            pass
    return {}


def judge_claim_faithfulness(
    claim: str,
    evidence_chunks: List[Dict[str, Any]],
    judge_llm: ChatOpenAI,
    inferential: bool = False,
) -> Dict[str, Any]:
    """LLM judges whether a claim is supported by evidence.

    For inferential claims (cross-paper synthesis), uses a modified prompt
    that doesn't expect verbatim evidence matches.
    """
    from src.retrieval.bm25_index import BM25Index
    bm25 = BM25Index()
    bm25.add_documents([ch["text"] for ch in evidence_chunks])
    best_results = bm25.query(claim, n_results=3)
    if best_results:
        best_evidence = " ||| ".join(str(r)[:1000] for r in best_results)
    else:
        best_evidence = "(no evidence)"

    from langchain_core.messages import HumanMessage, SystemMessage

    system = _INFERENTIAL_FAITHFULNESS_SYSTEM if inferential else _FAITHFULNESS_SYSTEM
    msgs = [
        SystemMessage(content=system),
        HumanMessage(content=_FAITHFULNESS_USER.format(
            claim=claim[:1000], evidence=best_evidence,
        )),
    ]

    try:
        response = judge_llm.invoke(msgs)
        result = _parse_json_response(str(response.content or ""))
        return {
            "claim": claim[:150],
            "faithfulness_score": result.get("score", 0),
            "reasoning": str(result.get("reasoning", ""))[:200],
            "best_evidence": best_evidence[:300],
        }
    except Exception as e:
        return {
            "claim": claim[:150],
            "faithfulness_score": 0,
            "reasoning": f"LLM judge failed: {e}",
            "best_evidence": "",
        }
    except Exception as e:
        return {
            "claim": claim[:150],
            "faithfulness_score": 0,
            "reasoning": f"LLM judge failed: {e}",
            "best_evidence": "",
        }


def judge_gap_quality(
    gap_question: str,
    context: str,
    judge_llm: ChatOpenAI,
) -> Dict[str, Any]:
    """LLM rates a gap question for novelty and actionability."""
    from langchain_core.messages import HumanMessage, SystemMessage

    msgs = [
        SystemMessage(content=_GAP_JUDGE_SYSTEM),
        HumanMessage(content=_GAP_JUDGE_USER.format(
            gap=gap_question, context=context[:3000],
        )),
    ]

    try:
        response = judge_llm.invoke(msgs)
        result = _parse_json_response(str(response.content or ""))
        return {
            "gap": gap_question[:150],
            "novelty": result.get("novelty", 0),
            "actionability": result.get("actionability", 0),
            "reasoning": str(result.get("reasoning", ""))[:200],
        }
    except Exception as e:
        return {
            "gap": gap_question[:150],
            "novelty": 0,
            "actionability": 0,
            "reasoning": f"LLM judge failed: {e}",
        }


# ── Calibration claims ────────────────────────────────────────────────────
# TRUE: verbatim or near-verbatim from Avery et al. / corpus papers
# FALSE: factually impossible or directly contradicted
# GRAY: oversimplifications or speculative extensions

_CALIBRATION_CLAIMS = [
    # ── TRUE (should score 4-5) ──
    {
        "claim": "CD4+ and CD8+ T cells reduce inflammation and promote bone healing "
                 "around titanium implants.",
        "expected": "high", "type": "true",
    },
    {
        "claim": "Rough-hydrophilic titanium surfaces skew macrophages toward an "
                 "anti-inflammatory M2 phenotype.",
        "expected": "high", "type": "true",
    },
    {
        "claim": "Obese mice exhibit elevated serum leptin and C-reactive protein "
                 "compared to lean controls.",
        "expected": "high", "type": "true",
    },
    {
        "claim": "T cell deficiency promotes a pro-inflammatory macrophage phenotype "
                 "around titanium implants.",
        "expected": "high", "type": "true",
    },
    # ── FALSE (should score 1-2) ──
    {
        "claim": "Titanium alloys dissolve completely within 48 hours in physiological "
                 "saline at body temperature.",
        "expected": "low", "type": "false",
    },
    {
        "claim": "Obesity eliminates all macrophage activity at the implant site, "
                 "preventing any immune response.",
        "expected": "low", "type": "false",
    },
    {
        "claim": "IFN-gamma is exclusively produced by B lymphocytes and has no role "
                 "in T cell biology.",
        "expected": "low", "type": "false",
    },
    {
        "claim": "Smooth titanium surfaces promote faster osseointegration than rough "
                 "surfaces in all animal models.",
        "expected": "low", "type": "false",
    },
    # ── GRAY (oversimplifications — should score 2-4) ──
    {
        "claim": "M1 macrophage polarization always inhibits bone formation regardless "
                 "of timing or context.",
        "expected": "mid", "type": "gray",
    },
    {
        "claim": "IL-10 alone is sufficient to reverse obesity-induced peri-implant "
                 "inflammation.",
        "expected": "mid", "type": "gray",
    },
]


def run_calibration(
    evidence: List[Dict[str, Any]],
    judge_llm: ChatOpenAI,
) -> Dict[str, Any]:
    """Run calibration claims through the judge to validate it discriminates.

    Returns calibration results plus a validity assessment.
    """
    logger.info("Calibration: judging %d calibration claims", len(_CALIBRATION_CLAIMS))
    results = []
    for cal in _CALIBRATION_CLAIMS:
        logger.info("  [calibration] %s...", cal["claim"][:60])
        result = judge_claim_faithfulness(cal["claim"], evidence, judge_llm)
        result["type"] = cal["type"]
        result["expected"] = cal["expected"]
        results.append(result)

    # Assess judge calibration
    true_scores = [r["faithfulness_score"] for r in results if r["type"] == "true"]
    false_scores = [r["faithfulness_score"] for r in results if r["type"] == "false"]
    gray_scores = [r["faithfulness_score"] for r in results if r["type"] == "gray"]

    true_ok = all(4 <= s <= 5 for s in true_scores) if true_scores else True
    false_ok = all(1 <= s <= 2 for s in false_scores) if false_scores else True
    # Gray: 2-4 range, at least some should not be 5
    gray_ok = all(2 <= s <= 4 for s in gray_scores) if gray_scores else True

    judge_valid = true_ok and false_ok

    issues = []
    if not true_ok:
        low_true = [s for s in true_scores if s < 4]
        issues.append(f"TRUE claims scored too low: {low_true}")
    if not false_ok:
        high_false = [s for s in false_scores if s > 2]
        issues.append(f"FALSE claims scored too high: {high_false}")
    if not gray_ok:
        issues.append("GRAY claims outside expected 2-4 range")

    return {
        "judge_valid": judge_valid,
        "judge_issues": issues,
        "true_claims": {"scores": true_scores, "avg": round(sum(true_scores) / max(len(true_scores), 1), 2)},
        "false_claims": {"scores": false_scores, "avg": round(sum(false_scores) / max(len(false_scores), 1), 2)},
        "gray_claims": {"scores": gray_scores, "avg": round(sum(gray_scores) / max(len(gray_scores), 1), 2)},
        "details": results,
    }


# ── Orchestration ──────────────────────────────────────────────────────────

def run_faithfulness_eval(
    theme_syntheses: Dict[str, Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    judge_llm: ChatOpenAI,
    sample: int = 0,
) -> Dict[str, Any]:
    """Evaluate faithfulness of grounded AND inferential claims separately."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ev_texts = [ch["text"] for ch in evidence]
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    ev_matrix = vectorizer.fit_transform(ev_texts)

    grounded_claims: List[Tuple[str, float]] = []
    inferential_claims: List[Tuple[str, float]] = []
    for ts in theme_syntheses.values():
        synth = scrub_unicode(ts.get("synthesis", "") or "")
        for claim in decompose_claims(synth):
            c_vec = vectorizer.transform([scrub_unicode(claim)])
            best_sim = float(cosine_similarity(c_vec, ev_matrix)[0].max())
            if best_sim >= 0.35:
                grounded_claims.append((claim, best_sim))
            else:
                inferential_claims.append((claim, best_sim))

    # Sample each
    grounded_claims.sort(key=lambda x: x[1], reverse=True)
    if sample > 0:
        import random
        random.seed(42)
        gr_sample = min(sample, len(grounded_claims))
        inf_sample = min(max(sample // 4, 3), len(inferential_claims))
        grounded_claims = random.sample(grounded_claims, gr_sample)
        if inferential_claims:
            inferential_claims = random.sample(inferential_claims, inf_sample)
    else:
        # If sample=0, evaluate ALL grounded claims, but cap inferential at 20
        if len(inferential_claims) > 20:
            import random
            random.seed(42)
            inferential_claims = random.sample(inferential_claims, 20)

    def _judge_claims(claims: List[Tuple[str, float]], label: str,
                      inferential: bool = False) -> Dict[str, Any]:
        logger.info("%s: %d claims to judge", label, len(claims))
        results = []
        scores = []
        for i, (claim, _) in enumerate(claims):
            logger.info("  [%d/%d] Judging: %s...", i + 1, len(claims), claim[:60])
            result = judge_claim_faithfulness(claim, evidence, judge_llm,
                                               inferential=inferential)
            results.append(result)
            if result["faithfulness_score"] > 0:
                scores.append(result["faithfulness_score"])
        avg = sum(scores) / max(len(scores), 1) if scores else 0.0
        return {
            "claims_evaluated": len(results),
            "avg_faithfulness": round(avg, 2),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "score_distribution": {
                "5 (verbatim match)": sum(1 for r in results if r["faithfulness_score"] == 5),
                "4 (strongly supported)": sum(1 for r in results if r["faithfulness_score"] == 4),
                "3 (extrapolated)": sum(1 for r in results if r["faithfulness_score"] == 3),
                "2 (weakly related)": sum(1 for r in results if r["faithfulness_score"] == 2),
                "1 (contradicts/fabricated)": sum(1 for r in results if r["faithfulness_score"] == 1),
            },
            "details": results,
        }

    return {
        "grounded": _judge_claims(grounded_claims, "Grounded", inferential=False),
        "inferential": _judge_claims(inferential_claims, "Inferential", inferential=True),
    }


def run_gap_quality_eval(
    gap_analysis: str,
    cross_theme_synthesis: str,
    judge_llm: ChatOpenAI,
) -> Dict[str, Any]:
    """Evaluate gap question quality using *judge_llm*."""
    # Extract gap questions
    questions = re.split(r"(?<=[?])", gap_analysis)
    questions = [q.strip() for q in questions if q.strip() and "?" in q]
    if not questions:
        questions = [l.strip() for l in gap_analysis.split("\n") if len(l.strip()) > 30]

    logger.info("Gap quality eval: %d questions to judge", len(questions))

    results = []
    novelty_scores = []
    actionability_scores = []
    for i, q in enumerate(questions):
        logger.info("  [%d/%d] Judging gap: %s...", i + 1, len(questions), q[:60])
        result = judge_gap_quality(q, cross_theme_synthesis[:3000], judge_llm)
        results.append(result)
        if result["novelty"] > 0:
            novelty_scores.append(result["novelty"])
        if result["actionability"] > 0:
            actionability_scores.append(result["actionability"])

    avg_novelty = sum(novelty_scores) / max(len(novelty_scores), 1) if novelty_scores else 0.0
    avg_actionability = sum(actionability_scores) / max(len(actionability_scores), 1) if actionability_scores else 0.0

    return {
        "questions_evaluated": len(results),
        "avg_novelty": round(avg_novelty, 2),
        "avg_actionability": round(avg_actionability, 2),
        "details": results,
    }


# ── Report ─────────────────────────────────────────────────────────────────

def print_scorecard(
    faithfulness: Dict[str, Any],
    gap_quality: Dict[str, Any],
    calibration: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  CORRECTNESS SCORECARD (LLM-as-Judge){RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Calibration ──
    if calibration:
        valid = calibration.get("judge_valid", False)
        vc = GREEN if valid else RED
        print(f"\n  {CYAN}Judge Calibration{RESET}: {vc}{'VALID' if valid else 'INVALID'}{RESET}")
        tc = calibration.get("true_claims", {})
        fc = calibration.get("false_claims", {})
        gc = calibration.get("gray_claims", {})
        print(f"    TRUE (expect 4-5):  avg {tc.get('avg',0):.1f}  scores: {tc.get('scores',[])}")
        print(f"    FALSE (expect 1-2): avg {fc.get('avg',0):.1f}  scores: {fc.get('scores',[])}")
        print(f"    GRAY (expect 2-4):  avg {gc.get('avg',0):.1f}  scores: {gc.get('scores',[])}")
        for issue in calibration.get("judge_issues", []):
            print(f"    {RED}⚠ {issue}{RESET}")

    # ── Grounded claims ──
    grounded = faithfulness.get("grounded", {})
    print(f"\n  {CYAN}Grounded Claim Faithfulness{RESET}")
    _print_faith_section(grounded)

    # ── Inferential claims ──
    inferential = faithfulness.get("inferential", {})
    if inferential and inferential.get("claims_evaluated", 0) > 0:
        print(f"\n  {CYAN}Inferential Claim Faithfulness{RESET}")
        print(f"    {YELLOW}(These claims synthesize across evidence — may lack direct chunk support){RESET}")
        _print_faith_section(inferential)
    else:
        print(f"\n  {CYAN}Inferential Claim Faithfulness{RESET}: none to evaluate")

    # ── Gap quality ──
    print(f"\n  {CYAN}Gap Question Quality{RESET}")
    n = gap_quality.get("avg_novelty", 0)
    a = gap_quality.get("avg_actionability", 0)
    nc = GREEN if n >= 3.5 else YELLOW if n >= 2.5 else RED
    ac = GREEN if a >= 3.5 else YELLOW if a >= 2.5 else RED
    print(f"    Avg novelty:     {nc}{n:.1f}/5{RESET}")
    print(f"    Avg actionability: {ac}{a:.1f}/5{RESET}")
    print(f"    Questions:        {gap_quality.get('questions_evaluated', 0)}")

    print(f"\n{BOLD}{'=' * 65}{RESET}\n")


def _print_faith_section(data: Dict[str, Any]) -> None:
    avg = data.get("avg_faithfulness", 0)
    color = GREEN if avg >= 4.0 else YELLOW if avg >= 3.0 else RED
    print(f"    Avg score: {color}{avg:.1f}/5{RESET} ({data.get('claims_evaluated', 0)} claims)")
    dist = data.get("score_distribution", {})
    for label, count in sorted(dist.items()):
        if count:
            print(f"    {label}: {count}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 6 — LLM-as-Judge Correctness Evaluation",
    )
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of grounded claims to judge (default: 20)")
    parser.add_argument("--judge-provider", default="deepseek",
                        help="Provider for judge LLM: deepseek (default), ollama")
    parser.add_argument("--judge-model", default="deepseek-chat",
                        help="Judge model (default: deepseek-chat; use deepseek-v4-pro for stronger reasoning)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration claims (TRUE/FALSE/GRAY) to validate judge before evaluating")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Dry run: show what would be evaluated, no LLM calls")
    parser.add_argument("--output", "-o", default=str(OUTPUT_PATH),
                        help="Output path for JSON scorecard")
    args = parser.parse_args()

    if not SURVEY_PATH.exists():
        print(f"{RED}No cached survey — run phase4_demo.py first.{RESET}")
        sys.exit(1)

    survey = json.loads(SURVEY_PATH.read_text(encoding="utf-8"))
    themes = survey.get("per_theme_syntheses", {})
    gap = survey.get("gap_analysis", "")
    cross = survey.get("cross_theme_synthesis", "")

    if args.skip_llm:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        evidence = _load_evidence()
        ev_texts = [ch["text"] for ch in evidence]
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        ev_matrix = vectorizer.fit_transform(ev_texts)

        grounded_count = 0
        inferential_count = 0
        for ts in themes.values():
            for claim in decompose_claims(scrub_unicode(ts.get("synthesis", "") or "")):
                c_vec = vectorizer.transform([scrub_unicode(claim)])
                best_sim = float(cosine_similarity(c_vec, ev_matrix)[0].max())
                if best_sim >= 0.35:
                    grounded_count += 1
                else:
                    inferential_count += 1

        gap_qs = [q.strip() for q in re.split(r"(?<=[?])", gap) if q.strip() and "?" in q]

        print(f"\n{BOLD}Dry run — would evaluate:{RESET}")
        print(f"  Grounded:     {grounded_count} claims")
        print(f"  Inferential:  {inferential_count} claims")
        print(f"  Gaps:         {len(gap_qs)} questions")
        print(f"  Calibration:  {len(_CALIBRATION_CLAIMS)} claims ({'yes' if args.calibrate else 'no -- use --calibrate to enable'})")
        print(f"  Judge model:  {args.judge_model} (provider: {args.judge_provider})")
        return

    judge_llm = _get_judge_llm(provider=args.judge_provider, model=args.judge_model)

    t0 = time.time()
    evidence = _load_evidence()

    print(f"\nRunning LLM-as-Judge evaluation:")
    print(f"  Judge: {args.judge_model} (provider: {args.judge_provider})")

    calibration = None
    if args.calibrate:
        calibration = run_calibration(evidence, judge_llm)

    faithfulness = run_faithfulness_eval(
        themes, evidence, judge_llm, sample=args.sample,
    )
    gap_quality = run_gap_quality_eval(gap, cross, judge_llm)

    elapsed = time.time() - t0
    print_scorecard(faithfulness, gap_quality, calibration, elapsed)

    scorecard = {
        "faithfulness": faithfulness,
        "gap_quality": gap_quality,
        "calibration": calibration,
        "metadata": {
            "judge_model": args.judge_model,
            "judge_provider": args.judge_provider,
            "elapsed_s": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Scorecard saved to {out_path}")


if __name__ == "__main__":
    main()
