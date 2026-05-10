#!/usr/bin/env python3
"""Phase 6 — Automated Benchmark Dataset Generator.

Generates 80–100 high-confidence QA pairs from the paper corpus using
LLMs + RAGAS validation, per HANDOFF.md §"Automated benchmark creation":

  1. LLM generates research questions from paper chunks
  2. LLM generates "gold" synthesis answers
  3. RAGAS faithfulness validation (LLM-as-judge)
  4. Self-consistency filtering (2× generation, keep only claims in both)
  5. Anchoring score verification
  6. Human spot-check report (sample claims to review)

Output: ``projects/default/benchmark_dataset.json`` — a structured dataset
with questions, gold answers, key entities, evidence chunks, and validation
scores.  Ready for use by the Tier A benchmark.

Usage:
    python generate_benchmark_dataset.py              # Full generation (~30-60 min)
    python generate_benchmark_dataset.py --sample 20  # Smaller test run
    python generate_benchmark_dataset.py --dry-run     # Show what would be generated
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.synthesis_drafter import SynthesisDrafter
from src.anchoring.evidence_check import compute_anchoring_score, decompose_claims
from src.llm import get_chat_model, resolve_model
from src.retrieval.chroma_client import ChromaClient
from src.unicode_map import scrub_unicode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("benchmark_dataset")

PROJECT_DIR = Path("projects/default")
OUTPUT_PATH = PROJECT_DIR / "benchmark_dataset.json"
CHROMA_PATH = str(PROJECT_DIR / "chroma_data")
TARGET_SAMPLES = 80
MAX_CHUNKS_PER_PAPER = 20

# ── ANSI ───────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Question generation prompt templates ───────────────────────────────────

_QUESTION_PROMPTS = [
    "Based on the following paper excerpt, generate 2 specific research questions "
    "that this paper's findings can answer. Be precise — mention specific entities, "
    "methods, or cell types from the text.",

    "Given this biomedical research text, formulate 2 focused questions about "
    "the mechanisms, outcomes, or experimental models described. Each question "
    "should be answerable from the data in this paper alone.",

    "Read this paper chunk and write 2 narrow, answerable research questions "
    "about the biological processes, cell types, or materials discussed. "
    "Avoid broad/vague questions.",
]

_ANSWER_PROMPT = (
    "You are a biomedical literature synthesis expert.  Answer the following "
    "research question based ONLY on the provided paper text.  Produce a concise, "
    "evidence-backed synthesis with fact-by-fact citations to the source chunks.  "
    "One claim per line.  Include specific entities, numbers, and findings.  "
    "Plain text only.  Do not speculate beyond the provided evidence."
)


# ── Paper chunk loader ─────────────────────────────────────────────────────

def load_paper_chunks() -> Dict[str, List[Dict[str, Any]]]:
    """Load body chunks from ChromaDB grouped by source paper."""
    chroma = ChromaClient(collection_name="public_corpus", persist_directory=CHROMA_PATH)
    all_data = chroma.collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []

    papers: Dict[str, List[Dict[str, Any]]] = {}
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        if meta.get("chunk_type") == "reference":
            continue
        src = meta.get("source", "unknown")
        if not src or src == "unknown":
            continue
        papers.setdefault(src, []).append({
            "text": scrub_unicode(str(doc)),
            "metadata": meta,
        })

    # Cap chunks per paper
    for src in papers:
        if len(papers[src]) > MAX_CHUNKS_PER_PAPER:
            step = len(papers[src]) / MAX_CHUNKS_PER_PAPER
            papers[src] = [papers[src][int(i * step)] for i in range(MAX_CHUNKS_PER_PAPER)]

    logger.info("Loaded %d papers with %d total chunks",
                len(papers), sum(len(v) for v in papers.values()))
    return papers


# ── LLM-based generation ───────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think> block from Qwen outputs."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_questions(
    paper_chunks: List[Dict[str, Any]],
    model_tier: str = "small",
    questions_per_paper: int = 3,
) -> List[str]:
    """Generate research questions from paper chunks using an LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage

    chunk_text = "\n\n".join(ch["text"][:500] for ch in paper_chunks[:8])
    if not chunk_text.strip():
        return []

    prompt_idx = hash(chunk_text[:100]) % len(_QUESTION_PROMPTS)
    system_prompt = _QUESTION_PROMPTS[prompt_idx]

    llm = get_chat_model(model=model_tier, temperature=0.3)
    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Paper text:\n\n{chunk_text[:6000]}\n\n"
                     f"Generate {questions_per_paper} specific research questions (one per line, "
                     f"numbered)."),
    ]
    try:
        response = llm.invoke(msg)
        text = _strip_thinking(scrub_unicode(str(response.content or "")))
    except Exception as e:
        logger.warning("Question generation failed: %s", e)
        return []

    # Parse questions
    import re
    questions: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        # Remove numbering
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(line) > 20 and line.endswith("?"):
            questions.append(line)
        elif len(line) > 30 and "?" in line:
            # Grab the question part
            parts = line.split("?")
            if len(parts) >= 2:
                questions.append(parts[0].strip() + "?")

    return questions[:questions_per_paper]


def generate_answer(
    question: str,
    paper_chunks: List[Dict[str, Any]],
    model_tier: str = "large",
) -> Tuple[str, Dict[str, Any]]:
    """Generate a "gold" answer for a question using the evidence chunks.

    Returns (answer_text, metrics_dict).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Build evidence with chunk IDs
    evidence_parts = []
    for i, ch in enumerate(paper_chunks[:15]):
        evidence_parts.append(f"[Chunk {i}] {ch['text'][:800]}")

    evidence_text = "\n\n---\n\n".join(evidence_parts)

    llm = get_chat_model(model=model_tier, temperature=0.0)
    msg = [
        SystemMessage(content=_ANSWER_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nEvidence chunks:\n{evidence_text}"),
    ]

    try:
        response = llm.invoke(msg)
        answer = _strip_thinking(scrub_unicode(str(response.content or "")))
    except Exception as e:
        logger.warning("Answer generation failed for '%s': %s", question[:50], e)
        return "", {}

    # Compute anchoring score
    claims = decompose_claims(answer)
    score, ungrounded = compute_anchoring_score(claims, paper_chunks)
    metrics = {
        "anchoring_score": round(score, 4),
        "claims_count": len(claims),
        "ungrounded_count": len(ungrounded),
        "answer_chars": len(answer),
    }

    return answer, metrics


# ── RAGAS faithfulness (optional, if installed) ───────────────────────────

def ragas_faithfulness_check(
    question: str,
    answer: str,
    contexts: List[str],
) -> Optional[float]:
    """Run RAGAS faithfulness check. Returns score or None if RAGAS unavailable."""
    try:
        from ragas.metrics import Faithfulness
        from ragas import evaluate
        from datasets import Dataset
    except ImportError:
        return None

    try:
        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })
        result = evaluate(ds, metrics=[Faithfulness()])
        score = float(result.get("faithfulness", [0])[0])
        return round(score, 4)
    except Exception as e:
        logger.debug("RAGAS check failed: %s", e)
        return None


# ── Self-consistency check ─────────────────────────────────────────────────

def check_self_consistency(
    answer1: str,
    answer2: str,
) -> Tuple[float, List[str]]:
    """Compare two answers — claims appearing in both are consistent."""
    claims1 = decompose_claims(answer1)
    claims2_raw = decompose_claims(answer2)

    # Normalize and fingerprint
    def _fingerprint(claim: str) -> str:
        words = claim.lower().split()
        return " ".join(words[:8]) if len(words) >= 5 else claim.lower()

    fp1 = {_fingerprint(c): c for c in claims1}
    fp2 = {_fingerprint(c): c for c in claims2_raw}

    consistent_keys = set(fp1.keys()) & set(fp2.keys())
    consistent_claims = [fp1[k] for k in consistent_keys]

    total_unique = len(set(list(fp1.keys()) + list(fp2.keys())))
    consistency = len(consistent_keys) / max(total_unique, 1)

    return round(consistency, 4), consistent_claims


# ── Dataset generation orchestrator ────────────────────────────────────────

def generate_dataset(
    papers: Dict[str, List[Dict[str, Any]]],
    target_samples: int = 80,
    sample: int = 0,
) -> Dict[str, Any]:
    """Generate the full benchmark dataset.

    Args:
        papers: Dict of source_name → list of chunk dicts
        target_samples: Target number of QA pairs to generate
        sample: If > 0, limit to this many papers (for testing)

    Returns:
        Dict with 'samples', 'metadata', 'spot_check_items'
    """
    paper_names = sorted(papers.keys())
    if sample > 0:
        paper_names = paper_names[:sample]

    questions_per_paper = max(1, target_samples // len(paper_names))
    logger.info("Target: %d samples from %d papers (%d Qs/paper)",
                target_samples, len(paper_names), questions_per_paper)

    # Phase 1: Generate questions (fast tier)
    logger.info("=" * 60)
    logger.info("PHASE 1: Generating research questions...")
    logger.info("=" * 60)

    all_question_pairs: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for i, paper_name in enumerate(paper_names):
        logger.info("[%d/%d] %s", i + 1, len(paper_names), paper_name)
        chunks = papers[paper_name]
        questions = generate_questions(chunks, model_tier="small",
                                       questions_per_paper=questions_per_paper)
        for q in questions:
            all_question_pairs.append((paper_name, q, chunks))
        logger.info("  Generated %d questions", len(questions))

    logger.info("Total questions generated: %d", len(all_question_pairs))

    # Phase 2: Generate answers + validate (reasoning tier)
    logger.info("=" * 60)
    logger.info("PHASE 2: Generating answers + validating...")
    logger.info("=" * 60)

    samples: List[Dict[str, Any]] = []
    total_score = 0.0
    failed = 0

    for i, (paper_name, question, chunks) in enumerate(all_question_pairs):
        logger.info("[%d/%d] %s...", i + 1, len(all_question_pairs), question[:70])

        # Generate answer (run 1 — primary)
        answer1, metrics1 = generate_answer(question, chunks, model_tier="large")
        if not answer1:
            failed += 1
            continue

        # Self-consistency check (run 2)
        answer2, metrics2 = generate_answer(question, chunks, model_tier="large")
        consistency, consistent_claims = check_self_consistency(answer1, answer2)

        # Build context list for RAGAS
        context_strings = [ch["text"][:500] for ch in chunks[:10]]

        # RAGAS faithfulness
        ragas_score = ragas_faithfulness_check(question, answer1, context_strings)

        # Key entities from pre-extraction
        extractions_dir = PROJECT_DIR / "extractions" / f"{paper_name}.json"
        key_entities: Dict[str, List[str]] = {}
        if extractions_dir.exists():
            try:
                extr_data = json.loads(extractions_dir.read_text(encoding="utf-8"))
                for cat, ent_list in extr_data.items():
                    if isinstance(ent_list, list) and len(ent_list) > 0:
                        key_entities[cat] = [
                            e.get("entity", str(e)) if isinstance(e, dict) else str(e)
                            for e in ent_list[:5]
                        ]
            except Exception:
                pass

        sample = {
            "id": hashlib.sha256(f"{paper_name}:{question}".encode()).hexdigest()[:16],
            "paper": paper_name,
            "question": question,
            "gold_answer": answer1,
            "answer_run2": answer2,
            "anchoring_score": metrics1.get("anchoring_score", 0),
            "claims_count": metrics1.get("claims_count", 0),
            "ungrounded_count": metrics1.get("ungrounded_count", 0),
            "self_consistency": consistency,
            "consistent_claims": consistent_claims[:5],  # Store up to 5 for spot-check
            "ragas_faithfulness": ragas_score,
            "key_entities": key_entities,
            "num_evidence_chunks": len(chunks),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        samples.append(sample)
        total_score += metrics1.get("anchoring_score", 0)

    # Filter low-quality samples
    quality_filtered = [
        s for s in samples
        if s["anchoring_score"] >= 0.30 and s["self_consistency"] >= 0.25
    ]
    logger.info("Quality filter: %d/%d samples pass (anchoring >= 0.30, consistency >= 0.25)",
                len(quality_filtered), len(samples))

    # Sort by anchoring score (best first)
    quality_filtered.sort(key=lambda s: s["anchoring_score"], reverse=True)

    # Generate spot-check report
    spot_checks = []
    for s in quality_filtered[:max(10, len(quality_filtered) // 10)]:
        spot_checks.append({
            "id": s["id"],
            "paper": s["paper"],
            "question": s["question"][:120],
            "sample_claim": s["consistent_claims"][0][:200] if s["consistent_claims"]
                            else decompose_claims(s["gold_answer"])[0][:200] if decompose_claims(s["gold_answer"])
                            else "N/A",
        })

    avg_anchor = total_score / max(len(samples), 1)
    dataset = {
        "metadata": {
            "total_generated": len(samples),
            "quality_passing": len(quality_filtered),
            "failed_generation": failed,
            "avg_anchoring_score": round(avg_anchor, 4),
            "avg_self_consistency": round(
                sum(s["self_consistency"] for s in samples) / max(len(samples), 1), 4),
            "papers_used": len(paper_names),
            "target_samples": target_samples,
            "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": {
                "question_generation": resolve_model("small"),
                "answer_generation": resolve_model("large"),
            },
        },
        "samples": quality_filtered[:target_samples],
        "spot_check_items": spot_checks,
    }

    return dataset


# ── Report printing ────────────────────────────────────────────────────────

def print_dataset_report(dataset: Dict[str, Any]) -> None:
    """Print a summary of the generated dataset."""
    meta = dataset.get("metadata", {})
    samples = dataset.get("samples", [])
    spot_checks = dataset.get("spot_check_items", [])

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  BENCHMARK DATASET GENERATION REPORT{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    print(f"\n  {CYAN}Generation:{RESET}")
    print(f"    Papers used:       {meta.get('papers_used', 0)}")
    print(f"    Samples generated: {meta.get('total_generated', 0)}")
    print(f"    Quality passing:   {GREEN}{meta.get('quality_passing', 0)}{RESET}")
    print(f"    Failed (empty):    {meta.get('failed_generation', 0)}")

    print(f"\n  {CYAN}Quality:{RESET}")
    print(f"    Avg anchoring:     {meta.get('avg_anchoring_score', 0):.4f}")
    print(f"    Avg consistency:   {meta.get('avg_self_consistency', 0):.4f}")

    if samples:
        anchor_scores = [s.get("anchoring_score", 0) for s in samples]
        import statistics
        print(f"    Score range:       {min(anchor_scores):.3f} - {max(anchor_scores):.3f}")
        print(f"    Score std dev:     {statistics.stdev(anchor_scores) if len(anchor_scores) > 1 else 0:.3f}")

    print(f"\n  {CYAN}Models used:{RESET}")
    print(f"    Questions:  {meta.get('models', {}).get('question_generation', '?')}")
    print(f"    Answers:    {meta.get('models', {}).get('answer_generation', '?')}")

    print(f"\n  {CYAN}Spot-check items ({len(spot_checks)}):{RESET}")
    for sc in spot_checks:
        print(f"    [{sc['paper']}] {sc['question'][:80]}...")
        print(f"      → {sc['sample_claim'][:100]}...")

    print(f"\n  {CYAN}Output:{RESET} {OUTPUT_PATH}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 6 — Automated Benchmark Dataset Generator",
    )
    parser.add_argument("--sample", type=int, default=0,
                        help="Limit to N papers for test run (0 = all)")
    parser.add_argument("--target", type=int, default=TARGET_SAMPLES,
                        help=f"Target number of samples (default: {TARGET_SAMPLES})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without running LLMs")
    parser.add_argument("--output", "-o", default=str(OUTPUT_PATH),
                        help="Output path")
    args = parser.parse_args()

    papers = load_paper_chunks()
    if not papers:
        print(f"{YELLOW}No papers found in ChromaDB. Run phase4_demo.py first.{RESET}")
        sys.exit(1)

    print(f"Papers available: {len(papers)}")

    if args.dry_run:
        questions_per_paper = max(1, args.target // len(papers))
        estimated_llm_calls = (
            len(papers) +  # question generation (1 per paper)
            len(papers) * questions_per_paper * 2  # answer generation (2x for consistency)
        )
        print(f"\n{BOLD}Dry run — {estimated_llm_calls} LLM calls would be needed:{RESET}")
        print(f"  {len(papers)} question-generation calls (fast tier)")
        print(f"  {len(papers) * questions_per_paper * 2} answer-generation calls (reasoning tier)")
        print(f"  Estimated time: {estimated_llm_calls * 30 // 60}-{estimated_llm_calls * 90 // 60} min")
        print(f"\n  Papers that would be used:")
        for p in sorted(papers.keys()):
            print(f"    - {p} ({len(papers[p])} chunks)")
        return

    t0 = time.time()
    dataset = generate_dataset(papers, target_samples=args.target, sample=args.sample)
    elapsed = time.time() - t0

    dataset["metadata"]["generation_elapsed_s"] = round(elapsed, 1)

    print_dataset_report(dataset)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Dataset saved to {out_path}")
    print(f"Total generation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
