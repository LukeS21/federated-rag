---
phase: all
status: hub
tags: [decisions, overview]
created: 2026-05-10
links:
  - vision-model-gemma4
  - classification-first
  - no-model-rotation
  - bm25-no-figures
  - drafter-citation-fix
  - cache-version-v3
  - num-predict-workaround
  - single-pass-debate
  - critic-threshold-0.50
  - hybrid-anchoring
  - chunk-level-matching
  - gliner-label-restriction
  - per-theme-parallel
  - same-model-parallel
  - dense-claim-format
---

# Decisions Overview

Hub note linking to all architecture and implementation decisions.

| Decision | Phase | Status | Rationale |
|----------|-------|--------|-----------|
| [[vision-model-gemma4]] | 7 | decided | gemma4:e4b beats alternatives on biomedical accuracy, already loaded |
| [[classification-first]] | 7 | decided | Trained classifier (65% weight) more reliable than heuristics |
| [[no-model-rotation]] | 7 | decided | gemma4:e4b already loaded + multimodal, saves 15-30s/query |
| [[bm25-no-figures]] | 7 | decided | AI-generated text doesn't belong in author-authored keyword index |
| [[drafter-citation-fix]] | 7 | decided | Removed hardcoded @author2025, added anti-hallucination constraint |
| [[cache-version-v3]] | 7 | decided | Bumped to v3 for stronger prompt constraints |
| [[num-predict-workaround]] | 7 | decided | Upstream Ollama bug; truncate output post-generation |
| [[single-pass-debate]] | 5.5 | decided | 2 Critic→Arbiter passes → 1, 67% fewer Critic calls |
| [[critic-threshold-0.50]] | 5.5 | decided | Calibrated for local models; do not revert without benchmarks |
| [[hybrid-anchoring]] | 6 | decided | BM25 + ChromaDB fusion fixes keyword-frequency bias |
| [[chunk-level-matching]] | 6 | decided | Chunk-level more honest than sentence-level (83-88% vs 99%) |
| [[gliner-label-restriction]] | 6 | decided | Precision over recall: FPR 58% → 12% |
| [[per-theme-parallel]] | 6.5 | decided | 2 workers, same-model, 23% faster wall clock |
| [[same-model-parallel]] | 5.5 | decided | Dual-model exhausted KV cache; same-model pipelined |
| [[dense-claim-format]] | 5.5 | active | One claim per line, 60-75% size reduction |
