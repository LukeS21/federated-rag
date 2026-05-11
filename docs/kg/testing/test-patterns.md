---
phase: all
status: reference
tags: [testing, patterns, mocking]
created: 2026-05-10
links: [test-suite-overview]
---

# Test Patterns

Common test patterns across the suite.

## Component Mocking Strategies

| Component | Instantiation | Mocking Strategy | Verification |
|---|---|---|---|
| PDFParser | Real instance | No mock — real Docling | Output structure validation |
| ChromaClient | Real instance | No mock | Persistence roundtrip |
| BM25Index | Real instance | No mock | Retrieval ordering |
| HybridRetriever | Real instance | No mock | Fusion scoring |
| ExtractionAgent | Real instance | `patch._call_llm` | Claim extraction shapes |
| SynthesisDrafter/Critic/Arbiter | Real instance | `patch ChatOpenAI.invoke` | Synthesis output validation |
| Anchoring | `MockBM25` | Mock retriever | Anchoring precision |
| ThematicClusterer | Real instance | `patch ChatOpenAI.invoke` | Cluster coherence |
| VisionDescriptor | Real instance | `patch requests.post` | Description output |

## Fixture Patterns

- Shared PDF fixtures in `tests/fixtures/`
- JSON cache fixtures for vision and synthesis
- Model registry fixtures for tier mapping

## Import-Order Caveat

`figure_embedder` monkey-patch must be applied before module import to avoid initialization errors.
