---
phase: all
status: open-minor
tags: [testing, gaps]
created: 2026-05-10
links: [test-suite-overview]
---

# Known Test Gaps

1. **Vision descriptions with real gemma4:e4b** — tested via mocked API. Full integration requires live Ollama.
2. **Sectioned survey end-to-end with live LLMs** — tested with mocked Drafter.
3. **Multi-query variance for baseline** — single run cached.
4. **Phase 9 literature discovery POC** — no unit tests for PubMed/S2 wrappers.
5. **Post-generation citation validation** — not tested (not yet implemented).

All gaps are minor — core logic is well-tested. Phase 8 scale testing is the priority.
