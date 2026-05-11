---
phase: 5.5
status: open-upstream
created: 2026-05-10
tags:
  - gaps
  - ollama
  - context
links:
  - "[[model-tiering]]"
---
`OLLAMA_CONTEXT_LENGTH=32768` hardcoded by Ollama. Cannot override via LangChain ChatOpenAI. Per-model context control requires native Ollama API or Modelfile. Minor impact at current scale.
