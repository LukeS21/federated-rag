---
phase: all
status: reference
created: 2026-05-10
tags:
  - dev
  - config
  - env
links:
  - "[[quick-start]]"
  - "[[model-tiering]]"
---
Full `.env` reference.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM provider backend |
| `OLLAMA_HOST` | — | Ollama server address |
| `OLLAMA_SMALL_MODEL` | `gemma4:e4b` | Small/fast model for classification and simple tasks |
| `OLLAMA_LARGE_MODEL` | `qwen3.6:35b` | Large model for synthesis and complex reasoning |
| `OLLAMA_ALT_MODEL` | — | Alternative model fallback |
| `OLLAMA_KEEP_ALIVE` | `60s` | Time to keep model loaded in memory |
| `LLM_TIMEOUT` | `900` | Request timeout in seconds |
| `GAP_ANALYSIS_MODEL` | — | Model used for gap analysis benchmarking |
| `LLM_MAX_TOKENS` | `4096` | Maximum tokens per response |
| `VISION_MODEL` | `gemma4:e4b` | Model used for figure description |
| `DEEPSEEK_API_KEY` | — | API key for DeepSeek models |
| `DEEPSEEK_BASE_URL` | — | Base URL for DeepSeek API |
| `ZOTERO_LIBRARY_ID` | — | Zotero library identifier |
| `ZOTERO_API_KEY` | — | Zotero API key |
| `PUBLIC_CORPUS_DIR` | — | Directory for public corpus documents |
| `SECURE_CORPUS_DIR` | — | Directory for secure/restricted corpus |
| `PROJECT_DIR` | — | Project root directory |
| `SECURITY_AUDIT_LOG` | — | Path to security audit log file |
| `BOUNDARY_SCRUB_PATTERNS` | — | Patterns for boundary scrubbing |
| `GLINER_PRIVACY_ENABLED` | — | Enable GLiNER privacy scanning |
| `PER_THEME_MAX_WORKERS` | `2` | Max parallel workers per theme |
| `INSTITUTIONAL_PROXY_URL` | — | Institutional proxy URL |
