---
phase: 5
status: reference
tags: [architecture, llm, ollama, deepseek]
created: 2026-05-10
links: [model-tiering, gemma4-e4b, qwen3-6-35b, security-modules]
---

# LLM Provider

Unified LLM interface for all model calls.

## Entry Point

All LLM calls go through `get_chat_model()` from `src/llm/__init__.py`. Never use `ChatOpenAI` directly.

## Provider Selection

`LLM_PROVIDER` environment variable:
- `ollama` (default) — local Ollama server via OpenAI-compatible API
- `deepseek` — DeepSeek cloud API

## Model Tiering

`resolve_model()` maps logical names to concrete models:

| Logical Name | Tier | Example Concrete |
|-------------|------|------------------|
| `"small"`, `"chat"` | Fast | gemma4:e4b |
| `"large"`, `"pro"` | Reasoning | qwen3.6:35b |

## Scope Routing

`get_chat_model_for_scope()` routes to public or secure Ollama hosts based on `query_scope`. Ensures secure corpus data never touches public endpoints.

## Security Constraint

Secure scope NEVER routes to DeepSeek. Attempting raises `RuntimeError`. This enforces the air-gap boundary at the code level.

## Ollama Interface

Ollama accessed via OpenAI-compatible API using `ChatOpenAI` with custom `base_url`. No Ollama-specific client library dependency.

## Instrumentation

`audited_invoke()` wrapper provides instrumented LLM calls: logs model, prompt length, token usage, latency, and scope for every invocation.
