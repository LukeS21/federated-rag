---
phase: all
status: reference
created: 2026-05-10
tags:
  - dev
  - ollama
  - setup
links:
  - "[[model-tiering]]"
  - "[[gemma4-e4b]]"
  - "[[qwen3-6-35b]]"
  - "[[quick-start]]"
---
Ollama setup instructions.

## Required Models

```bash
ollama pull gemma4:e4b
ollama pull qwen3.6:35b
```

## Optional Models

```bash
ollama pull llava:7b
ollama pull qwen3-vl:4b
```

## Memory Management

Set in `.env`:

| Variable | Value | Purpose |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | `60s` | Unload models from memory after 60s idle |
| `PER_THEME_MAX_WORKERS` | `2` | Limit parallel workers to control memory |

## Parallelism Server Settings

Set via environment or Ollama config:

| Setting | Value | Purpose |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `4` | Max concurrent requests |
| `OLLAMA_MAX_QUEUE` | `8` | Max queued requests |

## Model Sizes and Memory Usage

| Model | Size | RAM Usage (approx) |
|---|---|---|
| `gemma4:e4b` | ~3B params | ~4 GB |
| `qwen3.6:35b` | ~35B params | ~24 GB |
| `llava:7b` | ~7B params | ~6 GB |
| `qwen3-vl:4b` | ~4B params | ~3 GB |
