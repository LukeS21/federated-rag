---
phase: all
status: reference
created: 2026-05-10
tags:
  - dev
  - quick-start
  - commands
links:
  - "[[dashboard]]"
  - "[[environment-variables]]"
  - "[[ollama-setup]]"
---
All run commands organized by category.

## Test Suite

| Command | Description |
|---|---|
| `python -m pytest tests/` | Full test suite |
| `python -m pytest tests/test_correctness.py` | Correctness benchmarks |
| `python -m pytest tests/test_security.py` | Security tests |
| `python -m pytest tests/test_vision.py` | Vision pipeline tests |
| `python -m pytest tests/test_synthesis.py` | Synthesis quality tests |
| `python -m pytest tests/test_anchoring.py` | Evidence anchoring tests |
| `python -m pytest tests/test_retrieval.py` | Retrieval quality tests |
| `python -m pytest tests/test_graph.py` | Graph construction tests |
| `python -m pytest tests/test_baseline.py` | Baseline comparison |
| `python -m pytest tests/test_sectioned.py` | Sectioned survey tests |
| `python -m pytest tests/test_concurrency.py` | Concurrency tests |
| `python -m pytest tests/test_ragas.py` | RAGAS evaluation |

## Benchmarks

| Command | Description |
|---|---|
| `python bench/tier_a.py` | Tier A benchmark runs |
| `python bench/correctness.py` | Correctness benchmark |
| `python bench/security.py` | Security benchmark |
| `python bench/vision.py` | Vision pipeline benchmark |
| `python bench/baseline.py` | Baseline comparison |
| `python bench/literature_discovery.py` | Literature discovery benchmark |

## Vision Pipeline

| Command | Description |
|---|---|
| `python vision/benchmark.py` | Run vision benchmarks |
| `python vision/describe.py` | Describe figures in a PDF |
| `python vision/quality_check.py` | Check figure description quality |

## Sectioned Survey

| Command | Description |
|---|---|
| `python run_sectioned.py --query "..."` | Run sectioned survey |
| `python run_sectioned.py --query "..." --theme biology` | Run with theme filter |

## Streamlit UI

| Command | Description |
|---|---|
| `streamlit run app.py` | Launch Streamlit dashboard |

## Cached Results

| Command | Description |
|---|---|
| `python view_cache.py` | View cached LLM results |
| `python view_cache.py --clear` | Clear LLM cache |

## Clean Start

| Command | Description |
|---|---|
| `python clean_start.py` | Reset all caches and state |

## Model Pulls

| Command | Description |
|---|---|
| `ollama pull gemma4:e4b` | Pull small model |
| `ollama pull qwen3.6:35b` | Pull large model |
| `ollama pull llava:7b` | Pull optional vision model |
| `ollama pull qwen3-vl:4b` | Pull optional lightweight vision model |
