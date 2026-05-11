---
phase: 7
status: decided
tags: [decisions, vision, gemma4, architecture]
created: 2026-05-10
links: [model-rotation, vision-model-reuse, gemma4-e4b]
---

# No Model Rotation

## Decision

Eliminate model rotation entirely. Check existing model capabilities before pulling new models.

## Original Plan

The original architecture required a load/unload cycle:
1. Unload current model
2. Load vision-specific model
3. Process figures
4. Unload vision model
5. Reload primary model

## Discovery

`gemma4:e4b` was already loaded and supports multimodal (text + vision). No separate vision model needed.

## Savings

~15-30 seconds saved per query by avoiding unnecessary model rotation.

## Precedent

This decision sets a precedent: **always check existing model capabilities before pulling new models.** Many modern models support multimodal natively.
