---
phase: 7
status: decided
tags: [decisions, vision, bugs, ollama]
created: 2026-05-10
links: [num-predict-bug, vision-descriptor-api]
---

# num_predict Workaround

## Decision

Send `temperature` only in VisionDescriptor options. Enforce `max_tokens` via post-generation truncation.

## The Bug

Ollama's `num_predict` parameter is not respected by vision models in the current version. Setting it in the API call has no effect — the model generates until its internal stop condition.

## Workaround

1. Send only `temperature` in the Ollama API options
2. Truncate output to `max_tokens` after generation completes
3. This is functionally equivalent but slightly wasteful (extra tokens generated then discarded)

## Status

Not a project bug — this is an **upstream Ollama issue**. Re-test after Ollama updates and remove the workaround if the bug is fixed.
