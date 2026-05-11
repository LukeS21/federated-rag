---
phase: [3, 4, 7]
status: reference
tags: [architecture, human-in-the-loop, interrupt]
created: 2026-05-10
links: [deep-mode-graph, survey-mode-graph, sectioned-survey-graph]
---

# Human-in-the-Loop

Interrupt/resume checkpoints across all execution modes.

## Deep Mode Checkpoints

`interrupt_before=["category_discovery", "human_gate"]`

| Checkpoint | Purpose |
|------------|---------|
| After CategoryDiscovery | Review/edit discovered categories before NER begins |
| HumanGate | Final review when anchoring < 0.85 after second pass |

## Survey Mode Checkpoint

`interrupt_before=["survey_scrub"]`

| Action | Description |
|--------|-------------|
| Approve | Accept synthesis as-is |
| Edit-with-feedback | Modify output with inline feedback |
| Discard | Reject and re-run |

## Sectioned Survey Checkpoints

`interrupt_at` each section review:

| Section | Review Focus |
|---------|-------------|
| Introduction | Scope, background comprehensiveness |
| Methods | Completeness, reproducibility |
| Results | Evidence fidelity, figure integration |
| Discussion | Interpretation validity, gap coverage |

## Checkpointer

`MemorySaver` checkpointer enables interrupt/resume across all modes. State serialized to memory between interrupts. Survives process restarts within same session.
