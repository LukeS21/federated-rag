---
phase: 7
status: known-issue-workaround
tags: [vision, bugs, ollama, api]
created: 2026-05-10
links: [vision-descriptor-api, num-predict-workaround]
---

# `num_predict` Bug (Ollama Multimodal)

Known Ollama API issue: multimodal models return empty responses when `num_predict` is in the options dict.

## Symptom

When `num_predict` is included in the `options` field of a `POST /api/generate` request with a multimodal model, the response is:

```json
{
  "response": "",
  "done": true,
  "done_reason": "length"
}
```

The model immediately terminates with zero tokens generated.

## Affected Models

| Model | Affected? |
|-------|-----------|
| gemma4:e4b | Yes |
| qwen3-vl:4b | Yes |
| llava:7b | No (different architecture) |

All multimodal (vision-language) models appear affected. Text-only models are not.

## Investigation

Systematically tested combinations:

| `temperature` | `num_predict` | Result |
|---------------|---------------|--------|
| ✗ | ✗ | Works |
| ✓ | ✗ | Works |
| ✗ | ✓ | **Fails** |
| ✓ | ✓ | **Fails** |

Conclusion: `num_predict` is the sole trigger. `temperature` has no effect either way.

## Workaround

In `VisionDescriptor`, the `options` dict sent to Ollama contains **only** `temperature`:

```python
options = {"temperature": 0.2}
# num_predict is intentionally omitted
```

Output length is enforced via post-generation string truncation in Python (rough heuristic: ~4 characters per token):

```python
def _truncate(self, text: str, max_tokens: int) -> str:
    return text[:max_tokens * 4]
```

## Future

This may be fixed in a future Ollama release. Before removing the workaround, re-test with the exact same payload on the then-current Ollama version. Validate across all three multimodal models.

**Not a project bug** — this is an upstream Ollama API issue.
