---
phase: 7
status: complete
tags: [vision, api, ollama, gemma4]
created: 2026-05-10
links: [vision-ingest, num-predict-bug, gemma4-e4b, figure-extraction]
---

# Vision Descriptor API

Core API for describing scientific figures using local vision-language models via Ollama.

## Core Class: `VisionDescriptor`

Location: `src/vision/vision_descriptor.py`

Uses Ollama's **native REST API** (`POST /api/generate`), not LangChain. This gives full control over the request payload and avoids LangChain abstraction overhead.

## Key Methods

| Method | Description |
|--------|-------------|
| `pull_model()` | Pulls model if not cached locally |
| `unload_model()` | Frees GPU memory (keep-alive=0) |
| `describe(image, prompt)` | Describe a single PIL image |
| `describe_figures(figures, unload_first, reload_after)` | Batch describe extracted figures |

## Image Encoding

Images are encoded as base64 JPEG and sent in Ollama's `images` array:

```python
def _encode(self, image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

payload = {
    "model": self.model,        # gemma4:e4b by default
    "prompt": prompt,
    "images": [encoded_image],
    "stream": False,
    "options": {"temperature": 0.2}
}
```

## `num_predict` Workaround

Sending `num_predict` in the options dict causes multimodal models (gemma4, qwen3-vl) to return **empty responses** with `done_reason=length`. Only `temperature` is sent in options. Length enforcement is done via post-generation string truncation (~4 chars/token). See [[num-predict-bug]].

## Default Prompt

```
Describe this scientific figure in detail, including the
methodology, experimental groups, and key findings. Focus on
quantitative results, statistical comparisons, and biological
interpretation.
```

Customizable via the `prompt` parameter on `describe()` and `describe_figures()`.

## ASCII Scrubbing

Post-generation, responses are scrubbed of non-printable and non-ASCII characters that some models (especially qwen3-vl) emit.

## Caching

Descriptions are persisted to disk for offline review:

```python
VisionDescriptor.load_descriptions(path)   # Read cached JSON
VisionDescriptor.save_descriptions(path)    # Write current descriptions to JSON
```

Default path: `projects/default/figure_descriptions.json`

## Configuration

Model defaults to `VISION_MODEL` environment variable. If unset, uses `gemma4:e4b`.
