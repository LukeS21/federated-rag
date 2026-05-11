---
phase: 7
status: fixed
tags: [vision, bugs, docling, captions]
created: 2026-05-10
links: [figure-extraction, phase-7-vision]
---

# Caption Resolution Bug

All figure captions showed raw classification strings instead of real figure labels.

## Symptom

Every extracted figure displayed its docling classifier metadata as the caption:

```
kind='classification' provenance='DocumentPictureClassifier-v2.5'
prediction=[{'classification': {'confidence': 0.91, 'class_name': 'bar_chart'}}]
```

Instead of the actual figure label:

```
Fig. 1. Characterization of smooth, rough, and rough hydrophilic
Ti samples.
```

## Root Cause

In `figure_extractor.py`, the caption resolution logic checked `picture.annotations` first (which contains the classifier's prediction data). Because `picture.annotations` is always populated when classification is enabled, the code **never reached** `picture.captions`, which holds the real figure label references.

```python
# BEFORE (broken)
if picture.annotations:                    # Always true with classification
    caption = repr(picture.annotations)     # Raw classification dump
elif picture.captions:
    caption = resolve_caption(picture.captions)  # Never reached
```

## Fix

Reversed the priority — check `picture.captions` first, skip `picture.annotations` entirely for caption purposes:

```python
# AFTER (fixed)
if picture.captions:
    caption = resolve_caption(picture.captions)  # Real labels
else:
    caption = ""
```

## Caption Resolution Mechanics

`picture.captions` is a list of `RefItem` objects. Each contains a `cref` field with a JSON Pointer-style reference into `DoclingDocument.texts`:

```python
ref = picture.captions[0]               # RefItem
# ref.cref == "#/texts/42"
idx = int(ref.cref.lstrip("#/texts/"))  # 42
caption = doc.texts[idx].text           # "Fig. 1. Characterization of..."
```

## Detection

Caught by the `test_real_captions_extracted` test, which asserts captions start with patterns like "Fig." and do not contain raw classification strings.

## Impact

After the fix, [[figure-filtering]] caption scores improved significantly (avg relevance went from 0.823 → 0.951), since the caption scorer can now detect keywords like "Fig.", "Figure", and other label patterns.
