---
phase: 7
status: open-minor
created: 2026-05-10
tags:
  - gaps
  - vision
  - ingest
links:
  - "[[vision-ingest]]"
  - "[[figure-extraction]]"
---
Vision pipeline skips already-ingested PDFs. To force re-description, delete ChromaDB entry for that PDF. Not a bug — intentional (avoids redundant LLM calls). But means improved models won't retroactively update existing descriptions.
