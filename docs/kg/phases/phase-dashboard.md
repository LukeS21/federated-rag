# Phase Status Dashboard

Auto-generated from YAML frontmatter. Requires [[Dataview]] plugin.

## Phase Completion

```dataview
TABLE status, file.cday as "Updated"
FROM "phases"
SORT phase ASC
```

## Active Work (Phase 8)

```dataview
TABLE status
FROM "phases"
WHERE phase = 8
```

## All Notes by Phase

```dataview
TABLE file.folder as "Category"
FROM ""
WHERE phase
SORT phase ASC
```
