# Gaps Dashboard

Auto-generated from YAML frontmatter. Install [[Dataview]] plugin to see live queries.

## Open Gaps

```dataview
TABLE phase, status, tags, file.cday as "Created"
FROM "gaps"
WHERE status != "fixed" AND status != "closed"
SORT phase ASC
```

## Gaps by Status

```dataview
TABLE length(rows) as "Count"
FROM "gaps"
GROUP BY status
SORT rows.phase ASC
```

## Fixed Gaps

```dataview
TABLE phase, file.cday as "Fixed"
FROM "gaps"
WHERE status = "fixed" OR status = "closed"
SORT file.cday DESC
```

## Phase 8 Dependencies

```dataview
TABLE status, file.cday as "Created"
FROM "gaps"
WHERE phase = 8 OR contains(phase, 8)
SORT status ASC
```
