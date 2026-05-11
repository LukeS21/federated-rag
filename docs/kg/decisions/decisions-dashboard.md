# Decisions Dashboard

Auto-generated from YAML frontmatter. Requires [[Dataview]] plugin.

## All Decisions

```dataview
TABLE phase, status, file.cday as "Decided"
FROM "decisions"
WHERE status = "decided" OR status = "active"
SORT phase ASC
```

## Decisions by Phase

```dataview
TABLE length(rows) as "Count"
FROM "decisions"
GROUP BY phase
SORT key ASC
```

## Most Recent Decisions

```dataview
TABLE phase, status
FROM "decisions"
SORT file.cday DESC
LIMIT 10
```
