---
phase: 6
status: decided
tags: [decisions, security, gliner, privacy]
created: 2026-05-10
links: [security-modules]
---

# GLiNER Label Restriction

## Decision

Restrict GLiNER-PII labels to high-risk types only.

## Label Whitelist

| Category | Labels |
|----------|--------|
| Persons | person |
| Contact | phone, email |
| Financial | credit card, ssn |
| Medical | patient id |
| Network | url, ip |
| Government | id |

## Removed Labels

`medical condition`, `organization`, `location`, `date`, `address`, `hospital`

## Rationale

Biomedical text has high false positive rates for broad entity types:
- "Organization" matches research institutions, journals, funding bodies
- "Medical condition" matches every disease term in the corpus
- "Location" matches study sites, lab names

## Results

| Metric | Before | After |
|--------|--------|-------|
| False positive rate | 58% | 12% |
| Detection rate | 50% | 25% |

## Tradeoff

**Precision over recall.** Fewer PII entities detected, but far fewer false positives to manually review. Fine-tuning GLiNER on biomedical PHI is deferred to a future phase.
