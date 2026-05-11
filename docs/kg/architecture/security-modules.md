---
phase: 5
status: reference
tags: [architecture, security, air-gap]
created: 2026-05-10
links: [llm-provider, system-overview]
---

# Security Modules

Security hardening for air-gapped biomedical data processing.

## BoundaryScrubber

Regex-based redaction of sensitive patterns:
- SSN (XXX-XX-XXXX)
- Email addresses
- Phone numbers
- Medical Record Numbers (MRN)
- API keys and tokens
- Grant IDs
- IP addresses
- Date of birth (DOB)

## AuditLogger

JSON-structured logging of:
- Boundary crossings (public → secure scope transitions)
- Scope routing decisions
- LLM call parameters (model, scope, prompt length)
- Security events (scrub detections, access violations)

## GLiNER-PII

Privacy detection model (570M params, Apache 2.0 license):
- Labels restricted to high-risk PII types only
- False positive rate reduced from 58% → 12% after label restriction
- Runs as pre-filter before regex scrubbing

## Docker Air-Gap Architecture

Three services with network isolation:

| Service | Network | Description |
|---------|---------|-------------|
| `orchestrator` | public network | Query routing, results assembly |
| `public-corpus` | public network | Public biomedical literature |
| `secure-corpus` | `internal:true` | Sensitive/protected data |

Secure corpus network has `internal:true` — containers on this network cannot reach the internet.

## Security Fuzzer

`phase6_security_fuzzer.py`: 9 tests, 100% regex detection rate. Validates BoundaryScrubber against adversarial PII injection patterns.
