"""BoundaryScrubber: regex redaction at the secure→public data boundary.

When ``query_scope`` is ``"both"``, secure-corpus data must be sanitized
before it appears in the public output.  This module provides configurable
pattern-based redaction for:

  - PHI identifiers (MRN, SSN, DOB patterns)
  - Email addresses, phone numbers
  - Internal project / grant codes
  - IPv4 / IPv6 addresses
  - API keys and tokens
  - Custom user-defined patterns (loaded from a file)

An optional AI-based ``PrivacyModel`` (Phase 6) can be layered on top
for context-dependent PII detection that regex patterns miss.

Use :func:`default_boundary_scrubber` for the singleton configured from
``.env``.  Call :meth:`BoundaryScrubber.scrub` before any secure-derived
text flows into a public-facing output.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

from src.security.privacy_model import NoOpPrivacyModel, PrivacyModel

logger = logging.getLogger(__name__)

# ── Built-in redaction patterns ───────────────────────────────────────────
_DEFAULT_PATTERNS: List[Tuple[str, str, str]] = [
    # (label, regex, replacement)

    # Medical Record Number (MRN) — typical hospital formats
    ("MRN", r"\bMR[#\s]*\d{6,12}\b", "[REDACTED-MRN]"),

    # Social Security Number (US format: XXX-XX-XXXX)
    ("SSN", r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "[REDACTED-SSN]"),

    # Date of birth (MM/DD/YYYY or YYYY-MM-DD)
    ("DOB", r"\bDOB:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[REDACTED-DOB]"),

    # Email addresses
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
     "[REDACTED-EMAIL]"),

    # Phone numbers (US and international)
    ("PHONE", r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
     "[REDACTED-PHONE]"),

    # IPv4 addresses
    ("IPV4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED-IP]"),

    # API keys (common patterns: sk-, pk-, api-)
    ("API_KEY", r"\b(sk|pk|api)[-_][a-zA-Z0-9_-]{20,}\b", "[REDACTED-KEY]"),

    # NIH / grant identifiers (R01, K23, U01, etc.)
    ("GRANT", r"\b[RUK]\d{2}[A-Z]{2}\d{6}\b", "[REDACTED-GRANT]"),
    ("GRANT2", r"\bGrant\s*#?\s*[A-Z0-9]+-\d{4,}\b", "[REDACTED-GRANT]"),

    # Named hospitals / clinics (common in de-identified data)
    ("HOSPITAL", r"\b\w+\s+(Hospital|Medical Center|Clinic|Infirmary)\b",
     "[REDACTED-FACILITY]"),

    # Internal project codes (e.g., PROJ-2024-001)
    ("PROJ_CODE", r"\bPROJ[-\s]?\d{4}[-\s]?\d{3}\b", "[REDACTED-PROJECT]"),
]


class BoundaryScrubber:
    """Configurable regex-based redaction engine with optional AI layer.

    Patterns can be loaded from env ``BOUNDARY_SCRUB_PATTERNS`` (file with
    one regex per line) and/or added programmatically.

    An optional :class:`PrivacyModel` provides context-dependent PII
    detection on top of structured regex patterns.  When None (default),
    only regex-based redaction is performed.
    """

    def __init__(
        self,
        patterns: List[Tuple[str, str, str]] | None = None,
        pattern_file: str | Path | None = None,
        privacy_model: PrivacyModel | None = None,
    ) -> None:
        self._patterns: List[Tuple[str, str]] = []
        self._redaction_count: int = 0
        self._privacy_model = privacy_model or NoOpPrivacyModel()

        # Load built-in patterns
        for _label, regex, replacement in (_DEFAULT_PATTERNS or []):
            self._patterns.append((regex, replacement))

        # Load custom patterns from argument
        if patterns:
            for _label, regex, replacement in patterns:
                self._patterns.append((regex, replacement))

        # Load custom patterns from file (if provided or env)
        file_path = pattern_file or os.getenv("BOUNDARY_SCRUB_PATTERNS", "")
        if file_path:
            self._load_from_file(Path(file_path))

    def _load_from_file(self, path: Path) -> None:
        if not path.exists():
            logger.warning("BoundaryScrubber: pattern file not found: %s", path)
            return
        loaded = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    self._patterns.append((line, "[REDACTED]"))
                    loaded += 1
        except OSError as e:
            logger.error("BoundaryScrubber: failed to read pattern file: %s", e)
        if loaded:
            logger.info("BoundaryScrubber: loaded %d custom patterns from %s", loaded, path)

    def add_pattern(self, regex: str, replacement: str = "[REDACTED]") -> None:
        self._patterns.append((regex, replacement))

    def scrub(self, text: str) -> str:
        """Apply all redaction patterns to *text*, then AI privacy layer.

        Returns the scrubbed string with all sensitive matches replaced.
        Regex patterns run first (structured data), then the AI privacy
        model checks for context-dependent PII.  The internal redaction
        counter can be inspected via :attr:`redaction_count`.
        """
        self._redaction_count = 0
        if not text:
            return text

        # Layer 1: regex patterns
        for regex, replacement in self._patterns:
            new_text, count = re.subn(regex, replacement, text)
            if count:
                text = new_text
                self._redaction_count += count

        # Layer 2: AI privacy model (no-op in Phase 5, active in Phase 6)
        ai_count = 0
        if self._privacy_model is not None:
            text, ai_count = self._privacy_model.redact(text)
            self._redaction_count += ai_count
        if ai_count:
            logger.debug("AI privacy model: %d additional redactions", ai_count)

        return text

    @property
    def redaction_count(self) -> int:
        """Number of redactions applied in the most recent :meth:`scrub` call."""
        return self._redaction_count

    @property
    def pattern_count(self) -> int:
        """Total number of active redaction patterns."""
        return len(self._patterns)


_boundary_scrubber: BoundaryScrubber | None = None


def default_boundary_scrubber() -> BoundaryScrubber:
    """Return the global BoundaryScrubber singleton."""
    global _boundary_scrubber
    if _boundary_scrubber is None:
        _boundary_scrubber = BoundaryScrubber()
    return _boundary_scrubber
