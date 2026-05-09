"""PrivacyModel — abstract interface for AI-based PII detection (Phase 6).

This module defines the interface for adding a machine-learning privacy
detection layer on top of the regex-based BoundaryScrubber.  The interface
is designed for future drop-in models (GLiNER-PII-Edge, presidio-analyzer,
or a fine-tuned transformer).

Phase 5 provides a no-op implementation.  Phase 6 will ship a concrete
implementation using a lightweight open-source model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class PrivacyModel(ABC):
    """Abstract interface for AI-based privacy detection.

    Implementations detect context-dependent PII that regex patterns miss,
    such as named entities in biomedical text that could identify patients
    or proprietary research when combined with other metadata.
    """

    @abstractmethod
    def detect(self, text: str) -> List[Tuple[str, int, int, str]]:
        """Detect privacy-sensitive spans in *text*.

        Returns:
            List of tuples ``(entity_text, start_char, end_char, category)``
            where *category* is one of: ``"PERSON"``, ``"LOCATION"``,
            ``"ORGANIZATION"``, ``"DATE_TIME"``, ``"ID_NUMBER"``,
            ``"PROPRIETARY_TERM"``, or ``"OTHER"``.
        """
        ...

    @abstractmethod
    def redact(self, text: str) -> Tuple[str, int]:
        """Detect and redact privacy-sensitive spans.

        Returns:
            Tuple of ``(redacted_text, redaction_count)``.
        """
        ...


class NoOpPrivacyModel(PrivacyModel):
    """No-op implementation — returns text unchanged.

    Used when no AI privacy model is loaded (default in Phase 5).
    """

    def detect(self, text: str) -> List[Tuple[str, int, int, str]]:
        return []

    def redact(self, text: str) -> Tuple[str, int]:
        return text, 0
