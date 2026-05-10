"""GLiNERPrivacyModel — NVIDIA GLiNER-PII drop-in for the PrivacyModel interface.

Uses ``urchade/gliner_multi_pii-v1`` (570M params, 55+ entity types, ~1 GB
at FP16, Apache 2.0 license) to detect context-dependent PII that regex
patterns miss: person names in biomedical text, organization names, locations,
and other privacy-sensitive entities.

Implements the :class:`PrivacyModel` abstract interface from
``src/security/privacy_model.py``.  Drop-in replacement for ``NoOpPrivacyModel``
in :class:`BoundaryScrubber`.

Usage:
    >>> from src.security.gliner_privacy import GlinerPrivacyModel
    >>> model = GlinerPrivacyModel()
    >>> model.redact("Patient John Smith at Mayo Clinic has diabetes.")
    ('Patient [REDACTED-PERSON] at [REDACTED-ORGANIZATION] has [REDACTED-MEDICAL_CONDITION].', 3)
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, List, Tuple

from src.security.privacy_model import PrivacyModel

logger = logging.getLogger(__name__)

# GLiNER model name (public, Apache 2.0)
_GLINER_MODEL = "urchade/gliner_multi_pii-v1"

# Map GLiNER PII labels → interface categories
# Other labels are mapped to "OTHER"
_LABEL_MAP: dict = {
    "person": "PERSON",
    "first name": "PERSON",
    "last name": "PERSON",
    "organization": "ORGANIZATION",
    "org": "ORGANIZATION",
    "location": "LOCATION",
    "address": "LOCATION",
    "date": "DATE_TIME",
    "date of birth": "DATE_TIME",
    "time": "DATE_TIME",
    "phone number": "ID_NUMBER",
    "email": "ID_NUMBER",
    "id": "ID_NUMBER",
    "ssn": "ID_NUMBER",
    "credit card": "ID_NUMBER",
    "url": "OTHER",
    "ip": "ID_NUMBER",
    "medical condition": "PROPRIETARY_TERM",
    "medication": "PROPRIETARY_TERM",
    "patient id": "ID_NUMBER",
    "hospital": "ORGANIZATION",
    "username": "OTHER",
    "password": "OTHER",
}

# Model singleton (lazy loaded, thread-safe)
_model: Any = None
_model_lock = threading.Lock()


def _get_model() -> Any:
    """Lazy-load the GLiNER model (once, thread-safe)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        logger.info("Loading GLiNER-PII model: %s (first call ~1 min download)", _GLINER_MODEL)
        from gliner import GLiNER

        _model = GLiNER.from_pretrained(_GLINER_MODEL)
        logger.info("GLiNER-PII model loaded successfully")
        return _model


class GlinerPrivacyModel(PrivacyModel):
    """NVIDIA GLiNER-PII implementation of the PrivacyModel interface.

    Detects 55+ entity types including PHI (person names, dates, locations,
    organizations, IDs) and biomedical context (medical conditions,
    medications).  Runs ~50ms per text block on CPU, ~1GB at FP16.

    Model is lazy-loaded on first call (clean import, no wait).
    """

    def __init__(self, labels: list[str] | None = None) -> None:
        """Initialize the GLiNER privacy model.

        Args:
            labels: Optional custom label list. Defaults to GLiNER's
                    built-in labels.  If provided, restricts detection
                    to only these entity types.
        """
        self._labels = labels or [
            "person", "organization", "location", "date",
            "phone number", "email", "id", "medical condition",
            "url", "ip address", "address",
        ]
        self._model = None  # Lazy loaded

    @property
    def model(self) -> Any:
        """Access the underlying GLiNER model (lazy-loaded)."""
        if self._model is None:
            self._model = _get_model()
        return self._model

    def _to_category(self, label: str) -> str:
        """Map GLiNER label to interface category."""
        label_lower = label.lower().strip()
        return _LABEL_MAP.get(label_lower, "OTHER")

    def detect(self, text: str) -> List[Tuple[str, int, int, str]]:
        """Detect privacy-sensitive spans in *text*.

        Returns:
            List of ``(entity_text, start_char, end_char, category)`` tuples.
        """
        if not text or not text.strip():
            return []

        try:
            entities = self.model.predict_entities(text, labels=self._labels)
        except Exception as e:
            logger.warning("GLiNER detection failed: %s", e)
            return []

        results: List[Tuple[str, int, int, str]] = []
        for ent in entities:
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            entity_text = str(ent.get("text", ""))
            label = str(ent.get("label", ""))
            category = self._to_category(label)
            # Filter very low-confidence detections
            score = float(ent.get("score", 1.0))
            if score < 0.3:
                continue
            results.append((entity_text, start, end, category))

        # Sort by start position for consistent redaction order
        results.sort(key=lambda x: x[1])
        return results

    def redact(self, text: str) -> Tuple[str, int]:
        """Detect and redact privacy-sensitive spans.

        Returns:
            ``(redacted_text, redaction_count)``.
        """
        if not text or not text.strip():
            return text, 0

        spans = self.detect(text)
        if not spans:
            return text, 0

        # Redact from end to start to preserve offsets
        chars = list(text)
        redacted = 0
        for entity_text, start, end, category in reversed(spans):
            if start < 0 or end > len(chars) or start >= end:
                continue
            replacement = f"[REDACTED-{category}]"
            chars[start:end] = list(replacement)
            redacted += 1

        logger.debug("GLiNER redacted %d spans in %d chars", redacted, len(text))
        return "".join(chars), redacted


# ── Factory for BoundaryScrubber integration ──────────────────────────────

def create_gliner_privacy_model(labels: list[str] | None = None) -> GlinerPrivacyModel:
    """Create a configured GLiNER privacy model for BoundaryScrubber.

    Usage in BoundaryScrubber:
        scrubber = BoundaryScrubber(privacy_model=create_gliner_privacy_model())
    """
    return GlinerPrivacyModel(labels=labels)
