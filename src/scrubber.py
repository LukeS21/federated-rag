"""Final output scrubber utilities.

This project standardizes outputs to plain ASCII for deterministic downstream
processing and consistent rendering across terminals/clients.
"""

from __future__ import annotations

from src.unicode_map import scrub_unicode


def final_scrub(text: str) -> str:
    """Apply the project-wide final normalization to model outputs."""

    return scrub_unicode((text or "").strip())

