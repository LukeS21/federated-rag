"""LangChain callback handler for real-time token streaming with degradation detection.

Monitors the LLM output stream for signs of model degradation (KV-cache
corruption, Metal backend fragmentation) and signals the caller so the
batch can be aborted and retried with a fresh GPU state.
"""

import logging
import zlib
from io import TextIOBase
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

JUNK_LINE_THRESHOLD = 20
WORD_REPEAT_THRESHOLD = 10
HYPHEN_REPEAT_THRESHOLD = 10
COMPRESSION_RATIO_THRESHOLD = 8.0
DEGRADATION_CHECK_INTERVAL_CHARS = 100


class ModelDegradedException(Exception):
    """Raised after generation completes when degradation was detected.

    The *text* attribute carries whatever output was captured before the
    degradation was noticed (may be partial or complete).
    """

    def __init__(self, reason: str, text: str = "") -> None:
        super().__init__(reason)
        self.text = text


class TokenStreamHandler(BaseCallbackHandler):
    """Prints tokens as they arrive and monitors for degradation in real time.

    Degradation signals detected:

    * **Compression ratio** — tail text compresses far beyond normal
      (≥8:1 ratio, normal text is ~1.5‑3:1).  This is a *universal*
      degradation detector — any form of repetition (word, line, hyphen,
      or novel pattern) inflates the compression ratio.
    * **Word‑level repetition** — ≥10 consecutive identical space‑delimited
      words (e.g. ``Energy: Energy: Energy: …``).
    * **Hyphen‑level repetition** — ≥10 consecutive identical sub‑tokens
      within a hyphenated token (e.g. ``e-coli-coli-coli…``).
    * **Junk‑line streaks** — ≥20 consecutive non‑blank lines that lack
      the expected ``:`` format separator — the model has lost the
      line‑tagged output format entirely.

    When degradation is detected the handler sets ``self.degraded`` and
    records the reason.  The caller (``_call_llm``) checks this flag after
    the LLM invocation returns and raises :class:`ModelDegradedException`
    so the batch can be retried with a clean GPU.
    """

    def __init__(
        self,
        junk_line_threshold: int = JUNK_LINE_THRESHOLD,
        word_repeat_threshold: int = WORD_REPEAT_THRESHOLD,
        hyphen_repeat_threshold: int = HYPHEN_REPEAT_THRESHOLD,
        compression_ratio_threshold: float = COMPRESSION_RATIO_THRESHOLD,
        output_file: Optional[TextIOBase] = None,
    ) -> None:
        self.current_text = ""
        self.degraded = False
        self.degraded_reason = ""
        self._junk_line_threshold = junk_line_threshold
        self._word_repeat_threshold = word_repeat_threshold
        self._hyphen_repeat_threshold = hyphen_repeat_threshold
        self._compression_ratio_threshold = compression_ratio_threshold
        self._output_file = output_file
        self._chars_since_check = 0

    # ── LangChain callbacks ────────────────────────────────────────────

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self._output_file:
            self._output_file.write(token)
            self._output_file.flush()
        else:
            print(token, end="", flush=True)
        self.current_text += token
        self._chars_since_check += len(token)

        if "\n" in token:
            self._check_junk_lines()

        if self._chars_since_check >= DEGRADATION_CHECK_INTERVAL_CHARS:
            self._chars_since_check = 0
            self._check_degradation()

    def on_llm_end(self, response, **kwargs) -> None:
        if self._output_file:
            self._output_file.write("\n")
            self._output_file.flush()
        else:
            print()

    # ── Degradation detection ──────────────────────────────────────────

    def _mark_degraded(self, reason: str) -> None:
        """Record degradation — does NOT raise so LangChain can finish normally."""
        self.degraded = True
        self.degraded_reason = reason
        logger.warning("TokenStreamHandler: model degradation — %s", reason)

    def _check_junk_lines(self) -> None:
        """Scan lines from the end — count consecutive lines without ':'.

        A line without a ':' is junk because the extraction format requires
        ``KEY: VALUE`` on every non‑blank line.
        """
        lines = self.current_text.split("\n")
        junk = 0
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue  # blank line — separator between entity groups
            if ":" not in stripped:
                junk += 1
            else:
                break  # found a formatted line — reset perspective
        if junk >= self._junk_line_threshold:
            self._mark_degraded(
                f"{junk} consecutive junk lines (no ':' format separator)"
            )

    def _check_degradation(self) -> None:
        """Periodic check on the tail of accumulated text for repetition."""
        if len(self.current_text) < 200:
            return
        tail = self.current_text[-2000:]
        words = tail.split()

        # ── Word‑level repetition ──────────────────────────────────
        if len(words) >= self._word_repeat_threshold:
            recent = words[-self._word_repeat_threshold:]
            if len(set(w.lower().rstrip(":,;.") for w in recent)) == 1:
                self._mark_degraded(
                    f"{self._word_repeat_threshold}+ consecutive identical "
                    f"words ({recent[0]!r})"
                )
                return

        # ── Hyphen‑level repetition ────────────────────────────────
        for w in words:
            if "-" not in w or len(w) <= 20:
                continue
            parts = [p.strip().lower() for p in w.split("-") if p.strip()]
            if len(parts) < self._hyphen_repeat_threshold:
                continue
            run = 1
            for i in range(1, len(parts)):
                if parts[i] == parts[i - 1]:
                    run += 1
                    if run >= self._hyphen_repeat_threshold:
                        self._mark_degraded(
                            f"{run}+ consecutive identical hyphen‑subtokens "
                            f"({parts[i]!r})"
                        )
                        return
                else:
                    run = 1

        # ── Compression ratio (universal repetition detector) ──────
        if not self.degraded and len(tail) >= 400:
            try:
                compressed = zlib.compress(tail.encode("utf-8"))
                ratio = len(tail) / max(len(compressed), 1)
                if ratio >= self._compression_ratio_threshold:
                    self._mark_degraded(
                        f"Compression ratio {ratio:.1f}:1 indicates "
                        f"repetitive output (normal text ~1.5‑3:1)"
                    )
            except Exception:
                pass
