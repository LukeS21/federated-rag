"""
Vision model integration with Ollama model rotation.

Handles loading a lightweight multimodal model (LLaVA, Qwen-VL, Granite-Vision,
etc.) via Ollama, generating text descriptions of figures, then swapping back
to the text model to release memory.  Peak memory ~26 GB on M3 Max (text model
23 GB or vision model ~3‑5 GB — never both simultaneously).

Model lifecycle:
  1. Unload text model:  set keep_alive=0 on the text model
  2. Pull vision model:  ollama pull <model> (if not already pulled)
  3. Generate per figure:  POST /api/generate with base64 image
  4. Unload vision model:  keep_alive=0
  5. Reload text model:  next text LLM call reloads automatically

Uses Ollama's native REST API (not LangChain) for multimodal generation because
LangChain's ChatOpenAI adapter does not support image inputs.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "gemma4:e4b"


class VisionDescriptor:
    """Describe figures using a multimodal Ollama model.

    Usage::

        vd = VisionDescriptor(model="llava:7b")
        desc = vd.describe(pil_image, prompt="Describe this scientific figure.")
        print(desc)

        # Or use the high-level batch method:
        descriptions = vd.describe_figures(
            figures,
            unload_first="gemma4:e4b",
            reload_after=True,
        )
    """

    def __init__(
        self,
        model: str | None = None,
        ollama_host: str | None = None,
        timeout: int = 120,
    ):
        self.model = model or os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL)
        self.ollama_host = (
            (ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
            .rstrip("/")
        )
        self.timeout = int(timeout)

    def _api_url(self, endpoint: str) -> str:
        return f"{self.ollama_host}/api/{endpoint.lstrip('/')}"

    def pull_model(self) -> bool:
        """Ensure the vision model is pulled. Returns True if successful."""
        try:
            logger.info("Pulling vision model %s...", self.model)
            resp = requests.post(
                self._api_url("pull"),
                json={"name": self.model, "stream": False},
                timeout=self.timeout * 5,  # pulling can take a while
            )
            resp.raise_for_status()
            logger.info("Vision model %s pulled successfully.", self.model)
            return True
        except Exception as e:
            logger.error("Failed to pull vision model %s: %s", self.model, e)
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from Ollama memory by setting keep_alive=0."""
        try:
            logger.debug("Unloading model %s...", model_name)
            resp = requests.post(
                self._api_url("generate"),
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.debug("Unload request sent for %s (%d)", model_name, resp.status_code)
            return resp.status_code == 200
        except Exception as e:
            logger.warning("Failed to unload model %s: %s", model_name, e)
            return False

    def describe(
        self,
        image: Image.Image,
        prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 300,
    ) -> str:
        """Generate a text description of a single figure image.

        Args:
            image: PIL Image of the figure.
            prompt: System/user prompt.  If None, uses a default biomedical prompt.
            temperature: Generation temperature (0 = deterministic).
            max_tokens: Maximum output tokens for the description.

        Returns:
            Generated description text (ASCII-scrubbed).
        """
        if prompt is None:
            prompt = (
                "Describe this scientific figure in detail. Include what type "
                "of chart or image it is, what variables are shown, key trends, "
                "and any notable data points. Be concise but thorough. "
                "Output plain text only, no markdown."
            )

        b64_image = self._encode_image(image)

        start = time.monotonic()
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
            }
            # temperature works with vision models; num_predict does not
            # (known Ollama bug: num_predict causes empty responses on multimodal models)
            if temperature is not None:
                payload["options"] = {"temperature": temperature}

            resp = requests.post(
                self._api_url("generate"),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            description = data.get("response", "")

            # Truncate to approximate max_tokens (rough: ~4 chars/token)
            if max_tokens and len(description) > max_tokens * 4:
                description = description[: max_tokens * 4]

            elapsed = time.monotonic() - start
            logger.info(
                "Figure described in %.1fs (%d chars, model=%s)",
                elapsed, len(description), self.model,
            )
            return self._scrub_text(description)
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error("Vision model call failed after %.1fs: %s", elapsed, e)
            return ""

    def describe_figures(
        self,
        figures: List[Dict],
        unload_first: str | None = None,
        reload_after: bool = True,
        fallback_to_caption: bool = True,
    ) -> List[Dict]:
        """Generate descriptions for multiple figures with model rotation.

        Args:
            figures: List of figure dicts from FigureExtractor (each has
                     ``image``, ``caption``, ``file_path``, etc.).
            unload_first: Text model name to unload before loading the vision
                          model (e.g., ``"qwen3.6:35b"``).
            reload_after: If True, unload the vision model after all figures
                          are described (so text model can reload on next call).
            fallback_to_caption: If the vision model returns an empty
                                 description, fall back to the figure's caption.

        Returns:
            List of figure dicts with a new ``description`` key added.
        """
        # ── Unload text model ──
        if unload_first:
            logger.info("Unloading text model %s before vision model...", unload_first)
            self.unload_model(unload_first)

        # ── Pull vision model ──
        self.pull_model()

        # ── Describe each figure ──
        n = len(figures)
        for i, figure in enumerate(figures):
            image = figure.get("image")
            if image is None:
                # Try to load from file_path
                file_path = figure.get("file_path", "")
                if file_path and Path(file_path).exists():
                    image = Image.open(file_path)
                else:
                    logger.warning("No image for figure %d — skipping.", i)
                    figure["description"] = figure.get("caption", "")
                    figure["described_by"] = "skipped"
                    continue

            caption = figure.get("caption", "")
            logger.info("Describing figure %d/%d (page %d)...", i + 1, n, figure.get("page_no", 0))
            description = self.describe(image)

            if not description and fallback_to_caption and caption:
                description = caption
                logger.info("Falling back to caption for figure %d", i)

            figure["description"] = description
            figure["described_by"] = self.model

        # ── Unload vision model ──
        if reload_after:
            logger.info("Unloading vision model %s...", self.model)
            self.unload_model(self.model)

        return figures

    @staticmethod
    def _encode_image(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """Encode a PIL Image to a base64 string for Ollama's API."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format=format, quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _scrub_text(text: str) -> str:
        """Remove non-ASCII characters from the generated description."""
        return text.encode("ascii", errors="replace").decode("ascii").replace("?", "")

    # ── Static helper for offline / cached results ──

    @staticmethod
    def load_descriptions(cache_path: Path | str) -> Dict[str, str]:
        """Load previously cached figure descriptions from a JSON file.

        Returns a dict of ``{file_path: description}``.
        """
        cache_path = Path(cache_path)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def save_descriptions(cache_path: Path | str, descriptions: Dict[str, str]) -> None:
        """Save figure descriptions to a JSON cache file."""
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(descriptions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
