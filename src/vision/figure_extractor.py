"""
Figure extraction from biomedical PDFs via Docling's picture extraction pipeline.

Uses ``generate_picture_images=True`` to extract embedded figures, photos, diagrams,
and chart images from PDFs.  Saves figures as PNG files and returns structured
metadata (page number, bounding box, caption, PIL image, file path).

Delegates text extraction to the existing ``PDFParser`` for body text.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import PictureItem

from src.unicode_map import scrub_unicode

logger = logging.getLogger(__name__)

# ── Figure output directory ──────────────────────────────────────────────────
FIGURES_DIR = Path("projects/default/figures")


class FigureExtractor:
    """Extract figures and images from biomedical PDFs using Docling.

    Each extracted figure includes:
      - PIL Image object (for immediate use or encoding)
      - file_path: where the PNG was saved to disk
      - page_no: 1-based page number
      - bbox: bounding box coordinates (l, t, r, b) in PDF space
      - caption: figure caption text (if any)
      - width, height: image dimensions in pixels

    Usage::

        extractor = FigureExtractor()
        figures = extractor.extract(Path("data/paper.pdf"))
        for fig in figures:
            print(fig["file_path"], fig["page_no"], fig["caption"])
    """

    def __init__(self, output_dir: Path | str = FIGURES_DIR, images_scale: float = 1.0):
        """Initialize the figure extractor.

        Args:
            output_dir: Directory to save extracted figure images.
            images_scale: Scale factor for extracted images (1.0 = original, 2.0 = 2x).
        """
        self.output_dir = Path(output_dir)
        self.images_scale = float(images_scale)

    def _build_converter(self) -> DocumentConverter:
        """Create a Docling converter configured for picture extraction + classification."""
        opts = PdfPipelineOptions()
        opts.generate_picture_images = True
        opts.do_picture_classification = True
        opts.images_scale = self.images_scale
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=opts,
                    backend=DoclingParseDocumentBackend,
                    pipeline_cls=StandardPdfPipeline,
                ),
            }
        )

    def extract(self, pdf_path: Path) -> List[Dict]:
        """Extract all figures from a PDF.

        Returns a list of dicts, each with keys:
          ``file_path``, ``page_no``, ``bbox``, ``caption``, ``width``,
          ``height``, ``image`` (PIL), ``pdf_source``, ``figure_index``,
          ``classification`` (list of {class_name, confidence} dicts).
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        converter = self._build_converter()
        logger.info("Extracting figures from %s (scale=%.1f)...", pdf_path.name, self.images_scale)
        result = converter.convert(str(pdf_path))
        doc = result.document

        pdf_name = pdf_path.stem
        figures_dir = self.output_dir / pdf_name
        figures_dir.mkdir(parents=True, exist_ok=True)

        extracted: List[Dict] = []
        picture_items: List[PictureItem] = list(doc.pictures)

        for idx, picture in enumerate(picture_items):
            pil_img = picture.get_image(doc)
            if pil_img is None:
                continue

            prov = picture.prov[0] if picture.prov else None
            page_no = prov.page_no + 1 if prov else 0  # 0-based → 1-based
            bbox = {
                "l": round(prov.bbox.l, 1),
                "t": round(prov.bbox.t, 1),
                "r": round(prov.bbox.r, 1),
                "b": round(prov.bbox.b, 1),
            } if prov and prov.bbox else None

            caption = ""
            if picture.captions:
                caption_parts = []
                for caption_ref in picture.captions:
                    ref_str = getattr(caption_ref, "cref", "") or str(caption_ref)
                    if ref_str.startswith("#/texts/"):
                        try:
                            idx = int(ref_str.split("/")[-1])
                            if 0 <= idx < len(doc.texts):
                                text_item = doc.texts[idx]
                                if hasattr(text_item, "text") and text_item.text:
                                    caption_parts.append(scrub_unicode(text_item.text.strip()))
                        except (ValueError, IndexError, TypeError):
                            pass

                caption = " ".join(caption_parts)

            if not caption:
                caption = ""

            # ── Classification data from Docling's DocumentFigureClassifier ──
            classification = []
            meta = getattr(picture, "meta", None)
            if meta is not None:
                cls_data = getattr(meta, "classification", None)
                if cls_data is not None:
                    predictions = getattr(cls_data, "predictions", [])
                    for pred in predictions:
                        classification.append({
                            "class_name": getattr(pred, "class_name", "unknown"),
                            "confidence": round(getattr(pred, "confidence", 0.0), 6),
                        })

            file_name = f"{pdf_name}_fig_{idx:03d}.png"
            file_path = figures_dir / file_name
            pil_img.save(str(file_path), format="PNG")

            extracted.append({
                "file_path": str(file_path),
                "page_no": page_no,
                "bbox": bbox,
                "caption": caption.strip(),
                "width": pil_img.width,
                "height": pil_img.height,
                "image": pil_img,
                "pdf_source": pdf_path.name,
                "figure_index": idx,
                "classification": classification,
            })

        logger.info("Extracted %d figures from %s", len(extracted), pdf_path.name)
        return extracted

    def encode_base64(self, pil_image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """Encode a PIL Image to a base64 data URI string.

        Suitable for passing images to vision models via HTTP APIs.
        """
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format=format, quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = f"image/{format.lower()}"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def compute_figure_hash(pil_image: Image.Image) -> str:
        """Compute a perceptual-ish hash for figure deduplication.

        Resizes to 64x64, converts to grayscale, and hashes the raw pixel bytes.
        Not a proper perceptual hash but sufficient for detecting exact duplicate
        images across PDFs.
        """
        small = pil_image.resize((64, 64), Image.LANCZOS).convert("L")
        return hashlib.sha256(small.tobytes()).hexdigest()[:16]

    @classmethod
    def extract_from_pdfs(cls, pdf_paths: List[Path], **kwargs) -> Dict[str, List[Dict]]:
        """Extract figures from multiple PDFs. Returns ``{pdf_name: [figures]}``."""
        extractor = cls(**kwargs)
        results: Dict[str, List[Dict]] = {}
        for pdf_path in pdf_paths:
            try:
                figures = extractor.extract(pdf_path)
                results[pdf_path.name] = figures
            except Exception as e:
                logger.error("Failed to extract figures from %s: %s", pdf_path.name, e)
                results[pdf_path.name] = []
        return results
