"""
PDF ingestion using Docling, with immediate Unicode-to-ASCII scrubbing.
Now tags reference‑list chunks so the retriever can filter them later.
"""
import re
from pathlib import Path
from typing import List, Dict

from docling.document_converter import DocumentConverter
from src.unicode_map import scrub_unicode


class PDFParser:
    """Parses biomedical PDFs into clean, searchable text chunks and tables."""

    def __init__(self):
        self.converter = DocumentConverter()

    def _is_reference_section(self, text: str) -> bool:
        """
        Heuristic to detect if a text chunk is part of the reference list.
        Handles both plain '[1] ...' and markdown list items like '- [1] ...'.
        """
        # Matches optional whitespace, optional dash, optional whitespace, [number]
        return bool(re.match(r'^\s*(-\s*)?\[\d+\]', text.strip()))

    def parse(self, pdf_path: Path) -> List[Dict]:
        """
        Parse a PDF and return a list of chunk dicts.
        Each dict: {"text": "...", "metadata": {...}}
        """
        result = self.converter.convert(str(pdf_path))
        doc = result.document

        chunks = []

        # 1. Extract main body text
        body_text = doc.export_to_text()
        if body_text:
            paragraphs = [p.strip() for p in body_text.split("\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                clean_text = scrub_unicode(para)

                # Tag references vs body text
                if self._is_reference_section(para):
                    chunk_type = "reference"
                else:
                    chunk_type = "text"

                chunks.append({
                    "text": clean_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "chunk_type": chunk_type,
                        "chunk_index": i,
                    }
                })

        # 2. Extract tables
        tables = getattr(doc, "tables", [])
        for i, table in enumerate(tables):
            try:
                table_md = table.export_to_markdown()
                clean_table = scrub_unicode(table_md)
                chunks.append({
                    "text": clean_table,
                    "metadata": {
                        "source": pdf_path.name,
                        "chunk_type": "table",
                        "table_index": i,
                    }
                })
            except Exception:
                # Fallback to plain text if markdown export fails
                table_text = getattr(table, "text", "")
                if table_text:
                    chunks.append({
                        "text": scrub_unicode(table_text),
                        "metadata": {
                            "source": pdf_path.name,
                            "chunk_type": "table",
                            "table_index": i,
                        }
                    })

        return chunks