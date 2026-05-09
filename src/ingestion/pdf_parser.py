"""
PDF ingestion using Docling, with immediate Unicode-to-ASCII scrubbing.
Now tags reference‑list chunks so the retriever can filter them later.
Also provides content fingerprinting for deduplication across differently‑named PDFs.
"""
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

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


def compute_content_hash(chunks: List[Dict]) -> str:
    """Compute a content fingerprint from body text (references excluded).

    Used for deduplication: two PDFs with the same body text produce
    the same hash regardless of filename.  Hashing the first 8000 chars
    of body text is sufficient to uniquely identify a paper.

    Returns a 16‑character hex digest.
    """
    body = "\n".join(
        ch["text"] for ch in chunks
        if (ch.get("metadata", {}) or {}).get("chunk_type") != "reference"
    )
    return hashlib.sha256(body[:8000].encode("utf-8")).hexdigest()[:16]


def extract_title_from_chunks(chunks: List[Dict]) -> str:
    """Extract a tentative title from body text, skipping journal headers.

    Many biomedical PDFs lead with HHS Public Access notices, PMC headers,
    or journal masthead text before the actual paper title.  This function
    skips those and finds the first substantive block that looks like a title.
    """
    skip_patterns = [
        r"^HHS Public Access",
        r"^Author manuscript",
        r"available in PMC",
        r"Published in final edited form",
        r"Contents lists available at",
        r"^Acta Biomater\.",
        r"^NIH Public Access",
        r"^\d+\.\s+Introduction$",
        r"journal homepage",
        r"^www\.",
    ]
    for ch in chunks:
        if (ch.get("metadata", {}) or {}).get("chunk_type") == "reference":
            continue
        text = ch.get("text", "").strip()
        if len(text) < 30:
            continue
        # Skip known header patterns
        if any(re.match(p, text) for p in skip_patterns):
            continue
        # Found substantive text — return first 200 chars as title
        return text[:200]
    # Fallback: any non‑reference text
    for ch in chunks:
        if (ch.get("metadata", {}) or {}).get("chunk_type") != "reference":
            text = ch.get("text", "").strip()
            if len(text) > 10:
                return text[:200]
    return ""


# ── Content hash registry (JSON file, quick duplicate check) ────────────────
def _hash_registry_path() -> Path:
    return Path("projects/default/content_hashes.json")


def load_content_hashes() -> Dict[str, str]:
    """Load existing content hashes: hash → paper_filename."""
    p = _hash_registry_path()
    if p.exists():
        import json
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_content_hash(hash_val: str, filename: str) -> None:
    """Register a content hash for a paper."""
    import json
    hashes = load_content_hashes()
    hashes[hash_val] = filename
    _hash_registry_path().parent.mkdir(parents=True, exist_ok=True)
    _hash_registry_path().write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")


def check_content_duplicate(hash_val: str) -> Optional[str]:
    """Return the existing filename if this content hash is already known."""
    hashes = load_content_hashes()
    return hashes.get(hash_val)