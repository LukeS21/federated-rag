"""
JATS XML parser for Europe PMC full-text documents.

Parses JATS (Journal Article Tag Suite) XML into structured chunk dicts
compatible with the existing ingestion pipeline (same format as PDFParser output).

Extracts:
  - Title, abstract, authors, journal info
  - Body sections (recursively, handling nested sections)
  - Figure captions with labels and image URLs
  - References

Usage::

    from src.ingestion.pmc_xml_parser import PMCXMLParser

    parser = PMCXMLParser()
    chunks = parser.parse(xml_text, pmcid="PMC13059311", doi="10.1016/...")
    # chunks is a list of dicts with "text" and "metadata" keys
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# JATS namespace prefix patterns (handles namespaced and non-namespaced XML)
_JATS_NS = "{http://www.w3.org/1998/xlink}"
_XLINK_HREF = f"{_JATS_NS}href"


def _strip_ns(tag: str) -> str:
    """Strip XML namespace from tag but return only the local name."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _elem_text(elem: Optional[ET.Element]) -> str:
    """Extract all text from an element, stripping extra whitespace."""
    if elem is None:
        return ""
    return " ".join(elem.itertext()).strip()


def _iter_children_text(elem: ET.Element, join_str: str = " ") -> str:
    """Extract text from all child elements, joined."""
    parts = []
    for child in elem:
        text = _elem_text(child)
        if text:
            parts.append(text)
    return join_str.join(parts)


class PMCXMLParser:
    """Parse JATS XML from Europe PMC into chunk dicts."""

    MIN_CHUNK_WORDS = 20  # skip chunks shorter than this

    def parse(
        self,
        xml_text: str,
        pmcid: str = "",
        doi: str = "",
    ) -> List[Dict[str, Any]]:
        """Parse JATS XML into a list of chunk dicts.

        Returns:
            List of dicts, each with:
              - text: str          — section/figure/caption content
              - metadata: dict     — chunk_type, section_title, figure_label,
                                     pmcid, doi, source, cite_key
        """
        try:
            # Strip namespace prefixes for easier access
            xml_text = self._strip_namespaces(xml_text)
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning("XML parse error for %s: %s", pmcid, e)
            return []

        article = root.find(".//article") or root
        chunks: List[Dict[str, Any]] = []

        # — Front matter —
        title = self._extract_title(article)
        authors = self._extract_authors(article)
        abstract = self._extract_abstract(article)
        cite_key = self._make_cite_key(title, authors, doi, pmcid)

        # Title chunk
        if title:
            chunks.append(self._chunk(
                text=f"Title: {title}",
                chunk_type="title",
                section_title="Article Title",
                cite_key=cite_key,
                pmcid=pmcid,
                doi=doi,
            ))

        # Abstract chunk
        if abstract:
            chunks.append(self._chunk(
                text=f"Abstract: {abstract}",
                chunk_type="abstract",
                section_title="Abstract",
                cite_key=cite_key,
                pmcid=pmcid,
                doi=doi,
            ))

        # — Body sections —
        body = article.find(".//body")
        if body is not None:
            section_chunks = self._extract_sections(body, cite_key, pmcid, doi)
            chunks.extend(section_chunks)

        # — Figures —
        figure_chunks = self._extract_figures(article, cite_key, pmcid, doi)
        chunks.extend(figure_chunks)

        # — References —
        ref_chunks = self._extract_references(article, cite_key, pmcid, doi)
        chunks.extend(ref_chunks)

        # Add chunk_index to each chunk's metadata (required for unique IDs)
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = i

        return chunks

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _strip_namespaces(xml_text: str) -> str:
        """Remove namespace prefixes from XML for easier xpath matching.

        Handles both namespace declarations (xmlns:prefix) and prefixed
        attributes (prefix:attr) by converting them to unprefixed form.
        """
        # 1. Remove all xmlns:* attribute declarations
        xml_text = re.sub(r'\s+xmlns:\w+="[^"]*"', '', xml_text)
        xml_text = re.sub(r"\s+xmlns:\w+='[^']*'", '', xml_text)
        # Also remove default namespace
        xml_text = re.sub(r'\s+xmlns="[^"]*"', '', xml_text)
        xml_text = re.sub(r"\s+xmlns='[^']*'", '', xml_text)

        # 2. Convert prefixed attributes: prefix:attr → attr
        xml_text = re.sub(r'(\s)(\w+):(\w+)(=)', r'\1\3\4', xml_text)

        # 3. Convert prefixed tags: <prefix:tag → <tag, </prefix:tag → </tag
        #    Also handles self-closing tags: <prefix:tag/> → <tag/>
        xml_text = re.sub(r'<(\w+):(\w+)([\s/>])', r'<\2\3', xml_text)
        xml_text = re.sub(r'</\w+:(\w+)\s*>', r'</\1>', xml_text)

        return xml_text

    @staticmethod
    def _chunk(
        text: str,
        chunk_type: str,
        section_title: str = "",
        cite_key: str = "",
        pmcid: str = "",
        doi: str = "",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "chunk_type": chunk_type,
            "section_title": section_title,
            "cite_key": cite_key,
            "pmcid": pmcid,
            "doi": doi,
            "source": f"europe_pmc_xml_{pmcid}" if pmcid else "europe_pmc_xml",
        }
        if extra_meta:
            meta.update(extra_meta)
        return {"text": text.strip(), "metadata": meta}

    @staticmethod
    def _make_cite_key(title: str, authors: str, doi: str, pmcid: str) -> str:
        """Generate a citation key (e.g. '@smith2023')."""
        import hashlib

        if doi:
            return f"@ref_{hashlib.sha256(doi.lower().encode()).hexdigest()[:8]}"
        if pmcid:
            return f"@ref_{pmcid.lower()}"
        if title:
            slug = re.sub(r'[^a-z0-9]+', '_', title.lower().strip())[:30]
            return f"@{slug}"
        return "@ref_unknown"

    # --------------------------------------------------------- extractors

    @staticmethod
    def _extract_title(article: ET.Element) -> str:
        title = article.find(".//article-title")
        if title is not None:
            return _elem_text(title)
        # Fallback: try without namespace
        for t in article.iter():
            if _strip_ns(t.tag) == "article-title":
                return _elem_text(t)
        return ""

    @staticmethod
    def _extract_authors(article: ET.Element) -> str:
        authors = []
        for contrib in article.iter():
            if _strip_ns(contrib.tag) == "contrib":
                contrib_type = contrib.get("contrib-type", "")
                if contrib_type == "author" or not contrib_type:
                    surname = ""
                    given = ""
                    for name_el in contrib.iter():
                        name_tag = _strip_ns(name_el.tag)
                        if name_tag == "surname":
                            surname = _elem_text(name_el)
                        elif name_tag == "given-names":
                            given = _elem_text(name_el)
                    if surname:
                        authors.append(f"{surname} {given}".strip())
        return ", ".join(authors)

    @staticmethod
    def _extract_abstract(article: ET.Element) -> str:
        abstract = article.find(".//abstract")
        if abstract is not None:
            return _elem_text(abstract)
        for el in article.iter():
            if _strip_ns(el.tag) == "abstract":
                return _elem_text(el)
        return ""

    def _extract_sections(
        self,
        body: ET.Element,
        cite_key: str,
        pmcid: str,
        doi: str,
    ) -> List[Dict[str, Any]]:
        """Recursively extract sections from body element."""
        chunks: List[Dict[str, Any]] = []
        for child in body:
            tag = _strip_ns(child.tag)
            if tag == "sec":
                section_chunks = self._extract_sec(child, cite_key, pmcid, doi)
                chunks.extend(section_chunks)
            elif tag in ("p",):
                text = _elem_text(child)
                if len(text.split()) >= self.MIN_CHUNK_WORDS:
                    chunks.append(self._chunk(
                        text=text,
                        chunk_type="text",
                        section_title="Body",
                        cite_key=cite_key,
                        pmcid=pmcid,
                        doi=doi,
                    ))
        return chunks

    def _extract_sec(
        self,
        sec: ET.Element,
        cite_key: str,
        pmcid: str,
        doi: str,
        parent_title: str = "",
    ) -> List[Dict[str, Any]]:
        """Recursively extract a section."""
        chunks: List[Dict[str, Any]] = []
        sec_title = ""

        # Get section title
        title_el = sec.find("./title")
        if title_el is not None:
            sec_title = _elem_text(title_el)

        full_title = f"{parent_title} > {sec_title}" if parent_title else sec_title

        # Collect text from paragraphs and nested sections
        paragraphs: List[str] = []
        for child in sec:
            tag = _strip_ns(child.tag)
            if tag == "p":
                text = _elem_text(child)
                if len(text.split()) >= self.MIN_CHUNK_WORDS:
                    paragraphs.append(text)
            elif tag == "sec":
                nested = self._extract_sec(child, cite_key, pmcid, doi, full_title)
                chunks.extend(nested)

        # Combine paragraphs into one chunk for this section
        if paragraphs:
            combined = "\n\n".join(paragraphs)
            if len(combined.split()) >= self.MIN_CHUNK_WORDS:
                chunks.append(self._chunk(
                    text=combined,
                    chunk_type="text",
                    section_title=full_title or "Unnamed Section",
                    cite_key=cite_key,
                    pmcid=pmcid,
                    doi=doi,
                ))

        return chunks

    def _extract_figures(
        self,
        article: ET.Element,
        cite_key: str,
        pmcid: str,
        doi: str,
    ) -> List[Dict[str, Any]]:
        """Extract figure captions and image URLs."""
        chunks: List[Dict[str, Any]] = []

        for el in article.iter():
            tag = _strip_ns(el.tag)
            if tag != "fig":
                continue

            label = ""
            caption_text = ""
            image_url = ""

            label_el = el.find("./label")
            if label_el is not None:
                label = _elem_text(label_el)

            caption_el = el.find("./caption")
            if caption_el is not None:
                caption_text = _elem_text(caption_el)

            # Try to find image URL
            graphic = el.find("./graphic")
            if graphic is not None:
                image_url = graphic.get(_XLINK_HREF, "") or graphic.get("xlink:href", "")

            if caption_text:
                full = f"[{label}] {caption_text}".strip() if label else caption_text
                chunks.append(self._chunk(
                    text=full,
                    chunk_type="figure",
                    section_title="Figures",
                    cite_key=cite_key,
                    pmcid=pmcid,
                    doi=doi,
                    extra_meta={
                        "figure_label": label,
                        "figure_image_url": image_url,
                    },
                ))

        return chunks

    def _extract_references(
        self,
        article: ET.Element,
        cite_key: str,
        pmcid: str,
        doi: str,
    ) -> List[Dict[str, Any]]:
        """Extract reference list."""
        refs: List[str] = []
        ref_list = article.find(".//ref-list")
        if ref_list is None:
            return []

        for ref in ref_list.findall(".//ref"):
            citation = _elem_text(ref)
            if citation and len(citation) > 10:
                refs.append(citation)

        if refs:
            return [self._chunk(
                text="References:\n" + "\n".join(f"[{i+1}] {r}" for i, r in enumerate(refs)),
                chunk_type="reference",
                section_title="References",
                cite_key=cite_key,
                pmcid=pmcid,
                doi=doi,
            )]

        return []
