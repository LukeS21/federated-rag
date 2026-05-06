from pathlib import Path
import pytest
from src.ingestion.pdf_parser import PDFParser

TEST_PDF = Path("data/test.pdf")

@pytest.mark.skipif(not TEST_PDF.exists(), reason="No test PDF in data/")
def test_parser_returns_chunks():
    parser = PDFParser()
    chunks = parser.parse(TEST_PDF)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c
        assert "metadata" in c
        # Verify no Unicode left
        assert all(ord(char) < 128 for char in c["text"]), f"Non-ASCII in chunk: {c['text'][:100]}"