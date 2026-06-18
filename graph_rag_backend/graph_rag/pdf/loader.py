"""
PDF Loader — graph_rag/pdf/loader.py
======================================
Reads PDF files page by page using pdfplumber.
Produces a list of (page_number, text) tuples with page metadata.

Each page's text is preserved exactly as pdfplumber extracts it — 
no collapsing of line breaks, no reordering of pages.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """A single page from a PDF file."""
    page_number: int   # 1-indexed
    text:        str   # extracted text, may be empty for image-only pages


@dataclass
class PDFLoadResult:
    """Result of loading a PDF file."""
    source_pdf:  str           # basename of the PDF file
    pages:       list[PDFPage]
    total_pages: int


class PDFLoader:
    """
    Reads PDF files page by page using pdfplumber.

    Usage:
        loader = PDFLoader()
        result = loader.load("docs/people/alice_chen/medical_report.pdf")
        for page in result.pages:
            print(f"Page {page.page_number}: {len(page.text)} chars")
    """

    def load(self, path: str | Path) -> PDFLoadResult:
        """
        Load a PDF file and return its pages.

        Args:
            path — path to the PDF file

        Returns:
            PDFLoadResult with ordered 1-indexed pages

        Raises:
            ValueError — if file not found, not a PDF, or pdfplumber parse error
        """
        try:
            import pdfplumber
        except ImportError:
            raise ValueError(
                "pdfplumber not installed. Run: pip install pdfplumber"
            )

        path = Path(path)
        if not path.exists():
            raise ValueError(f"PDF file not found: {path.name}")

        try:
            with pdfplumber.open(path) as pdf:
                pages: list[PDFPage] = []
                for i, page in enumerate(pdf.pages):
                    page_number = i + 1  # 1-indexed
                    text = page.extract_text() or ""
                    if not text.strip():
                        logger.warning(
                            "PDF page %d has no extractable text: %s",
                            page_number, path.name
                        )
                    pages.append(PDFPage(page_number=page_number, text=text))

                return PDFLoadResult(
                    source_pdf  = path.name,
                    pages       = pages,
                    total_pages = len(pages),
                )

        except Exception as e:
            # Re-raise as ValueError with filename + reason
            # (catches pdfplumber.PDFSyntaxError and other parse errors)
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse PDF '{path.name}': {e}") from e
