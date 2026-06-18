"""
Document Loader — Step 3
=========================
Reads .txt files from disk and converts them into Document objects.

TEACHING NOTES
--------------
This is the entry point for raw text documents into the pipeline.
After this step, no other component ever touches the raw files directly —
they all work with Document objects.

Key responsibility: pre-compute line_offsets so the ProvenanceTracker
can later map any line number to an exact character position in the text.

What this does NOT do:
- Does not classify the document type (that's DocumentClassifier, Step 4)
- Does not extract entities (that's the Extractor, Step 5/6)
- Does not handle UiPath JSON files (that's UiPathExtractor, Step 6)
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from .models import Document, DocType

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads raw .txt files from disk into Document objects.

    Usage:
        loader = DocumentLoader()

        # Load a single file
        doc = loader.load_file("docs/people/alice_chen/birth_certificate.txt")

        # Load all .txt files in a directory
        docs = loader.load_directory("docs/people/alice_chen")

        # Load multiple specific files
        docs = loader.load(["birth_cert.txt", "license.txt"])
    """

    def load_file(self, path: str | Path) -> Optional[Document]:
        """
        Load a single .txt file into a Document object.

        TEACHING: This is where the real work happens.
        We read the file, split it three ways (lines, paragraphs, offsets),
        and package everything into a Document.

        Returns None if the file can't be read (so callers can skip it).
        """
        path = Path(path)

        # ---- Guard: file must exist and be readable ----
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            return None

        if not path.is_file():
            logger.warning("Path is not a file, skipping: %s", path)
            return None

        # ---- Read the file ----
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Cannot read file %s: %s — skipping", path, e)
            return None

        # ---- Check if empty ----
        # A document is "empty" if it has no actual content after stripping
        # whitespace. We still create a Document (for provenance purposes)
        # but mark it empty=True so extractors know to skip it.
        is_empty = len(text.strip()) == 0

        if is_empty:
            logger.warning("Empty file (no text content): %s", path)

        # ---- Split into lines ----
        # We split on \n (newline character).
        # "line 1\nline 2\nline 3" → ["line 1", "line 2", "line 3"]
        #
        # We do NOT strip trailing newlines from the text before splitting,
        # because that would shift line numbers and break char offsets.
        lines = text.split("\n")

        # ---- Compute line_offsets ----
        # This is the key computation for provenance tracking.
        #
        # For each line, we record: at what character position does it start?
        #
        # Example:
        #   text          = "CERT OF BIRTH\nFull Name: Alice\nDOB: March 15"
        #   lines         = ["CERT OF BIRTH", "Full Name: Alice", "DOB: March 15"]
        #   line_offsets  = [0,               14,                 31            ]
        #                    ↑                 ↑                   ↑
        #                    "C" is at pos 0   "F" is at pos 14    "D" is at pos 31
        #
        # How do we get 14?
        #   len("CERT OF BIRTH") = 13
        #   + 1 for the \n character = 14
        #
        # Invariant (always true):
        #   text[line_offsets[i] : line_offsets[i] + len(lines[i])] == lines[i]
        #
        line_offsets: list[int] = []
        pos = 0
        for line in lines:
            line_offsets.append(pos)
            pos += len(line) + 1  # +1 for the \n that separated this line

        # ---- Split into paragraphs ----
        # Paragraphs are separated by blank lines (two consecutive newlines).
        # "para 1 line 1\npara 1 line 2\n\npara 2 line 1"
        # → ["para 1 line 1\npara 1 line 2", "para 2 line 1"]
        #
        # We filter out empty strings that result from multiple blank lines.
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # ---- Assign a unique ID ----
        # UUID v4 = random UUID. Guaranteed unique across all documents.
        doc_id = str(uuid.uuid4())

        # ---- Build and return the Document ----
        doc = Document(
            doc_id       = doc_id,
            filename     = path.name,          # just the filename, not the full path
            text         = text,
            lines        = lines,
            paragraphs   = paragraphs,
            line_offsets = line_offsets,
            doc_type     = DocType.GENERIC,     # DocumentClassifier will update this
            doc_date     = None,                # DocumentClassifier may extract this
            empty        = is_empty,
            metadata     = {
                "full_path":  str(path.resolve()),
                "file_size":  path.stat().st_size,
            },
        )

        logger.debug(
            "Loaded: %s | %d lines | %d paragraphs | empty=%s",
            path.name, len(lines), len(paragraphs), is_empty
        )

        return doc

    def load(self, paths: list[str | Path]) -> list[Document]:
        """
        Load multiple files. Skips files that can't be read.

        TEACHING: This is the batch version of load_file().
        It collects results and skips failures gracefully.
        The pipeline should never crash because one file is bad.

        Returns:
            List of Document objects (may be shorter than paths if some failed)

        Raises:
            ValueError — if the paths list is completely empty
        """
        if not paths:
            raise ValueError(
                "No file paths provided. Pass at least one .txt file path."
            )

        docs: list[Document] = []
        skipped = 0

        for path in paths:
            doc = self.load_file(path)
            if doc is not None:
                docs.append(doc)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning("Skipped %d file(s) due to errors.", skipped)

        if not docs:
            raise ValueError(
                f"No documents could be loaded from the provided paths. "
                f"All {skipped} file(s) failed or were not found."
            )

        logger.info("Loaded %d document(s), skipped %d.", len(docs), skipped)
        return docs

    def load_directory(self, directory: str | Path) -> list[Document]:
        """
        Load all .txt files in a directory (non-recursive — top level only).

        TEACHING: Path.glob("*.txt") finds all .txt files in a directory.
        We sort them so the order is predictable (alphabetical).
        sorted() on Path objects sorts by filename.

        Returns:
            List of Document objects for all .txt files found

        Raises:
            ValueError — if directory doesn't exist or has no .txt files
        """
        directory = Path(directory)

        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Find all .txt files (non-recursive, top level only)
        txt_files = sorted(directory.glob("*.txt"))

        if not txt_files:
            raise ValueError(
                f"No .txt files found in: {directory}. "
                f"Files present: {[f.name for f in directory.iterdir()]}"
            )

        logger.info(
            "Found %d .txt file(s) in %s", len(txt_files), directory
        )

        return self.load(txt_files)


# ---------------------------------------------------------------------------
# HELPER: verify line_offsets are correct
# This is used in tests to validate the computation.
# ---------------------------------------------------------------------------

def verify_line_offsets(doc: Document) -> bool:
    """
    Verify that every line_offset correctly points to the start of its line.

    This is the core invariant of the provenance system:
        doc.text[doc.line_offsets[i] : doc.line_offsets[i] + len(doc.lines[i])]
        == doc.lines[i]

    Returns True if all offsets are correct, False otherwise.
    Useful in tests and debugging.
    """
    for i, (offset, line) in enumerate(zip(doc.line_offsets, doc.lines)):
        extracted = doc.text[offset : offset + len(line)]
        if extracted != line:
            logger.error(
                "line_offset mismatch at line %d: "
                "expected %r, got %r (offset=%d)",
                i + 1, line, extracted, offset
            )
            return False
    return True
