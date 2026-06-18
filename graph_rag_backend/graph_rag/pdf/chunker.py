"""
Semantic Chunker — graph_rag/pdf/chunker.py
=============================================
Splits PDF text into semantically coherent chunks using embedding-based
cosine similarity. Splits where topic changes (adjacent paragraph similarity
drops below a configurable threshold).

This is superior to fixed token windows because clinical notes have natural
sections: Chief Complaint, History, Medications, Assessment, Plan — each
a distinct semantic unit.
"""

import logging
from dataclasses import dataclass

from ..core.embeddings import EmbeddingEngine
from .loader import PDFLoadResult

logger = logging.getLogger(__name__)


@dataclass
class PDFChunk:
    """A semantically coherent chunk of text from a PDF."""
    source_pdf:   str    # basename of original PDF
    chunk_index:  int    # 0-based index within this PDF's chunks
    start_page:   int    # 1-based first page contributing to this chunk
    end_page:     int    # 1-based last page contributing to this chunk
    text:         str    # concatenated paragraph text (joined with '\n\n')


class SemanticChunker:
    """
    Splits PDF pages into semantically coherent chunks.
    
    Algorithm:
    1. Split each page's text into paragraphs (split on '\n\n')
    2. Embed each non-empty paragraph using EmbeddingEngine
    3. Compare adjacent paragraph embeddings with cosine similarity
    4. Insert chunk boundary where similarity < threshold (topic shift)
    5. Collect paragraphs within boundaries into PDFChunk objects
    
    Usage:
        chunker = SemanticChunker(embedding_engine, threshold=0.75)
        chunks = chunker.chunk(load_result)
    """

    def __init__(self, embedding_engine: EmbeddingEngine, threshold: float = 0.75):
        """
        Args:
            embedding_engine — existing EmbeddingEngine (reused, no new model)
            threshold        — cosine similarity below this = topic shift = chunk boundary
                               Default 0.75. 0.0 is valid (splits on every paragraph).
                               Value is stored EXACTLY as supplied — never substituted.
        """
        self._embedding_engine = embedding_engine
        self._threshold = threshold  # stored exactly, including 0.0

    def chunk(self, load_result: PDFLoadResult) -> list[PDFChunk]:
        """
        Split PDF pages into semantic chunks.
        
        Args:
            load_result — from PDFLoader.load()
        
        Returns:
            list of PDFChunk (at least 1, even if no topic shifts detected)
        """
        # Collect (page_number, paragraph_text) pairs
        # Discard empty paragraphs
        para_page_pairs: list[tuple[int, str]] = []
        for page in load_result.pages:
            for para in page.text.split('\n\n'):
                stripped = para.strip()
                if stripped:
                    para_page_pairs.append((page.page_number, stripped))

        if not para_page_pairs:
            # No text at all — return single empty-ish chunk
            return [PDFChunk(
                source_pdf  = load_result.source_pdf,
                chunk_index = 0,
                start_page  = 1,
                end_page    = load_result.total_pages or 1,
                text        = "",
            )]

        # Embed each paragraph (EmbeddingEngine caches by text hash)
        embeddings = [
            self._embedding_engine.embed(para)
            for _, para in para_page_pairs
        ]

        # Find chunk boundaries:
        # boundary before paragraph i+1 if similarity(i, i+1) < threshold
        boundaries: set[int] = {0}  # always start a chunk at index 0
        for i in range(len(embeddings) - 1):
            sim = self._embedding_engine.cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                boundaries.add(i + 1)

        # Build chunks from boundaries
        sorted_bounds = sorted(boundaries)
        chunks: list[PDFChunk] = []

        for b_idx, start in enumerate(sorted_bounds):
            end = sorted_bounds[b_idx + 1] if b_idx + 1 < len(sorted_bounds) else len(para_page_pairs)
            chunk_paras = [para for _, para in para_page_pairs[start:end]]
            chunk_pages = [pg for pg, _ in para_page_pairs[start:end]]

            chunks.append(PDFChunk(
                source_pdf  = load_result.source_pdf,
                chunk_index = len(chunks),
                start_page  = min(chunk_pages) if chunk_pages else 1,
                end_page    = max(chunk_pages) if chunk_pages else 1,
                text        = '\n\n'.join(chunk_paras),
            ))

        logger.info(
            "SemanticChunker: %d paragraphs → %d chunks (threshold=%.2f) for %s",
            len(para_page_pairs), len(chunks), self._threshold, load_result.source_pdf
        )

        return chunks
