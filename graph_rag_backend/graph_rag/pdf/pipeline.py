"""
PDF Ingest Pipeline — graph_rag/pdf/pipeline.py
================================================
Orchestrates the full PDF ingestion flow:
    PDFLoader → SemanticChunker → PDFEntityExtractor
    → KnowledgeGraphBuilder → EntityResolver

Each PDF chunk becomes a Document node in the NetworkX knowledge graph
(doc_type = MEDICAL_REPORT), and extracted PERSON entities are linked
via same_as edges to any matching identity-document entities already
in the graph.

Usage:
    pipeline = PDFIngestPipeline(
        graph_builder    = graph_builder,
        embedding_engine = embedding_engine,
        llm_provider     = llm_provider,
        documents        = documents_dict,
    )
    result = pipeline.ingest("docs/people/alice_chen/medical_report.pdf")
    print(result.entities_extracted, result.same_as_edges_added)
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..core.embeddings import EmbeddingEngine
from ..core.graph_builder import KnowledgeGraphBuilder
from ..core.models import DocType, Document, EdgeType
from ..llm.provider import LLMProvider
from .chunker import PDFChunk, SemanticChunker
from .extractor import PDFEntityExtractor
from .loader import PDFLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RESULT DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class PDFIngestResult:
    """
    Summary of a completed PDF ingestion.

    Fields:
        filename            — basename of the ingested PDF
        pages_processed     — total number of pages in the PDF
        chunks_created      — number of semantic chunks produced
        entities_extracted  — total entities extracted across all chunks
        same_as_edges_added — number of new same_as edges added by EntityResolver
    """
    filename:            str
    pages_processed:     int
    chunks_created:      int
    entities_extracted:  int
    same_as_edges_added: int


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

class PDFIngestPipeline:
    """
    Orchestrates the PDF ingestion pipeline end-to-end.

    Stages:
        1. PDFLoader.load()           — extract raw pages from PDF
        2. SemanticChunker.chunk()    — split pages into semantic chunks
        3. For each chunk:
           a. Create a Document node  — builds Document exactly like DocumentLoader
           b. graph_builder.add_document()
           c. documents[doc_id] = doc
           d. PDFEntityExtractor.extract() — LLM entity extraction
           e. embedding_engine.embed_entities() — set entity embeddings
           f. graph_builder.add_entity() for each entity
        4. Count same_as edges before EntityResolver.resolve()
        5. EntityResolver.resolve()   — link PDF entities to identity docs
        6. Add same_as edges, count delta
        7. Return PDFIngestResult

    Usage:
        pipeline = PDFIngestPipeline(graph_builder, embedding_engine, llm, docs)
        result = pipeline.ingest("path/to/report.pdf")
    """

    def __init__(
        self,
        graph_builder:    KnowledgeGraphBuilder,
        embedding_engine: EmbeddingEngine,
        llm_provider:     LLMProvider,
        documents:        dict[str, Document],
        threshold:        float = 0.75,
    ):
        """
        Args:
            graph_builder    — shared KnowledgeGraphBuilder (in-memory graph)
            embedding_engine — shared EmbeddingEngine (reused, cached)
            llm_provider     — LLMProvider for entity extraction
            documents        — shared dict mapping doc_id → Document (modified in place)
            threshold        — SemanticChunker similarity threshold (default 0.75)
        """
        self._graph_builder    = graph_builder
        self._embedding_engine = embedding_engine
        self._documents        = documents

        # Wire up the sub-components
        self._loader    = PDFLoader()
        self._chunker   = SemanticChunker(embedding_engine, threshold)
        self._extractor = PDFEntityExtractor(llm_provider)

    def ingest(self, path: str | Path) -> PDFIngestResult:
        """
        Run the full PDF ingestion pipeline for a single file.

        Args:
            path — path to the PDF file to ingest

        Returns:
            PDFIngestResult with counts of pages, chunks, entities, same_as edges

        Raises:
            ValueError — if PDFLoader.load() raises (bad file, missing, corrupt)
                         Re-raised as-is; callers (e.g. app.py) catch it for HTTP 400.
        """
        path = Path(path)

        # ----------------------------------------------------------------
        # Stage 1: Load PDF pages
        # ValueError from PDFLoader is intentionally NOT caught here —
        # it propagates up to the caller (e.g. the /ingest/pdf endpoint).
        # ----------------------------------------------------------------
        load_result = self._loader.load(path)
        logger.info(
            "PDFIngestPipeline: loaded '%s' (%d pages)",
            load_result.source_pdf, load_result.total_pages,
        )

        # ----------------------------------------------------------------
        # Stage 2: Semantic chunking
        # ----------------------------------------------------------------
        chunks = self._chunker.chunk(load_result)
        logger.info(
            "PDFIngestPipeline: '%s' → %d chunk(s)",
            load_result.source_pdf, len(chunks),
        )

        # ----------------------------------------------------------------
        # Stage 3: Process each chunk
        # ----------------------------------------------------------------
        total_entities = 0

        for chunk in chunks:
            # 3a. Build a Document object exactly like DocumentLoader does
            doc = self._build_document_from_chunk(chunk)

            # 3b. Add document node to graph
            self._graph_builder.add_document(doc)

            # 3c. Store in shared documents dict
            self._documents[doc.doc_id] = doc

            # 3d. Extract entities from chunk
            entities = self._extractor.extract(chunk, doc.doc_id)

            # 3e. Embed all extracted entities
            if entities:
                self._embedding_engine.embed_entities(entities)

            # 3f. Add each entity to the graph
            for entity in entities:
                self._graph_builder.add_entity(entity)

            total_entities += len(entities)

            logger.debug(
                "Chunk %d of '%s': %d entity/entities extracted",
                chunk.chunk_index, chunk.source_pdf, len(entities),
            )

        # ----------------------------------------------------------------
        # Stage 4: Count same_as edges BEFORE entity resolution
        # ----------------------------------------------------------------
        graph = self._graph_builder.get_graph()
        same_as_before = sum(
            1
            for _, _, data in graph.edges(data=True)
            if data.get("edge_type") == EdgeType.SAME_AS.value
        )

        # ----------------------------------------------------------------
        # Stage 5: Run EntityResolver
        # Following the exact pattern from app.py's _run_entity_resolution()
        # ----------------------------------------------------------------
        same_as_added = 0

        entity_count = len(self._graph_builder.get_entity_nodes())
        if entity_count >= 2:
            try:
                from ..core.resolver import EntityResolver

                resolver = EntityResolver(
                    self._extractor._llm,
                    self._embedding_engine,
                )
                pairs = resolver.resolve(graph)

                # ----------------------------------------------------------------
                # Stage 6: Add same_as edges
                # ----------------------------------------------------------------
                for pair in pairs:
                    self._graph_builder.add_same_as_edge(
                        pair.entity_id_a, pair.entity_id_b, pair
                    )

                # Count same_as edges AFTER resolution to get the delta
                same_as_after = sum(
                    1
                    for _, _, data in graph.edges(data=True)
                    if data.get("edge_type") == EdgeType.SAME_AS.value
                )
                same_as_added = same_as_after - same_as_before

                logger.info(
                    "PDFIngestPipeline: EntityResolver added %d same_as edge(s) "
                    "(before=%d after=%d)",
                    same_as_added, same_as_before, same_as_after,
                )

            except Exception as e:
                logger.warning(
                    "PDFIngestPipeline: entity resolution failed for '%s': %s",
                    load_result.source_pdf, e,
                )

        # ----------------------------------------------------------------
        # Stage 7: Return result
        # ----------------------------------------------------------------
        result = PDFIngestResult(
            filename            = load_result.source_pdf,
            pages_processed     = load_result.total_pages,
            chunks_created      = len(chunks),
            entities_extracted  = total_entities,
            same_as_edges_added = same_as_added,
        )

        logger.info(
            "PDFIngestPipeline: ingested '%s' — "
            "pages=%d chunks=%d entities=%d same_as_added=%d",
            result.filename,
            result.pages_processed,
            result.chunks_created,
            result.entities_extracted,
            result.same_as_edges_added,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_document_from_chunk(self, chunk: PDFChunk) -> Document:
        """
        Build a Document object from a PDFChunk, exactly mirroring how
        DocumentLoader.load_file() constructs documents.

        - doc_id       = str(uuid4())
        - filename     = f"{chunk.source_pdf}_chunk_{chunk.chunk_index}"
        - text         = chunk.text
        - lines        = chunk.text.split('\\n')
        - paragraphs   = [p.strip() for p in chunk.text.split('\\n\\n') if p.strip()]
        - line_offsets = computed cumulatively (same algorithm as DocumentLoader)
        - doc_type     = DocType.MEDICAL_REPORT
        - metadata     = source_pdf, chunk_index, start_page, end_page
        """
        text = chunk.text

        # Split into lines (same as DocumentLoader)
        lines = text.split("\n")

        # Compute line_offsets cumulatively (same as DocumentLoader)
        # Invariant: text[line_offsets[i] : line_offsets[i] + len(lines[i])] == lines[i]
        line_offsets: list[int] = []
        pos = 0
        for line in lines:
            line_offsets.append(pos)
            pos += len(line) + 1  # +1 for the '\n' separator

        # Split into paragraphs (same as DocumentLoader)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        return Document(
            doc_id       = str(uuid.uuid4()),
            filename     = f"{chunk.source_pdf}_chunk_{chunk.chunk_index}",
            text         = text,
            lines        = lines,
            paragraphs   = paragraphs,
            line_offsets = line_offsets,
            doc_type     = DocType.MEDICAL_REPORT,
            doc_date     = None,
            empty        = len(text.strip()) == 0,
            metadata     = {
                "source_pdf":   chunk.source_pdf,
                "chunk_index":  chunk.chunk_index,
                "start_page":   chunk.start_page,
                "end_page":     chunk.end_page,
            },
        )
