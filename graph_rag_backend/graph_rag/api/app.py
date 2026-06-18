"""
FastAPI REST API for the Graph RAG pipeline.
Exposes ingestion, querying, graph stats, visualization, and demo endpoints.
All state is in-memory; the graph resets when the server restarts.
"""

import logging
import os

# Prevent HuggingFace network calls — model already cached locally
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from pathlib import Path

# Load .env file on startup so all os.getenv() calls work correctly
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    # dotenv not installed — load .env manually
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..core.embeddings import EmbeddingEngine
from ..core.graph_builder import KnowledgeGraphBuilder
from ..core.models import Document
from ..extraction.classifier import DocumentClassifier
from ..extraction.langchain_extractor import LangChainExtractor
from ..extraction.uipath_extractor import UiPathExtractor
from ..llm.provider import create_llm_provider
from ..query.engine import QueryEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SNAPSHOT LOADING — load pre-built graph on startup
# ---------------------------------------------------------------------------
# If graph_snapshot.pkl exists, load it immediately so the app starts
# with all 48 people already ingested — no LLM calls needed at startup.
# The snapshot is built once by: python dataset_prep/build_snapshot.py

_SNAPSHOT_PATH = Path(__file__).parent.parent.parent / "graph_snapshot.pkl"


def _load_snapshot() -> bool:
    """
    Load the pre-built graph snapshot if it exists.
    Returns True if loaded successfully, False otherwise.
    """
    global _graph_builder, _documents
    if not _SNAPSHOT_PATH.exists():
        logger.info("No snapshot found at %s — starting with empty graph", _SNAPSHOT_PATH.name)
        return False
    try:
        import pickle
        logger.info("Loading graph snapshot from %s ...", _SNAPSHOT_PATH.name)
        with open(_SNAPSHOT_PATH, "rb") as f:
            snapshot = pickle.load(f)

        # Restore graph state
        graph = snapshot["graph"]
        docs  = snapshot["documents"]

        # Re-attach the loaded graph to the builder
        _graph_builder._graph = graph
        _documents.update(docs)

        stats = _graph_builder.stats()
        logger.info(
            "Snapshot loaded: %d docs, %d entities, %d same_as edges, %d conflicts",
            stats["documents"], stats["entities"],
            stats["same_as_edges"], stats["conflict_edges"],
        )
        return True
    except Exception as e:
        logger.warning("Failed to load snapshot: %s — starting with empty graph", e)
        return False


# ---------------------------------------------------------------------------
# GLOBAL PIPELINE STATE
# ---------------------------------------------------------------------------
# In a production system, this would be managed by a database or Redis.
# For this project, we keep everything in memory.

_graph_builder    = KnowledgeGraphBuilder()
_documents: dict[str, Document] = {}  # doc_id → Document
_llm              = None
_embedding_engine = EmbeddingEngine()
_extraction_mode  = "langchain"  # "langchain" or "uipath"
_query_engine     = None         # built on first query or after ingest


def _get_llm():
    """Lazily initialize the LLM provider."""
    global _llm
    if _llm is None:
        _llm = create_llm_provider()
    return _llm


def _get_query_engine():
    """Build or rebuild the query engine."""
    global _query_engine
    llm = _get_llm()
    _query_engine = QueryEngine(
        graph            = _graph_builder.get_graph(),
        llm_provider     = llm,
        embedding_engine = _embedding_engine,
        documents        = _documents,
    )
    return _query_engine


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Graph RAG API",
    description = "Knowledge graph-based retrieval-augmented generation pipeline",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ---- CORS middleware ----
# Allow all origins for development. In production, restrict to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Load the pre-built snapshot on startup (no-op if file doesn't exist)
_load_snapshot()

# ---------------------------------------------------------------------------
# PYDANTIC REQUEST/RESPONSE MODELS
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Request body for POST /ingest"""
    paths:     list[str]
    extractor: str = "langchain"   # "langchain" or "uipath"


class IngestResponse(BaseModel):
    """Response from POST /ingest"""
    documents_ingested:  int
    entities_extracted:  int
    extraction_mode:     str


class QueryRequest(BaseModel):
    """Request body for POST /query"""
    question:         str
    max_hops:         int  = 3
    temporal_context: str  = "current"


class ExtractionModeRequest(BaseModel):
    """Request body for POST /extraction/mode"""
    mode: str   # "langchain" or "uipath"


class PDFIngestResponse(BaseModel):
    """Response from POST /ingest/pdf"""
    filename:            str
    pages_processed:     int
    chunks_created:      int
    entities_extracted:  int
    same_as_edges_added: int
    warning:             Optional[str] = None


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _run_entity_resolution():
    """
    Run entity resolution, contradiction detection, and household detection.
    Called after each ingest to keep the graph up to date.
    """
    from ..core.resolver import EntityResolver
    from ..core.contradiction import ContradictionDetector
    from ..core.household import HouseholdDetector

    graph = _graph_builder.get_graph()

    entity_count = len(_graph_builder.get_entity_nodes())
    if entity_count < 2:
        return

    try:
        llm = _get_llm()
        resolver = EntityResolver(llm, _embedding_engine)

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(resolver.resolve, graph)
            try:
                pairs = future.result(timeout=30)  # max 30s for resolution
            except concurrent.futures.TimeoutError:
                logger.warning("Entity resolution timed out after 30s — skipping")
                pairs = []

        for pair in pairs:
            _graph_builder.add_same_as_edge(
                pair.entity_id_a, pair.entity_id_b, pair
            )
        logger.info("Resolution: added %d same_as edges", len(pairs))

        detector   = ContradictionDetector(graph)
        conflicts  = detector.detect()
        for conflict in conflicts:
            _graph_builder.add_conflict_edge(
                conflict.entity_id_a, conflict.entity_id_b, conflict
            )
        logger.info("Contradiction: found %d conflicts", len(conflicts))

        # Household detection — find people at same address
        household_detector = HouseholdDetector(graph)
        households = household_detector.detect()
        household_detector.add_lives_with_edges(households, _graph_builder)
        logger.info("Household: found %d households", len(households))

    except Exception as e:
        logger.warning("Resolution/contradiction/household detection failed: %s", e)


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status":  "ok",
        "service": "Graph RAG API",
        "docs":    "/docs",
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Ingest documents into the knowledge graph.

    Accepts .txt files (LangChain mode) or .json files (UiPath mode).
    Auto-detects mode based on the `extractor` field.

    Returns count of documents and entities successfully ingested.
    """
    global _extraction_mode

    extractor_mode = request.extractor.lower().strip()
    if extractor_mode not in ("langchain", "uipath"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid extractor: {extractor_mode}. Use 'langchain' or 'uipath'."
        )

    _extraction_mode = extractor_mode
    docs_ingested    = 0
    entities_total   = 0

    if extractor_mode == "uipath":
        extractor = UiPathExtractor()
        for path_str in request.paths:
            try:
                doc, entities = extractor.extract(path_str)
                if doc is None:
                    continue
                _documents[doc.doc_id] = doc
                _graph_builder.add_document(doc)
                _embedding_engine.embed_entities(entities)
                for entity in entities:
                    _graph_builder.add_entity(entity)
                docs_ingested  += 1
                entities_total += len(entities)
            except Exception as e:
                logger.warning("Failed to ingest %s: %s", path_str, e)

    else:  # langchain
        from ..core.loader import DocumentLoader
        llm        = _get_llm()
        classifier = DocumentClassifier(llm)
        extractor  = LangChainExtractor(llm, model_name=getattr(llm, "model_name", "unknown"))
        loader     = DocumentLoader()

        for path_str in request.paths:
            try:
                doc = loader.load_file(path_str)
                if doc is None:
                    continue
                doc_type, schema = classifier.classify(doc)
                entities = extractor.extract(doc, schema)
                _documents[doc.doc_id] = doc
                _graph_builder.add_document(doc)
                _embedding_engine.embed_entities(entities)
                for entity in entities:
                    _graph_builder.add_entity(entity)
                docs_ingested  += 1
                entities_total += len(entities)
            except Exception as e:
                logger.warning("Failed to ingest %s: %s", path_str, e)

    # Run entity resolution after ingestion
    _run_entity_resolution()
    # Refresh query engine
    _get_query_engine()

    return IngestResponse(
        documents_ingested = docs_ingested,
        entities_extracted = entities_total,
        extraction_mode    = extractor_mode,
    )


@app.post("/ingest/uipath", response_model=IngestResponse)
def ingest_uipath(request: IngestRequest):
    """
    Ingest UiPath JSON files specifically.
    Shorthand for POST /ingest with extractor='uipath'.
    """
    request.extractor = "uipath"
    return ingest(request)


@app.post("/ingest/pdf", response_model=PDFIngestResponse)
def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a medical report PDF into the knowledge graph.
    Runs: PDFLoader → SemanticChunker → PDFEntityExtractor → Graph → EntityResolver
    Returns chunk count, entity count, and new same_as edges created.
    """
    import tempfile
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        from ..pdf.pipeline import PDFIngestPipeline
        pipeline = PDFIngestPipeline(
            graph_builder    = _graph_builder,
            embedding_engine = _embedding_engine,
            llm_provider     = _get_llm(),
            documents        = _documents,
        )
        result = pipeline.ingest(tmp_path)
        _get_query_engine()

        warning = None
        if result.entities_extracted == 0:
            warning = "No entities extracted — PDF may contain only images or unstructured text."

        return PDFIngestResponse(
            filename            = file.filename or "uploaded.pdf",
            pages_processed     = result.pages_processed,
            chunks_created      = result.chunks_created,
            entities_extracted  = result.entities_extracted,
            same_as_edges_added = result.same_as_edges_added,
            warning             = warning,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@app.post("/query")
def query_endpoint(request: QueryRequest):
    """
    Query the knowledge graph with a natural language question.
    Returns QueryResult as a JSON dict.
    Returns 400 if the graph is empty.
    """
    stats = _graph_builder.stats()
    if stats["nodes"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Graph is empty. Ingest documents first via POST /ingest"
        )

    engine = _get_query_engine()

    try:
        result = engine.query(
            question         = request.question,
            max_hops         = request.max_hops,
            temporal_context = request.temporal_context,
        )
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return {
        "question":              result.question,
        "answer":                result.answer,
        "source_documents":      result.source_documents,
        "resolved_entities":     result.resolved_entities,
        "resolution_confidence": result.resolution_confidence,
        "hops_used":             result.hops_used,
        "provenance": [
            {
                "fact":             p.fact,
                "source_filename":  p.source_filename,
                "doc_type":         p.doc_type.value,
                "line_number":      p.line_number,
                "line_text":        p.line_text,
                "confidence":       p.confidence,
                "entity_id":        p.entity_id,
            }
            for p in result.provenance
        ],
        "conflicts": [
            {
                "conflict_type":  c.conflict_type,
                "attribute_key":  c.attribute_key,
                "value_a":        c.value_a,
                "value_b":        c.value_b,
                "severity":       c.severity,
                "source_doc_a":   c.source_doc_a,
                "source_doc_b":   c.source_doc_b,
            }
            for c in result.conflicts
        ],
        "has_conflicts":    result.has_conflicts,
        "temporal_context": result.temporal_context,
    }


@app.post("/smart-query")
def smart_query_endpoint(request: QueryRequest):
    """
    Intelligent query with fuzzy name matching, disambiguation,
    LLM synthesis, provenance, and conflict detection.
    """
    from .smart_query import run_smart_query

    stats = _graph_builder.stats()
    if stats["nodes"] == 0:
        return {
            "type":    "empty",
            "answer":  "No documents have been ingested yet. Please upload some documents first.",
            "options": [],
        }

    graph = _graph_builder.get_graph()
    entity_nodes = [
        (nid, data)
        for nid, data in graph.nodes(data=True)
        if data.get("node_type") == "entity"
    ]

    llm = _get_llm()
    return run_smart_query(request.question, graph, llm, entity_nodes)


@app.get("/graph/stats")
def graph_stats():
    """Return statistics about the current graph state."""
    return _graph_builder.stats()


@app.get("/graph/visualize", response_class=HTMLResponse)
def graph_visualize():
    """
    Return an interactive HTML visualization of the graph using Pyvis.
    Falls back to a simple HTML placeholder if Pyvis is not installed.
    """
    try:
        from ..visualization.visualizer import GraphVisualizer
        import tempfile
        visualizer = GraphVisualizer(_graph_builder.get_graph())
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name

        result_path = visualizer.render(output_path)
        if result_path:
            return HTMLResponse(content=Path(result_path).read_text(), status_code=200)
    except Exception as e:
        logger.warning("Visualization failed: %s", e)

    # Fallback HTML
    stats = _graph_builder.stats()
    return HTMLResponse(content=f"""
    <html><head><title>Graph RAG Visualization</title></head>
    <body>
    <h1>Graph RAG Knowledge Graph</h1>
    <p><b>Nodes:</b> {stats['nodes']} | <b>Edges:</b> {stats['edges']} |
       <b>Entities:</b> {stats['entities']} | <b>Documents:</b> {stats['documents']}</p>
    <p>Install pyvis for interactive visualization: <code>pip install pyvis</code></p>
    </body></html>
    """, status_code=200)


@app.get("/entities")
def list_entities():
    """Return all entity nodes in the graph as a list of dicts."""
    graph = _graph_builder.get_graph()
    entities = []
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") == "entity":
            # Exclude large/non-serializable fields
            entity_dict = {
                k: v for k, v in data.items()
                if k != "embedding" and isinstance(v, (str, int, float, bool, dict, list, type(None)))
            }
            entity_dict["node_id"] = node_id
            entities.append(entity_dict)
    return {"entities": entities, "count": len(entities)}


@app.delete("/graph")
def reset_graph():
    """Reset the knowledge graph (removes all nodes and edges)."""
    global _documents, _query_engine
    _graph_builder.reset()
    _documents    = {}
    _query_engine = None
    return {"message": "Graph reset", "stats": _graph_builder.stats()}


@app.get("/extraction/modes")
def get_extraction_modes():
    """Return available extraction modes and the currently active mode."""
    return {
        "modes":  ["langchain", "uipath"],
        "active": _extraction_mode,
    }


@app.post("/extraction/mode")
def set_extraction_mode(request: ExtractionModeRequest):
    """
    Switch the active extraction mode.

    Args:
        mode — "langchain" or "uipath"
    """
    global _extraction_mode
    mode = request.mode.lower().strip()
    if mode not in ("langchain", "uipath"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode!r}. Use 'langchain' or 'uipath'."
        )
    _extraction_mode = mode
    return {"mode": _extraction_mode, "message": f"Extraction mode set to {mode}"}


# ---------------------------------------------------------------------------
# FILE UPLOAD ENDPOINTS
# ---------------------------------------------------------------------------
# These endpoints accept actual uploaded files (multipart/form-data)
# instead of file paths. Used by the React frontend drag-and-drop.

import tempfile
import shutil

@app.post("/upload")
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    extractor: str = "langchain",
):
    """
    Upload one or more files and ingest them directly.

    Accepts:
        - .txt files  → LangChain mode (extractor=langchain)
        - .json files → UiPath mode   (extractor=uipath)

    The file is saved to a temp directory, ingested, then cleaned up.
    Returns ingestion summary.
    """
    global _extraction_mode
    extractor = extractor.lower().strip()
    if extractor not in ("langchain", "uipath"):
        raise HTTPException(400, detail=f"Invalid extractor: {extractor}")

    _extraction_mode = extractor
    docs_ingested  = 0
    entities_total = 0
    results        = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            # Save uploaded file to temp dir
            tmp_path = Path(tmpdir) / upload.filename
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            # Ingest
            file_result = {"filename": upload.filename, "status": "ok", "entities": 0}
            try:
                if extractor == "uipath":
                    from ..extraction.uipath_extractor import UiPathExtractor
                    ue  = UiPathExtractor()
                    doc, entities = ue.extract(tmp_path)
                    if doc:
                        _documents[doc.doc_id] = doc
                        _graph_builder.add_document(doc)
                        _embedding_engine.embed_entities(entities)
                        for entity in entities:
                            _graph_builder.add_entity(entity)
                        docs_ingested  += 1
                        entities_total += len(entities)
                        file_result["entities"] = len(entities)
                else:
                    from ..core.loader import DocumentLoader
                    from ..extraction.classifier import DocumentClassifier
                    from ..extraction.langchain_extractor import LangChainExtractor
                    llm        = _get_llm()
                    loader     = DocumentLoader()
                    classifier = DocumentClassifier(llm)
                    extractor_obj = LangChainExtractor(
                        llm, model_name=getattr(llm, "model_name", "unknown")
                    )
                    doc = loader.load_file(tmp_path)
                    if doc:
                        doc_type, schema = classifier.classify(doc)
                        entities = extractor_obj.extract(doc, schema)
                        _documents[doc.doc_id] = doc
                        _graph_builder.add_document(doc)
                        _embedding_engine.embed_entities(entities)
                        for entity in entities:
                            _graph_builder.add_entity(entity)
                        docs_ingested  += 1
                        entities_total += len(entities)
                        file_result["entities"] = len(entities)
                        file_result["doc_type"] = doc_type.value
            except Exception as e:
                file_result["status"] = f"error: {e}"
                logger.warning("Upload ingest failed for %s: %s", upload.filename, e)

            results.append(file_result)

    # Run resolution after all files uploaded
    _run_entity_resolution()
    _get_query_engine()

    return {
        "documents_ingested": docs_ingested,
        "entities_extracted": entities_total,
        "extraction_mode":    _extraction_mode,
        "files":              results,
        "graph_stats":        _graph_builder.stats(),
    }


@app.get("/testdata")
def list_test_data():
    """
    Return a tree of all available test dataset files.
    Used by the Test Dataset panel in the frontend.

    Returns:
        {
          "people": [
            {
              "name": "alice_chen",
              "files": [
                {"name": "birth_certificate.txt", "type": "txt", "path": "..."},
                {"name": "birth_certificate.json", "type": "json", "path": "..."},
                ...
              ]
            },
            ...
          ]
        }
    """
    people_dir = Path("docs/people")
    if not people_dir.exists():
        return {"people": []}

    people = []
    for person_dir in sorted(people_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        files = []
        for f in sorted(person_dir.iterdir()):
            if f.suffix in (".txt", ".json"):
                files.append({
                    "name": f.name,
                    "type": f.suffix.lstrip("."),
                    "path": str(f),
                    "size": f.stat().st_size,
                })
        if files:
            people.append({
                "name":  person_dir.name,
                "label": person_dir.name.replace("_", " ").title(),
                "files": files,
            })

    return {"people": people, "total": len(people)}


# ---------------------------------------------------------------------------
# DEMO DATA ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/demo")
def list_demo_data():
    """
    Return the demo people available to add live to the graph.
    These are NOT in the pre-built snapshot — they can be added during a demo.

    Returns:
        {
          "people": [
            {
              "name": "alice_chen",
              "label": "Alice Chen",
              "files": [{"name": "...", "type": "json|txt|pdf", "path": "..."}]
            },
            ...
          ]
        }
    """
    demo_dir = Path("docs/demo")
    if not demo_dir.exists():
        return {"people": [], "note": "docs/demo/ not found"}

    people = []
    for person_dir in sorted(demo_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        files = []
        for f in sorted(person_dir.iterdir()):
            if f.suffix in (".txt", ".json", ".pdf"):
                files.append({
                    "name": f.name,
                    "type": f.suffix.lstrip("."),
                    "path": str(f),
                    "size": f.stat().st_size,
                })
        if files:
            people.append({
                "name":  person_dir.name,
                "label": person_dir.name.replace("_", " ").title(),
                "files": files,
            })

    return {
        "people": people,
        "total":  len(people),
        "note":   "These people are not in the pre-built snapshot. Ingest them live to see the graph update.",
    }


@app.post("/demo/ingest")
def ingest_demo_person(body: dict):
    """
    Ingest all files for a demo person and run entity resolution.

    Body: {"name": "alice_chen", "extractor": "uipath"}

    Ingests all .json (or .txt) files + the .pdf (if present) for that person,
    runs entity resolution, and returns the updated graph stats.
    """
    person_name  = body.get("name", "").strip()
    extractor    = body.get("extractor", "uipath").lower()
    demo_dir     = Path("docs/demo") / person_name

    if not demo_dir.exists():
        raise HTTPException(404, detail=f"Demo person not found: {person_name}")

    ingested = 0
    entities_total = 0

    # ---- Ingest JSON/TXT files ----
    if extractor == "uipath":
        from ..extraction.uipath_extractor import UiPathExtractor
        ue = UiPathExtractor()
        for json_file in sorted(demo_dir.glob("*.json")):
            try:
                doc, entities = ue.extract(str(json_file))
                if doc is None:
                    continue
                _documents[doc.doc_id] = doc
                _graph_builder.add_document(doc)
                _embedding_engine.embed_entities(entities)
                for entity in entities:
                    _graph_builder.add_entity(entity)
                ingested += 1
                entities_total += len(entities)
            except Exception as e:
                logger.warning("Demo ingest failed %s: %s", json_file.name, e)
    else:
        from ..core.loader import DocumentLoader
        from ..extraction.classifier import DocumentClassifier
        from ..extraction.langchain_extractor import LangChainExtractor
        llm = _get_llm()
        loader = DocumentLoader()
        classifier = DocumentClassifier(llm)
        lc = LangChainExtractor(llm, model_name=getattr(llm, "model_name", "unknown"))
        for txt_file in sorted(demo_dir.glob("*.txt")):
            try:
                doc = loader.load_file(str(txt_file))
                if doc is None:
                    continue
                doc_type, schema = classifier.classify(doc)
                entities = lc.extract(doc, schema)
                _documents[doc.doc_id] = doc
                _graph_builder.add_document(doc)
                _embedding_engine.embed_entities(entities)
                for entity in entities:
                    _graph_builder.add_entity(entity)
                ingested += 1
                entities_total += len(entities)
            except Exception as e:
                logger.warning("Demo ingest failed %s: %s", txt_file.name, e)

    # ---- Ingest PDF if present ----
    pdf_files = list(demo_dir.glob("*.pdf"))
    if pdf_files:
        try:
            from ..pdf.pipeline import PDFIngestPipeline
            pdf_pipeline = PDFIngestPipeline(
                graph_builder    = _graph_builder,
                embedding_engine = _embedding_engine,
                llm_provider     = _get_llm(),
                documents        = _documents,
            )
            for pdf_file in pdf_files:
                result = pdf_pipeline.ingest(str(pdf_file))
                ingested += result.chunks_created
                entities_total += result.entities_extracted
        except Exception as e:
            logger.warning("Demo PDF ingest failed: %s", e)

    # ---- Run entity resolution to link demo person to existing graph ----
    _run_entity_resolution()
    _get_query_engine()

    return {
        "person":           person_name,
        "files_ingested":   ingested,
        "entities_added":   entities_total,
        "graph_stats":      _graph_builder.stats(),
    }


@app.post("/testdata/resolve")
def resolve_after_batch():
    """
    Run entity resolution in the background (non-blocking).
    Returns immediately — check /graph/stats to see when same_as_edges increase.
    """
    import threading

    def _resolve_bg():
        try:
            _run_entity_resolution()
            _get_query_engine()
            logger.info("Background resolution complete")
        except Exception as e:
            logger.error("Background resolution failed: %s", e)

    t = threading.Thread(target=_resolve_bg, daemon=True)
    t.start()
    return {
        "message":    "Resolution started in background",
        "hint":       "Poll GET /graph/stats — same_as_edges will increase when done",
        "graph_stats": _graph_builder.stats(),
    }


@app.post("/explore/resolve")
def explore_resolve():
    """Trigger resolution (background) — called from Explore panel."""
    import threading

    def _resolve_bg():
        try:
            _run_entity_resolution()
            _get_query_engine()
        except Exception as e:
            logger.error("Background resolution failed: %s", e)

    t = threading.Thread(target=_resolve_bg, daemon=True)
    t.start()
    return {"message": "Resolution running in background", "graph_stats": _graph_builder.stats()}


# ---------------------------------------------------------------------------
# UIPATH LIVE API ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/uipath/extract")
async def uipath_extract(
    file: UploadFile = File(...),
    extractor: str = "identity_documents",
):
    """
    Send a real scanned PDF/image to the UiPath Document Understanding API.
    Returns extracted fields and ingests into the knowledge graph.

    Requires UIPATH_CLIENT_ID and UIPATH_CLIENT_SECRET in .env

    Supported extractors:
        identity_documents — passports, licenses, national IDs
        invoices           — invoices
        receipts           — receipts
        contracts          — contracts
    """
    # Check credentials exist
    if not os.getenv("UIPATH_CLIENT_ID") or not os.getenv("UIPATH_CLIENT_SECRET"):
        raise HTTPException(
            status_code=400,
            detail=(
                "UiPath credentials not configured. "
                "Add UIPATH_CLIENT_ID and UIPATH_CLIENT_SECRET to your .env file. "
                "Get them from: cloud.uipath.com → Admin → External Applications"
            )
        )

    from ..extraction.uipath_api_connector import UiPathAPIConnector, UiPathAPIError

    connector = UiPathAPIConnector.from_env()

    # Save uploaded file to temp location
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            # Call UiPath API — get back a .json file
            json_path = connector.extract_to_json(
                file_path  = tmp_path,
                extractor  = extractor,
                output_dir = tmpdir,
            )

            # Ingest the resulting JSON into our pipeline
            from ..extraction.uipath_extractor import UiPathExtractor
            ue = UiPathExtractor()
            doc, entities = ue.extract(json_path)

            if doc is None:
                raise HTTPException(500, detail="UiPath extraction returned no document")

            _documents[doc.doc_id] = doc
            _graph_builder.add_document(doc)
            _embedding_engine.embed_entities(entities)
            for entity in entities:
                _graph_builder.add_entity(entity)

            _run_entity_resolution()
            _get_query_engine()

            # Return the extracted fields for display
            import json as _json
            pipeline_json = _json.loads(json_path.read_text())

        except UiPathAPIError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error("UiPath extraction error: %s", e)
            raise HTTPException(status_code=500, detail=f"UiPath error: {e}")

    return {
        "filename":          file.filename,
        "extractor":         extractor,
        "document_type":     pipeline_json.get("document_type"),
        "confidence":        pipeline_json.get("confidence"),
        "fields":            pipeline_json.get("fields", {}),
        "entities_extracted": len(entities),
        "graph_stats":        _graph_builder.stats(),
    }


@app.get("/uipath/status")
def uipath_status():
    """Check if UiPath credentials are configured and connection works."""
    client_id = os.getenv("UIPATH_CLIENT_ID", "")
    has_creds  = bool(client_id and os.getenv("UIPATH_CLIENT_SECRET"))

    if not has_creds:
        return {
            "configured": False,
            "message": (
                "UiPath credentials not set. "
                "Add UIPATH_CLIENT_ID and UIPATH_CLIENT_SECRET to .env"
            )
        }

    try:
        from ..extraction.uipath_api_connector import UiPathAPIConnector
        connector = UiPathAPIConnector.from_env()
        connected = connector.test_connection()
        return {
            "configured": True,
            "connected":  connected,
            "client_id":  client_id[:8] + "...",
            "message":    "UiPath API connected ✓" if connected else "Credentials set but connection failed",
        }
    except Exception as e:
        return {"configured": True, "connected": False, "message": str(e)}


# ---------------------------------------------------------------------------
# PERSONAL DOCUMENT UPLOAD — saves to docs/people/{person_name}/
# ---------------------------------------------------------------------------

@app.post("/person/upload")
async def upload_personal_document(
    file:        UploadFile = File(...),
    person_name: str        = "unknown_person",
    extractor:   str        = "langchain",
):
    """
    Upload a personal document (passport, license, etc.) using either
    the LangChain (LLM) or UiPath (API) extractor.

    Saves the extracted JSON to docs/people/{person_name}/ so it sits
    alongside other people in the test dataset.

    Args:
        file        — PDF, PNG, JPG, or TXT file to process
        person_name — folder name under docs/people/ (e.g. "thayaananthan_kanagaraj")
        extractor   — "langchain" (use local LLM) or "uipath" (use UiPath API)

    Returns:
        Extracted fields + entity info + path where JSON was saved
    """
    import re

    # Sanitize person_name: lowercase, spaces to underscores, no special chars
    person_name = re.sub(r"[^a-z0-9_]", "_", person_name.lower().strip())
    if not person_name or person_name == "_":
        raise HTTPException(400, detail="Invalid person_name")

    # Create person folder inside docs/people/
    person_dir = Path("docs") / "people" / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    extractor_mode = extractor.lower().strip()
    if extractor_mode not in ("langchain", "uipath"):
        raise HTTPException(400, detail="extractor must be 'langchain' or 'uipath'")

    # Save uploaded file temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc      = None
        entities = []
        saved_json_path = None
        doc_type_str    = "GENERIC"

        # ---- UiPath extraction ----
        if extractor_mode == "uipath":
            if not os.getenv("UIPATH_CLIENT_ID") or not os.getenv("UIPATH_CLIENT_SECRET"):
                raise HTTPException(400, detail="UiPath credentials not set in .env")

            from ..extraction.uipath_api_connector import UiPathAPIConnector, UiPathAPIError
            from ..extraction.uipath_extractor import UiPathExtractor

            # Determine extractor type from file content / name
            uipath_extractor_type = "identity_documents"  # passports, licenses, IDs

            connector = UiPathAPIConnector.from_env()
            try:
                json_path = connector.extract_to_json(
                    file_path   = tmp_path,
                    extractor   = uipath_extractor_type,
                    output_dir  = person_dir,  # save directly to person folder
                )
                # Rename to match document type
                stem = Path(file.filename).stem.lower().replace(" ", "_")
                target_path = person_dir / f"{stem}_uipath.json"
                json_path.rename(target_path)
                saved_json_path = str(target_path)

                # Parse with UiPath extractor
                ue = UiPathExtractor()
                doc, entities = ue.extract(target_path)
                if doc:
                    doc_type_str = doc.doc_type.value

            except UiPathAPIError as e:
                raise HTTPException(502, detail=f"UiPath API error: {e}")

        # ---- LangChain extraction ----
        else:
            from ..core.loader import DocumentLoader
            from ..extraction.classifier import DocumentClassifier
            from ..extraction.langchain_extractor import LangChainExtractor

            llm        = _get_llm()
            loader     = DocumentLoader()
            classifier = DocumentClassifier(llm)
            lc_extractor = LangChainExtractor(
                llm, model_name=getattr(llm, "model_name", "unknown")
            )

            doc = loader.load_file(tmp_path)
            if doc is None:
                raise HTTPException(400, detail=f"Could not read file: {file.filename}")

            doc_type, schema = classifier.classify(doc)
            entities = lc_extractor.extract(doc, schema)
            doc_type_str = doc_type.value

            # Also save a copy of the .txt and extracted JSON to person folder
            # Copy the original file
            dest_txt = person_dir / file.filename
            shutil.copy2(tmp_path, dest_txt)

            # Save extracted result as JSON
            stem = Path(file.filename).stem.lower().replace(" ", "_")
            json_data = {
                "document_type": doc_type_str,
                "confidence":    entities[0].confidence if entities else 0.0,
                "source_file":   file.filename,
                "fields": {
                    k: {
                        "value":        v,
                        "confidence":   entities[0].confidence if entities else 0.0,
                        "page":         1,
                        "bounding_box": [
                            entities[0].char_offset_start,
                            entities[0].line_number,
                            entities[0].char_offset_end,
                            entities[0].line_number,
                        ] if entities else [0, 0, 0, 0],
                    }
                    for k, v in (entities[0].attributes.items() if entities else {})
                }
            }
            saved_json_path = str(person_dir / f"{stem}.json")
            Path(saved_json_path).write_text(
                __import__("json").dumps(json_data, indent=2), encoding="utf-8"
            )

    # ---- Add to graph ----
    if doc and entities:
        _documents[doc.doc_id] = doc
        _graph_builder.add_document(doc)
        _embedding_engine.embed_entities(entities)
        for entity in entities:
            # Tag with person_name so we can find them later
            entity.attributes["_person_folder"] = person_name
            _graph_builder.add_entity(entity)

        # Run resolution to link this document to any existing ones for same person
        _run_entity_resolution()
        _get_query_engine()

    # Return summary
    extracted_attrs = entities[0].attributes if entities else {}
    extracted_attrs.pop("_person_folder", None)

    return {
        "person_name":       person_name,
        "person_folder":     str(person_dir),
        "filename":          file.filename,
        "extractor":         extractor_mode,
        "document_type":     doc_type_str,
        "saved_json":        saved_json_path,
        "extracted_fields":  extracted_attrs,
        "entities_extracted": len(entities),
        "entity_name":        entities[0].name if entities else None,
        "graph_stats":        _graph_builder.stats(),
    }


@app.get("/person/{person_name}")
def get_person_info(person_name: str):
    """
    Get all graph information about a specific person by folder name.
    Returns all entities, documents, same_as links, and conflicts.
    """
    graph = _graph_builder.get_graph()

    # Find all entities belonging to this person
    person_entities = []
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        attrs = data.get("attributes", {}) or {}
        if (attrs.get("_person_folder") == person_name
                or person_name.lower() in data.get("name", "").lower().replace(" ", "_")):
            person_entities.append({
                "node_id":    node_id,
                "name":       data.get("name"),
                "doc_type":   data.get("doc_type"),
                "attributes": {k: v for k, v in attrs.items() if not k.startswith("_")},
                "source_filename": data.get("source_filename"),
                "confidence": data.get("confidence"),
            })

    # Find same_as links between these entities
    entity_ids = {e["node_id"] for e in person_entities}
    links = []
    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") == "same_as" and u in entity_ids and v in entity_ids:
            links.append({
                "entity_a":   u[:8],
                "entity_b":   v[:8],
                "confidence": data.get("confidence"),
            })

    # Find conflicts
    conflicts = []
    for u, v, data in graph.edges(data=True):
        if data.get("edge_type") == "conflict" and (u in entity_ids or v in entity_ids):
            conflicts.append({
                "conflict_type": data.get("conflict_type"),
                "attribute_key": data.get("attribute_key"),
                "value_a":       data.get("value_a"),
                "value_b":       data.get("value_b"),
                "severity":      data.get("severity"),
            })

    return {
        "person_name":  person_name,
        "entities":     person_entities,
        "same_as_links": links,
        "conflicts":    conflicts,
        "resolved":     len(links) > 0,
    }


# ---------------------------------------------------------------------------
# DUAL EXTRACTION — same file → UiPath + LangChain/OCR simultaneously
# ---------------------------------------------------------------------------

def _check_person_exists(person_name: str) -> dict:
    """
    Check if a person with a similar name already exists in the graph.
    Returns match info.
    """
    graph = _graph_builder.get_graph()
    name_lower = person_name.lower().replace("_", " ")
    matches = []

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        existing_name = data.get("name", "").lower()
        attrs = data.get("attributes", {}) or {}
        folder = attrs.get("_person_folder", "")

        # Check name similarity
        if (name_lower in existing_name or existing_name in name_lower
                or folder == person_name):
            matches.append({
                "node_id":       node_id[:8],
                "name":          data.get("name"),
                "doc_type":      data.get("doc_type"),
                "source_file":   data.get("source_filename"),
                "person_folder": folder,
            })

    return {
        "exists":  len(matches) > 0,
        "matches": matches,
        "count":   len(matches),
    }


@app.post("/person/dual-extract")
async def dual_extract(
    file:        UploadFile = File(...),
    person_name: str        = "thayaananthan_kanagaraj",
):
    """
    Extract a document using BOTH UiPath API and LangChain OCR+LLM simultaneously.
    Returns both results for comparison and ingests both into the graph.

    Also checks if this person already exists in the graph.

    Args:
        file        — image (JPG, PNG) or PDF
        person_name — folder name under docs/people/

    Returns:
        {
          "already_exists": {...},     ← duplicate check
          "uipath_result":  {...},     ← UiPath extraction
          "langchain_result": {...},   ← OCR + LLM verification
          "agreement": {...},          ← which fields both agree on
          "conflicts": [...],          ← which fields disagree
          "saved_to": "docs/people/thayaananthan_kanagaraj/",
          "graph_stats": {...}
        }
    """
    import re as _re
    import json as _json

    # Sanitize name
    person_name = _re.sub(r"[^a-z0-9_]", "_", person_name.lower().strip())
    person_dir  = Path("docs") / "people" / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    # ---- Check if person already in graph ----
    existence_check = _check_person_exists(person_name)

    # Save uploaded file to person dir
    original_path = person_dir / file.filename
    file_content  = await file.read()
    original_path.write_bytes(file_content)

    uipath_result    = None
    langchain_result = None
    uipath_error     = None
    langchain_error  = None

    # ---- Run UiPath extraction ----
    if os.getenv("UIPATH_CLIENT_ID") and os.getenv("UIPATH_CLIENT_SECRET"):
        try:
            from ..extraction.uipath_api_connector import UiPathAPIConnector, UiPathAPIError
            from ..extraction.uipath_extractor import UiPathExtractor

            connector = UiPathAPIConnector.from_env()
            json_path = connector.extract_to_json(
                file_path  = original_path,
                extractor  = "identity_documents",
                output_dir = person_dir,
            )
            # Rename to clear name
            stem     = Path(file.filename).stem.lower()
            uipath_json_path = person_dir / f"{stem}_uipath.json"
            json_path.rename(uipath_json_path)

            ue = UiPathExtractor()
            doc_u, entities_u = ue.extract(uipath_json_path)

            if doc_u and entities_u:
                entities_u[0].attributes["_person_folder"] = person_name
                entities_u[0].attributes["_extractor"]     = "uipath"
                _documents[doc_u.doc_id] = doc_u
                _graph_builder.add_document(doc_u)
                _embedding_engine.embed_entities(entities_u)
                for e in entities_u:
                    _graph_builder.add_entity(e)

                uipath_result = {
                    "doc_type":       doc_u.doc_type.value,
                    "entity_name":    entities_u[0].name,
                    "confidence":     entities_u[0].confidence,
                    "fields":         {k: v for k, v in entities_u[0].attributes.items()
                                       if not k.startswith("_")},
                    "saved_json":     str(uipath_json_path),
                }

        except Exception as e:
            uipath_error = str(e)
            logger.warning("UiPath dual extract error: %s", e)
    else:
        uipath_error = "UiPath credentials not configured in .env"

    # ---- Run LangChain OCR+LLM extraction ----
    try:
        from ..extraction.ocr_extractor import OCRExtractor

        llm = _get_llm()
        ocr_extractor = OCRExtractor(llm, model_name=getattr(llm, "model_name", "unknown"))
        doc_l, entities_l = ocr_extractor.extract(original_path, person_name=person_name)

        if doc_l and entities_l:
            entities_l[0].attributes["_person_folder"] = person_name
            entities_l[0].attributes["_extractor"]     = "langchain_ocr"
            _documents[doc_l.doc_id] = doc_l
            _graph_builder.add_document(doc_l)
            _embedding_engine.embed_entities(entities_l)
            for e in entities_l:
                _graph_builder.add_entity(e)

            # Save LangChain result as JSON too
            stem = Path(file.filename).stem.lower()
            lc_json_path = person_dir / f"{stem}_langchain.json"
            lc_data = {
                "document_type": doc_l.doc_type.value,
                "confidence":    entities_l[0].confidence,
                "source_file":   file.filename,
                "fields": {
                    k: {"value": v, "confidence": entities_l[0].confidence,
                        "page": 1, "bounding_box": [0, 0, 0, 0]}
                    for k, v in entities_l[0].attributes.items()
                    if not k.startswith("_")
                }
            }
            lc_json_path.write_text(_json.dumps(lc_data, indent=2))

            langchain_result = {
                "doc_type":    doc_l.doc_type.value,
                "entity_name": entities_l[0].name,
                "confidence":  entities_l[0].confidence,
                "fields":      {k: v for k, v in entities_l[0].attributes.items()
                                if not k.startswith("_")},
                "saved_json":  str(lc_json_path),
            }
    except Exception as e:
        langchain_error = str(e)
        logger.warning("LangChain OCR dual extract error: %s", e)

    # ---- Run entity resolution to link both extractions ----
    _run_entity_resolution()
    _get_query_engine()

    # ---- Compare results ----
    agreement = {}
    field_conflicts = []

    if uipath_result and langchain_result:
        u_fields = uipath_result.get("fields", {})
        l_fields = langchain_result.get("fields", {})
        shared   = set(u_fields.keys()) & set(l_fields.keys())

        for key in shared:
            uv = str(u_fields[key]).strip().lower()
            lv = str(l_fields[key]).strip().lower()
            if uv == lv:
                agreement[key] = u_fields[key]
            else:
                field_conflicts.append({
                    "field":    key,
                    "uipath":   u_fields[key],
                    "langchain": l_fields[key],
                })

    return {
        "person_name":      person_name,
        "saved_to":         str(person_dir),
        "original_file":    str(original_path),
        "already_exists":   existence_check,
        "uipath_result":    uipath_result,
        "uipath_error":     uipath_error,
        "langchain_result": langchain_result,
        "langchain_error":  langchain_error,
        "agreement":        agreement,
        "field_conflicts":  field_conflicts,
        "agreement_count":  len(agreement),
        "conflict_count":   len(field_conflicts),
        "graph_stats":      _graph_builder.stats(),
    }


# ---------------------------------------------------------------------------
# PREVIEW EXTRACTION — extract fields WITHOUT adding to graph
# User reviews results and confirms before anything is saved
# ---------------------------------------------------------------------------

@app.post("/person/extract-preview")
async def extract_preview(
    file:        UploadFile = File(...),
    person_name: str        = "unknown_person",
    extractor:   str        = "langchain",
    dry_run:     str        = "true",
):
    """
    Extract fields from an image/PDF WITHOUT modifying the graph.
    Returns extracted fields for user review.
    After review, call POST /person/add-to-graph to actually add.

    Args:
        file        — image or PDF
        person_name — folder name
        extractor   — "langchain" (OCR+LLM) or "uipath" (API)
        dry_run     — always true here, kept for clarity
    """
    import re as _re
    extractor_mode = extractor.lower().strip()
    person_name    = _re.sub(r"[^a-z0-9_]", "_", person_name.lower().strip())

    # Save uploaded file to person folder
    person_dir   = Path("docs") / "people" / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    file_content = await file.read()
    save_path    = person_dir / file.filename
    save_path.write_bytes(file_content)

    doc      = None
    entities = []
    doc_type_str = "GENERIC"
    saved_json   = None

    try:
        if extractor_mode == "uipath":
            if not os.getenv("UIPATH_CLIENT_ID") or not os.getenv("UIPATH_CLIENT_SECRET"):
                raise HTTPException(400, detail="UiPath credentials not set in .env")

            from ..extraction.uipath_api_connector import UiPathAPIConnector, UiPathAPIError
            from ..extraction.uipath_extractor import UiPathExtractor

            connector = UiPathAPIConnector.from_env()
            json_path = connector.extract_to_json(
                file_path  = save_path,
                output_dir = person_dir,
            )
            stem             = save_path.stem.lower()
            uipath_json_path = person_dir / f"{stem}_uipath.json"
            json_path.rename(uipath_json_path)
            saved_json = str(uipath_json_path)

            ue = __import__("graph_rag.extraction.uipath_extractor",
                            fromlist=["UiPathExtractor"]).UiPathExtractor()
            doc, entities = ue.extract(uipath_json_path)
            if doc:
                doc_type_str = doc.doc_type.value

        else:  # langchain OCR
            from ..extraction.ocr_extractor import OCRExtractor
            llm = _get_llm()
            ocr = OCRExtractor(llm, model_name=getattr(llm, "model_name", "unknown"))
            doc, entities = ocr.extract(save_path, person_name=person_name)
            if doc:
                doc_type_str = doc.doc_type.value
                # Save extracted JSON for reference
                import json as _json
                stem = save_path.stem.lower()
                lc_path = person_dir / f"{stem}_langchain_preview.json"
                preview_data = {
                    "document_type": doc_type_str,
                    "confidence":    entities[0].confidence if entities else 0.0,
                    "source_file":   file.filename,
                    "fields": {
                        k: {"value": v, "confidence": entities[0].confidence if entities else 0.0,
                            "page": 1, "bounding_box": [0,0,0,0]}
                        for k, v in (entities[0].attributes.items() if entities else {})
                        if not k.startswith("_")
                    }
                }
                lc_path.write_text(_json.dumps(preview_data, indent=2))
                saved_json = str(lc_path)

    except Exception as e:
        raise HTTPException(500, detail=str(e))

    # Return preview — nothing added to graph yet
    fields_out = {}
    if entities:
        for k, v in entities[0].attributes.items():
            if not k.startswith("_"):
                fields_out[k] = {
                    "value":      str(v),
                    "confidence": entities[0].confidence,
                }

    return {
        "person_name":  person_name,
        "filename":     file.filename,
        "doc_type":     doc_type_str,
        "entity_name":  entities[0].name if entities else person_name,
        "confidence":   entities[0].confidence if entities else 0.0,
        "fields":       fields_out,
        "saved_json":   saved_json,
        "extractor":    extractor_mode,
        "added":        False,   # NOT added to graph yet
    }


@app.post("/person/add-to-graph")
def add_extraction_to_graph(body: dict):
    """
    After the user reviews and confirms extracted fields,
    add the entity to the knowledge graph.

    Body:
        {
          "extraction_result": {...},  ← from /person/extract-preview
          "person_name": "thayaananthan_kanagaraj"
        }
    """
    import uuid as _uuid
    from datetime import datetime, timezone

    extraction = body.get("extraction_result", {})
    person_name = body.get("person_name", "unknown")

    if not extraction:
        raise HTTPException(400, detail="extraction_result is required")

    # Rebuild a minimal Document and Entity from the extraction result
    from ..core.models import Document, Entity, DocType, EntityType

    doc_type_str = extraction.get("doc_type", "GENERIC")
    try:
        doc_type = DocType(doc_type_str)
    except ValueError:
        doc_type = DocType.GENERIC

    # Build Document
    doc = Document(
        doc_id       = str(_uuid.uuid4()),
        filename     = extraction.get("filename", "uploaded_doc"),
        text         = "",
        lines        = [],
        paragraphs   = [],
        line_offsets = [],
        doc_type     = doc_type,
        doc_date     = None,
        empty        = False,
        metadata     = {"person_folder": person_name, "extractor": extraction.get("extractor", "")},
    )

    # Build attributes from fields
    attributes: dict[str, str] = {"_person_folder": person_name}
    raw_fields = extraction.get("fields", {})
    for k, v in raw_fields.items():
        if k.startswith("_"):
            continue
        val = v.get("value") if isinstance(v, dict) else str(v)
        if val:
            attributes[k] = str(val)

    name = (
        attributes.get("name")
        or extraction.get("entity_name")
        or person_name.replace("_", " ").title()
    )

    entity = Entity(
        entity_id            = str(_uuid.uuid4()),
        name                 = name,
        entity_type          = EntityType.PERSON,
        attributes           = attributes,
        source_doc_id        = doc.doc_id,
        source_filename      = doc.filename,
        doc_type             = doc_type,
        line_number          = 0,
        line_text            = "",
        paragraph_index      = 0,
        paragraph_text       = "",
        char_offset_start    = 0,
        char_offset_end      = 0,
        extractor_model      = extraction.get("extractor", ""),
        extraction_timestamp = datetime.now(timezone.utc).isoformat(),
        confidence           = float(extraction.get("confidence", 1.0)),
        embedding            = None,
    )

    # Add to graph
    _documents[doc.doc_id] = doc
    _graph_builder.add_document(doc)
    _embedding_engine.embed_entities([entity])
    _graph_builder.add_entity(entity)

    # Run entity resolution to link with existing entities
    _run_entity_resolution()
    _get_query_engine()

    return {
        "added":       True,
        "person_name": person_name,
        "entity_name": name,
        "doc_type":    doc_type_str,
        "graph_stats": _graph_builder.stats(),
    }


# ---------------------------------------------------------------------------
# EXPLORE ENDPOINTS — households and conflicts
# ---------------------------------------------------------------------------

@app.get("/explore/households")
def get_households():
    """
    Return all detected same-household groups (people at the same address).

    A household = 2+ different people with matching addresses.
    """
    from ..core.household import HouseholdDetector
    graph     = _graph_builder.get_graph()
    detector  = HouseholdDetector(graph)
    households = detector.detect(min_members=2)

    return {
        "households": [
            {
                "address":      h.address,
                "member_count": h.member_count,
                "members": [
                    {
                        "name":        m["name"],
                        "doc_type":    m["doc_type"],
                        "source_file": m["source_file"],
                        "entity_id":   m["entity_id"],
                    }
                    for m in h.members
                ],
            }
            for h in households
        ],
        "total_households": len(households),
        "total_people_in_households": sum(h.member_count for h in households),
    }


@app.get("/explore/conflicts")
def get_conflicts():
    """
    Return all data conflicts from the graph.
    Reads conflict edges directly — no re-detection needed.
    """
    graph = _graph_builder.get_graph()
    result = []

    def get_line_info(node_id: str) -> dict:
        for neighbor in graph.neighbors(node_id):
            edge = graph.edges.get((node_id, neighbor)) or graph.edges.get((neighbor, node_id), {})
            if edge.get("edge_type") == "mentions":
                return {"line_number": edge.get("line_number", 0), "line_text": edge.get("line_text", "")}
        return {"line_number": 0, "line_text": ""}

    seen_pairs: set = set()

    for node_a, node_b, edge_data in graph.edges(data=True):
        if edge_data.get("edge_type") != "conflict":
            continue

        # Deduplicate — undirected graph may give us both (a,b) and (b,a)
        pair_key = tuple(sorted([node_a, node_b]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        entity_a = graph.nodes.get(node_a, {})
        entity_b = graph.nodes.get(node_b, {})
        prov_a   = get_line_info(node_a)
        prov_b   = get_line_info(node_b)

        # Each edge may have a 'conflicts' list or single conflict fields
        conflicts_list = edge_data.get("conflicts", [])
        if not conflicts_list:
            # Single conflict stored on edge
            conflicts_list = [{
                "conflict_type": edge_data.get("conflict_type", "mismatch"),
                "attribute_key": edge_data.get("attribute_key", ""),
                "value_a":       edge_data.get("value_a", ""),
                "value_b":       edge_data.get("value_b", ""),
                "severity":      edge_data.get("severity", "minor"),
            }]

        for conf in conflicts_list:
            attr_key = conf.get("attribute_key", "")
            val_a    = conf.get("value_a", "")
            val_b    = conf.get("value_b", "")

            # Generate a human-readable problem description
            severity  = conf.get("severity", "minor")
            conf_type = conf.get("conflict_type", "mismatch")

            if "dob" in attr_key.lower():
                problem = f"Date of birth mismatch: one document says '{val_a}', another says '{val_b}'"
            elif "name" in attr_key.lower():
                problem = f"Name mismatch: '{val_a}' vs '{val_b}'"
            elif "license" in attr_key.lower():
                problem = f"License number mismatch: '{val_a}' vs '{val_b}'"
            elif "passport" in attr_key.lower():
                problem = f"Passport number mismatch: '{val_a}' vs '{val_b}'"
            elif "address" in attr_key.lower():
                problem = f"Address differs: '{val_a}' vs '{val_b}'"
            else:
                problem = f"{attr_key} mismatch: '{val_a}' vs '{val_b}'"

            result.append({
                "conflict_type":  conf_type,
                "attribute_key":  attr_key,
                "severity":       severity,
                "problem":        problem,
                "entity_a": {
                    "entity_id":   node_a,
                    "name":        entity_a.get("name", ""),
                    "doc_type":    entity_a.get("doc_type", ""),
                    "source_file": entity_a.get("source_filename", ""),
                    "value":       val_a,
                    "line_number": prov_a["line_number"],
                    "line_text":   prov_a["line_text"],
                    "attributes":  {k: v for k, v in (entity_a.get("attributes", {}) or {}).items()
                                    if not k.startswith("_")},
                },
                "entity_b": {
                    "entity_id":   node_b,
                    "name":        entity_b.get("name", ""),
                    "doc_type":    entity_b.get("doc_type", ""),
                    "source_file": entity_b.get("source_filename", ""),
                    "value":       val_b,
                    "line_number": prov_b["line_number"],
                    "line_text":   prov_b["line_text"],
                    "attributes":  {k: v for k, v in (entity_b.get("attributes", {}) or {}).items()
                                    if not k.startswith("_")},
                },
            })

    critical = [r for r in result if r["severity"] == "critical"]
    minor    = [r for r in result if r["severity"] == "minor"]

    return {
        "conflicts":      result,
        "total":          len(result),
        "critical_count": len(critical),
        "minor_count":    len(minor),
        "critical":       critical,
        "minor":          minor,
    }


@app.get("/explore/entity/{entity_id}")
def get_entity_detail(entity_id: str):
    """
    Get full details for a specific entity — all attributes, documents,
    same_as links, conflicts, and household connections.
    """
    graph = _graph_builder.get_graph()

    if entity_id not in graph:
        raise HTTPException(404, detail=f"Entity {entity_id} not found")

    data = graph.nodes.get(entity_id, {})
    if data.get("node_type") != "entity":
        raise HTTPException(400, detail="Not an entity node")

    # Find all connected entities
    same_as_links = []
    conflict_links = []
    lives_with_links = []
    source_documents = []

    for neighbor in graph.neighbors(entity_id):
        edge = graph.edges.get((entity_id, neighbor)) or graph.edges.get((neighbor, entity_id), {})
        edge_type = edge.get("edge_type", "")
        n_data = graph.nodes.get(neighbor, {})

        if edge_type == "same_as":
            same_as_links.append({
                "entity_id":   neighbor,
                "name":        n_data.get("name", ""),
                "doc_type":    n_data.get("doc_type", ""),
                "source_file": n_data.get("source_filename", ""),
                "confidence":  edge.get("confidence", 0),
            })
        elif edge_type == "conflict":
            same_as_links.append({
                "entity_id":     neighbor,
                "name":          n_data.get("name", ""),
                "conflict_type": edge.get("conflict_type", ""),
                "attribute_key": edge.get("attribute_key", ""),
                "value_a":       edge.get("value_a", ""),
                "value_b":       edge.get("value_b", ""),
                "severity":      edge.get("severity", ""),
            })
            conflict_links.append({
                "entity_id":     neighbor,
                "name":          n_data.get("name", ""),
                "conflict_type": edge.get("conflict_type", ""),
                "attribute_key": edge.get("attribute_key", ""),
                "value_a":       edge.get("value_a", ""),
                "value_b":       edge.get("value_b", ""),
                "severity":      edge.get("severity", ""),
            })
        elif edge_type == "lives_with":
            lives_with_links.append({
                "entity_id":   neighbor,
                "name":        n_data.get("name", ""),
                "doc_type":    n_data.get("doc_type", ""),
                "source_file": n_data.get("source_filename", ""),
                "address":     edge.get("address", ""),
            })
        elif edge_type == "mentions":
            doc_data = graph.nodes.get(neighbor, {})
            if doc_data.get("node_type") == "document":
                source_documents.append({
                    "doc_id":    neighbor,
                    "filename":  doc_data.get("filename", ""),
                    "doc_type":  doc_data.get("doc_type", ""),
                    "doc_date":  doc_data.get("doc_date", ""),
                    "line_number": edge.get("line_number", 0),
                    "line_text":   edge.get("line_text", ""),
                })

    return {
        "entity_id":       entity_id,
        "name":            data.get("name", ""),
        "entity_type":     data.get("entity_type", ""),
        "doc_type":        data.get("doc_type", ""),
        "source_filename": data.get("source_filename", ""),
        "confidence":      data.get("confidence", 0),
        "extractor_model": data.get("extractor_model", ""),
        "attributes":      {k: v for k, v in (data.get("attributes", {}) or {}).items()
                            if not k.startswith("_")},
        "source_documents":  source_documents,
        "same_as_links":     same_as_links,
        "conflict_links":    conflict_links,
        "lives_with_links":  lives_with_links,
    }
