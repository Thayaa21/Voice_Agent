"""
Graph Snapshot Builder
========================
Pre-ingests all documents from docs/people/ into a Knowledge Graph
and serializes it to graph_snapshot.pkl so the app loads instantly.

What gets ingested:
    - All .json files  → UiPath extractor (fast, no LLM)
    - All .pdf files   → PDF pipeline (requires LLM, slow but done once)

The demo people (docs/demo/) are intentionally NOT ingested here.
They stay as raw files so they can be added live during a demo.

Usage:
    python dataset_prep/build_snapshot.py               # ingest all
    python dataset_prep/build_snapshot.py --no-pdf      # skip PDFs (faster)
    python dataset_prep/build_snapshot.py --out custom_snapshot.pkl

Output:
    graph_snapshot.pkl  — in the project root (QB1/)
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

# ---- Set offline mode before any HuggingFace imports ----
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ---- Load .env ----
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ---- Add project root to path ----
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("build_snapshot")


def build(
    people_dir:  Path,
    demo_dir:    Path,
    output_path: Path,
    include_pdf: bool = True,
    skip_resolution: bool = False,
) -> None:
    from graph_rag.core.embeddings import EmbeddingEngine
    from graph_rag.core.graph_builder import KnowledgeGraphBuilder
    from graph_rag.core.models import Document
    from graph_rag.core.resolver import EntityResolver
    from graph_rag.core.contradiction import ContradictionDetector
    from graph_rag.core.household import HouseholdDetector
    from graph_rag.extraction.patient_extractor import PatientExtractor
    from graph_rag.llm.provider import create_llm_provider

    graph_builder    = KnowledgeGraphBuilder()
    documents: dict[str, Document] = {}
    embedding_engine = EmbeddingEngine()
    llm              = create_llm_provider()

    # Collect all demo folder names so we skip them in the main ingest
    demo_folders = {p.name for p in demo_dir.iterdir() if p.is_dir()} if demo_dir.exists() else set()
    logger.info("Demo folders (excluded from snapshot): %s", sorted(demo_folders))

    # ---- 1. Ingest all patient.json files via PatientExtractor ----
    extractor = PatientExtractor()
    json_ok = 0
    json_fail = 0

    for person_dir in sorted(people_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        if person_dir.name in demo_folders:
            logger.info("SKIP (demo): %s", person_dir.name)
            continue

        patient_file = person_dir / "patient.json"
        if not patient_file.exists():
            continue

        try:
            doc, entities = extractor.extract(patient_file)
            if doc is None:
                continue
            documents[doc.doc_id] = doc
            graph_builder.add_document(doc)
            embedding_engine.embed_entities(entities)
            for entity in entities:
                graph_builder.add_entity(entity)
            json_ok += 1
        except Exception as e:
            logger.warning("Failed %s: %s", patient_file.name, e)
            json_fail += 1

    logger.info("Patient JSON ingest: %d OK, %d failed", json_ok, json_fail)

    # ---- 2. Ingest all .pdf files — embed text directly, link to person ----
    # No LLM extraction needed — person name+DOB is already in the header.
    # We load the PDF, chunk it semantically, embed each chunk as a Document node,
    # then create a PERSON entity from the header and link via same_as.
    pdf_ok = 0
    pdf_fail = 0

    if include_pdf:
        from graph_rag.pdf.loader import PDFLoader
        from graph_rag.pdf.chunker import SemanticChunker
        from graph_rag.core.models import Entity, EntityType, DocType, ResolvedPair
        import uuid as _uuid
        from datetime import datetime, timezone

        loader  = PDFLoader()
        chunker = SemanticChunker(embedding_engine, threshold=0.70)

        for person_dir in sorted(people_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            if person_dir.name in demo_folders:
                continue

            for pdf_file in sorted(person_dir.glob("*.pdf")):
                try:
                    # ── Step 1: load + chunk ──────────────────────────────
                    load_result = loader.load(pdf_file)
                    chunks = chunker.chunk(load_result)

                    # ── Step 2: find person name+DOB from the header ──────
                    # The header we injected looks like:
                    #   Patient Name:   Alice Chen
                    #   Date of Birth:  March 15, 1992
                    person_name = person_dir.name.replace("_", " ").title()
                    person_dob  = ""
                    first_page_text = load_result.pages[0].text if load_result.pages else ""
                    for line in first_page_text.splitlines():
                        ll = line.lower()
                        if "patient name" in ll or "patient:" in ll:
                            person_name = line.split(":", 1)[-1].strip()
                        if "date of birth" in ll or "dob" in ll:
                            person_dob = line.split(":", 1)[-1].strip()

                    # ── Step 3: add each chunk as a Document node ─────────
                    chunk_entity_ids = []
                    for chunk in chunks:
                        # Build Document
                        lines = chunk.text.split("\n")
                        line_offsets, pos = [], 0
                        for ln in lines:
                            line_offsets.append(pos)
                            pos += len(ln) + 1
                        paragraphs = [p.strip() for p in chunk.text.split("\n\n") if p.strip()]

                        doc = Document(
                            doc_id       = str(_uuid.uuid4()),
                            filename     = f"{pdf_file.name}_chunk_{chunk.chunk_index}",
                            text         = chunk.text,
                            lines        = lines,
                            paragraphs   = paragraphs,
                            line_offsets = line_offsets,
                            doc_type     = DocType.MEDICAL_REPORT,
                            doc_date     = None,
                            empty        = not chunk.text.strip(),
                            metadata     = {
                                "source_pdf":  pdf_file.name,
                                "chunk_index": chunk.chunk_index,
                                "start_page":  chunk.start_page,
                                "end_page":    chunk.end_page,
                                "person_folder": person_dir.name,
                            },
                        )
                        documents[doc.doc_id] = doc
                        graph_builder.add_document(doc)

                        # ── Step 4: create PERSON entity from header ──────
                        emb_text = f"{person_name} {chunk.text[:200]}"
                        entity = Entity(
                            entity_id            = str(_uuid.uuid4()),
                            name                 = person_name,
                            entity_type          = EntityType.PERSON,
                            attributes           = {
                                "dob":         person_dob,
                                "source_type": "medical_report",
                                "chunk_index": str(chunk.chunk_index),
                                "start_page":  str(chunk.start_page),
                                "end_page":    str(chunk.end_page),
                            },
                            source_doc_id        = doc.doc_id,
                            source_filename      = doc.filename,
                            doc_type             = DocType.MEDICAL_REPORT,
                            line_number          = 1,
                            line_text            = lines[0] if lines else "",
                            extractor_model      = "pdf-embed",
                            extraction_timestamp = datetime.now(timezone.utc).isoformat(),
                            confidence           = 0.95,
                        )
                        embedding_engine.embed_entities([entity])
                        graph_builder.add_entity(entity)
                        chunk_entity_ids.append(entity.entity_id)

                    pdf_ok += 1
                    logger.info(
                        "PDF %s: %d chunks embedded for '%s'",
                        pdf_file.name, len(chunks), person_name
                    )

                except Exception as e:
                    logger.warning("Failed PDF %s: %s", pdf_file.name, e)
                    pdf_fail += 1
        logger.info("PDF ingest: %d OK, %d failed", pdf_ok, pdf_fail)
    else:
        logger.info("PDF ingest skipped (--no-pdf)")

    # ---- 3. Entity resolution + contradiction + household ----
    graph = graph_builder.get_graph()
    entity_count = len(graph_builder.get_entity_nodes())
    logger.info("Total entities: %d", entity_count)

    if skip_resolution:
        logger.info("Entity resolution skipped (--skip-resolution)")
    elif entity_count >= 2:
        logger.info("Running entity resolution...")
        resolver = EntityResolver(llm, embedding_engine)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(resolver.resolve, graph)
            try:
                pairs = future.result(timeout=120)
            except concurrent.futures.TimeoutError:
                logger.warning("Resolution timed out — saving partial snapshot")
                pairs = []

        for pair in pairs:
            graph_builder.add_same_as_edge(pair.entity_id_a, pair.entity_id_b, pair)
        logger.info("same_as edges added: %d", len(pairs))

        detector = ContradictionDetector(graph)
        conflicts = detector.detect()
        for conflict in conflicts:
            graph_builder.add_conflict_edge(conflict.entity_id_a, conflict.entity_id_b, conflict)
        logger.info("Conflicts detected: %d", len(conflicts))

        hd = HouseholdDetector(graph)
        households = hd.detect()
        hd.add_lives_with_edges(households, graph_builder)
        logger.info("Households: %d", len(households))

    # ---- 4. Print stats ----
    stats = graph_builder.stats()
    logger.info("Graph stats: %s", stats)

    # ---- 5. Serialize to pkl ----
    snapshot = {
        "graph":     graph_builder.get_graph(),
        "documents": documents,
        "stats":     stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Snapshot saved: %s (%.1f MB)", output_path, size_mb)
    logger.info(
        "Done. Documents: %d  Entities: %d  same_as: %d  Conflicts: %d",
        stats["documents"], stats["entities"],
        stats["same_as_edges"], stats["conflict_edges"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build the graph snapshot pkl file.")
    parser.add_argument(
        "--people-dir", default="docs/people",
        help="Directory containing person subdirectories (default: docs/people)",
    )
    parser.add_argument(
        "--demo-dir", default="docs/demo",
        help="Directory containing demo-only people to exclude (default: docs/demo)",
    )
    parser.add_argument(
        "--out", default="graph_snapshot.pkl",
        help="Output pkl file path (default: graph_snapshot.pkl)",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF ingestion (faster, identity docs only)",
    )
    parser.add_argument(
        "--skip-resolution", action="store_true",
        help="Skip entity resolution, contradiction, and household detection (fast mode for large datasets)",
    )
    args = parser.parse_args()

    root = ROOT
    build(
        people_dir       = root / args.people_dir,
        demo_dir         = root / args.demo_dir,
        output_path      = root / args.out,
        include_pdf      = not args.no_pdf,
        skip_resolution  = args.skip_resolution,
    )


if __name__ == "__main__":
    main()
