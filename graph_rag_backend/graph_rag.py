#!/usr/bin/env python3
"""
Graph RAG Command-Line Interface — Step 17
===========================================
A click-based CLI for the Graph RAG pipeline.

TEACHING NOTES
--------------
click library:
    Click is a Python package for creating command-line interfaces.
    It uses decorators to define commands and options:

    @click.group()         — creates a group of subcommands
    @cli.command()         — creates a subcommand
    @click.option()        — adds an option (--flag value)
    @click.argument()      — adds a positional argument
    @click.pass_context()  — passes the click Context object to the function

State management with JSON:
    We persist pipeline state between CLI invocations using a JSON file.
    This lets you run `graph_rag ingest ...` and then `graph_rag query ...`
    without re-ingesting in the same Python process.

    The state file stores: list of ingested file paths, active extraction mode.
    The graph itself is rebuilt in memory on each command (from the state file's
    list of files). For large graphs, this would be slow — a production system
    would persist the graph too.

Pretty-printing:
    CLI output should be human-readable:
    - Answer in a box
    - Provenance as a numbered list with file+line info
    - Conflict warnings in red (or with ⚠ emoji)

sys.exit():
    We use sys.exit() in CLI code — this is OK.
    The rule against sys.exit() applies to LIBRARY code (graph_rag/core/*.py etc.)
    where unexpected exits would break callers.
    In a CLI entry point, sys.exit() is the correct way to set exit codes.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# State file location — in the current working directory
STATE_FILE = Path(".graph_rag_state.json")

# Set up logging for CLI usage
logging.basicConfig(
    level  = logging.WARNING,   # Only show warnings and errors in CLI mode
    format = "%(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# STATE HELPERS
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    """Load CLI state from the JSON state file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "ingested_files":  [],
        "extraction_mode": "langchain",
    }


def _save_state(state: dict) -> None:
    """Save CLI state to the JSON state file."""
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError as e:
        click.echo(f"Warning: could not save state: {e}", err=True)


def _build_pipeline(state: dict):
    """
    Rebuild the full pipeline from state (ingested files + mode).

    Returns (graph_builder, documents_dict, embedding_engine, llm) or
    raises click.ClickException on failure.
    """
    from graph_rag.core.embeddings import EmbeddingEngine
    from graph_rag.core.graph_builder import KnowledgeGraphBuilder
    from graph_rag.core.loader import DocumentLoader
    from graph_rag.core.models import Document
    from graph_rag.core.resolver import EntityResolver
    from graph_rag.core.contradiction import ContradictionDetector
    from graph_rag.extraction.classifier import DocumentClassifier
    from graph_rag.extraction.langchain_extractor import LangChainExtractor
    from graph_rag.extraction.uipath_extractor import UiPathExtractor
    from graph_rag.llm.provider import create_llm_provider

    ingested_files   = state.get("ingested_files", [])
    extraction_mode  = state.get("extraction_mode", "langchain")

    if not ingested_files:
        return None, {}, EmbeddingEngine(), None

    llm              = create_llm_provider()
    embedding_engine = EmbeddingEngine()
    graph_builder    = KnowledgeGraphBuilder()
    documents: dict[str, Document] = {}

    if extraction_mode == "uipath":
        extractor = UiPathExtractor()
        for path_str in ingested_files:
            try:
                doc, entities = extractor.extract(path_str)
                if doc is None:
                    continue
                documents[doc.doc_id] = doc
                graph_builder.add_document(doc)
                embedding_engine.embed_entities(entities)
                for entity in entities:
                    graph_builder.add_entity(entity)
            except Exception as e:
                click.echo(f"Warning: skipped {path_str}: {e}", err=True)
    else:
        loader     = DocumentLoader()
        classifier = DocumentClassifier(llm)
        extractor  = LangChainExtractor(llm, model_name=getattr(llm, "model_name", "unknown"))
        for path_str in ingested_files:
            try:
                doc = loader.load_file(path_str)
                if doc is None:
                    continue
                doc_type, schema = classifier.classify(doc)
                entities = extractor.extract(doc, schema)
                documents[doc.doc_id] = doc
                graph_builder.add_document(doc)
                embedding_engine.embed_entities(entities)
                for entity in entities:
                    graph_builder.add_entity(entity)
            except Exception as e:
                click.echo(f"Warning: skipped {path_str}: {e}", err=True)

    # Entity resolution
    graph = graph_builder.get_graph()
    try:
        resolver = EntityResolver(llm, embedding_engine)
        pairs    = resolver.resolve(graph)
        for pair in pairs:
            graph_builder.add_same_as_edge(pair.entity_id_a, pair.entity_id_b, pair)
        detector = ContradictionDetector(graph)
        conflicts = detector.detect()
        for conflict in conflicts:
            graph_builder.add_conflict_edge(conflict.entity_id_a, conflict.entity_id_b, conflict)
    except Exception as e:
        click.echo(f"Warning: resolution failed: {e}", err=True)

    return graph_builder, documents, embedding_engine, llm


# ---------------------------------------------------------------------------
# CLI GROUP
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="1.0.0", prog_name="graph_rag")
def cli():
    """
    Graph RAG — Knowledge graph-based document question answering.

    \b
    Workflow:
      1. graph_rag ingest --dir docs/people/alice_chen
      2. graph_rag query "What is Alice's license number?"
      3. graph_rag stats
    """


# ---------------------------------------------------------------------------
# COMMANDS
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--dir",   "directories", multiple=True, type=click.Path(exists=True),
              help="Directory of documents to ingest (can specify multiple)")
@click.option("--files", "files",       multiple=True, type=click.Path(exists=True),
              help="Individual files to ingest (can specify multiple)")
@click.option("--extractor", default="langchain", show_default=True,
              type=click.Choice(["langchain", "uipath"], case_sensitive=False),
              help="Extraction mode: langchain (raw .txt) or uipath (.json)")
def ingest(directories, files, extractor):
    """
    Ingest documents into the knowledge graph.

    Examples:

    \b
      # Ingest all .txt files in a directory
      graph_rag ingest --dir docs/people/alice_chen --extractor langchain

      # Ingest specific UiPath JSON files
      graph_rag ingest --files a.json b.json --extractor uipath

      # Ingest entire dataset
      graph_rag ingest --dir docs/people --extractor langchain
    """
    state = _load_state()
    state["extraction_mode"] = extractor.lower()

    # Collect file paths
    new_files: list[str] = []

    for dir_path in directories:
        dir_p = Path(dir_path)
        if extractor.lower() == "uipath":
            pattern = "**/*.json"
        else:
            pattern = "**/*.txt"
        found = sorted(dir_p.glob(pattern))
        new_files.extend(str(f) for f in found)

    for file_path in files:
        new_files.append(str(Path(file_path).resolve()))

    if not new_files:
        click.echo("No files found to ingest. Check --dir or --files arguments.")
        sys.exit(1)

    # Add to state (avoid duplicates)
    existing = set(state["ingested_files"])
    for f in new_files:
        if f not in existing:
            state["ingested_files"].append(f)
            existing.add(f)

    _save_state(state)

    click.echo(f"Added {len(new_files)} file(s) to ingestion list.")
    click.echo(f"Extraction mode: {extractor}")
    click.echo("Building graph (this may take a minute for LangChain mode)...")

    graph_builder, documents, embedding_engine, llm = _build_pipeline(state)
    if graph_builder is None:
        click.echo("No documents ingested.")
        return

    stats = graph_builder.stats()
    click.echo(f"\n✓ Graph built successfully!")
    click.echo(f"  Documents:     {stats['documents']}")
    click.echo(f"  Entities:      {stats['entities']}")
    click.echo(f"  Same-as edges: {stats['same_as_edges']}")
    click.echo(f"  Conflicts:     {stats['conflict_edges']}")


@cli.command()
@click.argument("question")
@click.option("--hops",     default=3,         show_default=True, help="Max BFS hops")
@click.option("--temporal", default="current", show_default=True,
              help="Temporal context: current, all, or ISO date (e.g. 2020-01-01)")
def query(question, hops, temporal):
    """
    Ask a question about the ingested documents.

    Examples:

    \b
      graph_rag query "What is Alice Chen's date of birth?"
      graph_rag query "What medications was James prescribed?" --temporal all
      graph_rag query "What was Alice's address in 2015?" --temporal 2015-12-31
    """
    state = _load_state()
    if not state.get("ingested_files"):
        click.echo("No documents ingested. Run 'graph_rag ingest' first.")
        sys.exit(1)

    click.echo("Building graph...")
    graph_builder, documents, embedding_engine, llm = _build_pipeline(state)
    if graph_builder is None:
        click.echo("Failed to build graph.")
        sys.exit(1)

    from graph_rag.query.engine import QueryEngine
    engine = QueryEngine(
        graph            = graph_builder.get_graph(),
        llm_provider     = llm,
        embedding_engine = embedding_engine,
        documents        = documents,
    )

    click.echo(f"\nQuerying: {question!r}\n")
    result = engine.query(question, max_hops=hops, temporal_context=temporal)

    # ---- Pretty-print the answer ----
    click.echo("=" * 70)
    click.echo("ANSWER")
    click.echo("=" * 70)
    click.echo(result.answer)

    # ---- Provenance ----
    if result.provenance:
        click.echo("\n" + "-" * 70)
        click.echo("SOURCES (Provenance)")
        click.echo("-" * 70)
        # Group by file for cleaner output
        by_file: dict[str, list] = {}
        for p in result.provenance:
            by_file.setdefault(p.source_filename, []).append(p)

        for filename, entries in by_file.items():
            click.echo(f"\n  📄 {filename}")
            seen_facts = set()
            for p in entries:
                if p.fact in seen_facts:
                    continue
                seen_facts.add(p.fact)
                if p.line_number > 0:
                    click.echo(f"     Line {p.line_number}: {p.fact}")
                    if p.line_text:
                        click.echo(f"       → \"{p.line_text}\"")
                else:
                    click.echo(f"     {p.fact}")

    # ---- Conflict warnings ----
    if result.has_conflicts:
        click.echo("\n" + "⚠" * 35)
        click.echo("⚠  CONFLICT WARNINGS")
        click.echo("⚠" * 35)
        for c in result.conflicts:
            click.echo(f"\n  [{c.severity.upper()}] {c.conflict_type}")
            click.echo(f"    {c.source_doc_a}: {c.value_a!r}")
            click.echo(f"    {c.source_doc_b}: {c.value_b!r}")

    # ---- Stats ----
    click.echo(f"\n  Sources used: {', '.join(result.source_documents) or 'none'}")
    click.echo(f"  Entities resolved: {len(result.resolved_entities)}")
    click.echo(f"  BFS hops: {result.hops_used}")
    click.echo(f"  Temporal context: {result.temporal_context}")


@cli.command()
@click.argument("fact")
def verify(fact):
    """
    Verify a fact — find which document(s) support it.

    Examples:

    \b
      graph_rag verify "dob: 1992-03-15"
      graph_rag verify "Alice Chen"
    """
    state = _load_state()
    if not state.get("ingested_files"):
        click.echo("No documents ingested. Run 'graph_rag ingest' first.")
        sys.exit(1)

    click.echo("Building graph...")
    graph_builder, documents, embedding_engine, _ = _build_pipeline(state)
    if graph_builder is None:
        click.echo("Failed to build graph.")
        sys.exit(1)

    from graph_rag.query.provenance import ProvenanceTracker
    tracker = ProvenanceTracker(graph_builder.get_graph(), documents)
    entries = tracker.verify(fact)

    if not entries:
        click.echo(f"\nNo provenance found for: {fact!r}")
        return

    click.echo(f"\nProvenance for: {fact!r}")
    click.echo("-" * 50)
    for p in entries:
        click.echo(f"\n  📄 {p.source_filename}")
        if p.line_number > 0:
            click.echo(f"     Line {p.line_number}: {p.fact}")
            if p.line_text:
                click.echo(f"     → \"{p.line_text}\"")
        else:
            click.echo(f"     {p.fact}")
        click.echo(f"     Confidence: {p.confidence:.2f}")


@cli.command()
@click.option("--output", default="graph.html", show_default=True,
              help="Output HTML file path")
def visualize(output):
    """
    Generate an interactive HTML visualization of the knowledge graph.

    Opens the generated file in your browser.
    Requires: pip install pyvis

    Example:
      graph_rag visualize --output my_graph.html
    """
    state = _load_state()
    if not state.get("ingested_files"):
        click.echo("No documents ingested. Run 'graph_rag ingest' first.")
        sys.exit(1)

    click.echo("Building graph...")
    graph_builder, _, _, _ = _build_pipeline(state)
    if graph_builder is None:
        click.echo("Failed to build graph.")
        sys.exit(1)

    try:
        from graph_rag.visualization.visualizer import GraphVisualizer
        viz = GraphVisualizer(graph_builder.get_graph())
        path = viz.render(output)
        if path:
            click.echo(f"✓ Visualization saved to: {path}")
            click.echo(f"  Open in browser: file://{Path(path).resolve()}")
        else:
            click.echo("Visualization failed. Install pyvis: pip install pyvis")
    except ImportError:
        click.echo("Visualization module not available. Install pyvis: pip install pyvis")


@cli.command()
@click.option("--port", default=8000, show_default=True, help="Port to run the API server on")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (development)")
def serve(port, host, reload):
    """
    Start the FastAPI REST API server.

    The API is available at http://HOST:PORT
    Interactive docs at http://HOST:PORT/docs

    Example:
      graph_rag serve --port 8000
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    click.echo(f"Starting Graph RAG API at http://{host}:{port}")
    click.echo(f"Interactive docs: http://{host}:{port}/docs")
    click.echo("Press Ctrl+C to stop.")

    uvicorn.run(
        "graph_rag.api.app:app",
        host   = host,
        port   = port,
        reload = reload,
    )


@cli.command()
def stats():
    """
    Show statistics about the current knowledge graph.

    Includes: node counts, edge counts, same-as links, conflicts.
    """
    state = _load_state()

    if not state.get("ingested_files"):
        click.echo("No documents ingested. Run 'graph_rag ingest' first.")
        click.echo("\nState file:", STATE_FILE)
        return

    click.echo("Building graph (for stats)...")
    graph_builder, _, _, _ = _build_pipeline(state)
    if graph_builder is None:
        click.echo("Failed to build graph.")
        return

    s = graph_builder.stats()
    click.echo("\n" + "=" * 40)
    click.echo("Knowledge Graph Statistics")
    click.echo("=" * 40)
    click.echo(f"  Total nodes:      {s['nodes']:>6}")
    click.echo(f"  Total edges:      {s['edges']:>6}")
    click.echo(f"  Document nodes:   {s['documents']:>6}")
    click.echo(f"  Entity nodes:     {s['entities']:>6}")
    click.echo(f"  Same-as edges:    {s['same_as_edges']:>6}")
    click.echo(f"  Conflict edges:   {s['conflict_edges']:>6}")
    click.echo(f"\nIngested files:    {len(state['ingested_files'])}")
    click.echo(f"Extraction mode:   {state['extraction_mode']}")
    click.echo(f"State file:        {STATE_FILE}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
