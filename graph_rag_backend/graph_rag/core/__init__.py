# Core package — models, loading, embeddings, graph, resolver, contradiction, temporal
from .models import (
    Document, Entity, ResolvedPair, ConflictRecord,
    ProvenanceEntry, QueryResult, ExtractionSource,
    DocType, EdgeType, EntityType,
)
from .loader import DocumentLoader, verify_line_offsets
from .embeddings import EmbeddingEngine
from .graph_builder import KnowledgeGraphBuilder
from .resolver import EntityResolver
from .contradiction import ContradictionDetector
from .temporal import TemporalFilter

__all__ = [
    # Enums
    "DocType", "EdgeType", "EntityType",
    # Data models
    "Document", "Entity", "ResolvedPair", "ConflictRecord",
    "ProvenanceEntry", "QueryResult", "ExtractionSource",
    # Utilities
    "DocumentLoader", "verify_line_offsets",
    "EmbeddingEngine",
    "KnowledgeGraphBuilder",
    "EntityResolver",
    "ContradictionDetector",
    "TemporalFilter",
]
