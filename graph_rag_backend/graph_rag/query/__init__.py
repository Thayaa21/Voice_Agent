"""
Query package — traversal, aggregation, provenance, and the query engine.
"""
from .traversal import MultiHopTraversal
from .aggregator import ContextAggregator, AggregatedContext
from .provenance import ProvenanceTracker
from .engine import QueryEngine

__all__ = [
    "MultiHopTraversal",
    "ContextAggregator",
    "AggregatedContext",
    "ProvenanceTracker",
    "QueryEngine",
]
