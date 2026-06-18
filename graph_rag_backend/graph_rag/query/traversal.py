"""
Multi-hop Traversal — Step 12
================================
BFS traversal through the knowledge graph, following same_as edges to collect
all documents about a queried entity across different files.

TEACHING NOTES
--------------
What is multi-hop traversal?
    "Hop" = crossing one edge. Multi-hop = crossing multiple edges.

    Example:
        Birth cert → [Alice Chen] ─same_as─► [Alice Chen in DL] ─same_as─► [Alice Chen in Insurance]
        Hop 1: birth cert → DL
        Hop 2: DL → insurance

    By following same_as edges, we collect ALL documents about Alice
    even though they're in separate files with no direct references.

BFS (Breadth-First Search):
    BFS explores nodes level by level:
    - Level 0 (seeds): Alice Chen in birth cert
    - Level 1 (hop 1): Alice Chen in drivers license
    - Level 2 (hop 2): Alice Chen in insurance

    Why BFS instead of DFS?
    BFS finds the SHORTEST path first. For entity resolution, shorter
    paths (more direct same_as links) are more reliable than long chains.

    Implementation: use a deque (double-ended queue) for O(1) pop from front.

Entity finding:
    First we need to find WHICH entities in the graph match the query.
    The query "What is Alice Chen's license number?" gives us the name "Alice Chen".

    We match entities using:
    1. RapidFuzz name similarity
    2. Cosine similarity of query embedding vs entity embedding
    Combined: match_score = 0.5 × name_score + 0.5 × semantic_score

    Entities above threshold=0.60 are "seed" nodes for BFS.

Confidence threshold for edge traversal:
    We only follow same_as edges with confidence ≥ 0.60.
    Low-confidence links might be false positives — following them
    would contaminate the context with wrong-person data.

max_hops constraint:
    Maximum 5 hops to prevent unbounded traversal in large graphs.
    Most real queries need 1-3 hops.
    Raising ValueError for invalid values (< 1 or > 5) fails fast.

Visited set:
    Prevents cycles. Once we've visited a node, we never visit it again.
    Without this, a graph with cycles (A→B→A) would loop forever.
"""

import logging
from collections import deque
from typing import Optional

import networkx as nx

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

from ..core.embeddings import EmbeddingEngine
from ..core.models import Document, EdgeType
from ..core.temporal import TemporalFilter

logger = logging.getLogger(__name__)


class MultiHopTraversal:
    """
    BFS traversal through the knowledge graph following same_as edges.

    Usage:
        traversal = MultiHopTraversal(graph, embedding_engine, temporal_filter)

        # Step 1: Find seed entities matching the query
        seed_ids = traversal.find_entities(
            names=["Alice Chen"],
            query_embedding=query_emb,
            threshold=0.60,
        )

        # Step 2: Expand via BFS through same_as edges
        all_ids = traversal.expand(seed_ids, max_hops=3)

        # Step 3: Get source documents
        docs = traversal.get_source_documents(all_ids)
    """

    def __init__(
        self,
        graph: nx.Graph,
        embedding_engine: EmbeddingEngine,
        temporal_filter: TemporalFilter,
    ):
        self._graph           = graph
        self._embedding_engine = embedding_engine
        self._temporal_filter  = temporal_filter

    def find_entities(
        self,
        names: list[str],
        query_embedding: list[float],
        threshold: float = 0.60,
    ) -> list[str]:
        """
        Find entity nodes that match the given names and/or query embedding.

        Uses a hybrid score:
            match_score = 0.5 × name_score + 0.5 × cosine_similarity(query, entity)

        Args:
            names           — list of person names to search for
                              (can be empty if relying on semantic search)
            query_embedding — 384-float embedding of the full query question
            threshold       — minimum match_score to include (default 0.60)

        Returns:
            List of entity node IDs above the threshold, sorted by score desc.
        """
        matches: list[tuple[float, str]] = []

        # Iterate over all entity nodes
        for node_id, data in self._graph.nodes(data=True):
            if data.get("node_type") != "entity":
                continue

            entity_name = str(data.get("name", ""))
            entity_emb  = data.get("embedding")

            # ---- Name similarity ----
            # Best score across all query names
            name_score = 0.0
            for query_name in names:
                if not query_name.strip():
                    continue
                ns = self._name_score(query_name, entity_name)
                name_score = max(name_score, ns)

            # ---- Semantic similarity ----
            if query_embedding and entity_emb:
                semantic_score = self._embedding_engine.cosine_similarity(
                    query_embedding, entity_emb
                )
            else:
                semantic_score = 0.0

            # ---- Hybrid score ----
            match_score = 0.5 * name_score + 0.5 * semantic_score

            # Also do a direct substring name match as fallback — very permissive
            # e.g. query "Alice" should match entity "Alice Chen"
            for query_name in names:
                qn_lower = query_name.lower().strip()
                en_lower = entity_name.lower().strip()
                if qn_lower and (qn_lower in en_lower or en_lower in qn_lower):
                    match_score = max(match_score, 0.85)  # force above threshold

            if match_score >= threshold:
                matches.append((match_score, node_id))

        # Sort by score descending, return just the IDs
        matches.sort(key=lambda x: x[0], reverse=True)

        result = [node_id for _, node_id in matches]
        logger.debug(
            "find_entities: found %d matches above threshold %.2f",
            len(result), threshold
        )
        return result

    def expand(
        self, entity_ids: list[str], max_hops: int = 3
    ) -> list[str]:
        """
        BFS expansion from seed entities, following same_as edges.

        Only traverses same_as edges with confidence >= 0.60.

        Args:
            entity_ids — starting entity node IDs (seeds)
            max_hops   — maximum BFS depth (must be 1–5, raises ValueError otherwise)

        Returns:
            All entity IDs reachable within max_hops via same_as edges,
            including the original seeds.

        Raises:
            ValueError — if max_hops < 1 or max_hops > 5

        TEACHING: BFS uses a queue (FIFO). We add seeds to the queue,
        then repeatedly pop the front node, visit its neighbors, and add
        unvisited neighbors to the back of the queue.
        """
        # Validate max_hops
        if max_hops < 1 or max_hops > 5:
            raise ValueError(
                f"max_hops must be between 1 and 5, got {max_hops}"
            )

        if not entity_ids:
            return []

        # ---- BFS ----
        visited: set[str]  = set()
        result:  list[str] = []

        # Queue entries: (node_id, current_depth)
        queue: deque[tuple[str, int]] = deque()

        # Initialize with seeds at depth 0
        for eid in entity_ids:
            if eid in self._graph and eid not in visited:
                visited.add(eid)
                queue.append((eid, 0))
                result.append(eid)

        while queue:
            current_id, depth = queue.popleft()

            # Don't traverse beyond max_hops
            if depth >= max_hops:
                continue

            # Look at all neighbors of current node
            for neighbor_id in self._graph.neighbors(current_id):
                if neighbor_id in visited:
                    continue

                # Only follow same_as edges (not mentions or conflict)
                edge_data = self._graph.edges.get(
                    (current_id, neighbor_id),
                    self._graph.edges.get((neighbor_id, current_id), {})
                )

                if edge_data.get("edge_type") != EdgeType.SAME_AS.value:
                    continue

                # Check confidence threshold
                confidence = float(edge_data.get("confidence", 0.0))
                if confidence < 0.60:
                    continue

                # Only follow entity nodes (not document nodes)
                neighbor_data = self._graph.nodes.get(neighbor_id, {})
                if neighbor_data.get("node_type") != "entity":
                    continue

                # Visit this neighbor
                visited.add(neighbor_id)
                result.append(neighbor_id)
                queue.append((neighbor_id, depth + 1))

        logger.debug(
            "expand: %d seeds → %d entities after %d-hop BFS",
            len(entity_ids), len(result), max_hops
        )
        return result

    def get_source_documents(self, entity_ids: list[str]) -> list[Document]:
        """
        For each entity, follow the 'mentions' edge to get its source Document.

        Deduplicates by doc_id — if two entities share a source document
        (shouldn't happen often but possible), we return that document once.

        Args:
            entity_ids — list of entity node IDs

        Returns:
            List of Document objects (deduplicated by doc_id)

        TEACHING: We reconstruct Document objects from the graph node attributes.
        The graph stores all Document fields as node attributes, so we can
        recreate the object without re-reading the file.
        """
        from ..core.models import DocType

        seen_doc_ids: set[str] = set()
        documents: list[Document] = []

        for entity_id in entity_ids:
            entity_data = self._graph.nodes.get(entity_id, {})
            if entity_data.get("node_type") != "entity":
                continue

            source_doc_id = entity_data.get("source_doc_id")
            if not source_doc_id:
                continue

            if source_doc_id in seen_doc_ids:
                continue

            # Look up the document node
            doc_data = self._graph.nodes.get(source_doc_id, {})
            if not doc_data or doc_data.get("node_type") != "document":
                continue

            seen_doc_ids.add(source_doc_id)

            # Reconstruct Document from graph node attributes
            try:
                doc_type_str = doc_data.get("doc_type", "GENERIC")
                try:
                    doc_type = DocType(doc_type_str)
                except ValueError:
                    doc_type = DocType.GENERIC

                doc = Document(
                    doc_id       = doc_data.get("doc_id", source_doc_id),
                    filename     = doc_data.get("filename", ""),
                    text         = doc_data.get("text", ""),
                    lines        = doc_data.get("text", "").split("\n"),
                    paragraphs   = [p.strip() for p in doc_data.get("text", "").split("\n\n") if p.strip()],
                    line_offsets = [],  # not stored on node — recomputed if needed
                    doc_type     = doc_type,
                    doc_date     = doc_data.get("doc_date"),
                    metadata     = doc_data.get("metadata", {}),
                )
                # Recompute line_offsets
                pos = 0
                for line in doc.lines:
                    doc.line_offsets.append(pos)
                    pos += len(line) + 1

                documents.append(doc)
            except Exception as e:
                logger.warning(
                    "Failed to reconstruct Document %s from graph: %s",
                    source_doc_id[:8], e
                )

        logger.debug(
            "get_source_documents: %d entities → %d unique documents",
            len(entity_ids), len(documents)
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _name_score(self, query_name: str, entity_name: str) -> float:
        """Compute normalized name similarity."""
        if not query_name or not entity_name:
            return 0.0
        if _RAPIDFUZZ:
            return rapidfuzz_fuzz.token_ratio(query_name, entity_name) / 100.0
        else:
            q = query_name.lower().strip()
            e = entity_name.lower().strip()
            if q == e:
                return 1.0
            if q in e or e in q:
                return 0.75
            return 0.0
