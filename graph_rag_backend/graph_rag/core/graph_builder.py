"""
Knowledge Graph Builder — Step 8
==================================
Builds and maintains the in-memory NetworkX knowledge graph.
Every document and entity becomes a node; relationships become edges.

TEACHING NOTES
--------------
What is a Knowledge Graph?
    A knowledge graph is a network of nodes (things) connected by edges
    (relationships). Unlike a flat table or list, it explicitly represents
    HOW things are related.

    Our graph has 3 node types:
        document  — one per ingested file
        entity    — one per extracted person/thing

    And 3 edge types:
        mentions  — entity was found in this document
        same_as   — these two entities are the same real-world person
        conflict  — these same_as entities have contradictory data

Why NetworkX?
    NetworkX is a pure-Python graph library. No database setup, no
    installation of a separate server. The entire graph lives in memory.
    Perfect for this learning project — real production systems would
    use Neo4j or similar.

    Key NetworkX concepts:
        nx.Graph()     — undirected graph (edges have no direction)
        G.add_node()   — add a node with attributes
        G.add_edge()   — add an edge between two nodes with attributes
        G.nodes()      — get all nodes
        G.edges()      — get all edges
        G[node]        — get a node's attributes
        G[a][b]        — get an edge's attributes

Node IDs:
    We use the entity_id or doc_id as the node key in NetworkX.
    These are UUID strings — guaranteed unique across all nodes.
    Using the same ID means: "update this node if it already exists".

Edge multiplicity:
    NetworkX's basic Graph() allows only ONE edge between two nodes.
    We use the entity_id/doc_id pair as the node keys, so each
    (entity, document) pair has exactly one 'mentions' edge.

    For multiple edge types between same nodes (same_as + conflict),
    we rely on the edge_type attribute to distinguish them.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from .models import (
    ConflictRecord,
    Document,
    EdgeType,
    Entity,
    ResolvedPair,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds and manages the in-memory NetworkX knowledge graph.

    The graph accumulates documents and entities as you ingest files.
    After all documents are ingested, EntityResolver adds same_as edges,
    and ContradictionDetector adds conflict edges.

    Usage:
        builder = KnowledgeGraphBuilder()

        # Ingest step: add documents and entities
        doc_node_id    = builder.add_document(doc)
        entity_node_id = builder.add_entity(entity)

        # Resolution step: add same_as links
        builder.add_same_as_edge(id_a, id_b, resolved_pair)

        # Conflict step: add conflict links
        builder.add_conflict_edge(id_a, id_b, conflict_record)

        # Query step: get the graph
        G = builder.get_graph()
        stats = builder.stats()
    """

    def __init__(self):
        """
        Initialize an empty NetworkX graph.

        TEACHING: nx.Graph() creates an undirected graph.
        "Undirected" means edge(A, B) == edge(B, A).
        Same-as relationships are symmetric: if A is the same person as B,
        then B is the same person as A.
        """
        self._graph = nx.Graph()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_document(self, document: Document) -> str:
        """
        Add a Document as a node in the graph.

        Returns the node_id (= document.doc_id) for later reference.

        If the node already exists (same doc_id), its attributes are updated.

        Node attributes stored:
            node_type, doc_id, filename, doc_type, doc_date, text, embedding
        """
        node_id = document.doc_id

        self._graph.add_node(
            node_id,
            node_type  = "document",
            doc_id     = document.doc_id,
            filename   = document.filename,
            doc_type   = document.doc_type.value,
            doc_date   = document.doc_date,
            text       = document.text,
            # Embedding starts as None; EmbeddingEngine sets it later
            embedding  = None,
            metadata   = document.metadata,
        )

        logger.debug(
            "Added document node: %s (%s)", document.filename, document.doc_type.value
        )
        return node_id

    def add_entity(self, entity: Entity) -> str:
        """
        Add an Entity as a node in the graph, and add a 'mentions' edge
        connecting it to its source document.

        Returns the node_id (= entity.entity_id).

        TEACHING: The 'mentions' edge carries all the provenance data:
        line_number, line_text, char offsets. This way the provenance
        lives on the edge, not just on the entity node.
        """
        node_id = entity.entity_id

        # Add entity node with all its fields as attributes
        self._graph.add_node(
            node_id,
            node_type          = "entity",
            entity_id          = entity.entity_id,
            name               = entity.name,
            entity_type        = entity.entity_type.value,
            attributes         = entity.attributes,
            source_doc_id      = entity.source_doc_id,
            source_filename    = entity.source_filename,
            doc_type           = entity.doc_type.value,
            line_number        = entity.line_number,
            line_text          = entity.line_text,
            paragraph_index    = entity.paragraph_index,
            paragraph_text     = entity.paragraph_text,
            char_offset_start  = entity.char_offset_start,
            char_offset_end    = entity.char_offset_end,
            extractor_model    = entity.extractor_model,
            extraction_timestamp = entity.extraction_timestamp,
            confidence         = entity.confidence,
            embedding          = entity.embedding,
        )

        # Add 'mentions' edge: entity → source document
        # This edge carries all provenance data for this entity
        if entity.source_doc_id in self._graph:
            self._graph.add_edge(
                node_id,
                entity.source_doc_id,
                edge_type         = EdgeType.MENTIONS.value,
                line_number       = entity.line_number,
                line_text         = entity.line_text,
                paragraph_index   = entity.paragraph_index,
                paragraph_text    = entity.paragraph_text,
                char_offset_start = entity.char_offset_start,
                char_offset_end   = entity.char_offset_end,
            )
        else:
            logger.warning(
                "Entity %s references unknown doc_id %s — mentions edge not created",
                entity.name, entity.source_doc_id
            )

        logger.debug(
            "Added entity node: %s (%s) from %s",
            entity.name, entity.entity_type.value, entity.source_filename
        )
        return node_id

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_same_as_edge(
        self, id_a: str, id_b: str, pair: ResolvedPair
    ) -> None:
        """
        Add a 'same_as' edge between two entity nodes.

        The edge carries all resolution metadata: confidence scores,
        whether the LLM confirmed it, and validity window.

        TEACHING: same_as edges are what makes multi-hop traversal possible.
        The traversal follows these edges to collect all documents about
        the same person across different files.
        """
        if id_a not in self._graph:
            logger.warning("same_as: node %s not in graph — skipping", id_a)
            return
        if id_b not in self._graph:
            logger.warning("same_as: node %s not in graph — skipping", id_b)
            return

        self._graph.add_edge(
            id_a, id_b,
            edge_type      = EdgeType.SAME_AS.value,
            confidence     = pair.confidence,
            name_score     = pair.name_score,
            semantic_score = pair.semantic_score,
            llm_confirmed  = pair.llm_confirmed,
            valid_from     = pair.valid_from,
            valid_until    = pair.valid_until,
        )

        logger.debug(
            "Added same_as edge: %s ↔ %s (confidence=%.2f)",
            id_a[:8], id_b[:8], pair.confidence
        )

    def add_conflict_edge(
        self, id_a: str, id_b: str, conflict: ConflictRecord
    ) -> None:
        """
        Add a 'conflict' edge between two entity nodes.

        Conflict edges are added AFTER same_as edges when ContradictionDetector
        finds mismatching attributes in linked entities.

        TEACHING: A pair of entities can have BOTH a same_as edge (they're
        the same person) AND a conflict edge (they disagree on some attribute).
        The QueryEngine surfaces these conflicts in its response.

        NetworkX's Graph() only supports one edge between two nodes.
        If both same_as and conflict need to exist, we use edge attributes to
        distinguish them (or add the conflict data to the same edge).
        Here we add a separate 'conflict' edge for each conflict record by
        using a composite key approach — we store all conflicts in a list
        attribute on a single conflict edge.
        """
        if id_a not in self._graph:
            logger.warning("conflict: node %s not in graph — skipping", id_a)
            return
        if id_b not in self._graph:
            logger.warning("conflict: node %s not in graph — skipping", id_b)
            return

        # If a conflict edge already exists between these nodes,
        # append to its conflicts list instead of overwriting
        edge_key = (id_a, id_b)
        rev_key  = (id_b, id_a)

        existing = self._graph.edges.get(edge_key) or self._graph.edges.get(rev_key)

        if existing and existing.get("edge_type") == EdgeType.CONFLICT.value:
            # Append this conflict to the existing conflict list
            existing.setdefault("conflicts", []).append({
                "conflict_type": conflict.conflict_type,
                "attribute_key": conflict.attribute_key,
                "value_a":       conflict.value_a,
                "value_b":       conflict.value_b,
                "severity":      conflict.severity,
            })
        else:
            # Create new conflict edge
            self._graph.add_edge(
                id_a, id_b,
                edge_type      = EdgeType.CONFLICT.value,
                conflict_type  = conflict.conflict_type,
                attribute_key  = conflict.attribute_key,
                value_a        = conflict.value_a,
                value_b        = conflict.value_b,
                severity       = conflict.severity,
                source_doc_a   = conflict.source_doc_a,
                source_doc_b   = conflict.source_doc_b,
                conflicts      = [{
                    "conflict_type": conflict.conflict_type,
                    "attribute_key": conflict.attribute_key,
                    "value_a":       conflict.value_a,
                    "value_b":       conflict.value_b,
                    "severity":      conflict.severity,
                }],
            )

        logger.debug(
            "Added conflict edge: %s ↔ %s (%s, severity=%s)",
            id_a[:8], id_b[:8], conflict.conflict_type, conflict.severity
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_graph(self) -> nx.Graph:
        """Return the raw NetworkX graph for direct traversal."""
        return self._graph

    def get_entity_nodes(self) -> list[str]:
        """
        Return all entity node IDs (excludes document nodes).

        TEACHING: We filter by the 'node_type' attribute we stored when
        adding nodes. This is how NetworkX lets you distinguish node types.
        """
        return [
            node_id
            for node_id, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == "entity"
        ]

    def get_document_nodes(self) -> list[str]:
        """Return all document node IDs (excludes entity nodes)."""
        return [
            node_id
            for node_id, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == "document"
        ]

    def stats(self) -> dict:
        """
        Return a summary of the graph's current state.

        Returns:
            {
                "nodes":          total number of nodes (documents + entities),
                "edges":          total number of edges,
                "entities":       number of entity nodes,
                "documents":      number of document nodes,
                "same_as_edges":  number of same_as edges,
                "conflict_edges": number of conflict edges,
            }
        """
        entity_count   = len(self.get_entity_nodes())
        document_count = len(self.get_document_nodes())

        same_as_count  = sum(
            1 for _, _, d in self._graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_AS.value
        )
        conflict_count = sum(
            1 for _, _, d in self._graph.edges(data=True)
            if d.get("edge_type") == EdgeType.CONFLICT.value
        )

        return {
            "nodes":          self._graph.number_of_nodes(),
            "edges":          self._graph.number_of_edges(),
            "entities":       entity_count,
            "documents":      document_count,
            "same_as_edges":  same_as_count,
            "conflict_edges": conflict_count,
        }

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def export_json(self, path: str | Path) -> None:
        """
        Export the graph as JSON (nodes + edges) to a file.

        The JSON format:
        {
          "nodes": [{"id": "...", "type": "entity", ...attrs}],
          "edges": [{"source": "...", "target": "...", "type": "same_as", ...attrs}]
        }

        TEACHING: NetworkX's node_link_data() exports the graph to a dict
        that can be serialized to JSON. We use it for visualization and
        debugging. Note: numpy arrays (embeddings) are converted to lists.
        """
        path = Path(path)

        # Build JSON-safe representation
        # Embeddings are lists of floats — JSON-serializable
        nodes = []
        for node_id, attrs in self._graph.nodes(data=True):
            node_dict = {"id": node_id}
            for k, v in attrs.items():
                # Convert any non-serializable types
                if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                    node_dict[k] = v
                else:
                    node_dict[k] = str(v)
            nodes.append(node_dict)

        edges = []
        for source, target, attrs in self._graph.edges(data=True):
            edge_dict = {"source": source, "target": target}
            for k, v in attrs.items():
                if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                    edge_dict[k] = v
                else:
                    edge_dict[k] = str(v)
            edges.append(edge_dict)

        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "stats": self.stats(),
        }

        path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
        logger.info("Graph exported to %s (%d nodes, %d edges)", path, len(nodes), len(edges))

    def reset(self) -> None:
        """Reset the graph to empty state."""
        self._graph = nx.Graph()
        logger.info("Graph reset to empty state.")
