"""
Provenance Tracker — Step 14
==============================
Links every extracted fact back to the exact line in its source document.
This is the "verify" feature: every answer can be traced to its origin.

TEACHING NOTES
--------------
Why provenance?
    When an LLM says "Alice's DOB is 1992-03-15", users need to know:
    - Which document did this come from?
    - What EXACTLY does the document say?
    - Is the document trustworthy?

    Provenance answers these questions by linking each fact to a specific
    line in a specific document. The ProvenanceList in the React frontend
    shows this as:
        📄 birth_certificate.txt — Line 5
           "Date of Birth: March 15, 1992"

How it works:
    1. For each entity, we have its graph node attributes including:
       - line_number (1-indexed)
       - line_text (verbatim line from source)
       - attributes dict {"dob": "1992-03-15", ...}
       - source_filename

    2. For each attribute in the entity, we create a ProvenanceEntry:
       - fact = "dob: 1992-03-15"
       - source_filename = "birth_certificate.txt"
       - line_number = 5
       - line_text = "Date of Birth: March 15, 1992"

    3. We optionally VERIFY the line_text against the actual document:
       - Get the actual line from document.lines[line_number - 1]
       - Check if it matches entity.line_text
       - If mismatch: log warning (LLM may have misquoted the line)

The verify() search function:
    Given a fact string, search all entity attributes for it.
    Used when the user queries "verify: DOB 1992-03-15" to find all
    places this fact appears.

Line number indexing:
    Our entities use 1-indexed line numbers (like text editors).
    Python lists are 0-indexed.
    So document.lines[line_number - 1] is the correct access.
    Special case: line_number = 0 means "not from a specific line"
    (e.g. UiPath entities — skip line verification).
"""

import logging
from typing import Optional

import networkx as nx

from ..core.models import Document, DocType, EntityType, ProvenanceEntry

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Extracts and verifies provenance for entity attributes.

    Usage:
        tracker = ProvenanceTracker(graph, documents)

        # Get provenance for all attributes of given entities
        entries = tracker.extract_provenance(entity_ids)

        # Search for a specific fact string
        entries = tracker.verify("dob: 1992-03-15")
    """

    def __init__(
        self,
        graph: nx.Graph,
        documents: dict[str, Document],
    ):
        """
        Args:
            graph     — NetworkX graph with entity nodes
            documents — dict mapping doc_id → Document object
        """
        self._graph     = graph
        self._documents = documents

    def extract_provenance(
        self, entity_ids: list[str]
    ) -> list[ProvenanceEntry]:
        """
        Create a ProvenanceEntry for each attribute of each entity.

        For each entity:
        1. Get its node attributes (line_number, line_text, attributes dict)
        2. For each key-value pair in attributes, create a ProvenanceEntry
        3. Optionally verify line_text against the actual document line

        Args:
            entity_ids — list of entity node IDs

        Returns:
            List of ProvenanceEntry objects (may be empty if entities have
            no attributes or if entity IDs are not in the graph)
        """
        entries: list[ProvenanceEntry] = []

        for entity_id in entity_ids:
            node_data = self._graph.nodes.get(entity_id, {})
            if not node_data or node_data.get("node_type") != "entity":
                continue

            entity_entries = self._extract_entity_provenance(entity_id, node_data)
            entries.extend(entity_entries)

        logger.debug(
            "extract_provenance: %d entities → %d provenance entries",
            len(entity_ids), len(entries)
        )
        return entries

    def verify(self, fact_str: str) -> list[ProvenanceEntry]:
        """
        Search all entity attributes for fact_str and return matching ProvenanceEntry objects.

        Case-insensitive substring match.

        Args:
            fact_str — the fact to verify, e.g. "dob: 1992-03-15" or "Alice Chen"

        Returns:
            List of ProvenanceEntry objects where fact contains fact_str
        """
        all_entity_ids = [
            node_id
            for node_id, data in self._graph.nodes(data=True)
            if data.get("node_type") == "entity"
        ]

        all_entries = self.extract_provenance(all_entity_ids)

        fact_lower = fact_str.lower()
        matching = [
            entry for entry in all_entries
            if fact_lower in entry.fact.lower()
        ]

        logger.debug(
            "verify('%s'): found %d matching entries from %d total",
            fact_str, len(matching), len(all_entries)
        )
        return matching

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_entity_provenance(
        self, entity_id: str, node_data: dict
    ) -> list[ProvenanceEntry]:
        """
        Create ProvenanceEntry objects for one entity's attributes.
        """
        entries: list[ProvenanceEntry] = []

        attributes     = node_data.get("attributes", {}) or {}
        source_doc_id  = node_data.get("source_doc_id", "")
        source_filename = node_data.get("source_filename", "")
        line_number    = int(node_data.get("line_number", 0))
        line_text      = str(node_data.get("line_text", ""))
        paragraph_index = int(node_data.get("paragraph_index", 0))
        paragraph_text  = str(node_data.get("paragraph_text", ""))
        char_offset_start = int(node_data.get("char_offset_start", 0))
        char_offset_end   = int(node_data.get("char_offset_end", 0))
        confidence        = float(node_data.get("confidence", 1.0))

        # Get doc_type
        doc_type_str = node_data.get("doc_type", "GENERIC")
        try:
            doc_type = DocType(doc_type_str)
        except ValueError:
            doc_type = DocType.GENERIC

        # Verify line_text against actual document
        # (skip verification for UiPath entities with line_number=0)
        verified_line_text = line_text
        if line_number > 0 and source_doc_id in self._documents:
            doc = self._documents[source_doc_id]
            if 1 <= line_number <= len(doc.lines):
                actual_line = doc.lines[line_number - 1]
                if actual_line != line_text and line_text:
                    logger.debug(
                        "Line text mismatch for entity %s at line %d: "
                        "graph=%r, doc=%r — using doc version",
                        entity_id[:8], line_number, line_text[:50], actual_line[:50]
                    )
                verified_line_text = actual_line

        # Create one ProvenanceEntry per attribute
        for attr_key, attr_value in attributes.items():
            fact = f"{attr_key}: {attr_value}"

            entries.append(ProvenanceEntry(
                fact             = fact,
                source_filename  = source_filename,
                doc_type         = doc_type,
                line_number      = line_number,
                line_text        = verified_line_text,
                paragraph_index  = paragraph_index,
                paragraph_text   = paragraph_text,
                char_offset_start = char_offset_start,
                char_offset_end   = char_offset_end,
                confidence        = confidence,
                entity_id         = entity_id,
            ))

        return entries
