"""
Temporal Filter — Step 11
===========================
Filters entity lists based on temporal context — useful for queries like
"What is Alice's CURRENT address?" where the most recent document wins.

TEACHING NOTES
--------------
Why temporal filtering?
    A person may have multiple documents from different dates:
        drivers_license_2015.txt  doc_date = 2015-06-10  address = "100 Oak Ave"
        drivers_license_2022.txt  doc_date = 2022-03-01  address = "204 Maple Street"

    The query "What is Alice's current address?" should return "204 Maple Street"
    — the most recent one — not both.

    Without temporal filtering, the system would return conflicting addresses
    and confuse the user.

Three temporal context modes:
    "current"     — for each same_as group, keep only the entity with the
                    latest doc_date. Entities without doc_date are excluded
                    (they can't be compared temporally).

    "all"         — return all entity_ids unchanged, regardless of date.
                    Used when you want the complete history.

    ISO date str  — e.g. "2020-01-01" — return entities whose doc_date <= that date.
                    "What was Alice's address as of 2020?"

How we determine groups:
    We use the graph's same_as edges to find which entities are linked.
    Within each connected component of same_as edges, we pick the most recent entity.

    TEACHING: A "connected component" is a set of nodes where every node
    can reach every other node via edges. All Alice Chen entities form one
    connected component via same_as edges.

Date comparison:
    We parse doc_date strings as dates for proper chronological comparison.
    String comparison of "2022-03-01" > "2015-06-10" happens to work for ISO
    dates, but we use proper date parsing for correctness.

    If doc_date is None or unparseable, the entity is:
    - Included in "all" mode
    - Excluded in "current" mode (we can't determine if it's the latest)
    - Excluded in ISO date mode (we can't determine if it predates the cutoff)
"""

import logging
from datetime import date, datetime
from typing import Any, Optional

import networkx as nx

from .models import EdgeType

logger = logging.getLogger(__name__)


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    Try to parse a date string into a date object.
    Returns None if parsing fails.

    TEACHING: We try ISO format first (most common in our dataset),
    then fall back to dateutil for other formats.
    """
    if not date_str:
        return None

    # Try ISO 8601 format first (YYYY-MM-DD)
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        pass

    # Try dateutil for more formats
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(date_str, fuzzy=False).date()
    except Exception:
        pass

    return None


class TemporalFilter:
    """
    Filters entity lists by temporal context.

    Usage:
        tf = TemporalFilter(graph)

        # Get only most recent entities for each same_as group
        current_ids = tf.filter_entities(entity_ids, "current")

        # Get all entities
        all_ids = tf.filter_entities(entity_ids, "all")

        # Get entities valid as of 2020
        historical_ids = tf.filter_entities(entity_ids, "2020-01-01")

        # Get the most recent value for an attribute
        entity_id, value = tf.get_most_recent(entity_ids, "address")
    """

    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph — NetworkX graph with entity nodes and same_as edges
        """
        self._graph = graph

    def filter_entities(
        self,
        entity_ids: list[str],
        temporal_context: str = "current",
    ) -> list[str]:
        """
        Filter a list of entity IDs based on temporal context.

        Args:
            entity_ids       — list of entity node IDs to filter
            temporal_context — "current", "all", or an ISO date string

        Returns:
            Filtered list of entity IDs

        "current" mode:
            Groups entity_ids by their same_as connected component.
            Within each group, keeps only the entity with the latest doc_date.
            Entities without doc_date are excluded from "current" results.

        "all" mode:
            Returns all entity_ids unchanged.

        ISO date string (e.g. "2020-01-01"):
            Returns entities whose doc_date <= that date.
            Entities without doc_date are excluded.
        """
        if not entity_ids:
            return []

        if temporal_context == "all":
            return list(entity_ids)

        if temporal_context == "current":
            return self._filter_current(entity_ids)

        # Try to parse as an ISO date cutoff
        cutoff_date = _parse_date(temporal_context)
        if cutoff_date is not None:
            return self._filter_by_date(entity_ids, cutoff_date)

        # Unknown mode — log warning and return all
        logger.warning(
            "Unknown temporal_context: %r — returning all entities", temporal_context
        )
        return list(entity_ids)

    def get_most_recent(
        self, entity_ids: list[str], attribute: str
    ) -> tuple[str, Any]:
        """
        Find the entity with the latest doc_date and return its attribute value.

        TEACHING: This is used to answer "current" queries for specific fields.
        "What is Alice's most recent address?" → find entity with max doc_date
        → return its 'address' attribute.

        Args:
            entity_ids — list of entity node IDs to search
            attribute  — the attribute key to retrieve

        Returns:
            (entity_id, value) where entity_id has the latest doc_date
            Returns ("", None) if no entity has the attribute or a doc_date.
        """
        best_id    = ""
        best_date  = None
        best_value = None

        for entity_id in entity_ids:
            node_data = self._graph.nodes.get(entity_id, {})
            attrs = node_data.get("attributes", {}) or {}

            if attribute not in attrs:
                continue

            # Get doc_date from the source document
            doc_date = self._get_entity_doc_date(entity_id, node_data)
            parsed   = _parse_date(doc_date)

            if parsed is None:
                continue

            if best_date is None or parsed > best_date:
                best_date  = parsed
                best_id    = entity_id
                best_value = attrs[attribute]

        return (best_id, best_value)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_current(self, entity_ids: list[str]) -> list[str]:
        """
        For each same_as connected component, keep only the most recent entity.

        TEACHING: We use a Union-Find-like approach:
        1. Build adjacency from same_as edges among the given entity_ids
        2. Find connected components (groups of same-person entities)
        3. Within each group, pick the entity with the latest doc_date
        """
        entity_id_set = set(entity_ids)

        # ---- Build groups via same_as edges ----
        # Start: each entity is in its own group
        parent: dict[str, str] = {eid: eid for eid in entity_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union entities connected by same_as edges
        for u, v, data in self._graph.edges(data=True):
            if (data.get("edge_type") == EdgeType.SAME_AS.value
                    and u in entity_id_set and v in entity_id_set):
                union(u, v)

        # ---- Group entities by component ----
        groups: dict[str, list[str]] = {}
        for eid in entity_ids:
            root = find(eid)
            groups.setdefault(root, []).append(eid)

        # ---- Pick most recent from each group ----
        result: list[str] = []
        for group in groups.values():
            best = self._pick_most_recent(group)
            if best:
                result.append(best)

        logger.debug(
            "TemporalFilter (current): %d entities → %d after filtering",
            len(entity_ids), len(result)
        )
        return result

    def _filter_by_date(
        self, entity_ids: list[str], cutoff: date
    ) -> list[str]:
        """
        Return entities whose doc_date is <= the cutoff date.
        Entities without a doc_date are excluded.
        """
        result = []
        for entity_id in entity_ids:
            node_data  = self._graph.nodes.get(entity_id, {})
            doc_date   = self._get_entity_doc_date(entity_id, node_data)
            parsed     = _parse_date(doc_date)

            if parsed is None:
                continue  # No date — exclude in date-specific mode

            if parsed <= cutoff:
                result.append(entity_id)

        return result

    def _pick_most_recent(self, entity_ids: list[str]) -> Optional[str]:
        """
        From a group of entity_ids, return the one with the latest doc_date.
        Returns None if no entity in the group has a doc_date.
        """
        best_id   = None
        best_date = None

        for entity_id in entity_ids:
            node_data = self._graph.nodes.get(entity_id, {})
            doc_date  = self._get_entity_doc_date(entity_id, node_data)
            parsed    = _parse_date(doc_date)

            if parsed is None:
                continue

            if best_date is None or parsed > best_date:
                best_date = parsed
                best_id   = entity_id

        # If no entity has a date, return None (excluded from "current" mode)
        return best_id

    def _get_entity_doc_date(
        self, entity_id: str, node_data: dict
    ) -> Optional[str]:
        """
        Get the doc_date for an entity by looking at its source document node.

        The entity node stores source_doc_id. We follow that to the document
        node and get its doc_date attribute.

        Falls back to checking if the entity itself has a doc_date attribute.
        """
        # Check source document
        source_doc_id = node_data.get("source_doc_id")
        if source_doc_id and source_doc_id in self._graph:
            doc_data = self._graph.nodes.get(source_doc_id, {})
            doc_date = doc_data.get("doc_date")
            if doc_date:
                return doc_date

        # Fallback: check entity's own attributes
        attrs = node_data.get("attributes", {}) or {}
        return attrs.get("issue_date") or attrs.get("date") or attrs.get("dob")
