"""
Household Detector
==================
Finds entities sharing the same address and links them
with a 'lives_with' edge — representing same-household relationships.

This is different from same_as (same person) —
lives_with means DIFFERENT people at the SAME address.

Examples:
- Two children and their parents at "123 Main St" → all live_with each other
- A person's license and birth cert have same address → same_as (same person)

Algorithm:
1. Collect all entities with an 'address' attribute
2. Normalize addresses (lowercase, strip punctuation)
3. Group by normalized address
4. For groups with 2+ DIFFERENT people (different names), add lives_with edges
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class HouseholdRecord:
    """A detected household — multiple people at the same address."""
    address:         str
    address_normalized: str
    members:         list[dict]   # list of {entity_id, name, doc_type, source_file}
    member_count:    int


def _normalize_address(address: str) -> str:
    """
    Normalize an address string for comparison.
    Removes punctuation, lowercases, strips zip code variations.

    Examples:
        "204 Maple Street, Vancouver, BC, V6B 2W9" → "204 maple street vancouver bc"
        "204 Maple St., Vancouver BC V6B2W9"       → "204 maple st vancouver bc"
    """
    if not address:
        return ""
    # Lowercase
    addr = address.lower()
    # Remove postal/zip codes (letter-digit patterns like V6B 2W9 or 85281-3885)
    addr = re.sub(r"\b[a-z]\d[a-z]\s?\d[a-z]\d\b", "", addr)  # Canadian postal
    addr = re.sub(r"\b\d{5}(?:-\d{4})?\b", "", addr)           # US zip
    # Remove punctuation except spaces
    addr = re.sub(r"[,.\-#]", " ", addr)
    # Collapse whitespace
    addr = re.sub(r"\s+", " ", addr).strip()
    # Normalize common abbreviations
    addr = addr.replace(" st ", " street ").replace(" ave ", " avenue ")
    addr = addr.replace(" rd ", " road ").replace(" dr ", " drive ")
    addr = addr.replace(" blvd ", " boulevard ").replace(" apt ", " apartment ")
    return addr


class HouseholdDetector:
    """
    Detects same-household relationships by matching addresses.

    Usage:
        detector = HouseholdDetector(graph)
        households = detector.detect()
        for h in households:
            print(f"Household at {h.address}: {[m['name'] for m in h.members]}")

        # Add lives_with edges to graph
        detector.add_lives_with_edges(households, graph_builder)
    """

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def detect(self, min_members: int = 2) -> list[HouseholdRecord]:
        """
        Find all households (groups of 2+ different people at same address).

        Args:
            min_members — minimum people to constitute a "household" (default 2)

        Returns:
            List of HouseholdRecord, sorted by member count descending
        """
        # Collect entities with addresses
        address_groups: dict[str, list[dict]] = {}

        for node_id, data in self._graph.nodes(data=True):
            if data.get("node_type") != "entity":
                continue

            attrs   = data.get("attributes", {}) or {}
            address = attrs.get("address", "")
            if not address:
                continue

            norm = _normalize_address(address)
            if not norm or len(norm) < 10:  # too short to be a real address
                continue

            member_info = {
                "entity_id":   node_id,
                "name":        data.get("name", "Unknown"),
                "doc_type":    data.get("doc_type", ""),
                "source_file": data.get("source_filename", ""),
                "address_raw": address,
            }
            address_groups.setdefault(norm, []).append(member_info)

        # Filter to groups with 2+ DIFFERENT people
        households = []
        for norm_addr, members in address_groups.items():
            # Deduplicate by name — same person shouldn't count twice
            seen_names = set()
            unique_members = []
            for m in members:
                name_key = m["name"].lower().strip()
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    unique_members.append(m)

            if len(unique_members) < min_members:
                continue

            # Use the most common raw address as display
            display_addr = members[0]["address_raw"]

            households.append(HouseholdRecord(
                address              = display_addr,
                address_normalized   = norm_addr,
                members              = unique_members,
                member_count         = len(unique_members),
            ))

        # Sort by member count descending
        households.sort(key=lambda h: h.member_count, reverse=True)

        logger.info(
            "HouseholdDetector: found %d households from %d address groups",
            len(households), len(address_groups)
        )
        return households

    def add_lives_with_edges(
        self,
        households: list[HouseholdRecord],
        graph_builder,
    ) -> int:
        """
        Add 'lives_with' edges between household members in the graph.

        Creates edges between every pair of members in each household.
        Does NOT create self-loops or duplicate edges.

        Returns number of edges added.
        """
        edges_added = 0

        for household in households:
            members = household.members
            # Add edge between every pair
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a = members[i]["entity_id"]
                    b = members[j]["entity_id"]

                    # Skip if already connected by same_as
                    existing = self._graph.edges.get((a, b)) or self._graph.edges.get((b, a))
                    if existing and existing.get("edge_type") == "same_as":
                        continue

                    # Add lives_with edge
                    if not self._graph.has_edge(a, b):
                        self._graph.add_edge(
                            a, b,
                            edge_type           = "lives_with",
                            address             = household.address,
                            address_normalized  = household.address_normalized,
                            confidence          = 0.95,
                        )
                        edges_added += 1

        logger.info("HouseholdDetector: added %d lives_with edges", edges_added)
        return edges_added
