"""
Contradiction Detector — Step 10
==================================
After entity resolution, scans every same_as-linked pair of entities for
attribute value mismatches and records them as ConflictRecords.

TEACHING NOTES
--------------
Why detect contradictions?
    When Alice Chen appears in her birth certificate AND her insurance record,
    the data SHOULD agree. But real-world data is messy:
    - Typos: "1992-03-15" vs "1992-03-25"
    - Format differences: "March 15 1992" vs "1992-03-15"
    - Genuine updates: old address vs new address
    - Errors: someone entered the wrong DOB on one form

    The Contradiction Detector finds these conflicts so the QueryEngine can
    warn the user: "⚠ CONFLICT: Two documents disagree on Alice's date of birth."

Algorithm:
    1. Find all same_as edges in the graph
    2. For each linked pair (entity_a, entity_b):
       a. Get their attributes dicts
       b. Find keys that exist in BOTH (shared attributes)
       c. Normalize values (strip whitespace, lowercase, normalize dates)
       d. Compare normalized values
       e. If they differ: create a ConflictRecord

Severity levels:
    CRITICAL: changes to identity-defining fields (dob, name, ID numbers)
              These are high-stakes — a DOB mismatch could mean fraud
    MINOR:    changes to less critical fields (address, phone)
              These could be legitimate updates (moved house)

Date normalization with dateutil:
    The same date appears in many formats:
        "March 15, 1992"  ←  natural language
        "15/03/1992"      ←  European format
        "03/15/1992"      ←  American format
        "1992-03-15"      ←  ISO 8601 (our preferred format)

    dateutil.parser.parse() recognizes all these formats and converts to
    a datetime object. We then format as YYYY-MM-DD for comparison.

    If parsing fails (not a date field), we fall back to string comparison.

Key insight:
    A conflict edge is ONLY added when two entities already have a same_as edge.
    We never flag two random entities as conflicting — only confirmed same-person pairs.
"""

import logging
from typing import Optional

import networkx as nx

try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    logging.warning(
        "python-dateutil not installed — date normalization disabled. "
        "Install with: pip install python-dateutil"
    )

from .models import ConflictRecord, EdgeType

logger = logging.getLogger(__name__)

# ---- Severity classification ----
# Fields where a mismatch is CRITICAL (identity-defining, high fraud risk)
CRITICAL_KEYS = {
    "dob",
    "name",
    "license_number",
    "passport_number",
    "policy_number",
    "registration_number",
    "patient_name",
}

# Fields that CAN legitimately differ (e.g. old address vs new address)
# Minor conflicts are shown in yellow, not red
MINOR_KEYS = {
    "address",
    "phone",
    "email",
}

# These fields differ by design across document types — NOT conflicts
# (a passport has a different expiry than a license; that's expected)
SKIP_COMPARISON_KEYS = {
    "parents",
    "place_of_birth",
    "place_of_issue",
    "nationality",
    "registration_number",   # BC-specific, no counterpart in other docs
    "patient_name",          # alias for 'name', redundant
    "_person_folder",
    "_extractor",
    # Document-lifecycle fields — intentionally different per doc type
    "expiry_date",
    "issue_date",
    "start_date",
    "vehicle_class",
    "coverage_type",
    "premium",
    "beneficiary",
    "doctor",
    "diagnosis",
    "medications",
    "procedures",
    "visit_date",
}


class ContradictionDetector:
    """
    Detects attribute conflicts between same_as-linked entity pairs.

    Usage:
        detector = ContradictionDetector(graph)
        conflicts = detector.detect()
        for conflict in conflicts:
            graph_builder.add_conflict_edge(
                conflict.entity_id_a, conflict.entity_id_b, conflict
            )
    """

    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph — NetworkX graph containing entity nodes and same_as edges
        """
        self._graph = graph

    def detect(self) -> list[ConflictRecord]:
        """
        Find all attribute conflicts between same_as-linked entity pairs.

        Returns:
            List of ConflictRecord objects (may be empty if no conflicts found)

        Never raises — errors in individual comparisons are logged and skipped.
        """
        conflicts: list[ConflictRecord] = []

        # Find all same_as edges
        same_as_edges = [
            (u, v, data)
            for u, v, data in self._graph.edges(data=True)
            if data.get("edge_type") == EdgeType.SAME_AS.value
        ]

        if not same_as_edges:
            logger.debug("ContradictionDetector: no same_as edges found.")
            return []

        logger.debug(
            "ContradictionDetector: scanning %d same_as edges for conflicts",
            len(same_as_edges)
        )

        for node_a, node_b, _edge_data in same_as_edges:
            try:
                pair_conflicts = self._compare_pair(node_a, node_b)
                conflicts.extend(pair_conflicts)
            except Exception as e:
                logger.warning(
                    "Error detecting conflicts for %s vs %s: %s — skipping",
                    node_a[:8], node_b[:8], e
                )

        logger.info(
            "ContradictionDetector: found %d conflict(s) in %d same_as pairs",
            len(conflicts), len(same_as_edges)
        )
        return conflicts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compare_pair(
        self, node_a: str, node_b: str
    ) -> list[ConflictRecord]:
        """
        Compare attributes of two same_as-linked entities.

        Returns a list of ConflictRecord for each attribute mismatch.
        """
        data_a = self._graph.nodes.get(node_a, {})
        data_b = self._graph.nodes.get(node_b, {})

        # Get attributes dicts (default to empty if missing)
        attrs_a: dict = data_a.get("attributes", {}) or {}
        attrs_b: dict = data_b.get("attributes", {}) or {}

        # Find shared keys (fields that appear in BOTH entities)
        shared_keys = set(attrs_a.keys()) & set(attrs_b.keys())

        if not shared_keys:
            return []

        # Get source filenames for ConflictRecord
        source_a = data_a.get("source_filename", "unknown")
        source_b = data_b.get("source_filename", "unknown")
        entity_id_a = data_a.get("entity_id", node_a)
        entity_id_b = data_b.get("entity_id", node_b)

        conflicts: list[ConflictRecord] = []

        for key in sorted(shared_keys):
            # Skip document-metadata fields that are never meaningful to compare
            if key in SKIP_COMPARISON_KEYS or key.startswith("_"):
                continue

            # Only flag conflicts on keys we explicitly classify as critical or minor
            # Unrecognised keys (document-type-specific) are silently skipped
            if key not in CRITICAL_KEYS and key not in MINOR_KEYS:
                continue

            raw_a = str(attrs_a[key]).strip()
            raw_b = str(attrs_b[key]).strip()

            # Skip empty values — a missing field is not a conflict
            if not raw_a or not raw_b:
                continue

            # Normalize values before comparing
            norm_a = self._normalize(raw_a)
            norm_b = self._normalize(raw_b)

            if norm_a == norm_b:
                # Values agree — no conflict
                continue

            # Values disagree — create a ConflictRecord
            severity = "critical" if key in CRITICAL_KEYS else "minor"
            conflict_type = f"{key}_mismatch"

            logger.debug(
                "CONFLICT: %s vs %s | key=%s | '%s' != '%s' | severity=%s",
                source_a, source_b, key, norm_a, norm_b, severity
            )

            conflicts.append(ConflictRecord(
                entity_id_a    = entity_id_a,
                entity_id_b    = entity_id_b,
                conflict_type  = conflict_type,
                attribute_key  = key,
                value_a        = raw_a,   # Store original (not normalized) for display
                value_b        = raw_b,
                source_doc_a   = source_a,
                source_doc_b   = source_b,
                severity       = severity,
            ))

        return conflicts

    def _normalize(self, value: str) -> str:
        """
        Normalize an attribute value for comparison.

        For names: strips middle initials so "Mei Lee" == "Mei N. Lee".
        For dates: normalizes all formats to YYYY-MM-DD.
        For everything else: lowercase, strip punctuation noise.
        """
        import re

        if not value:
            return ""

        # Try date parsing first
        if DATEUTIL_AVAILABLE:
            try:
                dt = dateutil_parser.parse(value, fuzzy=False)
                return dt.strftime("%Y-%m-%d")
            except (ValueError, OverflowError, TypeError):
                pass

        normalized = value.strip().lower()

        # ---- Name normalization: strip middle names/initials ----
        # "Mei N. Lee" → "Mei Lee"
        # "James Robert Lee" → "James Lee"
        # Strategy: keep first token and last token, drop everything in between
        tokens = normalized.split()
        if len(tokens) >= 3:
            # Keep only first and last token (strip middle name/initial)
            normalized = f"{tokens[0]} {tokens[-1]}"
        elif len(tokens) == 2:
            # Remove single-letter initials: "j. smith" → "smith" (keep both if not initial)
            if len(tokens[0]) == 1 or tokens[0].rstrip('.') == tokens[0][0]:
                normalized = tokens[1]
            else:
                normalized = ' '.join(tokens)
        # else: single token, keep as-is

        # Remove remaining periods, commas, hyphens
        normalized = re.sub(r'[.,\-]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized
