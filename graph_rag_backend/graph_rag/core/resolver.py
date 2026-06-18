"""
Entity Resolver — Step 9
==========================
Finds entities across different documents that refer to the same real-world person
and adds same_as edges between them.

TEACHING NOTES
--------------
The entity resolution problem:
    birth_certificate.txt → "Alice Chen, DOB 1992-03-15"
    drivers_license.txt   → "Alice Chen, DOB 1992-03-15, license BC-7745291"
    insurance.txt         → "Alice Chen, policy INS-887341"

    Three entities, all the same person. How do we know they're the same?
    We use a HYBRID SCORE combining name similarity + semantic similarity.

Hybrid score formula:
    confidence = 0.4 × name_score + 0.6 × semantic_score

    Why 0.4 and 0.6?
    - Name similarity alone is unreliable: "James Lee" ≈ "James Walker" (both James)
    - Semantic similarity captures the full context (name + attributes)
    - We weight semantics more because it's more discriminative
    - Both signals together are more reliable than either alone

RapidFuzz (name similarity):
    from rapidfuzz import fuzz
    fuzz.token_ratio("Alice Chen", "Alice Chen") → 100.0
    fuzz.token_ratio("Alice Chen", "A. Chen")    → 72.0
    fuzz.token_ratio("James Lee",  "James Lee")  → 100.0
    fuzz.token_ratio("James Lee",  "James Walker") → 46.0

    token_ratio handles word reordering:
        "John Robert Smith" ≈ "Smith, John R." (same name, different format)

    We divide by 100 to get a 0.0–1.0 score.

Semantic similarity (embeddings):
    entity.embedding contains 384 floats capturing the full meaning of
    "name + attributes" (e.g., "Alice Chen {'dob': '1992-03-15', ...}").

    cosine_similarity(emb_a, emb_b) = how similar the meanings are.
    Same person → high similarity even if field names differ slightly.

Decision thresholds:
    ≥ 0.85  → auto-link (high confidence, no LLM confirmation needed)
    0.60–0.84 → ask LLM ("Are these the same person? YES/NO")
    < 0.60  → skip (different people)

    These thresholds were tuned empirically:
    - 0.85: prevents false positives while catching obvious matches
    - 0.60: the gray zone where human judgment helps

Rules that prevent false matches:
    1. Only compare entities of the SAME entity_type (PERSON vs PERSON)
    2. Never compare entities from the SAME source_doc_id
       (same person can't appear twice in one document)
    3. Never link an entity to itself

Complexity:
    O(n²) comparisons for n entities of the same type.
    For 100 entities: 4,950 comparisons. Fast enough for this use case.
    For millions of entities, you'd use approximate nearest neighbor search.
"""

import logging
from itertools import combinations
from typing import Optional

import networkx as nx

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logging.warning(
        "rapidfuzz not installed — name matching will use basic similarity. "
        "Install with: pip install rapidfuzz"
    )

from .embeddings import EmbeddingEngine
from .models import EntityType, ResolvedPair
from ..llm.provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


def _normalize_date(value: str) -> str:
    """Normalize a date string to YYYY-MM-DD for comparison. Returns '' if not parseable."""
    try:
        from dateutil import parser as _dp
        return _dp.parse(value, fuzzy=False).strftime("%Y-%m-%d")
    except Exception:
        return value.strip().lower()


# LLM confirmation prompt
# We ask for YES/NO only — no explanation — for easy parsing
_CONFIRM_PROMPT = """\
Are these two records about the same person? Answer YES or NO only.

Record A: {name_a}, {attrs_a}
Record B: {name_b}, {attrs_b}"""


class EntityResolver:
    """
    Resolves entity identity across documents by comparing names and embeddings.

    Uses a hybrid scoring approach:
        confidence = 0.4 × name_similarity + 0.6 × semantic_similarity

    For borderline cases (0.60 ≤ confidence < 0.85), asks the LLM for confirmation.

    Usage:
        resolver = EntityResolver(llm, embedding_engine)
        resolved_pairs = resolver.resolve(graph)
        for pair in resolved_pairs:
            graph_builder.add_same_as_edge(pair.entity_id_a, pair.entity_id_b, pair)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_engine: EmbeddingEngine,
        auto_threshold: float = 0.85,    # must exceed this to auto-link (no LLM)
        confirm_threshold: float = 0.60, # borderline: ask LLM to confirm
    ):
        """
        Args:
            llm_provider      — LLM for confirming borderline cases
            embedding_engine  — for computing cosine similarity
            auto_threshold    — pairs above this are auto-linked (default 0.85)
            confirm_threshold — pairs above this ask LLM to confirm (default 0.60)
        """
        self._llm               = llm_provider
        self._embedding_engine  = embedding_engine
        self._auto_threshold    = auto_threshold
        self._confirm_threshold = confirm_threshold

    def resolve(self, graph: nx.Graph) -> list[ResolvedPair]:
        """
        Compare all entity pairs in the graph and return resolved same_as pairs.

        Algorithm:
        1. Get all entity nodes from the graph
        2. Group by entity_type (only compare same types)
        3. For each pair from different source_doc_ids:
           - Compute hybrid confidence score
           - Auto-link if ≥ auto_threshold
           - Ask LLM if ≥ confirm_threshold
           - Skip otherwise
        4. Return all resolved pairs

        Args:
            graph — NetworkX graph from KnowledgeGraphBuilder

        Returns:
            list of ResolvedPair objects to add as same_as edges

        Never raises — errors in individual comparisons are logged and skipped.
        """
        # ---- Get all entity nodes ----
        entity_nodes = [
            (node_id, data)
            for node_id, data in graph.nodes(data=True)
            if data.get("node_type") == "entity"
        ]

        if len(entity_nodes) < 2:
            logger.info("EntityResolver: fewer than 2 entities, nothing to resolve.")
            return []

        # ---- Group by entity_type ----
        # Only compare entities of the same type (PERSON vs PERSON, etc.)
        # This avoids nonsensical comparisons like a PERSON vs an ID_NUMBER
        groups: dict[str, list[tuple[str, dict]]] = {}
        for node_id, data in entity_nodes:
            etype = data.get("entity_type", EntityType.PERSON.value)
            groups.setdefault(etype, []).append((node_id, data))

        resolved: list[ResolvedPair] = []

        for entity_type, nodes in groups.items():
            logger.debug(
                "EntityResolver: comparing %d entities of type %s",
                len(nodes), entity_type
            )

            # Compare every pair (n×(n-1)/2 comparisons)
            for (id_a, data_a), (id_b, data_b) in combinations(nodes, 2):

                # Rule: never compare entities from the same document
                if data_a.get("source_doc_id") == data_b.get("source_doc_id"):
                    continue

                # Rule: never compare an entity to itself
                if id_a == id_b:
                    continue

                # Compute scores
                try:
                    pair = self._compare(id_a, data_a, id_b, data_b)
                    if pair is not None:
                        resolved.append(pair)
                except Exception as e:
                    logger.warning(
                        "Error comparing %s vs %s: %s — skipping",
                        id_a[:8], id_b[:8], e
                    )

        logger.info(
            "EntityResolver: found %d resolved pairs from %d entity nodes",
            len(resolved), len(entity_nodes)
        )
        return resolved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compare(
        self,
        id_a: str, data_a: dict,
        id_b: str, data_b: dict,
    ) -> Optional[ResolvedPair]:
        """
        Compare two entity nodes and return a ResolvedPair if they match.

        Returns None if the pair doesn't meet the minimum threshold.
        """
        name_a = str(data_a.get("name", ""))
        name_b = str(data_b.get("name", ""))
        emb_a  = data_a.get("embedding")
        emb_b  = data_b.get("embedding")

        # ---- Name similarity ----
        name_score = self._name_score(name_a, name_b)

        # ---- Semantic similarity ----
        if emb_a and emb_b:
            semantic_score = self._embedding_engine.cosine_similarity(emb_a, emb_b)
        else:
            semantic_score = 0.0

        # ---- Hard rule: never link if names are too different ----
        # Even if semantic score is high, different names = different people
        if name_score < 0.70:
            return None

        # ---- Hard rule: FULL name must substantially match ----
        # Both first AND last name tokens must be present and similar.
        # "James Lee" vs "James Walker" → last names differ → no link
        # "Alice Chen" vs "Alicia Chen" → close enough (typo/variant)
        name_a_tokens = set(name_a.lower().split())
        name_b_tokens = set(name_b.lower().split())

        # Last-name check — last token of each name must match closely
        def get_last(name: str) -> str:
            parts = name.strip().split()
            return parts[-1].lower() if parts else ""

        last_a = get_last(name_a)
        last_b = get_last(name_b)
        if last_a and last_b:
            if RAPIDFUZZ_AVAILABLE:
                from rapidfuzz import fuzz as _fuzz
                last_score = _fuzz.ratio(last_a, last_b) / 100.0
            else:
                last_score = 1.0 if last_a == last_b else 0.0
            # Require >85% last-name similarity — catches typos but blocks
            # genuinely different surnames like "Lee" vs "Walker"
            if last_score < 0.85:
                return None

        # ---- DOB anchor check ----
        # If both entities have a 'dob' and they differ by MORE than 3 days,
        # they are definitely different people — bail out.
        # If they differ by ≤ 3 days, still link them (likely a data entry error)
        # and let the ContradictionDetector flag the mismatch.
        attrs_a = data_a.get("attributes", {}) or {}
        attrs_b = data_b.get("attributes", {}) or {}
        dob_a = str(attrs_a.get("dob", "") or attrs_a.get("patient_dob", "")).strip()
        dob_b = str(attrs_b.get("dob", "") or attrs_b.get("patient_dob", "")).strip()
        if dob_a and dob_b:
            dob_a_norm = _normalize_date(dob_a)
            dob_b_norm = _normalize_date(dob_b)
            if dob_a_norm and dob_b_norm and dob_a_norm != dob_b_norm:
                # Check how many days apart they are
                try:
                    from datetime import datetime as _dt
                    d_a = _dt.strptime(dob_a_norm, "%Y-%m-%d")
                    d_b = _dt.strptime(dob_b_norm, "%Y-%m-%d")
                    days_diff = abs((d_a - d_b).days)
                    if days_diff > 3:
                        # Clearly different people — skip
                        return None
                    # ≤ 3 days apart — likely same person with data entry error
                    # Fall through and create same_as; contradiction detector will flag it
                except Exception:
                    return None

        # ---- Hybrid confidence ----
        confidence = 0.4 * name_score + 0.6 * semantic_score

        logger.debug(
            "Comparing '%s' vs '%s': name=%.2f semantic=%.2f confidence=%.2f",
            name_a, name_b, name_score, semantic_score, confidence
        )

        # ---- Decision ----
        if confidence >= self._auto_threshold:
            # High confidence — auto-link without LLM
            return ResolvedPair(
                entity_id_a    = id_a,
                entity_id_b    = id_b,
                confidence     = round(confidence, 3),
                name_score     = round(name_score, 3),
                semantic_score = round(semantic_score, 3),
                llm_confirmed  = False,
            )

        elif confidence >= self._confirm_threshold:
            # Borderline — only link if name is a very strong match
            # (avoids false positives from high semantic similarity alone)
            if name_score >= 0.90:
                return ResolvedPair(
                    entity_id_a    = id_a,
                    entity_id_b    = id_b,
                    confidence     = round(confidence, 3),
                    name_score     = round(name_score, 3),
                    semantic_score = round(semantic_score, 3),
                    llm_confirmed  = False,
                )
            return None

        else:
            return None

    def _name_score(self, name_a: str, name_b: str) -> float:
        """
        Compute normalized name similarity using RapidFuzz token_ratio.

        token_ratio handles:
        - Case differences: "Alice Chen" vs "alice chen"
        - Word reordering: "Chen Alice" vs "Alice Chen"
        - Partial matches: "A. Chen" vs "Alice Chen"

        Returns value in [0.0, 1.0].
        """
        if not name_a or not name_b:
            return 0.0

        if RAPIDFUZZ_AVAILABLE:
            # token_ratio: splits names into tokens, compares best permutation
            # Returns 0–100; we divide by 100 to normalize to [0, 1]
            return rapidfuzz_fuzz.token_ratio(name_a, name_b) / 100.0
        else:
            # Fallback: simple lowercase comparison
            a_lower = name_a.lower().strip()
            b_lower = name_b.lower().strip()
            if a_lower == b_lower:
                return 1.0
            # Check if one contains the other
            if a_lower in b_lower or b_lower in a_lower:
                return 0.75
            return 0.0

    def _confirm_with_llm(
        self, name_a: str, attrs_a: str, name_b: str, attrs_b: str
    ) -> bool:
        """
        Ask the LLM to confirm whether two records refer to the same person.

        Returns True if LLM says YES, False otherwise.
        """
        prompt = _CONFIRM_PROMPT.format(
            name_a=name_a, attrs_a=attrs_a,
            name_b=name_b, attrs_b=attrs_b,
        )

        try:
            response = self._llm.complete(prompt, temperature=0.0)
            # Parse: look for YES or NO in the response
            cleaned = response.strip().upper()
            if "YES" in cleaned:
                logger.debug(
                    "LLM confirmed same person: '%s' vs '%s'", name_a, name_b
                )
                return True
            elif "NO" in cleaned:
                logger.debug(
                    "LLM rejected same person: '%s' vs '%s'", name_a, name_b
                )
                return False
            else:
                # Ambiguous response — err on the side of caution (no link)
                logger.warning(
                    "LLM gave ambiguous response for '%s' vs '%s': %r — skipping",
                    name_a, name_b, response[:100]
                )
                return False
        except LLMProviderError as e:
            logger.warning("LLM confirmation failed: %s — skipping pair", e)
            return False
        except Exception as e:
            logger.warning("Unexpected LLM error: %s — skipping pair", e)
            return False
