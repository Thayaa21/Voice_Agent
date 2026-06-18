"""
Query Engine — Step 15
========================
The main pipeline orchestrator for question answering.
Combines all components into a single query() method.

TEACHING NOTES
--------------
Pipeline overview (what happens when you ask a question):

    1. LLM extracts entity names from the question
       "What is Alice Chen's license number?" → ["Alice Chen"]

    2. EmbeddingEngine embeds the question
       → [384 floats representing the question's meaning]

    3. MultiHopTraversal.find_entities() matches names + embeddings to graph nodes
       → entity_ids of matching nodes

    4. MultiHopTraversal.expand() BFS via same_as edges
       → all entity_ids for that person across all documents

    5. TemporalFilter.filter_entities() keeps only relevant time window
       → filtered entity_ids

    6. MultiHopTraversal.get_source_documents() follows mentions edges
       → Document objects

    7. ContextAggregator.aggregate() ranks and truncates documents
       → AggregatedContext (text + document list + scores)

    8. ProvenanceTracker.extract_provenance() creates provenance entries
       → list[ProvenanceEntry]

    9. ContradictionDetector.detect() finds conflicts in filtered entities
       → list[ConflictRecord]

   10. LLM synthesizes an answer from the aggregated context
       → natural language answer string

   11. Return QueryResult with everything

Error handling:
    Each step handles failures gracefully. If step 3 finds no entities,
    we return early with "No matching entities found." rather than
    letting downstream steps fail on empty input.

LLM calls:
    We make two LLM calls per query:
    1. Extract entity names (cheap, short prompt)
    2. Synthesize answer (expensive, long context)

    Both use temperature=0.0 for consistency.
"""

import logging

import networkx as nx

from ..core.contradiction import ContradictionDetector
from ..core.embeddings import EmbeddingEngine
from ..core.models import (
    ConflictRecord,
    Document,
    ProvenanceEntry,
    QueryResult,
)
from ..core.temporal import TemporalFilter
from ..llm.provider import LLMProvider, LLMProviderError
from .aggregator import ContextAggregator
from .provenance import ProvenanceTracker
from .traversal import MultiHopTraversal

logger = logging.getLogger(__name__)

# Prompt to extract entity names from the question
_EXTRACT_NAMES_PROMPT = """\
Extract person names from the following question.
Return ONLY a comma-separated list of names. No explanation, no punctuation.
If no person names are found, return NONE.

Question: {question}

Names:"""

# System prompt for final answer synthesis
_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "Be specific and cite relevant information from the context. "
    "If the context does not contain enough information to answer, say so."
)


class QueryEngine:
    """
    Main query pipeline orchestrator.

    Accepts a natural language question and returns a QueryResult with:
    - answer: the synthesized natural language answer
    - provenance: exact source lines for every fact
    - conflicts: any contradictions found in the data
    - source_documents: which files were used
    - resolved_entities: which entity names were matched
    - hops_used: depth of BFS traversal

    Usage:
        engine = QueryEngine(graph, llm, embedding_engine, documents)
        result = engine.query("What is Alice Chen's license number?")
        print(result.answer)
        for p in result.provenance:
            print(f"  {p.fact} ← {p.source_filename} line {p.line_number}")
    """

    def __init__(
        self,
        graph: nx.Graph,
        llm_provider: LLMProvider,
        embedding_engine: EmbeddingEngine,
        documents: dict[str, Document],
        token_budget: int = 4096,
    ):
        """
        Args:
            graph            — the knowledge graph (built by KnowledgeGraphBuilder)
            llm_provider     — LLM for name extraction and answer synthesis
            embedding_engine — for embedding the query and computing similarity
            documents        — dict mapping doc_id → Document object
            token_budget     — max tokens for context window (default 4096)
        """
        self._graph            = graph
        self._llm              = llm_provider
        self._embedding_engine = embedding_engine
        self._documents        = documents

        self._temporal_filter   = TemporalFilter(graph)
        self._traversal         = MultiHopTraversal(graph, embedding_engine, self._temporal_filter)
        self._aggregator        = ContextAggregator(token_budget=token_budget)
        self._provenance_tracker = ProvenanceTracker(graph, documents)

    def query(
        self,
        question: str,
        max_hops: int = 3,
        temporal_context: str = "current",
    ) -> QueryResult:
        """
        Answer a natural language question using the knowledge graph.

        Args:
            question         — the question to answer
            max_hops         — BFS depth for entity expansion (1–5)
            temporal_context — "current", "all", or ISO date string

        Returns:
            QueryResult with answer, provenance, conflicts, etc.
        """
        logger.info("QueryEngine: question='%s'", question[:100])

        # ---- Step 1: Extract entity names from question ----
        names = self._extract_names(question)
        logger.debug("Extracted names: %s", names)

        # ---- Step 2: Embed the question ----
        query_embedding = self._embedding_engine.embed(question)

        # ---- Step 3: Find matching entities ----
        matched_ids = self._traversal.find_entities(
            names=names,
            query_embedding=query_embedding,
            threshold=0.20,  # Very permissive — substring match boosts above this
        )

        # Fallback: if no match, try semantic-only with very low threshold
        if not matched_ids:
            logger.info("No matches with names — falling back to semantic-only search")
            matched_ids = self._traversal.find_entities(
                names=[],
                query_embedding=query_embedding,
                threshold=0.10,
            )

        if not matched_ids:
            logger.info("QueryEngine: no matching entities found.")
            return QueryResult(
                question              = question,
                answer                = "No matching entities found.",
                source_documents      = [],
                resolved_entities     = [],
                resolution_confidence = [],
                hops_used             = 0,
                provenance            = [],
                conflicts             = [],
                has_conflicts         = False,
                temporal_context      = temporal_context,
            )

        # ---- Step 4: Expand via same_as edges (BFS) ----
        expanded_ids = self._traversal.expand(matched_ids, max_hops=max_hops)
        hops_used    = max_hops  # We attempted max_hops

        # ---- Step 5: Temporal filtering ----
        filtered_ids = self._temporal_filter.filter_entities(
            expanded_ids, temporal_context
        )

        # If temporal filtering removed everything, fall back to all expanded
        if not filtered_ids:
            logger.debug("Temporal filter removed all entities — using expanded_ids")
            filtered_ids = expanded_ids

        # ---- Step 6: Get source documents ----
        source_docs = self._traversal.get_source_documents(filtered_ids)

        # ---- Step 7: Aggregate context ----
        agg_context = self._aggregator.aggregate(
            documents       = source_docs,
            query           = question,
            query_embedding = query_embedding,
            graph           = self._graph,
        )

        # ---- Step 8: Extract provenance ----
        provenance_entries = self._provenance_tracker.extract_provenance(
            filtered_ids
        )

        # ---- Step 9: Detect contradictions ----
        all_conflicts = ContradictionDetector(self._graph).detect()
        # Filter to only conflicts involving our filtered entities
        filtered_entity_set = set(filtered_ids)
        relevant_conflicts  = [
            c for c in all_conflicts
            if c.entity_id_a in filtered_entity_set
            or c.entity_id_b in filtered_entity_set
        ]

        # ---- Step 10: Synthesize answer ----
        if agg_context.text:
            answer = self._synthesize_answer(question, agg_context.text)
        else:
            answer = "No relevant context found to answer this question."

        # ---- Step 11: Build QueryResult ----
        # Collect entity names from matched nodes
        resolved_entities: list[str] = []
        resolution_confidence: list[float] = []

        for entity_id in filtered_ids:
            node_data = self._graph.nodes.get(entity_id, {})
            name = node_data.get("name", "")
            if name:
                resolved_entities.append(name)

        # Get confidence scores from same_as edges
        for u, v, data in self._graph.edges(data=True):
            if (data.get("edge_type") == "same_as"
                    and u in filtered_entity_set and v in filtered_entity_set):
                resolution_confidence.append(float(data.get("confidence", 0.0)))

        return QueryResult(
            question              = question,
            answer                = answer,
            source_documents      = [doc.filename for doc in agg_context.documents],
            resolved_entities     = resolved_entities,
            resolution_confidence = resolution_confidence,
            hops_used             = hops_used,
            provenance            = provenance_entries,
            conflicts             = relevant_conflicts,
            has_conflicts         = len(relevant_conflicts) > 0,
            temporal_context      = temporal_context,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_names(self, question: str) -> list[str]:
        """
        Use the LLM to extract person names from the question.

        Returns list of name strings, or [] if none found.
        """
        prompt = _EXTRACT_NAMES_PROMPT.format(question=question)

        try:
            response = self._llm.complete(prompt, temperature=0.0)
            cleaned  = response.strip()

            if not cleaned or cleaned.upper() == "NONE":
                return []

            # Split on commas, clean each name
            names = [n.strip() for n in cleaned.split(",") if n.strip()]
            return names

        except LLMProviderError as e:
            logger.warning("LLM name extraction failed: %s — using empty names", e)
            return []
        except Exception as e:
            logger.warning("Name extraction error: %s — using empty names", e)
            return []

    def _synthesize_answer(self, question: str, context: str) -> str:
        """
        Use the LLM to synthesize a natural language answer from the context.

        Uses chat() with a system message for better answer quality.
        Falls back to complete() if chat() is unavailable.
        """
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        try:
            return self._llm.chat(messages, temperature=0.0)
        except LLMProviderError as e:
            logger.warning("LLM synthesis failed: %s", e)
            return f"Error synthesizing answer: {e}"
        except Exception as e:
            logger.warning("Unexpected synthesis error: %s", e)
            return f"Error synthesizing answer: {e}"
