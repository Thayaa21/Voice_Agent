"""
Context Aggregator — Step 13
==============================
Takes the documents found by traversal, deduplicates them, ranks them by
hybrid relevance score, and truncates to fit the LLM's context window.

TEACHING NOTES
--------------
Why aggregate and rank?
    After BFS traversal, we might have 5-20 source documents.
    But the LLM's context window is limited (4096 tokens for qwen2.5).
    We can't just dump all documents into the prompt — we'd exceed the limit.

    We need to:
    1. Deduplicate (same doc_id → same document)
    2. Rank by relevance (how related is this doc to the query?)
    3. Truncate (keep only the top docs that fit in the budget)

Hybrid relevance score:
    relevance = 0.4 × graph_centrality + 0.6 × cosine_similarity(query, doc)

    graph_centrality — how "important" this document is in the graph
        Uses PageRank from NetworkX. Documents mentioned by many entities
        (or connected to highly-connected entities) score higher.
        Normalized to [0, 1].

        TEACHING: PageRank was originally developed by Larry Page for Google.
        It measures how important a web page is by counting how many other
        important pages link to it. We use the same idea for documents.

    cosine_similarity — how semantically similar is the document to the query
        If the document's text embedding is close to the query embedding,
        it's likely to contain the answer.
        Falls back to 0.0 if doc has no embedding (not yet set).

Token budget:
    We estimate tokens as: token_count ≈ word_count / 0.75
    (1 token ≈ 0.75 words is a rough approximation)

    We keep adding documents (in relevance order) until we'd exceed the budget.
    This ensures the context fits within the LLM's limit.

Output format:
    We format the context as numbered passages:
    "Document 1 (birth_certificate.txt):\nFull text...\n\n"

    Numbered references let the LLM cite its sources:
    "According to Document 1 (birth_certificate.txt), Alice's DOB is..."
"""

import logging
from dataclasses import dataclass, field

import networkx as nx

from ..core.embeddings import EmbeddingEngine
from ..core.models import Document

logger = logging.getLogger(__name__)


@dataclass
class AggregatedContext:
    """
    The ranked, truncated context ready for the LLM.

    text      — formatted context string to include in the LLM prompt
    documents — the Document objects included (after ranking/truncation)
    scores    — relevance score for each included document (same order)
    """
    text:      str
    documents: list[Document]
    scores:    list[float]


class ContextAggregator:
    """
    Aggregates, ranks, and truncates source documents for LLM context.

    Usage:
        aggregator = ContextAggregator(token_budget=4096)
        agg_ctx = aggregator.aggregate(docs, question, query_embedding, graph)
        # agg_ctx.text → formatted context string for LLM prompt
        # agg_ctx.documents → list of included Document objects
    """

    def __init__(self, token_budget: int = 4096):
        """
        Args:
            token_budget — maximum total tokens for context (default 4096)
                           This is passed to the LLM as the full context limit.
                           We use 90% of it for documents, leaving room for
                           the question and system prompt.
        """
        self._token_budget = int(token_budget * 0.9)  # 90% for documents

    def aggregate(
        self,
        documents: list[Document],
        query: str,
        query_embedding: list[float],
        graph: nx.Graph,
    ) -> AggregatedContext:
        """
        Rank and truncate documents for the LLM context.

        Args:
            documents       — source documents found by traversal
            query           — the original query string
            query_embedding — 384-float embedding of the query
            graph           — the knowledge graph (for PageRank centrality)

        Returns:
            AggregatedContext with text, documents, and scores.
        """
        if not documents:
            return AggregatedContext(text="", documents=[], scores=[])

        # ---- Deduplicate by doc_id ----
        seen: set[str] = set()
        unique_docs: list[Document] = []
        for doc in documents:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                unique_docs.append(doc)

        # ---- Compute PageRank centrality ----
        centrality = self._compute_centrality(graph)

        # ---- Score each document ----
        scored: list[tuple[float, Document]] = []
        for doc in unique_docs:
            score = self._score_document(
                doc, query_embedding, centrality
            )
            scored.append((score, doc))

        # ---- Sort by relevance descending ----
        scored.sort(key=lambda x: x[0], reverse=True)

        # ---- Truncate to token budget ----
        selected_docs:   list[Document] = []
        selected_scores: list[float]    = []
        total_words = 0

        for score, doc in scored:
            doc_word_count = len(doc.text.split())
            # Approximate token count: 1 token ≈ 0.75 words
            doc_tokens = doc_word_count / 0.75

            if total_words + doc_tokens > self._token_budget and selected_docs:
                # Would exceed budget (but always include at least 1 doc)
                break

            selected_docs.append(doc)
            selected_scores.append(score)
            total_words += doc_tokens

        # ---- Format context ----
        context_text = self._format_context(selected_docs)

        logger.info(
            "ContextAggregator: %d docs → %d selected (%.0f tokens estimated)",
            len(unique_docs), len(selected_docs), total_words
        )

        return AggregatedContext(
            text      = context_text,
            documents = selected_docs,
            scores    = selected_scores,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_centrality(self, graph: nx.Graph) -> dict[str, float]:
        """
        Compute PageRank centrality for all graph nodes.

        Returns dict mapping node_id → normalized pagerank score [0, 1].

        TEACHING: PageRank assigns high scores to nodes that are linked
        to by many other nodes (or by nodes that are themselves highly linked).
        Documents that many entities "mention" will have higher centrality.

        We normalize to [0, 1] by dividing by the max score.
        """
        if graph.number_of_nodes() == 0:
            return {}

        try:
            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
            max_pr   = max(pagerank.values()) if pagerank else 1.0
            if max_pr == 0:
                max_pr = 1.0
            return {k: v / max_pr for k, v in pagerank.items()}
        except Exception as e:
            logger.warning("PageRank failed: %s — using uniform centrality", e)
            # Fallback: uniform centrality
            n = graph.number_of_nodes()
            uniform = 1.0 / n if n > 0 else 0.5
            return {node: uniform for node in graph.nodes()}

    def _score_document(
        self,
        doc: Document,
        query_embedding: list[float],
        centrality: dict[str, float],
    ) -> float:
        """
        Compute hybrid relevance score for a document.

        Score = 0.4 × graph_centrality + 0.6 × cosine_similarity(query, doc)
        """
        # Graph centrality (how connected this document is)
        graph_cent = centrality.get(doc.doc_id, 0.5)

        # Cosine similarity (how semantically relevant to the query)
        doc_emb = getattr(doc, "embedding", None)
        if doc_emb and query_embedding:
            # Compute cosine similarity manually (EmbeddingEngine might not be
            # available here, so we compute inline)
            import math
            dot = sum(a * b for a, b in zip(query_embedding, doc_emb))
            mag_q = math.sqrt(sum(x * x for x in query_embedding))
            mag_d = math.sqrt(sum(x * x for x in doc_emb))
            if mag_q > 0 and mag_d > 0:
                sem_score = max(0.0, min(1.0, dot / (mag_q * mag_d)))
            else:
                sem_score = 0.0
        else:
            sem_score = 0.0

        return 0.4 * graph_cent + 0.6 * sem_score

    def _format_context(self, documents: list[Document]) -> str:
        """
        Format documents as numbered passages for the LLM prompt.

        Format:
            Document 1 (birth_certificate.txt):
            [full text]

            Document 2 (drivers_license.txt):
            [full text]
        """
        parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            header = f"Document {i} ({doc.filename}):"
            parts.append(f"{header}\n{doc.text}")

        return "\n\n".join(parts)
