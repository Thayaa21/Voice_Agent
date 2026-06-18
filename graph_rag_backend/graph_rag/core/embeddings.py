"""
Embedding Engine — Step 7
==========================
Converts text into dense vector representations (embeddings) using
sentence-transformers. These vectors capture semantic meaning, letting us
measure similarity between entities even when they use different wording.

TEACHING NOTES
--------------
What is an embedding?
    An embedding is a list of numbers (a vector) that represents the
    *meaning* of a piece of text. Texts with similar meanings produce
    similar vectors. Texts with different meanings produce very different vectors.

    Example:
        "Alice Chen, born 1992-03-15"  → [0.23, -0.41, 0.87, ...]  (384 numbers)
        "Alice Chen, DOB March 1992"   → [0.24, -0.40, 0.86, ...]  (similar!)
        "John Smith, born 1988-07-04"  → [-0.31, 0.22, -0.44, ...] (very different)

Why sentence-transformers?
    The `all-MiniLM-L6-v2` model:
    - Generates 384-dimensional vectors (384 floats per text)
    - Runs 100% locally — no API key, no internet required
    - Processes ~1000 sentences per second on a CPU
    - Is specifically trained for semantic similarity tasks

    It's perfect for comparing entity names + attributes across documents.

What is cosine similarity?
    Cosine similarity measures the angle between two vectors, NOT their distance.
    - 1.0 = identical direction = same meaning
    - 0.0 = perpendicular = unrelated
    - -1.0 = opposite direction = opposite meaning

    We use it to compare:
    - Two entity embeddings: are they the same person?
    - Query embedding vs document embedding: is this doc relevant?

SHA-256 caching:
    Generating embeddings takes ~1ms each. For a dataset of 1000 entities
    with the same name, you'd recompute the same embedding 1000 times.
    We cache by SHA-256 hash of the input text to avoid redundant computation.

    hashlib.sha256("Alice Chen".encode()).hexdigest()
    → "a1b2c3..." (a 64-char hex string, unique to this exact text)

Fallback behavior:
    If sentence-transformers is not installed, we log a warning and return
    zero vectors (384 zeros). This lets the pipeline run in environments
    where sentence-transformers can't be installed, with degraded similarity.
"""

import hashlib
import logging
import math
from typing import Optional

from .models import Entity

logger = logging.getLogger(__name__)

# Dimension of all-MiniLM-L6-v2 embeddings
EMBEDDING_DIM = 384


class EmbeddingEngine:
    """
    Manages text embeddings using sentence-transformers.

    Loads the model lazily (on first use) to keep startup fast.
    Caches embeddings by SHA-256 hash of the input text.

    Usage:
        engine = EmbeddingEngine()

        # Embed a single string
        vector = engine.embed("Alice Chen, born 1992-03-15")  # 384 floats

        # Embed multiple strings efficiently (batch processing)
        vectors = engine.embed_batch(["Alice Chen", "John Smith"])

        # Measure semantic similarity
        score = engine.cosine_similarity(vector_a, vector_b)  # 0.0 to 1.0

        # Embed an entity (uses name + attributes)
        entity_vec = engine.embed_entity(entity)

        # Embed all entities in a list (modifies entity.embedding in place)
        engine.embed_entities(entities)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingEngine.

        Args:
            model_name — sentence-transformers model identifier.
                         Default: "all-MiniLM-L6-v2" (384-dimensional, fast)

        TEACHING: We don't load the model here. We load it lazily
        (on first call to embed()) so that importing this module is fast.
        Model loading takes ~1 second and downloads ~100MB on first use.
        """
        self._model_name = model_name
        self._model = None          # loaded lazily in _get_model()
        self._model_available = None  # None = untested, True/False = known

        # Cache: sha256_hex → list[float]
        # Prevents re-embedding the same text multiple times
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        """
        Embed a single string into a 384-float vector.

        Uses SHA-256 hash for caching — identical text returns cached result
        without calling the model again.

        Args:
            text — any string to embed

        Returns:
            list of 384 floats (the embedding vector)

        Never raises — falls back to zero vector if model unavailable.
        """
        # ---- Compute cache key ----
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # ---- Return cached result if available ----
        if cache_key in self._cache:
            return self._cache[cache_key]

        # ---- Check model availability ----
        model = self._get_model()
        if model is None:
            # Fallback: return zeros — similarity will be 0.0 for all pairs
            zero_vec = [0.0] * EMBEDDING_DIM
            self._cache[cache_key] = zero_vec
            return zero_vec

        # ---- Generate embedding ----
        try:
            # sentence_transformers returns numpy arrays; we convert to Python list
            # encode() accepts a string or list of strings
            embedding = model.encode(text, convert_to_numpy=True)
            result = embedding.tolist()
        except Exception as e:
            logger.error("Error generating embedding: %s — returning zeros", e)
            result = [0.0] * EMBEDDING_DIM

        # ---- Cache and return ----
        self._cache[cache_key] = result
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings efficiently.

        Filters out already-cached texts and batches the rest into a single
        model.encode() call (much faster than calling embed() in a loop).

        Args:
            texts — list of strings to embed

        Returns:
            list of 384-float vectors, in the same order as input texts
        """
        if not texts:
            return []

        model = self._get_model()

        # ---- Separate cached from uncached ----
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # ---- Batch-embed uncached texts ----
        if uncached_texts and model is not None:
            try:
                # batch_size=64: process 64 texts at a time (memory vs speed tradeoff)
                embeddings = model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    batch_size=64,
                    show_progress_bar=False,
                )
                for idx, emb in zip(uncached_indices, embeddings):
                    vec = emb.tolist()
                    # Cache each result
                    cache_key = hashlib.sha256(
                        texts[idx].encode("utf-8")
                    ).hexdigest()
                    self._cache[cache_key] = vec
                    results[idx] = vec
            except Exception as e:
                logger.error("Batch embed error: %s — using zeros", e)
                for idx in uncached_indices:
                    results[idx] = [0.0] * EMBEDDING_DIM
        elif uncached_texts:
            # Model unavailable — fill with zeros
            for idx in uncached_indices:
                results[idx] = [0.0] * EMBEDDING_DIM

        # At this point all results should be filled in
        return [r if r is not None else [0.0] * EMBEDDING_DIM for r in results]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Formula:
            cosine_similarity(a, b) = dot(a, b) / (|a| × |b|)

        Returns:
            float in [0.0, 1.0] — where:
                1.0 = identical direction (same meaning)
                0.0 = perpendicular (unrelated)
                (We clamp to [0.0, 1.0] since negative similarities
                 are theoretically possible but rare in practice for text)

        TEACHING: Why cosine and not Euclidean distance?
            Cosine similarity only measures the DIRECTION of vectors,
            not their magnitude. Two sentences with the same meaning
            but different lengths produce vectors pointing in the same
            direction but with different magnitudes.
            Cosine handles this correctly; Euclidean distance does not.
        """
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            logger.warning("Embedding dimension mismatch: %d vs %d", len(a), len(b))
            return 0.0

        # Dot product: sum(a[i] * b[i])
        dot = sum(x * y for x, y in zip(a, b))

        # Magnitudes: sqrt(sum(x^2))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0

        # Clamp to [0.0, 1.0] — small numerical errors can push past 1.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def embed_entity(self, entity: Entity) -> list[float]:
        """
        Embed an Entity using its name and attributes combined.

        The embedding represents "who this entity is": their name plus
        all their key-value attributes as a natural language string.

        Example:
            entity.name = "Alice Chen"
            entity.attributes = {"dob": "1992-03-15", "license_number": "BC-7745291"}
            → embed("Alice Chen {'dob': '1992-03-15', 'license_number': 'BC-7745291'}")

        TEACHING: Including attributes makes the embedding much more
        discriminative. Two people named "James Lee" have similar name
        embeddings, but their combined name+attributes embeddings will
        differ if their DOBs or document types differ.
        """
        text = f"{entity.name} {str(entity.attributes)}"
        return self.embed(text)

    def embed_entities(self, entities: list[Entity]) -> None:
        """
        Embed all entities in a list and set entity.embedding in place.

        Uses batch processing for efficiency.
        Modifies each entity's embedding field directly.

        Args:
            entities — list of Entity objects to embed

        Side effect:
            Sets entity.embedding = list[float] for each entity in the list.

        TEACHING: "in place" means we modify the objects that were passed in.
        The caller doesn't need to assign the return value.
        Python passes objects by reference — changing entity.embedding here
        changes it everywhere the same entity object is used.
        """
        if not entities:
            return

        # Build list of text strings to batch-embed
        texts = [
            f"{entity.name} {str(entity.attributes)}"
            for entity in entities
        ]

        # Batch embed (much faster than calling embed() in a loop)
        vectors = self.embed_batch(texts)

        # Set embedding on each entity in place
        for entity, vector in zip(entities, vectors):
            entity.embedding = vector

        logger.info(
            "Embedded %d entities (cache size now: %d)",
            len(entities), len(self._cache)
        )

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the embedding cache (useful for testing or memory management)."""
        self._cache.clear()
        logger.debug("Embedding cache cleared.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """
        Lazily load the sentence-transformers model.

        Returns the model on success, None if not available.

        TEACHING: Lazy loading means we only load the model when actually
        needed. This keeps import time fast and avoids loading the model
        in tests that mock embeddings.

        We remember if the model is unavailable (_model_available = False)
        to avoid retrying the import on every call.
        """
        # Already loaded
        if self._model is not None:
            return self._model

        # Already known to be unavailable
        if self._model_available is False:
            return None

        # Try to load
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading sentence-transformers model: %s", self._model_name
            )
            self._model = SentenceTransformer(self._model_name)
            self._model_available = True
            logger.info(
                "Embedding model loaded. Dimension: %d", EMBEDDING_DIM
            )
            return self._model
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Embeddings will be zero vectors. "
                "Install with: pip install sentence-transformers"
            )
            self._model_available = False
            return None
        except Exception as e:
            logger.warning(
                "Failed to load embedding model '%s': %s — "
                "embeddings will be zero vectors.",
                self._model_name, e
            )
            self._model_available = False
            return None
