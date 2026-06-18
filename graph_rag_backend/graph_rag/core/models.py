"""
Data Models for Graph RAG
==========================

These are the core data structures that every component in the pipeline
passes around. Think of them as agreed-upon "contracts" between components.

TEACHING NOTES
--------------
@dataclass   — Python decorator that auto-generates __init__, __repr__, etc.
field()      — lets you set default values for dataclass fields
Optional[X]  — means the value can be X or None (i.e. it might not exist)
list[X]      — a list where every item is type X
dict         — a dictionary (key-value pairs, like JSON)
Enum         — a fixed set of named constants (e.g. DocType.PASSPORT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------------
# An Enum is a fixed set of named values.
# Instead of using raw strings like "birth_certificate" everywhere
# (which is error-prone — a typo silently breaks things), we use an Enum
# so Python will catch mistakes at the point of use.
# ---------------------------------------------------------------------------

class DocType(str, Enum):
    """
    The six document types the system can classify and extract from.

    Why inherit from str?
    Inheriting from str means DocType.PASSPORT == "PASSPORT" is True,
    so you can use it in JSON serialization without extra conversion.
    """
    BIRTH_CERTIFICATE = "BIRTH_CERTIFICATE"
    DRIVERS_LICENSE   = "DRIVERS_LICENSE"
    PASSPORT          = "PASSPORT"
    INSURANCE         = "INSURANCE"
    MEDICAL_RECORD    = "MEDICAL_RECORD"
    MEDICAL_REPORT    = "MEDICAL_REPORT"
    GENERIC           = "GENERIC"


class EdgeType(str, Enum):
    """
    The types of edges (relationships) that can exist in the knowledge graph.

    mentions     — an Entity was found in a Document
    same_as      — two Entities refer to the same real-world person
    conflict     — two same_as Entities have contradictory attribute values
    """
    MENTIONS = "mentions"
    SAME_AS  = "same_as"
    CONFLICT = "conflict"


class EntityType(str, Enum):
    """
    The types of entities the extractor can identify.
    """
    PERSON       = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION     = "LOCATION"
    ID_NUMBER    = "ID_NUMBER"
    DATE         = "DATE"


# ---------------------------------------------------------------------------
# DOCUMENT
# ---------------------------------------------------------------------------
# Represents a single ingested file.
# Created by: DocumentLoader
# Used by: EntityExtractor, KnowledgeGraphBuilder, ProvenanceTracker
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    A single ingested document (e.g. one .txt or .json file).

    Fields:
        doc_id      — Unique ID (UUID v4) assigned at load time
        filename    — Original file name, e.g. "birth_certificate.txt"
        text        — The full raw text content of the file
        lines       — text split by newline, e.g. ["CERTIFICATE OF BIRTH", ""]
        paragraphs  — text split by blank lines (double newline)
        line_offsets— char position where each line starts in `text`
                      e.g. line 0 starts at char 0, line 1 starts at char 22
        doc_type    — classified type (BIRTH_CERTIFICATE, etc.) — set after classification
        doc_date    — the document's date in ISO 8601 format, e.g. "1992-04-02"
                      used by the Temporal Filter for "current address" queries
        empty       — True if the file had no text content
        metadata    — any extra key-value data you want to store
    """
    doc_id:       str
    filename:     str
    text:         str
    lines:        list[str]
    paragraphs:   list[str]
    line_offsets: list[int]          # char index of each line's first character
    doc_type:     DocType            = DocType.GENERIC
    doc_date:     Optional[str]      = None
    empty:        bool               = False
    metadata:     dict               = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ENTITY
# ---------------------------------------------------------------------------
# Represents a named thing extracted from a document.
# Created by: LangChainExtractor or UiPathExtractor
# Used by: KnowledgeGraphBuilder, EntityResolver, ProvenanceTracker
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    A named entity extracted from a Document.

    Example: from a birth certificate we'd extract one PERSON entity:
        name       = "Alice Chen"
        entity_type = PERSON
        attributes  = {"dob": "1992-03-15", "place_of_birth": "Vancouver"}
        line_number = 5          ← line 5 of the source file
        line_text   = "Full Name: Alice Chen"  ← verbatim text of that line

    Provenance fields (the "verify" feature):
        line_number       — 1-indexed line in the source file
        line_text         — the exact verbatim text of that line
        paragraph_index   — which paragraph (0-indexed) it's in
        paragraph_text    — the full paragraph text
        char_offset_start — character position of the entity start in full text
        char_offset_end   — character position of the entity end in full text

    Extraction metadata:
        extractor_model   — which model extracted this: "langchain", "gpt-4o",
                            "uipath-document-understanding", etc.
        extraction_timestamp — when extraction happened (ISO 8601 UTC)
        confidence        — 0.0 to 1.0, how confident the extractor was

    Embedding:
        embedding         — a list of 384 floats (dense vector representation)
                            generated by sentence-transformers. Used for semantic
                            similarity matching during entity resolution.
    """
    # Identity
    entity_id:    str
    name:         str
    entity_type:  EntityType
    attributes:   dict               # e.g. {"dob": "1992-03-15", "license_number": "BC-7745291"}

    # Source
    source_doc_id:   str
    source_filename: str
    doc_type:        DocType

    # Provenance — where exactly in the document was this found?
    line_number:      int            = 0
    line_text:        str            = ""
    paragraph_index:  int            = 0
    paragraph_text:   str            = ""
    char_offset_start: int           = 0
    char_offset_end:   int           = 0

    # Extraction metadata
    extractor_model:       str       = ""
    extraction_timestamp:  str       = ""
    confidence:            float     = 1.0

    # Semantic embedding (set by EmbeddingEngine)
    embedding: Optional[list[float]] = field(default=None)


# ---------------------------------------------------------------------------
# RESOLVED PAIR
# ---------------------------------------------------------------------------
# Represents a decision by EntityResolver: "these two entities are the same person"
# Created by: EntityResolver
# Used by: KnowledgeGraphBuilder (to create same_as edges)
# ---------------------------------------------------------------------------

@dataclass
class ResolvedPair:
    """
    The result of EntityResolver deciding two entities refer to the same person.

    confidence    — combined score = 0.4 * name_score + 0.6 * semantic_score
    name_score    — RapidFuzz string similarity (0.0 to 1.0)
    semantic_score— cosine similarity of embedding vectors (0.0 to 1.0)
    llm_confirmed — True if the LLM was asked to confirm (borderline cases)
    valid_from    — ISO 8601 date: when did this link become valid?
                    set to the earlier document's doc_date
    """
    entity_id_a:   str
    entity_id_b:   str
    confidence:    float
    name_score:    float
    semantic_score: float
    llm_confirmed: bool
    valid_from:    Optional[str] = None
    valid_until:   Optional[str] = None  # None means "still valid / open-ended"


# ---------------------------------------------------------------------------
# CONFLICT RECORD
# ---------------------------------------------------------------------------
# Represents a detected contradiction between two same_as-linked entities.
# Created by: ContradictionDetector
# Used by: KnowledgeGraphBuilder (to create conflict edges), QueryEngine
# ---------------------------------------------------------------------------

@dataclass
class ConflictRecord:
    """
    A detected data conflict between two entities that were resolved as same_as.

    Example: Alice Chen's birth certificate says DOB = March 15, 1992
             Alice Chen's insurance record says DOB  = March 22, 1992
             → conflict_type = "dob_mismatch", severity = "critical"

    severity:
        "critical" — dob, name, license_number, passport_number, policy_number
        "minor"    — address, phone, email
    """
    entity_id_a:   str
    entity_id_b:   str
    conflict_type: str       # e.g. "dob_mismatch", "address_mismatch"
    attribute_key: str       # e.g. "dob"
    value_a:       str       # value from entity A's document
    value_b:       str       # value from entity B's document
    source_doc_a:  str       # filename of entity A's source
    source_doc_b:  str       # filename of entity B's source
    severity:      str       # "critical" | "minor"


# ---------------------------------------------------------------------------
# PROVENANCE ENTRY
# ---------------------------------------------------------------------------
# Links a specific fact in an answer back to the exact line it came from.
# Created by: ProvenanceTracker
# Used by: QueryEngine, REST API response, React ProvenanceList component
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceEntry:
    """
    The "verify" feature — every fact in the answer has one of these.

    Example:
        fact           = "dob: 1992-03-15"
        source_filename= "birth_certificate.txt"
        line_number    = 5
        line_text      = "Date of Birth: March 15, 1992"

    This is what the React frontend displays in the ProvenanceList:
        📄 birth_certificate.txt — Line 5
           "Date of Birth: March 15, 1992"
    """
    fact:             str    # e.g. "dob: 1992-03-15"
    source_filename:  str
    doc_type:         DocType
    line_number:      int    # 1-indexed line number in the source file
    line_text:        str    # verbatim exact line from the document
    paragraph_index:  int
    paragraph_text:   str
    char_offset_start: int
    char_offset_end:  int
    confidence:       float
    entity_id:        str


# ---------------------------------------------------------------------------
# QUERY RESULT
# ---------------------------------------------------------------------------
# The final output of the entire pipeline — what the user gets back.
# Created by: QueryEngine
# Used by: REST API (/query endpoint), CLI (query command), React AnswerDisplay
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """
    The complete response to a user's natural language question.

    question         — the original question asked
    answer           — the synthesized natural language answer
    source_documents — which files were used to construct the answer
    resolved_entities— the entity names that were linked across documents
    resolution_confidence — confidence score per same_as edge traversed
    hops_used        — how deep the multi-hop traversal went (1 = single doc)
    provenance       — list of ProvenanceEntry, one per fact in the answer
    conflicts        — any contradictions found in the traversed entities
    has_conflicts    — quick True/False flag (so frontend can show warning banner)
    temporal_context — "current", "all", or an ISO 8601 date string
    """
    question:              str
    answer:                str
    source_documents:      list[str]
    resolved_entities:     list[str]
    resolution_confidence: list[float]
    hops_used:             int
    provenance:            list[ProvenanceEntry]
    conflicts:             list[ConflictRecord]
    has_conflicts:         bool
    temporal_context:      str


# ---------------------------------------------------------------------------
# EXTRACTION SOURCE
# ---------------------------------------------------------------------------
# Passed to the ExtractionProvider to tell it what to process.
# Used by: LangChainExtractor, UiPathExtractor
# ---------------------------------------------------------------------------

@dataclass
class ExtractionSource:
    """
    Tells the ExtractionProvider what to process and in which mode.

    mode          — "langchain" (raw .txt) or "uipath" (structured .json)
    file_path     — path to the .txt file (langchain mode)
    uipath_json_path — path to the UiPath JSON file (uipath mode)
    """
    mode:              str
    file_path:         Optional[str] = None
    uipath_json_path:  Optional[str] = None
