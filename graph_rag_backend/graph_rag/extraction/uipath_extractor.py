"""
UiPath Extractor — Step 6
==========================
Parses UiPath Document Understanding JSON output into Document + Entity objects.

TEACHING NOTES
--------------
UiPath vs LangChain mode:
    LangChain mode: raw .txt → LLM reads it → extracts entities (slow, flexible)
    UiPath mode:    .json → parse directly  → entities in milliseconds (fast, structured)

    UiPath Document Understanding is a commercial product that processes
    scanned/printed documents with computer vision, giving us pixel-level
    bounding boxes instead of character offsets.

Bounding box → char offsets:
    UiPath returns bounding boxes: [x1, y1, x2, y2] in pixels.
    Our Entity model uses char_offset_start / char_offset_end (character positions).
    
    For UiPath entities we store:
        char_offset_start = bounding_box[0]   (left edge x1)
        char_offset_end   = bounding_box[2]   (right edge x2)
    
    This is a convention — the char offsets have pixel semantics for UiPath entities.
    The ProvenanceTracker will skip line verification for entities with line_number=0.

Why create a fake Document?
    All downstream components expect a Document object. When given a .json file
    we create a Document where text = pretty-printed JSON content.
    This means the graph still works — document nodes and mentions edges are valid.

Fallback on malformed JSON:
    If the file is missing, unreadable, or has bad JSON, we return (None, []).
    The pipeline logs a warning and skips the file without crashing.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.models import Document, DocType, Entity, EntityType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DOCUMENT TYPE MAPPING
# ---------------------------------------------------------------------------
# Map UiPath's document_type strings to our DocType enum.
# UiPath uses uppercase with underscores — so does our enum — easy mapping.

_UIPATH_DOCTYPE_MAP: dict[str, DocType] = {
    "BIRTH_CERTIFICATE": DocType.BIRTH_CERTIFICATE,
    "DRIVERS_LICENSE":   DocType.DRIVERS_LICENSE,
    "DRIVER_LICENSE":    DocType.DRIVERS_LICENSE,   # alternate spelling UiPath uses
    "PASSPORT":          DocType.PASSPORT,
    "INSURANCE":         DocType.INSURANCE,
    "MEDICAL_RECORD":    DocType.MEDICAL_RECORD,
    "GENERIC":           DocType.GENERIC,
}

# ---------------------------------------------------------------------------
# FIELD NAME → PRIMARY NAME KEY
# ---------------------------------------------------------------------------
# For each document type, which field holds the person's name?
# This determines entity.name.

_PRIMARY_NAME_FIELD: dict[str, str] = {
    "BIRTH_CERTIFICATE": "name",
    "DRIVERS_LICENSE":   "name",
    "DRIVER_LICENSE":    "name",
    "PASSPORT":          "name",
    "INSURANCE":         "name",
    "MEDICAL_RECORD":    "patient_name",
    "GENERIC":           "name",
}


class UiPathExtractor:
    """
    Parses UiPath Document Understanding JSON files into (Document, list[Entity]).

    Usage:
        extractor = UiPathExtractor()
        doc, entities = extractor.extract("docs/people/alice_chen/birth_certificate.json")
        for entity in entities:
            print(entity.name, entity.attributes)
            print(f"  Bounding box x: {entity.char_offset_start} → {entity.char_offset_end}")
    """

    def extract(
        self, json_path: str | Path
    ) -> tuple[Optional[Document], list[Entity]]:
        """
        Parse a UiPath JSON file and return (Document, [Entity]).

        Args:
            json_path — path to the .json file produced by UiPath

        Returns:
            (Document, list[Entity]) on success
            (None, []) on failure (file not found, malformed JSON, etc.)

        Never raises — all errors are caught and logged.

        TEACHING: Returning (None, []) instead of raising an exception
        lets the pipeline continue processing other files even if one fails.
        The caller checks for None and skips gracefully.
        """
        json_path = Path(json_path)

        # ---- Guard: file must exist ----
        if not json_path.exists():
            logger.warning("UiPath JSON not found, skipping: %s", json_path)
            return None, []

        if not json_path.is_file():
            logger.warning("Not a file, skipping: %s", json_path)
            return None, []

        # ---- Read and parse JSON ----
        try:
            raw_text = json_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Cannot read file %s: %s — skipping", json_path, e)
            return None, []
        except json.JSONDecodeError as e:
            logger.warning("Malformed JSON in %s: %s — skipping", json_path, e)
            return None, []

        # ---- Validate top-level structure ----
        if not isinstance(data, dict):
            logger.warning("Expected JSON object in %s — skipping", json_path)
            return None, []

        # ---- Build Document ----
        doc = self._build_document(data, json_path, raw_text)

        # ---- Build Entities ----
        entities = self._build_entities(data, doc, json_path)

        logger.info(
            "UiPath extracted: %s → %d entities (doc_type=%s)",
            json_path.name, len(entities), doc.doc_type.value
        )

        return doc, entities

    def extract_batch(
        self, json_paths: list[str | Path]
    ) -> list[tuple[Optional[Document], list[Entity]]]:
        """
        Extract from multiple UiPath JSON files.

        Returns a list of (Document, [Entity]) tuples in the same order
        as the input paths. Failed files return (None, []).
        """
        results = []
        for path in json_paths:
            result = self.extract(path)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_document(
        self, data: dict, json_path: Path, raw_text: str
    ) -> Document:
        """
        Build a Document object from the parsed JSON data.

        TEACHING: Even though this is a structured JSON file (not raw text),
        we still create a Document. This keeps the graph builder happy —
        every entity needs a document node to attach to via a 'mentions' edge.

        The document text is set to the raw JSON string so ProvenanceTracker
        can still work with it (though line_number=0 means it skips verification).
        """
        # Detect document type
        raw_doc_type = str(data.get("document_type", "GENERIC")).upper().strip()
        doc_type = _UIPATH_DOCTYPE_MAP.get(raw_doc_type, DocType.GENERIC)

        # Extract document confidence
        doc_confidence = data.get("confidence", 1.0)

        # Build "text" as the raw JSON for provenance storage
        text = raw_text

        # Split into lines (for Document invariants)
        lines = text.split("\n")
        line_offsets: list[int] = []
        pos = 0
        for line in lines:
            line_offsets.append(pos)
            pos += len(line) + 1

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Try to find a doc_date from the fields
        doc_date = self._extract_doc_date(data.get("fields", {}))

        return Document(
            doc_id       = str(uuid.uuid4()),
            filename     = json_path.name,
            text         = text,
            lines        = lines,
            paragraphs   = paragraphs,
            line_offsets = line_offsets,
            doc_type     = doc_type,
            doc_date     = doc_date,
            empty        = False,
            metadata     = {
                "full_path":       str(json_path.resolve()),
                "file_size":       json_path.stat().st_size,
                "source_file":     data.get("source_file", ""),
                "doc_confidence":  doc_confidence,
                "extractor":       "uipath-document-understanding",
            },
        )

    def _build_entities(
        self, data: dict, doc: Document, json_path: Path
    ) -> list[Entity]:
        """
        Build Entity objects from the 'fields' section of the UiPath JSON.

        UiPath typically returns one logical entity per document.
        We aggregate all field values into one Entity's attributes dict.

        TEACHING: We create ONE PERSON entity per document, collecting all
        fields as attributes. This mirrors what LangChainExtractor does —
        one PERSON entity per identity document.
        """
        fields = data.get("fields", {})
        if not fields:
            logger.warning("No fields in UiPath JSON: %s", json_path)
            return []

        # ---- Determine the name field ----
        raw_doc_type = str(data.get("document_type", "GENERIC")).upper().strip()
        name_key = _PRIMARY_NAME_FIELD.get(raw_doc_type, "name")

        # Build attributes dict from all fields
        attributes: dict[str, str] = {}
        field_confidences: list[float] = []

        for field_name, field_data in fields.items():
            if not isinstance(field_data, dict):
                continue
            value = field_data.get("value")
            if value is not None and str(value).strip():
                attributes[field_name] = str(value).strip()
                conf = field_data.get("confidence", 1.0)
                if isinstance(conf, (int, float)):
                    field_confidences.append(float(conf))

        # Get the primary name
        name = attributes.get(name_key) or attributes.get("name") or "Unknown"

        # Get provenance from the name field
        name_field_data = fields.get(name_key, fields.get("name", {}))
        if isinstance(name_field_data, dict):
            bbox = name_field_data.get("bounding_box", [0, 0, 0, 0])
            name_confidence = float(name_field_data.get("confidence", 1.0))
        else:
            bbox = [0, 0, 0, 0]
            name_confidence = 1.0

        # Compute bounding box offsets
        # UiPath bounding box: [x1, y1, x2, y2] in pixels
        # We store x1 as char_offset_start, x2 as char_offset_end
        # (this is a pixel-based convention for UiPath entities)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
            char_offset_start = int(bbox[0])
            char_offset_end   = int(bbox[2])
        else:
            char_offset_start = 0
            char_offset_end   = 0

        # Compute overall confidence as average of field confidences
        if field_confidences:
            overall_confidence = round(
                sum(field_confidences) / len(field_confidences), 3
            )
        else:
            overall_confidence = float(data.get("confidence", 1.0))

        entity = Entity(
            entity_id    = str(uuid.uuid4()),
            name         = name,
            entity_type  = EntityType.PERSON,
            attributes   = attributes,
            source_doc_id    = doc.doc_id,
            source_filename  = doc.filename,
            doc_type         = doc.doc_type,

            # UiPath gives bounding boxes, not line numbers
            # line_number = 0 tells ProvenanceTracker to skip line verification
            line_number      = 0,
            line_text        = "",
            paragraph_index  = 0,
            paragraph_text   = "",

            # Store bounding box x-coords as char offsets
            char_offset_start = char_offset_start,
            char_offset_end   = char_offset_end,

            extractor_model      = "uipath-document-understanding",
            extraction_timestamp = datetime.now(timezone.utc).isoformat(),
            confidence           = overall_confidence,

            # Embedding will be set by EmbeddingEngine later
            embedding = None,
        )

        return [entity]

    def _extract_doc_date(self, fields: dict) -> Optional[str]:
        """
        Try to extract a document date from common date fields.
        Used to populate Document.doc_date for temporal filtering.
        """
        # Date fields to check, in priority order
        date_candidates = [
            "issue_date", "date", "start_date", "expiry_date", "dob"
        ]

        for key in date_candidates:
            field_data = fields.get(key, {})
            if isinstance(field_data, dict):
                value = field_data.get("value")
                if value and str(value).strip():
                    # Try to return something that looks like a date
                    return str(value).strip()

        return None
