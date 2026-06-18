"""
PDF Entity Extractor — graph_rag/pdf/extractor.py
===================================================
Extracts structured medical entities from a single PDFChunk using the LLM
and MEDICAL_SCHEMA. Follows the same patterns as LangChainExtractor:
  - _number_lines() for line-numbered prompts
  - JSON parsing with one retry
  - Entity construction with cross-referenced provenance

Key differences from LangChainExtractor:
  - Operates on PDFChunk rather than Document
  - Uses DocType.MEDICAL_REPORT
  - Maps "patient_name" → entity.name
  - extractor_model = "pdf-langchain"
  - confidence = filled_fields / 7 (total Medical_Schema fields)
  - source_filename = f"{chunk.source_pdf}_chunk_{chunk.chunk_index}"
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from ..core.models import DocType, Entity, EntityType
from ..llm.provider import LLMProvider, LLMProviderError
from ..extraction.langchain_extractor import (
    _number_lines,
    _extract_json_from_response,
    _build_fields_list,
    _build_fields_json_template,
    _EXTRACT_PROMPT,
    _RETRY_PROMPT,
)
from .chunker import PDFChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MEDICAL SCHEMA
# ---------------------------------------------------------------------------

MEDICAL_SCHEMA: dict[str, str] = {
    "patient_name": "Full name of the patient",
    "dob":          "Patient date of birth in ISO format YYYY-MM-DD",
    "diagnosis":    "Primary medical diagnosis or condition",
    "medications":  "List of prescribed medications as a comma-separated string",
    "doctor":       "Name of the attending physician",
    "procedures":   "List of medical procedures as a comma-separated string",
    "visit_date":   "Date of the clinical encounter in ISO format YYYY-MM-DD",
}

# Total number of schema fields — used for confidence calculation
_SCHEMA_FIELD_COUNT = len(MEDICAL_SCHEMA)  # 7


# ---------------------------------------------------------------------------
# ENTITY BUILDER
# ---------------------------------------------------------------------------

def _build_entity_from_chunk(
    parsed: dict,
    chunk: PDFChunk,
    doc_id: str,
) -> Entity:
    """
    Convert the LLM's parsed JSON output into an Entity object for a PDFChunk.

    Provenance is cross-referenced against chunk.text (NOT trusted from LLM):
    - line_number: 1-indexed line reported by LLM, validated against chunk lines
    - line_text:   chunk.text.split('\\n')[line_number - 1] (verbatim from chunk)
    - char_offset_start / char_offset_end: positions within chunk.text
    """
    fields = parsed.get("fields", {})
    chunk_lines = chunk.text.split("\n")

    # Build attributes dict from all non-null field values
    attributes: dict[str, str] = {}
    for field_name in MEDICAL_SCHEMA:
        field_data = fields.get(field_name, {})
        value = field_data.get("value") if isinstance(field_data, dict) else None
        if value is not None:
            attributes[field_name] = str(value)

    # Map patient_name → entity.name
    name = attributes.get("patient_name") or "Unknown"

    # Get provenance from the 'patient_name' field (LLM-reported line number)
    name_field_data = fields.get("patient_name", {})
    if isinstance(name_field_data, dict):
        raw_line_num = name_field_data.get("line_number") or 0
        if isinstance(raw_line_num, list):
            raw_line_num = raw_line_num[0] if raw_line_num else 0
        line_number = int(raw_line_num)
    else:
        line_number = 0

    # Cross-reference line_number with actual chunk text
    # The LLM's reported line number is validated; we always use the actual line.
    if 1 <= line_number <= len(chunk_lines):
        actual_line_text = chunk_lines[line_number - 1]
        # Compute char_offset_start as cumulative char offset to this line
        char_offset_start = sum(len(chunk_lines[i]) + 1 for i in range(line_number - 1))
        char_offset_end = char_offset_start + len(actual_line_text)
    else:
        # Invalid line number from LLM — search for patient_name in chunk text
        actual_line_text, line_number, char_offset_start, char_offset_end = (
            _find_name_in_chunk(name, chunk_lines)
        )

    return Entity(
        entity_id            = str(uuid.uuid4()),
        name                 = name,
        entity_type          = EntityType.PERSON,
        attributes           = attributes,
        source_doc_id        = doc_id,
        source_filename      = f"{chunk.source_pdf}_chunk_{chunk.chunk_index}",
        doc_type             = DocType.MEDICAL_REPORT,
        line_number          = line_number,
        line_text            = actual_line_text,
        paragraph_index      = 0,   # chunk is already the paragraph unit
        paragraph_text       = chunk.text,
        char_offset_start    = char_offset_start,
        char_offset_end      = char_offset_end,
        extractor_model      = "pdf-langchain",
        extraction_timestamp = datetime.now(timezone.utc).isoformat(),
        confidence           = 1.0,  # updated by extract() after building
        embedding            = None,  # set by EmbeddingEngine after extraction
    )


def _find_name_in_chunk(
    name: str,
    chunk_lines: list[str],
) -> tuple[str, int, int, int]:
    """
    Fallback: search for the patient name string in the chunk lines.
    Returns (line_text, line_number, char_offset_start, char_offset_end).
    """
    if not name or name == "Unknown":
        return ("", 0, 0, 0)

    cumulative_offset = 0
    for i, line in enumerate(chunk_lines):
        if name.lower() in line.lower():
            return (line, i + 1, cumulative_offset, cumulative_offset + len(line))
        cumulative_offset += len(line) + 1  # +1 for the '\n' separator

    return ("", 0, 0, 0)


# ---------------------------------------------------------------------------
# MAIN EXTRACTOR CLASS
# ---------------------------------------------------------------------------

class PDFEntityExtractor:
    """
    Extracts medical entities from a single PDFChunk using the LLM.

    Uses MEDICAL_SCHEMA (7 fields) and follows the same prompt/retry/parse
    pattern as LangChainExtractor. Differences:
      - source is PDFChunk, not Document
      - doc_type = MEDICAL_REPORT
      - patient_name → entity.name
      - extractor_model = "pdf-langchain"
      - confidence = filled_fields / 7

    Usage:
        extractor = PDFEntityExtractor(llm_provider)
        entities = extractor.extract(chunk, doc_id)
        for entity in entities:
            print(entity.name, entity.confidence)
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str = "unknown"):
        self._llm        = llm_provider
        self._model_name = model_name

    def extract(
        self,
        chunk: PDFChunk,
        doc_id: str,
    ) -> list[Entity]:
        """
        Extract medical entities from a PDFChunk using the LLM.

        Args:
            chunk  — PDFChunk to extract entities from
            doc_id — doc_id of the Document node created for this chunk

        Returns:
            List containing one Entity (on success) or empty list (on failure).
            Never raises — all errors are caught and logged.
        """
        if not chunk.text.strip():
            logger.warning(
                "Skipping empty chunk %d from '%s'",
                chunk.chunk_index, chunk.source_pdf,
            )
            return []

        # 1. Number lines in chunk text (same helper as LangChainExtractor)
        numbered_text = _number_lines(chunk.text)

        # 2. Build prompt components using MEDICAL_SCHEMA
        fields_list          = _build_fields_list(MEDICAL_SCHEMA)
        fields_json_template = _build_fields_json_template(MEDICAL_SCHEMA)

        # 3. First extraction attempt
        prompt = _EXTRACT_PROMPT.format(
            doc_type             = "MEDICAL_REPORT",
            fields_list          = fields_list,
            numbered_text        = numbered_text,
            fields_json_template = fields_json_template,
        )

        raw    = self._call_llm(chunk.source_pdf, prompt)
        parsed = self._parse_json(raw)

        # 4. Retry once on malformed JSON
        if parsed is None:
            logger.warning(
                "First extraction attempt failed for chunk %d of '%s', retrying...",
                chunk.chunk_index, chunk.source_pdf,
            )
            retry_prompt = _RETRY_PROMPT.format(
                doc_type             = "MEDICAL_REPORT",
                numbered_text        = numbered_text,
                fields_json_template = fields_json_template,
            )
            raw2   = self._call_llm(chunk.source_pdf, retry_prompt)
            parsed = self._parse_json(raw2)

        # 5. Both attempts failed — log WARNING and return []
        if parsed is None:
            logger.warning(
                "Extraction failed after retry for '%s' chunk %d — skipping entity.",
                chunk.source_pdf, chunk.chunk_index,
            )
            return []

        # 6. Build Entity with cross-referenced provenance
        try:
            entity = _build_entity_from_chunk(parsed, chunk, doc_id)

            # 7. Set extractor_model (already set in _build_entity_from_chunk,
            #    but override with actual model_name if provided)
            entity.extractor_model = "pdf-langchain"

            # 8. Compute confidence = filled_fields / 7
            filled = sum(1 for k in MEDICAL_SCHEMA if entity.attributes.get(k))
            entity.confidence = round(filled / _SCHEMA_FIELD_COUNT, 2)

            logger.info(
                "Extracted entity from '%s' chunk %d: name='%s' "
                "confidence=%.2f fields=%d/%d",
                chunk.source_pdf, chunk.chunk_index,
                entity.name, entity.confidence, filled, _SCHEMA_FIELD_COUNT,
            )
            return [entity]

        except Exception as e:
            logger.error(
                "Failed to build entity from '%s' chunk %d: %s",
                chunk.source_pdf, chunk.chunk_index, e,
            )
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, source_pdf: str, prompt: str) -> str:
        """Call LLM, return empty string on failure."""
        try:
            return self._llm.complete(prompt, temperature=0.0)
        except LLMProviderError as e:
            logger.error("LLM error extracting '%s': %s", source_pdf, e)
            return ""
        except Exception as e:
            logger.error("Unexpected error extracting '%s': %s", source_pdf, e)
            return ""

    def _parse_json(self, raw: str) -> dict | None:
        """
        Try to parse raw LLM response as JSON.
        Returns parsed dict or None if parsing fails.
        """
        if not raw:
            return None

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from noisy response (strips markdown fences etc.)
        try:
            cleaned = _extract_json_from_response(raw)
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.debug("JSON parse failed. Raw response: %r", raw[:200])
            return None
