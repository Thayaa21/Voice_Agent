"""
LangChain Extractor — Step 5
==============================
Uses the LLM to extract named entities from a Document, including
exact provenance (line number + verbatim line text) for every value.

TEACHING NOTES
--------------
Why "LangChain" in the name?
    In this project, "LangChain mode" means using an LLM to read raw .txt
    files and extract entities. The name distinguishes it from "UiPath mode"
    where structured JSON is parsed directly. We don't actually use the
    LangChain library here — we call our own LLM provider directly.

Why number the lines in the prompt?
    If we just send raw text, the LLM can extract values but can't tell us
    WHERE in the document they came from. By prepending line numbers, the
    LLM can say "name is on line 4" and we can verify that exactly.

    This is the key technique that makes provenance tracking possible.

Why JSON output?
    Structured JSON is easy to parse programmatically. We ask the LLM to
    return a specific JSON shape and then parse it with json.loads().
    If the JSON is malformed, we retry once with a stricter prompt.

What is an Entity in this context?
    A single "person record" per document. From a birth certificate we
    get ONE entity of type PERSON with attributes {name, dob, place_of_birth...}.
    From a medical record we might also get a PERSON entity for the doctor.

    Attributes come from the schema (defined in classifier.py):
    BIRTH_CERTIFICATE schema → {name, dob, place_of_birth, parents, registration_number}

Retry logic:
    LLMs occasionally return malformed JSON. We:
    1. Try parsing the response
    2. If it fails, try to extract JSON from the response (LLMs sometimes
       add text before/after the JSON block)
    3. Retry once with an even stricter "JSON only" prompt
    4. If retry also fails, log a warning and return empty list
    Never crash the pipeline over one bad extraction.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone

from ..core.models import Document, DocType, Entity, EntityType
from ..llm.provider import LLMProvider, LLMProviderError
from .classifier import DOC_TYPE_SCHEMAS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """\
You are a document entity extractor. Extract structured information from the \
document below.

DOCUMENT TYPE: {doc_type}

FIELDS TO EXTRACT:
{fields_list}

RULES:
1. Return ONLY a valid JSON object — no explanation, no markdown, no code blocks
2. For each field, include:
   - "value": the extracted value as a string (use ISO date format YYYY-MM-DD for dates)
   - "line_number": the 1-indexed line number where this value appears
   - "line_text": the EXACT verbatim text of that line (copy it precisely)
3. If a field is not found, set "value" to null and "line_number" to 0
4. Dates MUST be in ISO format YYYY-MM-DD

DOCUMENT (with line numbers):
{numbered_text}

Return this exact JSON structure:
{{
  "entity_type": "PERSON",
  "fields": {{
{fields_json_template}
  }}
}}"""

_RETRY_PROMPT = """\
Return ONLY valid JSON. No explanation. No markdown. No code fences.

Extract from this {doc_type} document:
{numbered_text}

Required JSON (fill in values):
{{
  "entity_type": "PERSON",
  "fields": {{
{fields_json_template}
  }}
}}"""


def _build_fields_list(schema: dict[str, str]) -> str:
    """Format schema fields as a numbered list for the prompt."""
    lines = []
    for i, (field, description) in enumerate(schema.items(), 1):
        lines.append(f"  {i}. {field}: {description}")
    return "\n".join(lines)


def _build_fields_json_template(schema: dict[str, str]) -> str:
    """Build the JSON template section showing the expected output shape."""
    lines = []
    for field in schema:
        lines.append(
            f'    "{field}": {{"value": null, "line_number": 0, "line_text": ""}}'
        )
    return ",\n".join(lines)


def _number_lines(text: str) -> str:
    """
    Prepend line numbers to each line of text.

    "CERT OF BIRTH\nFull Name: Alice" →
    " 1: CERT OF BIRTH\n 2: Full Name: Alice"

    TEACHING: The line numbers let the LLM reference specific lines
    when reporting provenance. Without them, the LLM would have to
    guess or count lines itself, which is unreliable.
    """
    lines = text.split("\n")
    width = len(str(len(lines)))  # pad to consistent width
    numbered = []
    for i, line in enumerate(lines, 1):
        numbered.append(f"{i:>{width}}: {line}")
    return "\n".join(numbered)


def _extract_json_from_response(raw: str) -> str:
    """
    Try to extract a JSON object from a response that may contain extra text.

    LLMs sometimes say:
        "Here is the extracted data: {\"entity_type\": ...}"
    or wrap it in markdown:
        ```json
        {...}
        ```

    This function finds the first { and last } and returns what's between them.
    """
    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

    # Find JSON object boundaries
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw.strip()


# ---------------------------------------------------------------------------
# ENTITY BUILDER
# ---------------------------------------------------------------------------

def _combine_passport_name(fields: dict) -> str | None:
    """
    Passports list 'Surname' and 'Given Names' separately.
    Combine them into a full name: "Given Names Surname".
    """
    surname = None
    given   = None
    for key, val in fields.items():
        v = val.get("value") if isinstance(val, dict) else None
        if not v:
            continue
        k = key.lower()
        if "surname" in k or "last" in k:
            surname = v
        elif "given" in k or "first" in k:
            given = v
    if given and surname:
        return f"{given} {surname}"
    return None


def _build_entity(
    parsed: dict,
    document: Document,
    schema: dict[str, str],
) -> Entity:
    """
    Convert the LLM's parsed JSON output into an Entity object.

    TEACHING: This is where the LLM's raw JSON becomes a proper Entity
    with all metadata fields populated. We cross-reference the line_number
    from the LLM with document.lines to get the verbatim line text and
    compute the char_offset_start from document.line_offsets.

    The entity name is derived from the 'name' or 'patient_name' field,
    which is always the primary identifier for a person.
    """
    fields = parsed.get("fields", {})

    # Build attributes dict from all non-null field values
    attributes: dict[str, str] = {}
    for field_name in schema:
        field_data = fields.get(field_name, {})
        value = field_data.get("value") if isinstance(field_data, dict) else None
        if value is not None:
            attributes[field_name] = str(value)

    # Determine the primary name field
    # Birth certs / licenses use "name"; medical records use "patient_name"
    # Passports split into surname + given names — combine them
    name = (
        attributes.get("name")
        or attributes.get("patient_name")
        or _combine_passport_name(fields)
        or "Unknown"
    )

    # Get provenance from the 'name' or 'patient_name' field
    name_key = "name" if "name" in fields else "patient_name"
    name_field_data = fields.get(name_key, {})
    if isinstance(name_field_data, dict):
        raw_line_num = name_field_data.get("line_number") or 0
        # LLM sometimes returns a list — take first element
        if isinstance(raw_line_num, list):
            raw_line_num = raw_line_num[0] if raw_line_num else 0
        line_number = int(raw_line_num)
        llm_line_text = name_field_data.get("line_text") or ""
        if isinstance(llm_line_text, list):
            llm_line_text = llm_line_text[0] if llm_line_text else ""
    else:
        line_number = 0
        llm_line_text = ""

    # Cross-reference with actual document lines
    # The LLM might slightly misquote the line — we use the actual line text
    # from the document for perfect accuracy (provenance invariant).
    if 1 <= line_number <= len(document.lines):
        actual_line_text = document.lines[line_number - 1]
        char_offset_start = document.line_offsets[line_number - 1]
        char_offset_end   = char_offset_start + len(actual_line_text)
    else:
        # LLM gave an invalid line number — try to find the name in the text
        actual_line_text, line_number, char_offset_start, char_offset_end = (
            _find_value_in_document(name, document)
        )

    # Determine paragraph index from line number
    paragraph_index, paragraph_text = _get_paragraph_for_line(
        line_number, document
    )

    # Determine EntityType
    # For now all extracted entities are PERSON — future versions could
    # extract ORGANIZATIONs, LOCATIONs etc. from the same document
    entity_type_str = parsed.get("entity_type", "PERSON").upper()
    try:
        entity_type = EntityType(entity_type_str)
    except ValueError:
        entity_type = EntityType.PERSON

    return Entity(
        entity_id            = str(uuid.uuid4()),
        name                 = name,
        entity_type          = entity_type,
        attributes           = attributes,
        source_doc_id        = document.doc_id,
        source_filename      = document.filename,
        doc_type             = document.doc_type,
        line_number          = line_number,
        line_text            = actual_line_text,
        paragraph_index      = paragraph_index,
        paragraph_text       = paragraph_text,
        char_offset_start    = char_offset_start,
        char_offset_end      = char_offset_end,
        extractor_model      = "",   # set by LangChainExtractor after creation
        extraction_timestamp = datetime.now(timezone.utc).isoformat(),
        confidence           = 1.0,  # updated by extractor
        embedding            = None, # set by EmbeddingEngine in Step 7
    )


def _find_value_in_document(
    value: str, document: Document
) -> tuple[str, int, int, int]:
    """
    Fallback: search for a value string in the document lines.
    Returns (line_text, line_number, char_offset_start, char_offset_end).
    """
    if not value or value == "Unknown":
        return ("", 0, 0, 0)

    for i, line in enumerate(document.lines):
        if value.lower() in line.lower():
            offset = document.line_offsets[i]
            return (line, i + 1, offset, offset + len(line))

    return ("", 0, 0, 0)


def _get_paragraph_for_line(
    line_number: int, document: Document
) -> tuple[int, str]:
    """
    Find which paragraph a given line_number belongs to.
    Returns (paragraph_index, paragraph_text).
    """
    if line_number <= 0 or not document.paragraphs:
        return (0, "")

    # Reconstruct which paragraph each line belongs to
    current_line = 1
    for para_idx, para in enumerate(document.paragraphs):
        para_lines = para.split("\n")
        para_line_count = len(para_lines)
        if current_line <= line_number < current_line + para_line_count:
            return (para_idx, para)
        current_line += para_line_count + 1  # +1 for blank line between paragraphs

    return (0, document.paragraphs[0] if document.paragraphs else "")


# ---------------------------------------------------------------------------
# MAIN EXTRACTOR CLASS
# ---------------------------------------------------------------------------

class LangChainExtractor:
    """
    Extracts entities from .txt documents using the LLM.

    Takes a classified Document + its schema and returns a list of Entity
    objects with full provenance metadata.

    Usage:
        extractor = LangChainExtractor(llm, model_name="qwen2.5:14b")
        entities  = extractor.extract(doc, schema)
        for entity in entities:
            print(entity.name, entity.attributes)
            print(f"  Found at line {entity.line_number}: {entity.line_text}")
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str = "unknown"):
        self._llm        = llm_provider
        self._model_name = model_name

    def extract(
        self,
        document: Document,
        schema: dict[str, str],
    ) -> list[Entity]:
        """
        Extract entities from a document using the LLM.

        Args:
            document — classified Document (doc_type must be set)
            schema   — dict of {field_name: description} from classifier

        Returns:
            List of Entity objects (usually one per document for identity docs).
            Returns empty list if extraction fails after retry.

        Never raises — all errors are caught and logged.
        """
        if document.empty:
            logger.warning("Skipping empty document: %s", document.filename)
            return []

        # Build numbered text for the prompt
        numbered_text = _number_lines(document.text)
        doc_type_str  = document.doc_type.value

        # Build prompt components
        fields_list          = _build_fields_list(schema)
        fields_json_template = _build_fields_json_template(schema)

        # First attempt
        prompt = _EXTRACT_PROMPT.format(
            doc_type             = doc_type_str,
            fields_list          = fields_list,
            numbered_text        = numbered_text,
            fields_json_template = fields_json_template,
        )

        raw = self._call_llm(document.filename, prompt)
        parsed = self._parse_json(raw)

        # Retry if first attempt failed
        if parsed is None:
            logger.warning(
                "First extraction attempt failed for '%s', retrying...",
                document.filename
            )
            retry_prompt = _RETRY_PROMPT.format(
                doc_type             = doc_type_str,
                numbered_text        = numbered_text,
                fields_json_template = fields_json_template,
            )
            raw2   = self._call_llm(document.filename, retry_prompt)
            parsed = self._parse_json(raw2)

        if parsed is None:
            logger.error(
                "Extraction failed after retry for '%s' — skipping entity.",
                document.filename
            )
            return []

        # Build Entity from parsed JSON
        try:
            entity = _build_entity(parsed, document, schema)
            entity.extractor_model = self._model_name

            # Compute confidence from how many fields were successfully extracted
            total  = len(schema)
            filled = sum(1 for k in schema if entity.attributes.get(k))
            entity.confidence = round(filled / total, 2) if total > 0 else 0.0

            logger.info(
                "Extracted entity from '%s': name='%s' confidence=%.2f fields=%d/%d",
                document.filename, entity.name, entity.confidence, filled, total
            )
            return [entity]

        except Exception as e:
            logger.error(
                "Failed to build entity from '%s': %s", document.filename, e
            )
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, filename: str, prompt: str) -> str:
        """Call LLM, return empty string on failure."""
        try:
            return self._llm.complete(prompt, temperature=0.0)
        except LLMProviderError as e:
            logger.error("LLM error extracting '%s': %s", filename, e)
            return ""
        except Exception as e:
            logger.error("Unexpected error extracting '%s': %s", filename, e)
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

        # Try extracting JSON from noisy response
        try:
            cleaned = _extract_json_from_response(raw)
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.debug("JSON parse failed. Raw response: %r", raw[:200])
            return None
