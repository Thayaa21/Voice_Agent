"""
Document Classifier — Step 4
==============================
Uses the LLM to detect the type of a document from its first 500 characters.

TEACHING NOTES
--------------
Why only 500 characters?
    The title and first few lines of a document are almost always enough
    to identify its type. "CERTIFICATE OF BIRTH", "DRIVER'S LICENSE",
    "PASSPORT", etc. appear right at the top.
    Using the full document wastes LLM tokens (= time + money).

What is a schema?
    A schema defines WHAT to extract from a document of a given type.
    A birth certificate schema says: "look for name, dob, place_of_birth..."
    A driver's license schema says: "look for name, dob, license_number..."

    The schema is passed to the LangChain Extractor in the next step.
    Without classification first, the extractor wouldn't know which fields
    to look for.

Fallback to GENERIC:
    If the LLM returns something unexpected (or fails), we don't crash.
    We assign DocType.GENERIC and use a minimal schema.
    The document still gets processed — just with best-effort extraction.

Why strip and uppercase the response?
    LLMs sometimes return extra whitespace, lowercase, or punctuation.
    "birth_certificate\n" → strip() → "birth_certificate" → upper() → "BIRTH_CERTIFICATE"
    This makes matching robust regardless of LLM formatting quirks.
"""

import logging
from typing import Optional

from ..core.models import Document, DocType
from ..llm.provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EXTRACTION SCHEMAS
# ---------------------------------------------------------------------------
# Each schema is a dict of:
#   field_name → description
#
# The description is what the LangChain Extractor will include in its
# prompt to tell the LLM exactly what to look for and what it means.
#
# TEACHING: This is the "contract" between the Classifier and the Extractor.
# Classifier says "this is a DRIVERS_LICENSE".
# Extractor looks up DOC_TYPE_SCHEMAS["DRIVERS_LICENSE"] and extracts those fields.
# ---------------------------------------------------------------------------

DOC_TYPE_SCHEMAS: dict[str, dict[str, str]] = {
    "BIRTH_CERTIFICATE": {
        "name":                "Full legal name of the person",
        "dob":                 "Date of birth in ISO format YYYY-MM-DD",
        "place_of_birth":      "City, province/state, and country of birth",
        "parents":             "Names of father and mother",
        "registration_number": "Official registration or certificate number",
    },
    "DRIVERS_LICENSE": {
        "name":           "Full legal name of the license holder",
        "dob":            "Date of birth in ISO format YYYY-MM-DD",
        "license_number": "Driver's license number",
        "address":        "Full residential address including city, province, postal code",
        "vehicle_class":  "Class or category of vehicle license",
        "issue_date":     "Date the license was issued in ISO format YYYY-MM-DD",
        "expiry_date":    "Date the license expires in ISO format YYYY-MM-DD",
    },
    "PASSPORT": {
        "name":            "Full name — combine Given Names and Surname into one full name string",
        "dob":             "Date of birth in ISO format YYYY-MM-DD",
        "passport_number": "Passport document number",
        "nationality":     "Country of nationality",
        "expiry_date":     "Passport expiry date in ISO format YYYY-MM-DD",
        "place_of_issue":  "City or office where passport was issued",
    },
    "INSURANCE": {
        "name":          "Full name of the policyholder",
        "dob":           "Date of birth of the policyholder in ISO format YYYY-MM-DD",
        "policy_number": "Insurance policy number",
        "coverage_type": "Type of insurance coverage",
        "premium":       "Monthly or annual premium amount",
        "start_date":    "Policy start date in ISO format YYYY-MM-DD",
        "beneficiary":   "Named beneficiary of the policy",
    },
    "MEDICAL_RECORD": {
        "patient_name": "Full name of the patient",
        "dob":          "Patient date of birth in ISO format YYYY-MM-DD",
        "diagnosis":    "Medical diagnosis or condition",
        "doctor":       "Name of the attending physician",
        "date":         "Date of the medical visit in ISO format YYYY-MM-DD",
        "medications":  "List of medications prescribed",
    },
    "MEDICAL_REPORT": {
        "patient_name": "Full name of the patient",
        "dob":          "Patient date of birth in ISO format YYYY-MM-DD",
        "diagnosis":    "Primary medical diagnosis or condition",
        "medications":  "List of prescribed medications as a comma-separated string",
        "doctor":       "Name of the attending physician",
        "procedures":   "List of medical procedures as a comma-separated string",
        "visit_date":   "Date of the clinical encounter in ISO format YYYY-MM-DD",
    },
    "GENERIC": {
        "name":       "Full name of the person if mentioned",
        "id_number":  "Any identification number found",
        "date":       "Any significant date mentioned",
        "address":    "Any address mentioned",
    },
}

# Prompt template for classification
# We use a strict, simple prompt that asks for exactly one word.
# The fewer words the LLM has to produce, the less chance of noise.
_CLASSIFY_PROMPT = """\
You are a document type classifier. Read the beginning of the document below
and respond with EXACTLY ONE of these labels — nothing else, no explanation:

BIRTH_CERTIFICATE
DRIVERS_LICENSE
PASSPORT
INSURANCE
MEDICAL_RECORD
MEDICAL_REPORT
GENERIC

Use GENERIC only if the document does not match any of the other types.

--- DOCUMENT START ---
{text}
--- DOCUMENT END ---

Your response (one label only):"""


class DocumentClassifier:
    """
    Classifies a Document into one of the six DocType values using the LLM.

    After classification, the Document's doc_type field is updated and
    the appropriate extraction schema is returned for use by the Extractor.

    Usage:
        classifier = DocumentClassifier(llm)
        doc_type, schema = classifier.classify(doc)
        # doc.doc_type is now updated
        # schema is the dict of fields to extract
    """

    # How many characters from the start of the document to send to the LLM.
    # 500 characters is enough to capture the header and first few fields.
    PREVIEW_CHARS = 500

    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    def classify(self, document: Document) -> tuple[DocType, dict[str, str]]:
        """
        Classify a document and return its type + extraction schema.

        Args:
            document — a Document object (created by DocumentLoader)

        Returns:
            (doc_type, schema) tuple where:
                doc_type — the detected DocType enum value
                schema   — dict of {field_name: description} for extraction

        Side effect:
            Updates document.doc_type in place.

        Never raises — falls back to GENERIC on any error.
        """
        # ---- Handle empty documents ----
        # Empty documents can't be classified. Assign GENERIC immediately.
        if document.empty or not document.text.strip():
            logger.warning(
                "Document is empty, assigning GENERIC: %s", document.filename
            )
            document.doc_type = DocType.GENERIC
            return DocType.GENERIC, DOC_TYPE_SCHEMAS["GENERIC"]

        # ---- Build the prompt ----
        # Take only the first PREVIEW_CHARS characters.
        # This is the "header" of the document — enough to identify its type.
        preview = document.text[:self.PREVIEW_CHARS].strip()
        prompt  = _CLASSIFY_PROMPT.format(text=preview)

        # ---- Call the LLM ----
        raw_response = self._call_llm(document.filename, prompt)

        # ---- Parse the response ----
        doc_type = self._parse_response(raw_response, document.filename)

        # ---- Update the document in place ----
        document.doc_type = doc_type

        schema = DOC_TYPE_SCHEMAS[doc_type.value]

        logger.info(
            "Classified '%s' → %s (raw LLM response: %r)",
            document.filename, doc_type.value, raw_response
        )

        return doc_type, schema

    def classify_batch(
        self, documents: list[Document]
    ) -> list[tuple[DocType, dict[str, str]]]:
        """
        Classify multiple documents. Processes them one by one.

        Returns a list of (doc_type, schema) tuples in the same order
        as the input documents.

        Never raises — failed classifications fall back to GENERIC.
        """
        results = []
        for doc in documents:
            result = self.classify(doc)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, filename: str, prompt: str) -> str:
        """
        Call the LLM and return the raw string response.
        Returns empty string on failure (caller handles fallback).
        """
        try:
            return self._llm.complete(prompt, temperature=0.0)
        except LLMProviderError as e:
            logger.error(
                "LLM error while classifying '%s': %s — defaulting to GENERIC",
                filename, e
            )
            return ""
        except Exception as e:
            logger.error(
                "Unexpected error while classifying '%s': %s — defaulting to GENERIC",
                filename, e
            )
            return ""

    def _parse_response(self, raw: str, filename: str) -> DocType:
        """
        Parse the LLM's raw string response into a DocType.

        Handles:
        - Extra whitespace: "  BIRTH_CERTIFICATE  " → "BIRTH_CERTIFICATE"
        - Lowercase: "birth_certificate" → "BIRTH_CERTIFICATE"
        - Partial match: "This is a DRIVERS_LICENSE" → "DRIVERS_LICENSE"
        - Unknown value: falls back to GENERIC with a warning
        - Empty string (LLM failed): falls back to GENERIC with a warning

        TEACHING: LLMs are not perfectly obedient. They sometimes add
        extra words, use different casing, or include punctuation.
        Robust parsing handles all these cases gracefully.
        """
        if not raw:
            logger.warning(
                "Empty LLM response for '%s' — defaulting to GENERIC", filename
            )
            return DocType.GENERIC

        # Clean up the response
        cleaned = raw.strip().upper()

        # Direct match — the ideal case
        valid_types = {dt.value for dt in DocType}
        if cleaned in valid_types:
            return DocType(cleaned)

        # Partial match — LLM included extra words
        # e.g. "This appears to be a BIRTH_CERTIFICATE document"
        for dt in DocType:
            if dt.value in cleaned:
                logger.debug(
                    "Partial match for '%s': found '%s' in response %r",
                    filename, dt.value, raw
                )
                return dt

        # No match found — fall back to GENERIC
        logger.warning(
            "Unrecognized classification for '%s': %r — defaulting to GENERIC",
            filename, raw
        )
        return DocType.GENERIC
