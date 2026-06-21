"""
PatientExtractor
================
Reads the new patient.json format (built from Synthea CSV data) and produces
Document + Entity objects compatible with the existing graph pipeline.

Each patient.json contains:
  - patient_id, display_name, dob, age, gender, city, state
  - insurance: {payer, member_since, payer_id}
  - active_conditions: [{code, description, onset}]
  - encounters: [{encounter_id, date, type, description, reason, procedure_code,
                  provider, payer, billed, covered, patient_owes, claim_status}]
  - claim_summary: {total_billed, total_covered, total_owed, denied_claims, pending_claims}

Graph nodes created per patient:
  1. One PERSON entity — identity + insurance + claim summary
  2. One entity per active condition — linked to the person
  3. One entity per encounter — linked to the person

This replaces the old multi-file UiPath approach (birth_cert + passport + etc.)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.models import Document, DocType, Entity, EntityType

logger = logging.getLogger(__name__)


class PatientExtractor:
    """
    Extracts structured data from patient.json files into (Document, list[Entity]).

    Usage:
        extractor = PatientExtractor()
        doc, entities = extractor.extract("sample_data/aiden_garcia/patient.json")
    """

    def extract(
        self, json_path: str | Path
    ) -> tuple[Optional[Document], list[Entity]]:
        json_path = Path(json_path)

        if not json_path.exists():
            logger.warning("Patient JSON not found: %s", json_path)
            return None, []

        try:
            raw_text = json_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except Exception as e:
            logger.warning("Failed to read %s: %s", json_path, e)
            return None, []

        if not isinstance(data, dict) or "patient_id" not in data:
            logger.warning("Not a valid patient.json: %s", json_path)
            return None, []

        doc       = self._build_document(data, json_path, raw_text)
        entities  = self._build_entities(data, doc)

        logger.info(
            "PatientExtractor: %s → %d entities",
            json_path.parent.name, len(entities)
        )
        return doc, entities

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    def _build_document(self, data: dict, json_path: Path, raw_text: str) -> Document:
        lines        = raw_text.split("\n")
        line_offsets = []
        pos = 0
        for line in lines:
            line_offsets.append(pos)
            pos += len(line) + 1
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        return Document(
            doc_id       = str(uuid.uuid4()),
            filename     = json_path.name,
            text         = raw_text,
            lines        = lines,
            paragraphs   = paragraphs,
            line_offsets = line_offsets,
            doc_type     = DocType.MEDICAL_RECORD,
            doc_date     = data.get("dob"),
            empty        = False,
            metadata     = {
                "full_path":   str(json_path.resolve()),
                "patient_id":  data.get("patient_id", ""),
                "extractor":   "patient-json",
            },
        )

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def _build_entities(self, data: dict, doc: Document) -> list[Entity]:
        entities = []
        now = datetime.now(timezone.utc).isoformat()

        name        = data.get("display_name", "Unknown")
        patient_id  = data.get("patient_id", "")
        insurance   = data.get("insurance", {})
        summary     = data.get("claim_summary", {})
        conditions  = data.get("active_conditions", [])
        encounters  = data.get("encounters", [])

        # ── 1. Primary PERSON entity ──────────────────────────────────
        person_attrs = {
            "patient_id":     patient_id,
            "dob":            data.get("dob", ""),
            "age":            str(data.get("age", "")),
            "gender":         data.get("gender", ""),
            "city":           data.get("city", ""),
            "state":          data.get("state", ""),
            "insurance_payer": insurance.get("payer", ""),
            "insurance_since": insurance.get("member_since", ""),
            "total_billed":   str(summary.get("total_billed", 0)),
            "total_covered":  str(summary.get("total_covered", 0)),
            "total_owed":     str(summary.get("total_owed", 0)),
            "denied_claims":  str(summary.get("denied_claims", 0)),
            "pending_claims": str(summary.get("pending_claims", 0)),
            "active_conditions_count": str(len(conditions)),
            "encounters_count": str(len(encounters)),
        }

        entities.append(Entity(
            entity_id            = str(uuid.uuid4()),
            name                 = name,
            entity_type          = EntityType.PERSON,
            attributes           = person_attrs,
            source_doc_id        = doc.doc_id,
            source_filename      = doc.filename,
            doc_type             = DocType.MEDICAL_RECORD,
            line_number          = 0,
            extractor_model      = "patient-json",
            extraction_timestamp = now,
            confidence           = 1.0,
        ))

        # ── 2. One entity per active condition ────────────────────────
        for cond in conditions:
            cond_name = cond.get("description", "Unknown condition")
            entities.append(Entity(
                entity_id            = str(uuid.uuid4()),
                name                 = name,          # link back to patient name
                entity_type          = EntityType.PERSON,
                attributes           = {
                    "condition_code":        cond.get("code", ""),
                    "condition_description": cond_name,
                    "condition_onset":       cond.get("onset", ""),
                    "record_type":           "condition",
                },
                source_doc_id        = doc.doc_id,
                source_filename      = doc.filename,
                doc_type             = DocType.MEDICAL_RECORD,
                line_number          = 0,
                extractor_model      = "patient-json",
                extraction_timestamp = now,
                confidence           = 1.0,
            ))

        # ── 3. One entity per encounter ───────────────────────────────
        for enc in encounters:
            entities.append(Entity(
                entity_id            = str(uuid.uuid4()),
                name                 = name,
                entity_type          = EntityType.PERSON,
                attributes           = {
                    "encounter_id":    enc.get("encounter_id", ""),
                    "encounter_date":  enc.get("date", ""),
                    "encounter_type":  enc.get("type", ""),
                    "description":     enc.get("description", ""),
                    "reason":          enc.get("reason", ""),
                    "procedure_code":  enc.get("procedure_code", ""),
                    "provider":        enc.get("provider", ""),
                    "payer":           enc.get("payer", ""),
                    "billed":          str(enc.get("billed", 0)),
                    "covered":         str(enc.get("covered", 0)),
                    "patient_owes":    str(enc.get("patient_owes", 0)),
                    "claim_status":    enc.get("claim_status", ""),
                    "record_type":     "encounter",
                },
                source_doc_id        = doc.doc_id,
                source_filename      = doc.filename,
                doc_type             = DocType.MEDICAL_RECORD,
                line_number          = 0,
                extractor_model      = "patient-json",
                extraction_timestamp = now,
                confidence           = 1.0,
            ))

        return entities
