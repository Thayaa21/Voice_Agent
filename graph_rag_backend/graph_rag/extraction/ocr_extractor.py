"""
OCR Extractor — Image/PDF to Entity
=====================================
Processes image and PDF files using:

STEP 1: pytesseract (OCR) — fast Python, NO LLM needed
    - Converts image/PDF pixels → raw text
    - Rule-based regex parsing extracts common fields
    - Takes ~0.5-2 seconds per page

STEP 2: LLM (verifier only) — light verification pass
    - Receives: raw OCR text + regex-extracted fields
    - Task: verify/correct fields, NOT re-extract from scratch
    - Much cheaper than full extraction (shorter prompt, focused task)
    - Takes ~3-5 seconds (vs 15-30s for full LLM extraction)

Why this approach?
    Full LLM extraction: LLM reads raw pixels/text and extracts everything
    → slow, expensive, sometimes hallucinates

    OCR + LLM verify: Python does the heavy lifting (fast, deterministic),
    LLM just double-checks and fixes errors
    → fast, cheap, accurate
"""

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.models import Document, DocType, Entity, EntityType
from ..llm.provider import LLMProvider, LLMProviderError
from .classifier import DOC_TYPE_SCHEMAS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _image_to_text(file_path: Path) -> str:
    """
    Convert an image or PDF to raw text using pytesseract OCR.
    Supports: PNG, JPG, JPEG, TIFF, BMP, PDF.
    Upscales images smaller than 1000px on either dimension (minimum 2x).
    """
    suffix = file_path.suffix.lower()

    try:
        import pytesseract
        from PIL import Image

        if suffix == ".pdf":
            # Try pdf2image first (needs poppler: brew install poppler)
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(file_path), dpi=300)
                texts = []
                for page in pages:
                    texts.append(pytesseract.image_to_string(page, lang="eng"))
                return "\n".join(texts)
            except Exception as pdf_err:
                logger.warning("pdf2image failed (%s), trying PIL fallback", pdf_err)
                try:
                    img = Image.open(file_path)
                    return pytesseract.image_to_string(img, lang="eng")
                except Exception:
                    raise RuntimeError(
                        f"Cannot read PDF. Install poppler: brew install poppler\n{pdf_err}"
                    )
        else:
            img = Image.open(file_path)
            # Upscale small images for better OCR accuracy
            w, h = img.size
            if w < 1000 or h < 1000:
                scale = max(1000 / w, 1000 / h, 2.0)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            # Grayscale improves tesseract accuracy
            if img.mode != "L":
                img = img.convert("L")
            return pytesseract.image_to_string(img, lang="eng", config="--psm 3 --oem 3")

    except ImportError:
        raise RuntimeError(
            "pytesseract or Pillow not installed. "
            "Run: pip install pytesseract Pillow pdf2image"
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"OCR failed for {file_path.name}: {e}")


# ---------------------------------------------------------------------------
# Rule-based field extraction (no LLM)
# ---------------------------------------------------------------------------

def _regex_extract(text: str) -> dict[str, str]:
    """
    Fast rule-based extraction using regex patterns tuned for real OCR output.

    Handles OCR artifacts specific to scanned AZ driver licenses and Indian
    passports:
      - "saoen" / "saiss" labels → license number / issue / expiry
      - "poe" / "poc" artifact   → date of birth field
      - "exe" artifact           → expiry date
      - "iss" artifact           → issue date
      - MRZ lines P<IND...  and  WO462878<71ND...

    Priority: text-extracted fields take precedence; MRZ fills gaps only.
    """
    fields: dict[str, str] = {}

    # Split into lines and build a clean single-line version for most patterns
    raw_lines   = text.split("\n")
    clean_lines = [ln.strip() for ln in raw_lines if ln.strip()]
    clean_text  = " ".join(clean_lines)

    # -----------------------------------------------------------------------
    # 1. DOCUMENT TEXT PATTERNS — run first so they take priority
    # -----------------------------------------------------------------------

    # License number: letter + 7-9 digits (e.g. U10112277)
    m = re.search(r"\b([A-Z][0-9]{7,9})\b", clean_text)
    if m:
        fields["license_number"] = m.group(1).upper()

    # Collect all MM/DD/YYYY dates for context-aware assignment
    all_dates = re.findall(r"\b(\d{1,2}\/\d{1,2}\/\d{4})\b", clean_text)

    # --- DOB: "poe", "poc", "dob" followed by a date ---
    dob_m = re.search(
        r"\b(?:poe|poc|dob|date\s+of\s+birth|birth\s+date)\b[^0-9]*(\d{1,2}\/\d{1,2}\/\d{4})",
        clean_text, re.IGNORECASE
    )
    if dob_m:
        fields["dob"] = dob_m.group(1)

    # --- Expiry: "exe", "exp", "expiry", "date of expiry" followed by a date ---
    exp_m = re.search(
        r"\b(?:exe|exp|expires?|expiry|date\s+of\s+expiry)\b[^0-9]*(\d{1,2}\/\d{1,2}\/\d{4})",
        clean_text, re.IGNORECASE
    )
    if exp_m:
        fields["expiry_date"] = exp_m.group(1)

    # --- Issue date: "iss", "issued", "issue date", "date of issue", "date of tssue" ---
    iss_m = re.search(
        r"\b(?:iss|saiss|issued?|issue\s+date|date\s+of\s+[it]ssue)\b[^0-9]*(\d{1,2}\/\d{1,2}\/\d{4})",
        clean_text, re.IGNORECASE
    )
    if iss_m:
        fields["issue_date"] = iss_m.group(1)

    # Fallback date assignment: use year-based ordering when labels are unclear.
    # The latest year is expiry, the earliest year is dob (or issue), the middle is issue.
    # NOTE: This always runs and OVERRIDES label-based dates when we have exactly 2
    # dates — because label detection fails when labels and dates are on different lines
    # (e.g. Indian passports where "Date of Issue ... Date of Expiry" is one line and
    # "12/07/2022 11/07/2032" is the next).
    if all_dates:
        parsed: list[tuple[int, str]] = []
        for d in all_dates:
            d_parts = d.split("/")
            if len(d_parts) == 3:
                try:
                    parsed.append((int(d_parts[2]), d))
                except ValueError:
                    pass
        if len(parsed) >= 2:
            parsed.sort()
            # With exactly 2 dates: largest year = expiry, smallest = issue (or dob)
            # With 3 dates: largest = expiry, smallest = dob, middle = issue
            if not fields.get("dob") and len(parsed) == 3:
                fields["dob"] = parsed[0][1]           # earliest year = birth
            # Always set expiry from year ordering (most reliable signal)
            fields["expiry_date"] = parsed[-1][1]      # latest year  = expiry
            if not fields.get("issue_date"):
                if len(parsed) == 3:
                    fields["issue_date"] = parsed[1][1]  # middle year = issue
                elif len(parsed) == 2:
                    # Only 2 dates: the earlier one is likely issue date
                    fields["issue_date"] = parsed[0][1]

    # -----------------------------------------------------------------------
    # 2. NAME FROM OCR LINES — run before MRZ so clean line-based name wins
    # -----------------------------------------------------------------------
    # Strategy: scan lines for those that, after stripping leading noise, are a
    # single ALL-CAPS word (3-25 alpha chars). Excludes known document labels.
    #
    # AZ license OCR produces lines like:
    #   ": KANAGARAJ"          → strip leading ": " → "KANAGARAJ"
    #   "2 THAYAANANTHAN"      → strip leading "2 " → "THAYAANANTHAN"
    #
    # Indian passport OCR produces lines like:
    #   "} ss KANAGARA\"       → strip + fix trailing \ → "KANAGARAJ"
    #   "»  THAYAANANTHAN om"  → strip leading noise, drop trailing noise
    SKIP_WORDS = {
        "ARIZONA", "DRIVER", "LICENSE", "CLASS", "NONE",
        "ENDORSEMENTS", "RESTRICTIONS", "OPERATOR", "LIMITED",
        "TERM", "IND", "INDIAN", "PASSPORT", "VISA", "TYPE",
        "CODE", "SEX", "HGT", "WGT", "REV", "DOB", "ISS",
        "EXP", "DL", "ID", "ENON", "COIMBATORE",
    }

    caps_name_lines: list[str] = []
    for ln in clean_lines:
        # Strip leading noise: digits, punctuation, symbols
        stripped = re.sub(r"^[\d\W_]+", "", ln).strip()
        # Strip leading short lowercase noise like "ss ", "p " remnants
        stripped = re.sub(r"^[a-z]{1,3}\s+", "", stripped).strip()
        # Take only the first ALL-CAPS word (3-25 chars)
        word_only = re.match(r"^([A-Z]{3,25})", stripped)
        if not word_only:
            continue
        word = word_only.group(1)
        # Reject lines where substantial content follows the word — these are
        # body-text lines that happen to start with a caps word (e.g. "WRITS f fecadl...")
        # Only accept if the remainder after the word has ≤3 alphabetic characters
        remainder       = stripped[len(word):].strip()
        remainder_alpha = re.sub(r"[^a-zA-Z]", "", remainder)
        if len(remainder_alpha) > 3:
            continue
        # Fix common OCR artifact: trailing backslash = J was misread as \
        # e.g. "} ss KANAGARA\" → KANAGARAJ
        if word.endswith("A") and ln.rstrip().endswith("\\"):
            word += "J"
        if word not in SKIP_WORDS:
            caps_name_lines.append(word)

    if len(caps_name_lines) >= 2:
        # First CAPS word = surname, second = given name (license/passport layout)
        surname = caps_name_lines[0].title()
        given   = caps_name_lines[1].title()
        fields["name"] = f"{given} {surname}"
    elif len(caps_name_lines) == 1:
        fields["name"] = caps_name_lines[0].title()

    # -----------------------------------------------------------------------
    # 3. MRZ PARSING — fills gaps not already extracted from text
    # -----------------------------------------------------------------------

    # MRZ line 1: P<INDKANAGARAJ<<THAYAANANTHAN<<<<...
    # OCR sometimes appends 1-2 char noise before the << fillers
    mrz1 = re.search(r"P<([A-Z]{3})([A-Z<]{5,})", clean_text)
    if mrz1:
        country   = mrz1.group(1)
        name_part = mrz1.group(2)
        parts     = name_part.split("<<")
        # Surname: first part before any filler <
        surname_raw = parts[0].split("<")[0].strip()
        # Given names: second part — longest contiguous uppercase token ≥4 chars
        # (this filters out 1-2 char OCR noise like "KX")
        mrz_given = ""
        if len(parts) > 1:
            given_tokens = [t for t in parts[1].split("<") if re.match(r"^[A-Z]{4,}$", t)]
            if given_tokens:
                mrz_given = max(given_tokens, key=len).title()
        mrz_surname = surname_raw.title()
        # Only use MRZ name if we didn't already get a name from OCR lines
        if "name" not in fields and mrz_surname and mrz_given:
            fields["name"] = f"{mrz_given} {mrz_surname}"
        # Nationality always from MRZ (reliable)
        country_map = {"IND": "Indian", "USA": "American", "CAN": "Canadian",
                       "GBR": "British", "AUS": "Australian"}
        fields["nationality"] = country_map.get(country, country)

    # MRZ line 2: WO462878<71ND0301039M3207110...
    # Structure: passport_num(≤9)<check(1)country(3)dob(6)dobcheck(1)sex(1)expiry(6)
    # Note: OCR often reads 'I' as '1' in the country code (1ND instead of IND)
    mrz2 = re.search(
        r"([A-Z][A-Z0-9]{6,8})<\d[1I][A-Z]{2}(\d{6})\d[MF](\d{6})",
        clean_text
    )
    if mrz2:
        passport_num = mrz2.group(1)
        dob_raw      = mrz2.group(2)   # YYMMDD
        exp_raw      = mrz2.group(3)   # YYMMDD
        fields["passport_number"] = passport_num
        # Only fill dob/expiry from MRZ if not already extracted from document text
        if "dob" not in fields:
            yy, mm, dd = dob_raw[:2], dob_raw[2:4], dob_raw[4:6]
            year = f"20{yy}" if int(yy) <= 30 else f"19{yy}"
            fields["dob"] = f"{mm}/{dd}/{year}"
        if "expiry_date" not in fields:
            exp_yy, exp_mm, exp_dd = exp_raw[:2], exp_raw[2:4], exp_raw[4:6]
            exp_year = f"20{exp_yy}" if int(exp_yy) <= 50 else f"19{exp_yy}"
            fields["expiry_date"] = f"{exp_mm}/{exp_dd}/{exp_year}"

    # -----------------------------------------------------------------------
    # 4. PLACE OF BIRTH (passports)
    # -----------------------------------------------------------------------
    if "place_of_birth" not in fields:
        # Explicit label first
        m = re.search(
            r"(?:place\s+of\s+birth|birthplace|pob)[:\s]+([A-Za-z][A-Za-z\s,]+?)(?:\s{2,}|$|\n)",
            text, re.IGNORECASE
        )
        if m:
            fields["place_of_birth"] = m.group(1).strip().title()
        else:
            # Fallback: look for known city names appearing in OCR
            pob_m = re.search(
                r"\b(COIMBATORE(?:[,\s]+TAMIL\s+NADU)?)\b",
                clean_text, re.IGNORECASE
            )
            if pob_m:
                fields["place_of_birth"] = pob_m.group(1).strip().title()

    # -----------------------------------------------------------------------
    # 5. LICENSE CLASS
    # -----------------------------------------------------------------------
    if "class" not in fields:
        m = re.search(r"\bCLASS\s+([A-D])\b", clean_text, re.IGNORECASE)
        if m:
            fields["class"] = m.group(1).upper()

    # -----------------------------------------------------------------------
    # 6. STATE
    # -----------------------------------------------------------------------
    if "state" not in fields:
        m = re.search(r"\b(AZ|CA|NY|TX|FL|BC|ON|AB|QC|ARIZONA)\b", clean_text)
        if m:
            v = m.group(1)
            fields["state"] = "AZ" if v == "ARIZONA" else v

    # -----------------------------------------------------------------------
    # 7. HEIGHT
    # -----------------------------------------------------------------------
    if "height" not in fields:
        m = re.search(r"\bHGT\s+([0-9]{2,3})\b", clean_text, re.IGNORECASE)
        if not m:
            m = re.search(r"(?:height|hgt)[:\s]+([0-9]{2,3})", clean_text, re.IGNORECASE)
        if m:
            ht = m.group(1).strip()
            if len(ht) == 2:
                fields["height"] = f"{ht[0]}'{ht[1]}\""
            elif len(ht) == 3:
                fields["height"] = f"{ht[0]}'{ht[1:]}\""
            else:
                fields["height"] = ht

    # -----------------------------------------------------------------------
    # 8. WEIGHT
    # -----------------------------------------------------------------------
    if "weight" not in fields:
        m = re.search(r"\bWGT\s+([0-9]{2,3})\b", clean_text, re.IGNORECASE)
        if not m:
            m = re.search(r"(?:weight|wgt)[:\s]+([0-9]{2,3})\s*(?:lbs?|kg)?",
                          clean_text, re.IGNORECASE)
        if m:
            fields["weight"] = f"{m.group(1).strip()} lbs"

    # -----------------------------------------------------------------------
    # 9. EYE COLOR
    # -----------------------------------------------------------------------
    if "eye_color" not in fields:
        m = re.search(r"\b(BRN|BLU|GRN|HAZ|GRY|BLK|AMB)\b", clean_text)
        if not m:
            m = re.search(
                r"(?:eyes?|eye\s*color)[:\s]+([A-Z]{3,8})\b",
                clean_text, re.IGNORECASE
            )
        if m:
            color_map = {"BRN": "Brown", "BLU": "Blue", "GRN": "Green",
                         "HAZ": "Hazel", "GRY": "Gray", "BLK": "Black",
                         "AMB": "Amber"}
            val = m.group(1).upper()
            fields["eye_color"] = color_map.get(val, val.title())

    # -----------------------------------------------------------------------
    # 10. SEX
    # -----------------------------------------------------------------------
    if "sex" not in fields:
        m = re.search(r"\bSEX\s+([MF])\b", clean_text)
        if not m:
            m = re.search(r"(?:sex|gender)[:\s]+([MF](?:ale|emale)?)\b",
                          clean_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().upper()[0]
            fields["sex"] = "Male" if val == "M" else "Female"

    # -----------------------------------------------------------------------
    # 11. ADDRESS
    # -----------------------------------------------------------------------
    if "address" not in fields:
        m = re.search(
            r"([A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)",
            clean_text
        )
        if m:
            fields["address"] = m.group(1).strip()

    return fields


# ---------------------------------------------------------------------------
# LLM verifier prompt
# ---------------------------------------------------------------------------

_VERIFY_PROMPT = """\
You are a document verification assistant for identity documents.

Below is:
1. RAW OCR TEXT from a scanned document
2. FIELDS already extracted by a regex parser

Your task:
- Review the extracted fields and CORRECT errors
- Fix OCR artifacts: 0→O, 1→l, broken names across lines, etc.
- For NAMES: combine multi-line OCR names (e.g. "KANAGARAJ" + "THAYAANANTHAN" → "Thayaananthan Kanagaraj")
- For HEIGHT: normalize to feet/inches (e.g. "604" → "6'04\\"", "6-04" → "6'04\\"")
- For dates: use MM/DD/YYYY format as it appears on the document
- Do NOT hallucinate — only use information visible in the OCR text
- Return ONLY a JSON object, no explanation

OCR TEXT (first 1500 chars):
{ocr_text}

REGEX-EXTRACTED FIELDS:
{regex_fields}

DOCUMENT TYPE: {doc_type}

JSON only:"""


# ---------------------------------------------------------------------------
# Main OCR Extractor class
# ---------------------------------------------------------------------------

class OCRExtractor:
    """
    Extracts entities from images/PDFs using OCR + LLM verification.

    Pipeline:
    1. pytesseract converts image → raw text (fast, no LLM)
    2. regex patterns extract common fields (fast, no LLM)
    3. LLM verifies/corrects the regex results (light, fast)
    4. Returns Document + Entity with provenance

    Usage:
        extractor = OCRExtractor(llm_provider)
        doc, entities = extractor.extract("passport.jpg", person_name="thayaananthan")
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str = "unknown"):
        self._llm        = llm_provider
        self._model_name = model_name

    def extract(
        self,
        file_path:     str | Path,
        person_name:   str = "",
        doc_type_hint: Optional[DocType] = None,
    ) -> tuple[Optional[Document], list[Entity]]:
        """
        Extract entities from an image or PDF file.

        Args:
            file_path      — path to image/PDF
            person_name    — optional hint for the person's name
            doc_type_hint  — optional hint for document type

        Returns:
            (Document, list[Entity]) — ready for KnowledgeGraphBuilder
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            return None, []

        # Step 1: OCR → raw text
        try:
            raw_text = _image_to_text(file_path)
            logger.info("OCR extracted %d chars from %s", len(raw_text), file_path.name)
        except RuntimeError as e:
            logger.error("OCR failed: %s", e)
            return None, []

        if not raw_text.strip():
            logger.warning("OCR returned empty text for %s", file_path.name)
            return None, []

        # Step 2: Rule-based extraction (fast, no LLM)
        regex_fields = _regex_extract(raw_text)
        logger.info("Regex extracted %d fields: %s", len(regex_fields), list(regex_fields.keys()))

        # Step 3: Detect document type from OCR text
        doc_type = doc_type_hint or self._detect_doc_type(raw_text)

        # Step 4: LLM verification (light pass — corrects, doesn't re-extract)
        verified_fields = self._verify_with_llm(raw_text, regex_fields, doc_type)
        logger.info("After LLM verify: %d fields", len(verified_fields))

        # Step 5: Build Document object
        lines = raw_text.split("\n")
        line_offsets: list[int] = []
        pos = 0
        for line in lines:
            line_offsets.append(pos)
            pos += len(line) + 1
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        doc = Document(
            doc_id       = str(uuid.uuid4()),
            filename     = file_path.name,
            text         = raw_text,
            lines        = lines,
            paragraphs   = paragraphs,
            line_offsets = line_offsets,
            doc_type     = doc_type,
            doc_date     = None,
            empty        = False,
            metadata     = {
                "full_path":      str(file_path.resolve()),
                "ocr_char_count": len(raw_text),
                "extractor":      "ocr+llm-verify",
                "person_name":    person_name,
            },
        )

        # Step 6: Build Entity
        name = (
            verified_fields.get("name")
            or person_name.replace("_", " ").title()
            or "Unknown"
        )

        line_number, line_text = self._find_in_lines(name, lines)
        char_offset_start = line_offsets[line_number - 1] if line_number > 0 else 0

        schema     = DOC_TYPE_SCHEMAS.get(doc_type.value, DOC_TYPE_SCHEMAS["GENERIC"])
        n_found    = sum(1 for k in schema if verified_fields.get(k))
        n_total    = len(schema)
        confidence = round(n_found / n_total, 2) if n_total > 0 else 0.5

        entity = Entity(
            entity_id            = str(uuid.uuid4()),
            name                 = name,
            entity_type          = EntityType.PERSON,
            attributes           = verified_fields,
            source_doc_id        = doc.doc_id,
            source_filename      = file_path.name,
            doc_type             = doc_type,
            line_number          = line_number,
            line_text            = line_text,
            paragraph_index      = 0,
            paragraph_text       = paragraphs[0] if paragraphs else "",
            char_offset_start    = char_offset_start,
            char_offset_end      = char_offset_start + len(line_text),
            extractor_model      = f"ocr+{self._model_name}",
            extraction_timestamp = datetime.now(timezone.utc).isoformat(),
            confidence           = confidence,
            embedding            = None,
        )

        return doc, [entity]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_doc_type(self, text: str) -> DocType:
        """Heuristic doc type detection from OCR text keywords."""
        text_lower = text.lower()

        license_signals = [
            "driver", "licence", "license", "driving", "motor vehicle",
            "dmv", "class", "endorsements", "restrictions", "lic no",
            "license no", "ltd-term", "limited-term",
        ]
        if any(w in text_lower for w in license_signals):
            return DocType.DRIVERS_LICENSE

        passport_signals = [
            "passport", "travel document", "p<", "nationality",
            "place of birth", "date of issue", "authority",
        ]
        if any(w in text_lower for w in passport_signals):
            return DocType.PASSPORT

        bc_signals = [
            "birth certificate", "certificate of birth", "registration no",
            "place of birth", "father", "mother", "registrar",
        ]
        if any(w in text_lower for w in bc_signals):
            return DocType.BIRTH_CERTIFICATE

        ins_signals = [
            "insurance", "policy", "premium", "coverage",
            "beneficiary", "insured", "policyholder",
        ]
        if any(w in text_lower for w in ins_signals):
            return DocType.INSURANCE

        med_signals = [
            "patient", "diagnosis", "prescription", "hospital",
            "physician", "medications", "clinic",
        ]
        if any(w in text_lower for w in med_signals):
            return DocType.MEDICAL_RECORD

        # Heuristic fallback
        has_date  = bool(re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b", text))
        has_state = bool(re.search(r"\b(AZ|CA|NY|TX|FL|BC|ON|AB|QC)\b", text))
        has_dob   = bool(re.search(r"dob|birth|born|poe|poc", text_lower))
        if has_date and (has_state or has_dob):
            return DocType.DRIVERS_LICENSE

        return DocType.GENERIC

    def _verify_with_llm(
        self,
        ocr_text:     str,
        regex_fields: dict[str, str],
        doc_type:     DocType,
    ) -> dict[str, str]:
        """
        Ask LLM to verify and correct regex-extracted fields.
        Only sends first 1500 chars of OCR to keep the prompt short.
        Falls back gracefully to regex_fields if LLM fails.
        """
        import json

        ocr_preview = ocr_text[:1500].strip()
        prompt = _VERIFY_PROMPT.format(
            ocr_text     = ocr_preview,
            regex_fields = json.dumps(regex_fields, indent=2),
            doc_type     = doc_type.value,
        )

        try:
            response = self._llm.complete(prompt, temperature=0.0)
            start = response.find("{")
            end   = response.rfind("}") + 1
            if start >= 0 and end > start:
                corrected = json.loads(response[start:end])
                merged    = {**regex_fields, **corrected}
                return {k: str(v) for k, v in merged.items() if v and str(v).strip()}
        except (json.JSONDecodeError, LLMProviderError, Exception) as e:
            logger.warning("LLM verification failed: %s — using regex fields only", e)

        return regex_fields

    def _find_in_lines(self, value: str, lines: list[str]) -> tuple[int, str]:
        """
        Find a value in document lines.
        Returns (line_number 1-indexed, line_text).
        Returns (0, "") if not found.
        """
        if not value:
            return 0, ""
        for i, line in enumerate(lines):
            if value.lower() in line.lower():
                return i + 1, line
        return 0, ""
