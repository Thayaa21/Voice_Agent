"""UiPath Document Understanding API Connector"""
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

UIPATH_TOKEN_URL = "https://cloud.uipath.com/identity_/connect/token"
SCOPE = "Du.Classification.Api Du.Digitization.Api Du.Extraction.Api"
POLL_INTERVAL = 2
MAX_POLL = 30

UIPATH_DOCTYPE_MAP = {
    "id_cards": "DRIVERS_LICENSE",
    "passports": "PASSPORT",
    "invoices": "INSURANCE",
}


class UiPathAPIError(Exception):
    pass


class UiPathAPIConnector:
    def __init__(self, client_id: str, client_secret: str,
                 org: str = "", tenant: str = "DefaultTenant"):
        self._client_id = client_id
        self._client_secret = client_secret
        self._org = org
        self._tenant = tenant
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0

    @classmethod
    def from_env(cls) -> "UiPathAPIConnector":
        cid = os.getenv("UIPATH_CLIENT_ID", "")
        cs = os.getenv("UIPATH_CLIENT_SECRET", "")
        if not cid or not cs:
            raise UiPathAPIError(
                "UIPATH_CLIENT_ID and UIPATH_CLIENT_SECRET must be set in .env"
            )
        return cls(
            client_id=cid,
            client_secret=cs,
            org=os.getenv("UIPATH_ORG", ""),
            tenant=os.getenv("UIPATH_TENANT", "DefaultTenant"),
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """OAuth2 client credentials, cached with expiry."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        resp = requests.post(UIPATH_TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": SCOPE,
        }, timeout=30)
        if not resp.ok:
            raise UiPathAPIError(
                f"UiPath auth failed ({resp.status_code}): {resp.text[:200]}"
            )
        data = resp.json()
        if "access_token" not in data:
            raise UiPathAPIError(f"No access_token in response: {data}")
        self._token = data["access_token"]
        self._token_expiry = time.time() + int(data.get("expires_in", 3600))
        return self._token

    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/json",
            "X-UIPATH-TenantName": self._tenant,
        }

    def _base_url(self) -> str:
        return (
            f"https://cloud.uipath.com/{self._org}/{self._tenant}"
            "/du_/api/framework"
        )

    def test_connection(self) -> bool:
        """Test that a valid token can be obtained."""
        try:
            return bool(self._get_token())
        except Exception as exc:
            logger.error("Connection test failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_to_json(
        self,
        file_path: "str | Path",
        extractor: str = "identity_documents",
        output_dir: "Optional[str | Path]" = None,
    ) -> Path:
        """
        Full pipeline: get project_id → digitize → poll dig result →
        extract with id_cards/passports/generative_extractor → poll
        extraction result → convert → save JSON.
        Returns path to the saved JSON file.
        """
        raw = self.extract_raw(file_path, extractor=extractor)
        file_path = Path(file_path)
        pipeline_json = self._convert_result(raw, str(file_path))

        out_dir = Path(output_dir) if output_dir else file_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (file_path.stem + "_uipath.json")
        out_path.write_text(json.dumps(pipeline_json, indent=2))
        logger.info("Saved extraction result: %s", out_path)
        return out_path

    def extract_raw(
        self,
        file_path: "str | Path",
        extractor: str = "identity_documents",
    ) -> dict:
        """
        Full pipeline returning raw dict.
        For identity_documents: tries id_cards, then passports,
        then generative_extractor.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        base = self._base_url()
        headers = self._auth_headers()

        # Step 1: get project id
        project_id = self._get_project_id(base, headers)
        logger.info("Using project_id: %s", project_id)

        # Step 2: digitize via multipart/form-data → get doc_id + resultUrl
        dig_data = self._digitize(base, headers, project_id, file_path)
        doc_id = dig_data.get("documentId") or dig_data.get("id", "")
        if not doc_id:
            raise UiPathAPIError(
                f"No documentId in digitization response: {dig_data}"
            )
        logger.info("Got doc_id: %s", doc_id)

        # Step 3: poll digitization resultUrl until Succeeded
        result_url = dig_data.get("resultUrl", "")
        if result_url:
            logger.info("Polling digitization result...")
            self._poll_url(result_url, headers, expected_key="status")

        # Step 4: determine which extractors to try
        if extractor == "identity_documents":
            extractor_ids = ["id_cards", "passports", "generative_extractor"]
        else:
            extractor_ids = [extractor]

        # Step 5: extract fields (tries each extractor in order)
        return self._extract_fields(base, headers, project_id, doc_id, extractor_ids)

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _get_project_id(self, base_url: str, headers: dict) -> str:
        """Returns first project id or the predefined project UUID."""
        url = f"{base_url}/projects?api-version=1"
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.ok:
            try:
                data = resp.json()
                projects = data.get("projects", [])
                if projects:
                    pid = projects[0].get("id", "00000000-0000-0000-0000-000000000000")
                    logger.info("Found project: %s", pid)
                    return pid
            except Exception:
                pass
        logger.warning(
            "Could not fetch projects (%s), using predefined project",
            resp.status_code,
        )
        return "00000000-0000-0000-0000-000000000000"

    def _digitize(
        self, base_url: str, headers: dict, project_id: str, file_path: Path
    ) -> dict:
        """
        Upload file via multipart/form-data.
        Returns dict with at least documentId and optionally resultUrl.
        """
        suffix = file_path.suffix.lower()
        mime = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }.get(suffix, "application/octet-stream")

        url = (
            f"{base_url}/projects/{project_id}"
            "/digitization/start?api-version=1"
        )
        # Omit Content-Type so requests sets the multipart boundary automatically
        upload_headers = {
            "Authorization": headers["Authorization"],
            "Accept": "application/json",
            "X-UIPATH-TenantName": self._tenant,
        }

        with open(file_path, "rb") as fh:
            files = {"File": (file_path.name, fh, mime)}
            resp = requests.post(
                url, headers=upload_headers, files=files, timeout=60
            )

        if resp.status_code not in (200, 202):
            raise UiPathAPIError(
                f"Digitization failed ({resp.status_code}): {resp.text[:400]}"
            )

        try:
            data = resp.json()
        except Exception:
            raise UiPathAPIError(
                f"Digitization returned non-JSON ({resp.status_code}): "
                f"{resp.text[:200]}"
            )

        logger.info("Digitization response: %s", data)
        return data

    def _poll_url(self, url: str, headers: dict, expected_key: str = "status") -> dict:
        """
        Generic polling until status=Succeeded (or similar terminal state).
        Treats NotStarted/Running/Pending as still in progress.
        """
        for attempt in range(MAX_POLL):
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.ok:
                try:
                    data = resp.json()
                except Exception:
                    time.sleep(POLL_INTERVAL)
                    continue

                status = str(data.get("status", "")).lower()
                logger.debug("Poll %d: status=%s", attempt + 1, status)

                if status in ("succeeded", "completed"):
                    return data
                elif status in ("failed", "error"):
                    raise UiPathAPIError(f"Operation failed: {data}")
                # notstarted / running / pending / "" → keep polling
            else:
                logger.warning(
                    "Poll attempt %d returned %s", attempt + 1, resp.status_code
                )

            time.sleep(POLL_INTERVAL)

        raise UiPathAPIError(
            f"Polling timed out after {MAX_POLL} attempts: {url}"
        )

    def _extract_fields(
        self,
        base_url: str,
        headers: dict,
        project_id: str,
        doc_id: str,
        extractor_ids: "list[str]",
    ) -> dict:
        """Tries each extractor in order, returns the first successful result."""
        last_error = None

        for ext_id in extractor_ids:
            logger.info("Trying extractor: %s", ext_id)
            url = (
                f"{base_url}/projects/{project_id}/extractors"
                f"/{ext_id}/extraction/start?api-version=1"
            )
            payload = {"documentId": doc_id}
            post_headers = {**headers, "Content-Type": "application/json"}

            try:
                resp = requests.post(
                    url, headers=post_headers, json=payload, timeout=60
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Extractor %s request error: %s", ext_id, exc)
                continue

            if resp.status_code == 404:
                logger.info("Extractor %s not found (404), skipping", ext_id)
                continue

            if resp.status_code not in (200, 202):
                last_error = (
                    f"Extractor {ext_id}: {resp.status_code} {resp.text[:200]}"
                )
                logger.warning(last_error)
                continue

            try:
                data = resp.json()
            except Exception:
                last_error = f"Extractor {ext_id}: non-JSON response"
                logger.warning(last_error)
                continue

            logger.info(
                "Extractor %s: operationId=%s resultUrl=%s",
                ext_id,
                data.get("operationId"),
                data.get("resultUrl", "")[:60],
            )

            # Poll the resultUrl until Succeeded
            result_url = data.get("resultUrl", "")
            if result_url:
                try:
                    result = self._poll_url(result_url, headers, expected_key="status")
                    result["_extractor_id"] = ext_id
                    return result
                except UiPathAPIError as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Extractor %s poll failed: %s", ext_id, exc
                    )
                    continue
            else:
                # Synchronous result
                data["_extractor_id"] = ext_id
                return data

        raise UiPathAPIError(f"All extractors failed. Last error: {last_error}")

    # ------------------------------------------------------------------
    # Convert to pipeline JSON format
    # ------------------------------------------------------------------

    def _convert_result(self, raw: dict, source_file: str) -> dict:
        """
        Convert UiPath extraction result to pipeline JSON format.
        Handles ResultsDocument.Fields array format.
        """
        # Navigate to the Fields array:
        # raw["result"]["extractionResult"]["ResultsDocument"]["Fields"]
        result_obj = raw.get("result", raw)
        extraction_result = result_obj.get("extractionResult", result_obj)
        results_doc = extraction_result.get("ResultsDocument", {})
        fields_array = results_doc.get("Fields", [])

        doc_type_id_from_result = results_doc.get("DocumentTypeId", "")
        extractor_id = raw.get("_extractor_id", doc_type_id_from_result)

        # Build raw_fields map: FieldName → {value, confidence}
        raw_fields: dict = {}
        for field in fields_array:
            field_name = field.get("FieldName", "")
            is_missing = field.get("IsMissing", False)
            values = field.get("Values", [])
            if is_missing or not values:
                continue
            best = values[0]
            value = best.get("Value") or best.get("DerivedValue") or ""
            conf = float(best.get("Confidence", 1.0))
            if str(value).strip():
                raw_fields[field_name] = {
                    "value": str(value).strip(),
                    "confidence": conf,
                }

        # FieldName → output key mapping
        FIELD_MAP = {
            "Last Name": "_last_name",
            "First Name": "_first_name",
            "Birth Date": "dob",
            "Expiration Date": "expiry_date",
            "Issued Date": "issue_date",
            "ID Number": "license_number",
            "Document Number": "passport_number",
            "Passport Number": "passport_number",
            "Address": "address",
            "Sex": "sex",
            "State": "state",
            "Document Type": "_doc_type_raw",
            # Passport variants
            "Surname": "_last_name",
            "Given Names": "_first_name",
            "Date of Birth": "dob",
            "Date of Expiry": "expiry_date",
            "Date of Issue": "issue_date",
            "Nationality": "nationality",
            "Place of Birth": "place_of_birth",
            "MRZ": "mrz",
            "MRZ Line 1": "mrz_line1",
            "MRZ Line 2": "mrz_line2",
        }

        fields: dict = {}
        last_name = ""
        first_name = ""
        doc_type_raw = ""
        last_name_conf = 0.0
        first_name_conf = 0.0

        for field_name, fdata in raw_fields.items():
            mapped = FIELD_MAP.get(
                field_name, field_name.lower().replace(" ", "_")
            )
            val = fdata["value"]
            conf = fdata["confidence"]

            if mapped == "_last_name":
                last_name = val
                last_name_conf = conf
            elif mapped == "_first_name":
                first_name = val
                first_name_conf = conf
            elif mapped == "_doc_type_raw":
                doc_type_raw = val
            else:
                fields[mapped] = {
                    "value": val,
                    "confidence": round(min(1.0, max(0.0, conf)), 3),
                    "page": 1,
                    "bounding_box": [0, 0, 0, 0],
                }

        # Combine First Name + Last Name → "name"
        if first_name or last_name:
            # Title-case names that are in all-caps
            def _title(s: str) -> str:
                return s.title() if s.isupper() else s

            combined = f"{_title(first_name)} {_title(last_name)}".strip()
            avg_conf = (
                (last_name_conf + first_name_conf) / 2
                if (first_name and last_name)
                else (last_name_conf or first_name_conf)
            )
            fields["name"] = {
                "value": combined,
                "confidence": round(min(1.0, max(0.0, avg_conf)), 3),
                "page": 1,
                "bounding_box": [0, 0, 0, 0],
            }

        # Auto-detect document type
        doc_type = _detect_doc_type(doc_type_raw, extractor_id)

        # Overall confidence: average of field confidences
        all_confs = [f["confidence"] for f in fields.values()]
        overall_conf = (
            round(sum(all_confs) / len(all_confs), 3) if all_confs else 1.0
        )

        return {
            "document_type": doc_type,
            "confidence": overall_conf,
            "source_file": str(source_file),
            "fields": fields,
        }


def _detect_doc_type(doc_type_raw: str, extractor_id: str) -> str:
    """Auto-detect document type from Document Type field value or extractor id."""
    raw_lower = doc_type_raw.lower()
    if "passport" in raw_lower:
        return "PASSPORT"
    if "driver" in raw_lower or "license" in raw_lower or "licence" in raw_lower:
        return "DRIVERS_LICENSE"

    ext_lower = str(extractor_id).lower()
    if "passport" in ext_lower:
        return "PASSPORT"
    if (
        "id_card" in ext_lower
        or "id-card" in ext_lower
        or ext_lower == "id_cards"
    ):
        return "DRIVERS_LICENSE"
    if "invoice" in ext_lower:
        return "INSURANCE"

    return UIPATH_DOCTYPE_MAP.get(extractor_id, "GENERIC")
