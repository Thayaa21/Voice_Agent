"""
build_patient_graph.py
======================
Converts Synthea CSV dataset into per-patient JSON files for the Graph RAG backend.

What it does:
- Reads patients, encounters, conditions, payers, payer_transitions, providers
- Assigns a synthetic display name (Option A pseudonymization) to each patient UUID
- Outputs one folder per patient under graph_rag_backend/sample_data/
- Each folder contains a single patient.json with only clinically/financially relevant fields
- Drops: driver's license, passport, birth certificate (irrelevant to hospital ops)
- Keeps: demographics, insurance, encounter history, active conditions, claim financials

Claim status logic:
- PAID:         payer_coverage >= total_claim_cost * 0.95
- DENIED_NO_PA: payer_coverage == 0 and total_claim_cost > 500
- PENDING_P2P:  payer_coverage > 0 and payer_coverage < total_claim_cost * 0.5 and cost > 1000
- PARTIAL:      everything else with a coverage gap

Usage:
    python dataset/build_patient_graph.py
    python dataset/build_patient_graph.py --limit 100   # only first 100 patients
    python dataset/build_patient_graph.py --out graph_rag_backend/sample_data
"""

import csv
import json
import argparse
import random
import re
import shutil
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# SYNTHETIC NAME POOLS (Option A pseudonymization)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Aiden", "Sofia", "Marcus", "Elena", "James", "Priya", "Noah", "Layla",
    "Ethan", "Zara", "Liam", "Fatima", "Lucas", "Mei", "Oliver", "Amara",
    "Henry", "Ingrid", "Samuel", "Yuki", "Daniel", "Nadia", "Matthew", "Sara",
    "Nathan", "Chloe", "Isaac", "Leila", "Owen", "Mia", "Leo", "Ava",
    "Caleb", "Hannah", "Ryan", "Grace", "Dylan", "Lily", "Evan", "Aria",
    "Logan", "Nora", "Tyler", "Ella", "Jordan", "Scarlett", "Alex", "Violet",
    "Casey", "Aurora", "Morgan", "Stella", "Riley", "Hazel", "Taylor", "Willow",
    "Cameron", "Ivy", "Blake", "Luna", "Skylar", "Nova", "Quinn", "Maya",
    "Avery", "Cora", "Peyton", "Alice", "Reese", "Ruby", "Hayden", "Ellie",
    "Kendall", "Isla", "Sydney", "Naomi", "Paige", "Lydia", "Brooke", "Penelope",
]

LAST_NAMES = [
    "Chen", "Patel", "Kim", "Nguyen", "Singh", "Ali", "Martinez", "Johnson",
    "Williams", "Brown", "Davis", "Wilson", "Anderson", "Taylor", "Thomas",
    "Jackson", "White", "Harris", "Martin", "Garcia", "Thompson", "Lee",
    "Perez", "Walker", "Hall", "Young", "Allen", "King", "Wright", "Scott",
    "Torres", "Rivera", "Phillips", "Campbell", "Parker", "Evans", "Edwards",
    "Collins", "Stewart", "Morris", "Yamamoto", "Mueller", "Okonkwo", "Petrov",
    "Santos", "Rossi", "Kowalski", "Ahmed", "Sharma", "Tanaka", "Park", "Wu",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def derive_claim_status(total_cost: float, payer_coverage: float) -> str:
    if total_cost == 0:
        return "PAID"
    coverage_ratio = payer_coverage / total_cost
    if coverage_ratio >= 0.95:
        return "PAID"
    elif payer_coverage == 0 and total_cost > 100:
        # Zero coverage on any real encounter = denied
        return "DENIED_NO_PA"
    elif payer_coverage > 0 and coverage_ratio < 0.50 and total_cost > 300:
        # Partial coverage under 50% on a meaningful claim = stuck in review
        return "PENDING_P2P"
    else:
        return "PARTIAL"


import re

def clean_provider_name(raw: str) -> str:
    """Remove Synthea numeric suffixes like 'Maile198 Frami345' → 'Dr. Maile Frami'"""
    if not raw or raw == "Unknown Provider":
        return "Unknown Provider"
    cleaned = re.sub(r'\d+', '', raw).strip()
    parts = cleaned.split()
    parts = [p for p in parts if len(p) > 1]
    if not parts:
        return "Unknown Provider"
    return "Dr. " + " ".join(p.capitalize() for p in parts)


def safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def format_date(dt_str: str) -> str:
    """Convert ISO datetime or date string to YYYY-MM-DD."""
    if not dt_str:
        return ""
    return dt_str[:10]


def age_from_dob(dob: str) -> int:
    try:
        born = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# MAIN BUILD
# ---------------------------------------------------------------------------

def build(dataset_dir: Path, out_dir: Path, limit: int | None, seed: int):
    rng = random.Random(seed)

    print("Loading CSVs...")
    patients        = load_csv(dataset_dir / "patients.csv")
    encounters_raw  = load_csv(dataset_dir / "encounters.csv")
    conditions_raw  = load_csv(dataset_dir / "conditions.csv")
    payers_raw      = load_csv(dataset_dir / "payers.csv")
    transitions_raw = load_csv(dataset_dir / "payer_transitions.csv")
    providers_raw   = load_csv(dataset_dir / "providers.csv")

    # Build lookup dicts
    payer_map     = {p["Id"]: p["NAME"] for p in payers_raw}
    provider_map  = {p["Id"]: p["NAME"] for p in providers_raw}

    # Group encounters by patient
    enc_by_patient: dict[str, list] = {}
    for e in encounters_raw:
        enc_by_patient.setdefault(e["PATIENT"], []).append(e)

    # Group conditions by patient
    cond_by_patient: dict[str, list] = {}
    for c in conditions_raw:
        cond_by_patient.setdefault(c["PATIENT"], []).append(c)

    # Group payer transitions by patient
    trans_by_patient: dict[str, list] = {}
    for t in transitions_raw:
        trans_by_patient.setdefault(t["PATIENT"], []).append(t)

    # Generate unique synthetic names
    used_names = set()
    def gen_name() -> tuple[str, str]:
        for _ in range(1000):
            first = rng.choice(FIRST_NAMES)
            last  = rng.choice(LAST_NAMES)
            if (first, last) not in used_names:
                used_names.add((first, last))
                return first, last
        # fallback with index suffix
        first = rng.choice(FIRST_NAMES)
        last  = rng.choice(LAST_NAMES)
        return first, last + str(rng.randint(2, 99))

    # Clean and recreate output directory
    if out_dir.exists():
        print(f"Clearing existing output: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if limit:
        patients = patients[:limit]

    print(f"Building {len(patients)} patient records...")
    manifest = []

    for patient in patients:
        pid        = patient["Id"]
        first, last = gen_name()
        display_name = f"{first} {last}"
        folder_name  = f"{first.lower()}_{last.lower()}"

        dob    = format_date(patient["BIRTHDATE"])
        gender = patient["GENDER"]  # M/F
        city   = patient["CITY"]
        state  = patient["STATE"]
        age    = age_from_dob(dob)

        # --- Current insurance (latest payer transition) ---
        transitions = sorted(
            trans_by_patient.get(pid, []),
            key=lambda x: int(x["END_YEAR"]) if x["END_YEAR"] else 9999,
            reverse=True
        )
        current_payer_id   = transitions[0]["PAYER"] if transitions else ""
        current_payer_name = payer_map.get(current_payer_id, "NO_INSURANCE")
        insurance_since    = transitions[0]["START_YEAR"] if transitions else ""

        # --- Encounters: last 20, sorted newest first ---
        encounters = sorted(
            enc_by_patient.get(pid, []),
            key=lambda x: x["START"],
            reverse=True
        )[:20]

        encounter_list = []
        for e in encounters:
            total_cost     = safe_float(e["TOTAL_CLAIM_COST"])
            payer_coverage = safe_float(e["PAYER_COVERAGE"])
            patient_owes   = round(total_cost - payer_coverage, 2)
            claim_status   = derive_claim_status(total_cost, payer_coverage)
            payer_name     = payer_map.get(e["PAYER"], "Unknown")
            provider_name  = clean_provider_name(provider_map.get(e["PROVIDER"], ""))

            encounter_list.append({
                "encounter_id":    e["Id"],
                "date":            format_date(e["START"]),
                "type":            e["ENCOUNTERCLASS"],          # ambulatory, inpatient, wellness, etc.
                "description":     e["DESCRIPTION"],
                "reason":          e["REASONDESCRIPTION"] or "",
                "procedure_code":  e["CODE"],
                "provider":        provider_name,
                "payer":           payer_name,
                "billed":          total_cost,
                "covered":         payer_coverage,
                "patient_owes":    patient_owes,
                "claim_status":    claim_status,
            })

        # --- Active conditions (no STOP date = still active) ---
        all_conditions = cond_by_patient.get(pid, [])
        active_conditions = [
            {
                "code":        c["CODE"],
                "description": c["DESCRIPTION"],
                "onset":       format_date(c["START"]),
            }
            for c in all_conditions
            if not c.get("STOP", "").strip()
        ]

        # Deduplicate conditions by code
        seen_codes = set()
        deduped_conditions = []
        for c in active_conditions:
            if c["code"] not in seen_codes:
                seen_codes.add(c["code"])
                deduped_conditions.append(c)

        # --- Claim summary stats ---
        total_billed   = sum(safe_float(e["TOTAL_CLAIM_COST"]) for e in enc_by_patient.get(pid, []))
        total_covered  = sum(safe_float(e["PAYER_COVERAGE"])   for e in enc_by_patient.get(pid, []))
        total_owed     = round(total_billed - total_covered, 2)
        denied_count   = sum(
            1 for e in enc_by_patient.get(pid, [])
            if derive_claim_status(safe_float(e["TOTAL_CLAIM_COST"]), safe_float(e["PAYER_COVERAGE"])) == "DENIED_NO_PA"
        )
        pending_count  = sum(
            1 for e in enc_by_patient.get(pid, [])
            if derive_claim_status(safe_float(e["TOTAL_CLAIM_COST"]), safe_float(e["PAYER_COVERAGE"])) == "PENDING_P2P"
        )

        # --- Build final patient record ---
        record = {
            "patient_id":    pid,                # internal UUID, not shown to caller
            "display_name":  display_name,       # what the graph and agent use
            "dob":           dob,
            "age":           age,
            "gender":        gender,
            "city":          city,
            "state":         state,
            "insurance": {
                "payer":           current_payer_name,
                "member_since":    insurance_since,
                "payer_id":        current_payer_id,
            },
            "active_conditions": deduped_conditions,
            "encounters":        encounter_list,
            "claim_summary": {
                "total_billed":   round(total_billed, 2),
                "total_covered":  round(total_covered, 2),
                "total_owed":     total_owed,
                "denied_claims":  denied_count,
                "pending_claims": pending_count,
            },
        }

        # Write to output folder
        person_dir = out_dir / folder_name
        person_dir.mkdir(exist_ok=True)
        (person_dir / "patient.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

        manifest.append({
            "folder":       folder_name,
            "display_name": display_name,
            "patient_id":   pid,
            "dob":          dob,
            "insurance":    current_payer_name,
            "conditions":   len(deduped_conditions),
            "encounters":   len(encounter_list),
            "denied":       denied_count,
            "pending":      pending_count,
        })

        print(f"  ✓ {display_name:30s} | {current_payer_name:25s} | "
              f"{len(deduped_conditions):2d} conditions | "
              f"{len(enc_by_patient.get(pid,[]))} encounters | "
              f"denied={denied_count} pending={pending_count}")

    # Write manifest
    manifest_path = out_dir.parent / "patient_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Print summary
    total_denied  = sum(m["denied"]  for m in manifest)
    total_pending = sum(m["pending"] for m in manifest)
    print(f"\n{'='*60}")
    print(f"Done. {len(manifest)} patients written to {out_dir}/")
    print(f"  Denied claims (DENIED_NO_PA):  {total_denied}")
    print(f"  Pending P2P reviews:           {total_pending}")
    print(f"  Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build patient graph data from Synthea CSVs")
    parser.add_argument("--dataset", default="dataset",
                        help="Path to folder containing Synthea CSVs (default: dataset)")
    parser.add_argument("--out", default="graph_rag_backend/sample_data",
                        help="Output folder for patient JSON files (default: graph_rag_backend/sample_data)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N patients (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for name generation (default: 42)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent  # voice_agent/
    build(
        dataset_dir = root / args.dataset,
        out_dir     = root / args.out,
        limit       = args.limit,
        seed        = args.seed,
    )
