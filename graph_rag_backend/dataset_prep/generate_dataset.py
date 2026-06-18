"""
Graph RAG Synthetic Dataset Generator
======================================
Generates realistic fake documents for testing cross-document entity resolution,
multi-hop reasoning, contradiction detection, and disambiguation.

Usage:
    python generate_dataset.py                   # generates default 10 people
    python generate_dataset.py --count 20        # generates 20 people
    python generate_dataset.py --seed 42         # reproducible output
    python generate_dataset.py --out custom_dir  # custom output directory

No real personal data. All names, addresses, and IDs are fictional.
"""

import json
import os
import random
import argparse
import hashlib
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# PEOPLE DATA POOL
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "James", "Maria", "David", "Sarah", "Michael", "Emily", "Robert",
    "Priya", "Daniel", "Aisha", "Thomas", "Mei", "Carlos", "Sofia", "William",
    "Fatima", "Oliver", "Hannah", "Liam", "Yuki", "Nathan", "Zara", "Ethan",
    "Amara", "Lucas", "Chloe", "Ravi", "Ingrid", "Hassan",
]

LAST_NAMES = [
    "Chen", "Lee", "Walker", "Smith", "Patel", "Kim", "Johnson", "Garcia",
    "Nguyen", "Brown", "Singh", "Wilson", "Martinez", "Anderson", "Taylor",
    "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Yamamoto",
    "Rossi", "Kowalski", "Santos", "Mueller", "Okonkwo", "Petrov", "Ali", "Park",
]

CITIES = [
    ("Vancouver", "British Columbia", "Canada", "V6B"),
    ("Toronto", "Ontario", "Canada", "M5V"),
    ("Calgary", "Alberta", "Canada", "T2P"),
    ("Montreal", "Quebec", "Canada", "H3A"),
    ("Ottawa", "Ontario", "Canada", "K1A"),
    ("Edmonton", "Alberta", "Canada", "T5J"),
    ("Winnipeg", "Manitoba", "Canada", "R3C"),
    ("Halifax", "Nova Scotia", "Canada", "B3J"),
    ("Victoria", "British Columbia", "Canada", "V8W"),
    ("Saskatoon", "Saskatchewan", "Canada", "S7K"),
]

PROVINCE_ABBR = {
    "British Columbia": "BC",
    "Ontario": "ON",
    "Alberta": "AB",
    "Quebec": "QC",
    "Manitoba": "MB",
    "Nova Scotia": "NS",
    "Saskatchewan": "SK",
}

STREET_NAMES = [
    "Maple Street", "Oak Avenue", "Cedar Lane", "Pine Road", "Elm Drive",
    "Rideau Crescent", "King Street", "Queen Street", "Bay Avenue", "Park Boulevard",
    "River Road", "Lake Drive", "Hill Street", "Valley Way", "Forest Path",
]

DOCTORS = [
    "Dr. Sarah Nguyen", "Dr. James Patel", "Dr. Emily Chen", "Dr. Robert Kim",
    "Dr. Maria Santos", "Dr. David Wilson", "Dr. Priya Sharma", "Dr. Michael Brown",
]

DIAGNOSES = [
    ("Seasonal allergic rhinitis", "Cetirizine 10mg daily, Fluticasone nasal spray"),
    ("Type 2 diabetes mellitus", "Metformin 500mg twice daily"),
    ("Hypertension", "Lisinopril 10mg daily, dietary sodium restriction"),
    ("Anxiety disorder", "Sertraline 50mg daily, cognitive behavioural therapy"),
    ("Asthma", "Salbutamol inhaler as needed, Budesonide inhaler twice daily"),
    ("Hypothyroidism", "Levothyroxine 75mcg daily"),
    ("Migraine", "Sumatriptan 50mg as needed, Topiramate 25mg nightly"),
    ("Iron deficiency anaemia", "Ferrous sulfate 325mg three times daily"),
]

COVERAGE_TYPES = [
    "Comprehensive Health Insurance",
    "Extended Health and Dental",
    "Life and Disability Insurance",
    "Critical Illness Coverage",
    "Travel Health Insurance",
]

VEHICLE_CLASSES = [
    "Class 5 (Passenger Vehicle)",
    "Class 5 (Passenger Vehicle)",
    "Class 5 (Passenger Vehicle)",
    "Class 7 (Learner — Passenger Vehicle)",
    "Class 3 (Truck)",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def seeded_random(seed_str: str) -> float:
    """Deterministic float 0-1 from a string seed."""
    h = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0


def rand_date(rng: random.Random, start_year: int, end_year: int) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def format_date_long(d: date) -> str:
    return d.strftime("%B %d, %Y")


def format_date_iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def rand_id(rng: random.Random, prefix: str, length: int = 7) -> str:
    digits = "".join([str(rng.randint(0, 9)) for _ in range(length)])
    return f"{prefix}-{digits}"


def bbox(x1: int, y1: int, w: int = 300, h: int = 20) -> list:
    return [x1, y1, x1 + w, y1 + h]


def street_address(rng: random.Random, city: str, province: str, postal_prefix: str) -> str:
    number = rng.randint(10, 999)
    street = rng.choice(STREET_NAMES)
    abbr = PROVINCE_ABBR.get(province, province[:2].upper())
    postal = f"{postal_prefix} {rng.randint(1, 9)}{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randint(0,9)}"
    return f"{number} {street}, {city}, {abbr}, {postal}"


# ---------------------------------------------------------------------------
# PERSON GENERATOR
# ---------------------------------------------------------------------------

class Person:
    def __init__(self, rng: random.Random, first: str, last: str,
                 dob: date, city_info: tuple, contradiction: bool = False,
                 name_variant: str = None):
        self.rng = rng
        self.first = first
        self.last = last
        self.full_name = f"{first} {last}"
        self.dob = dob
        self.city, self.province, self.country, self.postal_prefix = city_info
        self.address = street_address(rng, self.city, self.province, self.postal_prefix)
        self.sex = rng.choice(["Male", "Female", "Male", "Female", "Non-binary"])

        # IDs
        self.bc_reg = rand_id(rng, f"BC-{dob.year}")
        self.license_num = rand_id(rng, PROVINCE_ABBR.get(self.province, "CA"))
        self.passport_num = rand_id(rng, "CA", 8)
        self.policy_num = rand_id(rng, f"INS-{rng.randint(2015, 2023)}", 6)

        # Parents
        father_last = last
        mother_maiden = rng.choice(LAST_NAMES)
        self.father = f"{rng.choice(FIRST_NAMES)} {father_last}"
        self.mother = f"{rng.choice(FIRST_NAMES)} {last} (nee {mother_maiden})"

        # License dates
        self.license_issue = rand_date(rng, max(dob.year + 16, 2005), 2022)
        self.license_expiry = date(dob.year + (rng.randint(35, 45)), dob.month, dob.day)
        if self.license_expiry < date(2025, 1, 1):
            self.license_expiry = date(self.license_expiry.year + 10, dob.month, dob.day)

        # Passport dates
        self.passport_issue = rand_date(rng, max(dob.year + 18, 2010), 2023)
        self.passport_expiry = date(self.passport_issue.year + 10,
                                    self.passport_issue.month,
                                    self.passport_issue.day)
        self.passport_issue_city = rng.choice(CITIES)[0] + " Passport Office"

        # Insurance
        self.insurance_start = rand_date(rng, 2015, 2022)
        self.coverage = rng.choice(COVERAGE_TYPES)
        self.premium = f"${rng.randint(8, 35) * 10}.00 per month"
        self.beneficiary = self.father

        # Medical
        self.visit_date = rand_date(rng, 2020, 2024)
        self.doctor = rng.choice(DOCTORS)
        diagnosis_info = rng.choice(DIAGNOSES)
        self.diagnosis, self.medications = diagnosis_info
        self.hospital = f"{self.city} General Hospital"

        # Contradiction: intentionally wrong DOB in insurance
        if contradiction:
            wrong_day = (dob.day % 28) + 1  # shift day by 1, keep valid
            self.insurance_dob = date(dob.year, dob.month, wrong_day)
        else:
            self.insurance_dob = dob

        # Name variant for passport/medical (e.g. "James R. Lee" vs "James Lee")
        middle_initial = rng.choice("ABCDEFGHJKLMNPRSTW")
        self.name_variant_passport = f"{first} {middle_initial}. {last}"
        self.name_variant_medical = f"{first} {middle_initial[0]}. {last}"


# ---------------------------------------------------------------------------
# TXT WRITERS
# ---------------------------------------------------------------------------

def write_bc_txt(p: Person, path: Path):
    abbr = PROVINCE_ABBR.get(p.province, p.province)
    content = f"""CERTIFICATE OF BIRTH
Registration No: {p.bc_reg}

Full Name: {p.full_name}
Date of Birth: {format_date_long(p.dob)}
Place of Birth: {p.city}, {p.province}, {p.country}
Sex: {p.sex}

Father's Name: {p.father}
Mother's Name: {p.mother}

Issued by: Vital Statistics, {p.province}
Date of Issue: {format_date_long(p.dob + timedelta(days=rng_offset(p, 15, 30)))}
"""
    path.write_text(content)


def write_license_txt(p: Person, path: Path):
    abbr = PROVINCE_ABBR.get(p.province, p.province)
    content = f"""DRIVER'S LICENSE
Province of {p.province}

Full Name: {p.full_name}
Date of Birth: {format_date_long(p.dob)}
License Number: {p.license_num}
Address: {p.address}
Vehicle Class: {p.rng.choice(VEHICLE_CLASSES)}
Issue Date: {format_date_long(p.license_issue)}
Expiry Date: {format_date_long(p.license_expiry)}
"""
    path.write_text(content)


def write_passport_txt(p: Person, path: Path):
    content = f"""PASSPORT
Government of Canada / Gouvernement du Canada

Surname: {p.last}
Given Names: {p.name_variant_passport.replace(p.last, "").strip()}
Date of Birth: {format_date_long(p.dob)}
Place of Birth: {p.city}, {p.province}, {p.country}
Nationality: Canadian
Passport Number: {p.passport_num}
Date of Issue: {format_date_long(p.passport_issue)}
Date of Expiry: {format_date_long(p.passport_expiry)}
Place of Issue: {p.passport_issue_city}
"""
    path.write_text(content)


def write_insurance_txt(p: Person, path: Path):
    content = f"""INSURANCE POLICY DOCUMENT
Policy Number: {p.policy_num}

Policyholder Name: {p.full_name}
Date of Birth: {format_date_long(p.insurance_dob)}
Coverage Type: {p.coverage}
Premium: {p.premium}
Policy Start Date: {format_date_long(p.insurance_start)}
Beneficiary: {p.beneficiary}

Note: This policy was issued based on the information provided at enrollment.
Issued by: Pacific Shield Insurance Group
"""
    path.write_text(content)


def write_medical_txt(p: Person, path: Path):
    content = f"""MEDICAL RECORD
{p.hospital}

Patient Name: {p.name_variant_medical}
Date of Birth: {format_date_long(p.dob)}
Date of Visit: {format_date_long(p.visit_date)}
Attending Physician: {p.doctor}
Department: General Practice

Diagnosis: {p.diagnosis}
Medications Prescribed: {p.medications}
Follow-up: 3 months

Emergency Contact: {p.mother.split(' (')[0]} — {p.rng.randint(416, 778)}-555-{p.rng.randint(1000,9999)}
"""
    path.write_text(content)


def rng_offset(p: Person, lo: int, hi: int) -> int:
    return p.rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# JSON WRITERS (UiPath format)
# ---------------------------------------------------------------------------

def uipath_json(doc_type: str, source_file: str, fields: dict, overall_conf: float) -> dict:
    return {
        "document_type": doc_type,
        "confidence": round(overall_conf, 2),
        "source_file": source_file,
        "fields": fields,
    }


def field(value: str, confidence: float, y: int, x1: int = 72, w: int = 380) -> dict:
    return {
        "value": value,
        "confidence": round(confidence, 2),
        "page": 1,
        "bounding_box": bbox(x1, y, w),
    }


def write_bc_json(p: Person, path: Path):
    data = uipath_json("BIRTH_CERTIFICATE", "birth_certificate.txt", {
        "name":               field(p.full_name,                                       0.99, 120),
        "dob":                field(format_date_iso(p.dob),                            0.99, 145),
        "place_of_birth":     field(f"{p.city}, {p.province}, {p.country}",           0.97, 170, w=480),
        "parents":            field(f"{p.father}, {p.mother}",                         0.95, 220, w=480),
        "registration_number": field(p.bc_reg,                                         0.99, 80),
    }, 0.98)
    path.write_text(json.dumps(data, indent=2))


def write_license_json(p: Person, path: Path):
    data = uipath_json("DRIVERS_LICENSE", "drivers_license.txt", {
        "name":           field(p.full_name,                                    0.99, 100),
        "dob":            field(format_date_iso(p.dob),                         0.99, 125),
        "license_number": field(p.license_num,                                  0.98, 150),
        "address":        field(p.address,                                      0.96, 175, w=480),
        "vehicle_class":  field(p.rng.choice(VEHICLE_CLASSES),                  0.97, 200),
        "issue_date":     field(format_date_iso(p.license_issue),               0.98, 225),
        "expiry_date":    field(format_date_iso(p.license_expiry),              0.98, 250),
    }, 0.97)
    path.write_text(json.dumps(data, indent=2))


def write_passport_json(p: Person, path: Path):
    given = p.name_variant_passport.replace(p.last, "").strip()
    data = uipath_json("PASSPORT", "passport.txt", {
        "name":            field(p.name_variant_passport,                       0.98, 120),
        "dob":             field(format_date_iso(p.dob),                        0.99, 145),
        "passport_number": field(p.passport_num,                                0.99, 80),
        "nationality":     field("Canadian",                                    0.99, 170),
        "expiry_date":     field(format_date_iso(p.passport_expiry),            0.99, 195),
        "place_of_issue":  field(p.passport_issue_city,                        0.97, 220, w=420),
    }, 0.99)
    path.write_text(json.dumps(data, indent=2))


def write_insurance_json(p: Person, path: Path):
    data = uipath_json("INSURANCE", "insurance.txt", {
        "name":          field(p.full_name,                              0.99, 100),
        "dob":           field(format_date_iso(p.insurance_dob),         0.94, 125),
        "policy_number": field(p.policy_num,                             0.99, 80),
        "coverage_type": field(p.coverage,                               0.97, 150, w=480),
        "premium":       field(p.premium,                                0.98, 175),
        "start_date":    field(format_date_iso(p.insurance_start),       0.99, 200),
        "beneficiary":   field(p.beneficiary,                            0.95, 225, w=420),
    }, 0.96)
    path.write_text(json.dumps(data, indent=2))


def write_medical_json(p: Person, path: Path):
    data = uipath_json("MEDICAL_RECORD", "medical_record.txt", {
        "patient_name": field(p.name_variant_medical,                    0.97, 100),
        "dob":          field(format_date_iso(p.dob),                    0.99, 125),
        "diagnosis":    field(p.diagnosis,                               0.95, 175, w=480),
        "doctor":       field(p.doctor,                                  0.98, 150),
        "date":         field(format_date_iso(p.visit_date),             0.99, 200),
        "medications":  field(p.medications,                             0.93, 220, w=520),
    }, 0.95)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# DOCUMENT SET DEFINITIONS
# Writers: list of (filename_stem, txt_writer, json_writer)
# ---------------------------------------------------------------------------

ALL_DOC_TYPES = {
    "birth_certificate": (write_bc_txt, write_bc_json),
    "drivers_license":   (write_license_txt, write_license_json),
    "passport":          (write_passport_txt, write_passport_json),
    "insurance":         (write_insurance_txt, write_insurance_json),
    "medical_record":    (write_medical_txt, write_medical_json),
}

# Assign doc sets per person — cycles through meaningful combos
DOC_SETS = [
    ["birth_certificate", "drivers_license", "insurance"],          # common set
    ["birth_certificate", "passport", "medical_record"],            # 3-hop chain
    ["birth_certificate", "drivers_license", "passport"],           # travel profile
    ["birth_certificate", "insurance", "medical_record"],           # health profile
    ["birth_certificate", "drivers_license", "passport", "insurance"],  # full set
    ["birth_certificate", "passport", "insurance", "medical_record"],   # full set 2
]


# ---------------------------------------------------------------------------
# GENERATE PEOPLE
# ---------------------------------------------------------------------------

def generate_people(count: int, base_seed: int) -> list:
    rng = random.Random(base_seed)
    used_names = set()
    people = []

    # Intentionally reuse first names to test disambiguation
    # ~30% of people share a first name with someone else
    first_pool = FIRST_NAMES.copy()
    shared_firsts = rng.sample(FIRST_NAMES, max(1, count // 3))

    for i in range(count):
        # Pick names
        if i < len(shared_firsts) and rng.random() < 0.4:
            first = shared_firsts[i % len(shared_firsts)]
        else:
            first = rng.choice(FIRST_NAMES)

        last = rng.choice(LAST_NAMES)
        while (first, last) in used_names:
            last = rng.choice(LAST_NAMES)
        used_names.add((first, last))

        dob = rand_date(rng, 1970, 2000)
        city_info = rng.choice(CITIES)
        contradiction = (i % 4 == 0)  # every 4th person has a DOB contradiction

        person_rng = random.Random(base_seed + i * 1000)
        p = Person(person_rng, first, last, dob, city_info, contradiction=contradiction)
        doc_set = DOC_SETS[i % len(DOC_SETS)]
        people.append((p, doc_set))

    return people


# ---------------------------------------------------------------------------
# WRITE ALL FILES
# ---------------------------------------------------------------------------

def generate_dataset(count: int, out_dir: str, seed: int):
    out = Path(out_dir)
    people_dir = out / "people"
    people_dir.mkdir(parents=True, exist_ok=True)

    people = generate_people(count, seed)
    manifest = []

    for p, doc_set in people:
        folder_name = f"{p.first.lower()}_{p.last.lower()}"
        person_dir = people_dir / folder_name
        person_dir.mkdir(exist_ok=True)

        contradiction = p.insurance_dob != p.dob
        person_manifest = {
            "name": p.full_name,
            "dob": format_date_iso(p.dob),
            "folder": folder_name,
            "contradiction": contradiction,
            "contradiction_detail": (
                f"Insurance DOB={format_date_iso(p.insurance_dob)} "
                f"vs actual DOB={format_date_iso(p.dob)}"
            ) if contradiction else None,
            "documents": doc_set,
            "ids": {
                "license_number": p.license_num if "drivers_license" in doc_set else None,
                "passport_number": p.passport_num if "passport" in doc_set else None,
                "policy_number": p.policy_num if "insurance" in doc_set else None,
                "bc_registration": p.bc_reg,
            }
        }

        for doc_type in doc_set:
            txt_writer, json_writer = ALL_DOC_TYPES[doc_type]
            txt_writer(p, person_dir / f"{doc_type}.txt")
            json_writer(p, person_dir / f"{doc_type}.json")

        manifest.append(person_manifest)
        print(f"  ✓ {p.full_name:30s} — {', '.join(doc_set)}"
              + (" ⚠ contradiction" if contradiction else ""))

    # Write manifest JSON
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Write README
    write_readme(out, manifest, count)

    print(f"\nDataset written to: {out}/people/")
    print(f"  {count} people × 2 formats (txt + json) = {sum(len(m['documents'])*2 for m in manifest)} files")
    print(f"  {sum(1 for m in manifest if m['contradiction'])} people with DOB contradictions")
    print(f"  manifest.json written for reference")


def write_readme(out: Path, manifest: list, count: int):
    lines = [
        "# Synthetic Test Dataset",
        "",
        "No real personal data. All names, addresses, and IDs are fictional.",
        f"Generated by `generate_dataset.py` — {count} people.",
        "",
        "---",
        "",
        "## People",
        "",
        "| Folder | Name | DOB | Documents | Contradiction? |",
        "|--------|------|-----|-----------|----------------|",
    ]
    for m in manifest:
        docs = ", ".join(m["documents"])
        flag = f"⚠ {m['contradiction_detail']}" if m["contradiction"] else "—"
        lines.append(f"| `{m['folder']}` | {m['name']} | {m['dob']} | {docs} | {flag} |")

    lines += [
        "",
        "---",
        "",
        "## File Formats",
        "",
        "- **`.txt`** — raw text, use with `--extractor langchain`",
        "- **`.json`** — UiPath Document Understanding format, use with `--extractor uipath`",
        "",
        "---",
        "",
        "## Test Query Examples",
        "",
        "```",
        "# Cross-document query",
        'python graph_rag.py query "What is [name]\'s license number and insurance policy?"',
        "",
        "# Multi-hop query",
        'python graph_rag.py query "What medication was prescribed to the person with passport [number]?"',
        "",
        "# Contradiction check",
        'python graph_rag.py query "What is [name]\'s date of birth?"',
        "# → Should return answer AND conflict warning if DOB mismatch exists",
        "",
        "# Disambiguation",
        'python graph_rag.py query "What is [first_name] [last_name1]\'s license number?"',
        "# → Should return that person's license, not the other James/Maria/etc.",
        "```",
        "",
        "---",
        "",
        "## manifest.json",
        "The `manifest.json` file lists all people, their documents, IDs, and contradiction flags.",
        "Use it to write automated test assertions.",
    ]

    (out / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Graph RAG test dataset")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of people to generate (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--out", type=str, default="docs",
                        help="Output directory (default: docs)")
    args = parser.parse_args()

    print(f"Generating {args.count} people (seed={args.seed}) → {args.out}/\n")
    generate_dataset(args.count, args.out, args.seed)
