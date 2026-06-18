"""
Medical PDF Dataset Generator
==============================
Integrates real MTSamples transcriptions (Kaggle) with our existing
docs/people/ dataset.

For each person in docs/manifest.json:
  1. Pick a matching MTSamples row by specialty
  2. Prepend a header with the person's real name + DOB
  3. Save as docs/people/<folder>/medical_report.pdf

The medical TEXT is 100% real MTSamples content.
Only the patient header (name, DOB) is injected to link it to our graph.

Prerequisites:
    pip install fpdf2 pandas

Dataset:
    Download mtsamples.csv from Kaggle:
    https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
    Place it at:  dataset_prep/mtsamples.csv

Usage:
    python dataset_prep/generate_medical_pdfs.py
    python dataset_prep/generate_medical_pdfs.py --csv dataset_prep/mtsamples.csv
    python dataset_prep/generate_medical_pdfs.py --out docs/people
    python dataset_prep/generate_medical_pdfs.py --seed 42
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# SPECIALTY MAP  — folder name → preferred MTSamples medical_specialty value
# ---------------------------------------------------------------------------

SPECIALTY_MAP = {
    "alice_chen":              "Cardiovascular / Pulmonary",
    "alice_garcia":            "Orthopedic",
    "alice_harris":            "Neurology",
    "aisha_nguyen":            "Gastroenterology",
    "aisha_walker":            "Endocrinology",
    "amara_johnson":           "Obstetrics / Gynecology",
    "amara_mueller":           "Cardiovascular / Pulmonary",
    "amara_singh":             "Nephrology",
    "chloe_santos":            "Dermatology",
    "david_anderson":          "Orthopedic",
    "david_patel":             "Urology",
    "emily_martinez":          "Neurology",
    "emily_patel":             "Psychiatry / Psychology",
    "emily_rossi":             "Allergy / Immunology",
    "ethan_johnson":           "Cardiovascular / Pulmonary",
    "ethan_petrov":            "Gastroenterology",
    "fatima_yamamoto":         "Obstetrics / Gynecology",
    "hassan_anderson":         "Orthopedic",
    "hassan_kim":              "Ophthalmology",
    "ingrid_petrov":           "Endocrinology",
    "ingrid_white":            "Hematology - Oncology",
    "james_lee":               "Cardiovascular / Pulmonary",
    "james_smith":             "Neurology",
    "james_walker":            "Orthopedic",
    "james_wilson":            "Gastroenterology",
    "liam_anderson":           "Urology",
    "liam_nguyen":             "Psychiatry / Psychology",
    "lucas_ali":               "Allergy / Immunology",
    "lucas_garcia":            "Nephrology",
    "mei_johnson":             "Obstetrics / Gynecology",
    "mei_lee":                 "Dermatology",
    "mei_singh":               "Endocrinology",
    "mei_smith":               "Hematology - Oncology",
    "michael_anderson":        "Cardiovascular / Pulmonary",
    "nathan_garcia":           "Orthopedic",
    "nathan_walker":           "Neurology",
    "oliver_brown":            "Gastroenterology",
    "oliver_smith":            "Urology",
    "ravi_okonkwo":            "Psychiatry / Psychology",
    "sarah_brown":             "Obstetrics / Gynecology",
    "sarah_jackson":           "Allergy / Immunology",
    "sofia_harris":            "Cardiovascular / Pulmonary",
    "sofia_nguyen":            "Orthopedic",
    "thayaananthan_kanagaraj": "Cardiovascular / Pulmonary",
    "william_johnson":         "Neurology",
    "william_yamamoto":        "Orthopedic",
    "yuki_patel":              "Endocrinology",
    "zara_kim":                "Hematology - Oncology",
    "zara_thomas":             "Psychiatry / Psychology",
}

# ---------------------------------------------------------------------------
# LOAD MTSAMPLES CSV
# ---------------------------------------------------------------------------

def load_mtsamples(csv_path: Path) -> dict[str, list[str]]:
    """
    Load mtsamples.csv and return a dict: specialty → list of transcription texts.
    Skips rows with empty transcription text.
    """
    import pandas as pd

    if not csv_path.exists():
        print(f"ERROR: MTSamples CSV not found at {csv_path}")
        print("Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Column names in the Kaggle version
    spec_col  = "medical_specialty"
    text_col  = "transcription"

    if spec_col not in df.columns or text_col not in df.columns:
        print(f"ERROR: Expected columns '{spec_col}' and '{text_col}' in CSV.")
        print(f"Found: {list(df.columns)}")
        sys.exit(1)

    # Drop rows with no transcription text
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip() != ""]

    # Build specialty → [transcription, ...] mapping
    by_specialty: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        spec = str(row[spec_col]).strip()
        text = str(row[text_col]).strip()
        by_specialty.setdefault(spec, []).append(text)

    print(f"Loaded {len(df)} MTSamples records across {len(by_specialty)} specialties.")
    return by_specialty


# ---------------------------------------------------------------------------
# PDF GENERATION
# ---------------------------------------------------------------------------

def _format_dob(dob_iso: str) -> str:
    """Convert '1992-03-15' → 'March 15, 1992'"""
    try:
        return datetime.strptime(dob_iso, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        return dob_iso


def make_pdf(person: dict, transcription: str, output_path: Path) -> None:
    """
    Create a PDF with a patient header (name + DOB) followed by the real
    MTSamples transcription text. Nothing else is modified.
    """
    from fpdf import FPDF

    name = person["name"]
    dob  = _format_dob(person["dob"])

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Header ----
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "MEDICAL CONSULTATION REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, f"Patient Name:   {name}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.cell(0, 7, f"Date of Birth:  {dob}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.cell(0, 7, f"Report Date:    {datetime.today().strftime('%B %d, %Y')}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(4)

    # Separator line
    pdf.set_draw_color(150, 150, 150)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # ---- Body: real MTSamples text, unchanged ----
    pdf.set_font("Helvetica", "", 10)
    for line in transcription.splitlines():
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        # Section headers (ALL CAPS lines) get bold styling
        if line.isupper() and len(line) < 60:
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 5, line)
            pdf.set_font("Helvetica", "", 10)
        else:
            # Encode safely — strip non-latin1 chars that FPDF can't handle
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe_line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate medical PDFs for our dataset.")
    parser.add_argument("--csv",  default="dataset_prep/mtsamples.csv",
                        help="Path to mtsamples.csv (default: dataset_prep/mtsamples.csv)")
    parser.add_argument("--out",  default="docs/people",
                        help="Output root directory (default: docs/people)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible specialty assignment (default: 42)")
    parser.add_argument("--manifest", default="docs/manifest.json",
                        help="Path to manifest.json (default: docs/manifest.json)")
    args = parser.parse_args()

    random.seed(args.seed)

    csv_path      = Path(args.csv)
    out_root      = Path(args.out)
    manifest_path = Path(args.manifest)

    # Load manifest
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found at {manifest_path}")
        sys.exit(1)
    people = json.loads(manifest_path.read_text())
    print(f"Found {len(people)} people in manifest.")

    # Load MTSamples
    by_specialty = load_mtsamples(csv_path)
    all_texts    = [t for texts in by_specialty.values() for t in texts]

    generated = 0
    skipped   = 0

    for person in people:
        folder   = person["folder"]
        out_path = out_root / folder / "medical_report.pdf"

        if out_path.exists():
            print(f"  SKIP  {folder}/medical_report.pdf (already exists)")
            skipped += 1
            continue

        # Pick best-matching specialty
        preferred = SPECIALTY_MAP.get(folder)
        candidates = by_specialty.get(preferred, []) if preferred else []

        # Fall back to any available transcription if specialty not found
        if not candidates:
            candidates = all_texts

        transcription = random.choice(candidates)

        make_pdf(person, transcription, out_path)
        print(f"  OK    {folder}/medical_report.pdf  [{preferred or 'general'}]")
        generated += 1

    print(f"\nDone. Generated: {generated}  |  Skipped: {skipped}")


if __name__ == "__main__":
    main()
