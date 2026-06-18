# Dataset Preparation

Scripts that generate the dataset used by the Graph RAG system.
These run **once** to create the data — they are not part of the inference pipeline.

---

## Scripts

### `generate_dataset.py`
Generates the synthetic identity documents (birth certificates, driver's licenses,
passports, insurance records, medical records) for 49 fictional people.

```bash
python dataset_prep/generate_dataset.py                 # default 49 people
python dataset_prep/generate_dataset.py --count 100     # scale up
python dataset_prep/generate_dataset.py --seed 42       # reproducible
python dataset_prep/generate_dataset.py --out docs      # custom output dir
```

Output: `docs/people/<name>/{birth_certificate,drivers_license,...}.txt/.json`

---

### `generate_medical_pdfs.py`
Integrates real **MTSamples** medical transcriptions (Kaggle) with our identity
dataset. For each person, it picks a real clinical transcription matching their
assigned specialty, prepends their real name + DOB as a patient header, and
saves the result as a PDF.

**The medical text is 100% real MTSamples content — only the patient header is injected.**

#### Step 1 — Get MTSamples

Download `mtsamples.csv` from Kaggle:
```
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
```
Place it at: `dataset_prep/mtsamples.csv`

#### Step 2 — Install dependencies

```bash
pip install fpdf2 pandas
```

#### Step 3 — Run

```bash
python dataset_prep/generate_medical_pdfs.py
```

This reads `docs/manifest.json` (all 49 people with name + DOB) and writes:
```
docs/people/alice_chen/medical_report.pdf
docs/people/david_anderson/medical_report.pdf
...
```

Optional flags:
```bash
--csv      path/to/mtsamples.csv     (default: dataset_prep/mtsamples.csv)
--out      docs/people               (default: docs/people)
--seed     42                        (random seed for reproducibility)
--manifest docs/manifest.json        (default: docs/manifest.json)
```

---

## What gets generated

After running both scripts, `docs/people/` contains:

| File | Source | Used for |
|---|---|---|
| `birth_certificate.txt/.json` | `generate_dataset.py` | Identity extraction (LangChain / UiPath) |
| `drivers_license.txt/.json` | `generate_dataset.py` | Identity extraction |
| `passport.txt/.json` | `generate_dataset.py` | Identity extraction |
| `insurance.txt/.json` | `generate_dataset.py` | Identity extraction |
| `medical_record.txt/.json` | `generate_dataset.py` | Short structured medical record |
| `medical_report.pdf` | `generate_medical_pdfs.py` | Long-form PDF RAG (MTSamples) |

The PDF is linked to the identity graph because its header contains the person's
exact name and DOB — the EntityResolver draws a `same_as` edge automatically.
