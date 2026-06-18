# Graph RAG Backend

This is the knowledge graph backend that powers the voice agent.
It is a standalone FastAPI server — the voice agent queries it via HTTP.

## What It Contains

```
graph_rag/              ← Python package (the full pipeline)
├── api/
│   └── app.py          ← FastAPI server (all endpoints)
├── core/
│   ├── models.py       ← Data models
│   ├── graph_builder.py← NetworkX graph
│   ├── embeddings.py   ← sentence-transformers
│   ├── resolver.py     ← Entity resolution (same_as edges)
│   └── contradiction.py← DOB mismatch detection
├── extraction/
│   ├── uipath_extractor.py  ← Parse UiPath JSON
│   └── langchain_extractor.py ← LLM extraction from text
├── pdf/
│   ├── loader.py       ← PDF text extraction
│   └── chunker.py      ← Semantic chunking
└── query/
    └── engine.py       ← Query orchestrator

dataset_prep/
├── generate_dataset.py ← Generate synthetic identity docs
├── generate_medical_pdfs.py ← Generate medical report PDFs
└── build_snapshot.py   ← Pre-ingest everything into graph_snapshot.pkl

sample_data/            ← 45 people with identity docs + medical PDFs
graph_snapshot.pkl      ← Pre-built graph (load instantly, no LLM needed)
graph_rag.py            ← CLI entry point
```

## Quick Start

### 0. Build the graph snapshot (FIRST TIME ONLY — do this before starting the server)

The `graph_snapshot.pkl` is **not included in the repo** (large binary, in `.gitignore`).
Without it, the server starts but returns empty results for all queries.

```bash
cd graph_rag_backend
cp .env.example .env
# Edit .env: set LLM_PROVIDER=ollama and start Ollama locally
# OR set LLM_PROVIDER=openai and add OPENAI_API_KEY

python dataset_prep/build_snapshot.py
# Takes ~5-10 minutes on first run
# Ingests all 45 people's JSON + PDF files and saves graph_snapshot.pkl
```

### 1. Install dependencies

```bash
pip install networkx rapidfuzz sentence-transformers fastapi uvicorn \
            pdfplumber python-dotenv openai langchain click pydantic
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env: set LLM_PROVIDER=ollama and start Ollama
# OR set LLM_PROVIDER=openai and add OPENAI_API_KEY
```

### 3. Start the server

```bash
# The graph_snapshot.pkl loads automatically — instant startup
uvicorn graph_rag.api.app:app --port 8000 --reload
```

Server is live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## The Key Endpoint (what the voice agent uses)

```
POST /smart-query
```

```bash
curl -X POST http://localhost:8000/smart-query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is Mei Lee diagnosis and doctor", "max_hops": 3}'
```

Response:
```json
{
  "type": "answer",
  "person": "Mei Lee",
  "answer": "Mei Lee's diagnosis is Type 2 diabetes mellitus. Her doctor is Dr. Emily Chen.",
  "facts": [
    {"attribute_key": "diagnosis", "value": "Type 2 diabetes mellitus", "source_filename": "medical_record.json"},
    {"attribute_key": "doctor",    "value": "Dr. Emily Chen",           "source_filename": "medical_record.json"}
  ],
  "has_conflicts": false
}
```

Response types:
- `"answer"` — single person found, answer ready
- `"disambiguation"` — multiple people match, returns options
- `"not_found"` — no match

## All Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/smart-query` | **Main endpoint** — fuzzy name match + graph query + LLM answer |
| `GET`  | `/graph/stats` | Counts: docs, entities, links, conflicts |
| `GET`  | `/entities` | All people in the graph |
| `GET`  | `/explore/conflicts` | All detected DOB mismatches |
| `DELETE` | `/graph` | Reset graph (reload snapshot on restart) |

## Dataset

`sample_data/` contains 45 fictional people. Each person has:
- `birth_certificate.json` — name, DOB, place of birth
- `drivers_license.json` — name, DOB, license number, address
- `insurance.json` — policy number, coverage, premium, beneficiary
- `passport.json` — passport number, nationality, expiry
- `medical_report.pdf` — real clinical transcription (MTSamples)

8 people have **intentional DOB conflicts** between their insurance and other docs — the system detects and surfaces these.

## Regenerate the Dataset

```bash
# Regenerate identity docs (synthetic)
python dataset_prep/generate_dataset.py --count 45 --out sample_data

# Regenerate medical PDFs (needs mtsamples.csv from Kaggle)
# Download: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
python dataset_prep/generate_medical_pdfs.py --csv dataset_prep/mtsamples.csv

# Rebuild the graph snapshot
python dataset_prep/build_snapshot.py
```

## Graph Architecture

```
[birth_certificate.json]     [insurance.json]          [medical_report.pdf]
        │                           │                           │
   [Mei Lee]  ←── same_as ──  [Mei Lee]  ←── same_as ──  [Mei Lee]
   BIRTH_CERT                  INSURANCE                  MEDICAL_REPORT
   dob: 1982-01-31             dob: 1982-01-31             diagnosis: T2 diabetes
                                                            doctor: Dr. Emily Chen
```

Entity resolution links all documents for the same person via `same_as` edges.
When you query "Mei Lee", the graph traverses all three documents and returns a unified answer.
