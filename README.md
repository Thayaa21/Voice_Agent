# Voice Agent — Hospital Appointment Assistant

## What This Is

A voice calling agent for hospital operations. Patients call in, the agent verifies their identity, looks up their records from the Graph RAG knowledge graph, and answers questions or reschedules appointments — all by voice.

## Standalone Project Structure

```
voice_agent/                    ← This is the full standalone repo
├── README.md                   ← This file
├── .env.example                ← All API keys needed
│
├── graph_rag_backend/          ← Knowledge graph server (runs on port 8000)
│   ├── graph_rag/              ← Full Graph RAG Python package
│   ├── dataset_prep/           ← Scripts to generate the dataset
│   ├── sample_data/            ← 45 people with identity docs + medical PDFs
│   ├── graph_snapshot.pkl      ← Pre-built graph (instant load, no LLM needed)
│   ├── graph_rag.py            ← CLI entry point
│   ├── requirements.txt
│   └── README.md               ← How to start the backend
│
├── rishi/                      ← Voice infrastructure (runs on port 8001)
│   ├── README.md               ← Step-by-step: Twilio, Deepgram, ElevenLabs
│   └── requirements.txt
│
└── chetan/                     ← Agent intelligence (runs on port 9001)
    ├── README.md               ← Step-by-step: state machine, graph queries
    └── requirements.txt
```

## How the Three Services Work Together

```
Patient calls Twilio number
        │
        ▼
Rishi's server (port 8001)  ← Handles call, STT, TTS
        │  sends text
        ▼
Chetan's agent (port 9001)  ← Conversation logic, identity, questions
        │  queries
        ▼
Graph RAG backend (port 8000) ← Knowledge graph with 45 people's records
```

## Quick Start (run everything)

```bash
# Terminal 1 — Graph RAG backend
cd graph_rag_backend
pip install -r requirements.txt
uvicorn graph_rag.api.app:app --port 8000

# Terminal 2 — Chetan's agent
cd chetan
pip install -r requirements.txt
uvicorn agent:app --port 9001

# Terminal 3 — Rishi's voice server
cd rishi
pip install -r requirements.txt
uvicorn server:app --port 8001
# Then: ngrok http 8001 → put URL in Twilio console
```

## Architecture (designed by Thayaa)

```
Incoming call (Twilio)
        │
        ▼
STT — Speech to Text (Deepgram)
        │
        ▼
Identity Verification
  "What's your name and date of birth?"
  → POST /smart-query on the Graph RAG backend
        │
        ▼
Graph RAG Backend (already built)
  → Returns: doctor name, diagnosis, appointment, insurance, conflicts
        │
        ▼
LLM Response Generation
  → Synthesizes a natural spoken answer
        │
        ▼
TTS — Text to Speech (ElevenLabs or Deepgram)
        │
        ▼
Caller hears the answer
```

## Demo Script

```
Patient calls: +1-xxx-xxx-xxxx

Agent: "Hello, thank you for calling. I'm the hospital's automated assistant.
        Can I get your full name and date of birth?"

Patient: "Mei Lee, January 31 1982"

Agent: "Thank you Mei Lee. I found your records.
        Your last visit was September 16 2020 with Dr. Emily Chen.
        Your diagnosis is Type 2 diabetes mellitus and you are on Metformin 500mg.
        How can I help you today?"

Patient: "I want to reschedule my appointment"

Agent: "Of course. What date works best for you?"
```

## Work Split

| Person | Service | Port | Responsibility |
|--------|---------|------|---------------|
| **Rishi** | `rishi/` | 8001 | Twilio call receiving, Deepgram STT, ElevenLabs TTS |
| **Chetan** | `chetan/` | 9001 | Conversation state machine, identity verification, Graph RAG queries |
| **Thayaa** | `graph_rag_backend/` | 8000 | Knowledge graph (already built) |

## Tech Stack

| Layer | Tool |
|-------|------|
| Inbound calls | Twilio Programmable Voice |
| Speech to Text | Deepgram Nova-2 |
| Text to Speech | ElevenLabs |
| Agent logic | Python FastAPI |
| Knowledge graph | Graph RAG (NetworkX + sentence-transformers) |
| LLM | Ollama (local) or OpenAI |


```
Incoming call (Twilio)
        │
        ▼
STT — Speech to Text (Deepgram)
        │
        ▼
Identity Verification
  "What's your name and date of birth?"
  → POST /smart-query on the Graph RAG backend
        │
        ▼
Graph RAG Backend (already built — QB1 project)
  → Returns: doctor name, diagnosis, appointment, insurance, conflicts
        │
        ▼
LLM Response Generation
  → Synthesizes a natural spoken answer
        │
        ▼
TTS — Text to Speech (ElevenLabs or Deepgram)
        │
        ▼
Caller hears the answer
```

## The Data (already exists in QB1)

The Graph RAG project at `QB1/` has:
- 200 entities across 45 people
- Each person has: birth certificate, passport, license, insurance, medical record
- Medical records include: diagnosis, medications, doctor name, visit date
- Conflicts already detected: DOB mismatches between insurance and birth cert

All of this is queryable via the `/smart-query` endpoint running at `http://localhost:8000`.

## How to Query the Existing Backend

```bash
# Start the backend (from QB1/)
./start.sh

# Test a query
curl -X POST http://localhost:8000/smart-query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is Mei Lee diagnosis and doctor"}'
```

Response:
```json
{
  "type": "answer",
  "person": "Mei Lee",
  "answer": "Mei Lee's diagnosis is Type 2 diabetes mellitus. Her doctor is Dr. Emily Chen.",
  "facts": [...]
}
```

## Work Split

| Person | Responsibility |
|--------|---------------|
| **Rishi** | Voice infrastructure — Twilio inbound calls, Deepgram STT, ElevenLabs TTS, call flow state machine |
| **Chetan** | Agent intelligence — identity verification flow, Graph RAG integration, conversation logic, LLM response generation |

## Tech Stack to Use

| Layer | Tool |
|-------|------|
| Inbound calls | Twilio Programmable Voice |
| Speech to Text | Deepgram Nova-2 |
| Text to Speech | ElevenLabs (or Deepgram TTS) |
| Agent logic | Python FastAPI (new service) |
| Knowledge backend | QB1 Graph RAG `/smart-query` API |
| LLM | Ollama (local) or OpenAI |

## Demo Script

```
Patient calls: +1-xxx-xxx-xxxx

Agent: "Hello, thank you for calling. I'm the hospital's automated assistant.
        Can I get your full name and date of birth?"

Patient: "Mei Lee, January 31 1982"

Agent: "Thank you Mei Lee. I found your records.
        Your last visit was September 16 2020 with Dr. Emily Chen.
        Your diagnosis is Type 2 diabetes mellitus and you are on Metformin 500mg.
        How can I help you today?"

Patient: "I want to reschedule my appointment"

Agent: "Of course. What date works best for you?"
```

## Folders

- `rishi/` — Voice infrastructure spec and tasks
- `chetan/` — Agent intelligence spec and tasks
