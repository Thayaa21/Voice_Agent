# Chetan — Implementation Notes

This is the agent-intelligence service (port 9001) described in `README.md`. It's
the "brain": Rishi's voice server sends it what the caller said, it figures out
who they are, queries Thayaa's Graph RAG backend, and returns a short line to speak.

## Files

- `agent.py` — the service. FastAPI app + LangGraph state machine + the 4 tools.
- `requirements.txt` — runtime deps.
- `dev_test_backend.py` — **dev only.** A local stand-in for the Graph RAG backend
  that replays the real `graph_snapshot.pkl` through the real backend functions, so
  you can run/test the agent without installing torch or running an LLM.
- `smoke_test.sh` — runs the README tests 1–5 plus disambiguation and the retry path.

## How it works

`POST /agent {text, call_id}` → `{response, end_call}`. One HTTP call = one turn.

The conversation is a LangGraph `StateGraph` with two nodes:

```
START ──(call_state == "handle")──▶ handle_query ──▶ END
   └────(otherwise / first turn)──▶ verify_identity ─▶ END
```

State per call is stored in a LangGraph `MemorySaver` keyed by `call_id`
(`thread_id`), so multi-turn memory is automatic — `person`, `retries`, the
disambiguation `pending` list, and the full message `history` all persist between
turns. The agent never re-asks who you are once you're verified.

**verify_identity** calls `lookup_patient` (`/smart-query`) and branches on the
backend's response type:
- `answer` → save the person, greet, move to `handle_query`.
- `disambiguation` → if the same utterance already contained a DOB that uniquely
  matches one candidate's `dob_hint`, resolve immediately; otherwise ask for the
  DOB and resolve on the next turn.
- `suggestion` (weak fuzzy near-miss) → "did you mean X?"; counts as a failed attempt.
- `not_found` / error → failed attempt.
- After **2 failed attempts** → `end_call: true`.

**handle_query** routes the question to one of the 4 tools and synthesises a reply:
- `explain_claim_status` → `/hospital/claim-status`
- `calculate_patient_owes` → `/hospital/cost-estimate` (estimate) or
  `/hospital/bill-explanation` (when they ask *why* they owe / about a bill)
- `get_active_conditions` → `/hospital/pre-procedure` (stop meds / fasting / prep)
  or `/smart-query` (list active conditions)
- general questions → `/smart-query`

The four `/hospital/*` endpoints already return spoken-ready text, so the agent's
job is mainly to condense it to ≤ 3 sentences. If `OPENAI_API_KEY` (or `OLLAMA_URL`)
is set, an LLM does that condensing and reads intent; if not, the agent falls back
to keyword routing + first-3-sentences trimming so it **always runs**.

## Run it

```bash
cd chetan
pip install -r requirements.txt

# Terminal A — backend. EITHER the real one (Thayaa's, needs the ML stack):
#   cd ../graph_rag_backend && uvicorn graph_rag.api.app:app --port 8000
# OR the lightweight dev replay (no torch/LLM needed):
python -m uvicorn dev_test_backend:app --port 8000

# Terminal B — the agent
export GRAPH_RAG_URL=http://localhost:8000
# optional: export OPENAI_API_KEY=sk-...   (otherwise deterministic mode)
uvicorn agent:app --port 9001 --reload

# Terminal C — smoke test
bash smoke_test.sh
```

## Design decisions

- **LangGraph over if/else**, as the spec asks: states are explicit, memory is
  handled by the checkpointer, and adding a state later (e.g. `reschedule`) is a
  one-node change.
- **LLM is optional.** The hospital endpoints are already natural language, so the
  agent is useful even with no model configured. With a model it gets shorter and
  more conversational. This keeps the service runnable for the whole team during dev.
- **DOB tie-break for disambiguation.** `/smart-query` matches on name only, so for
  common names it returns several people. The agent reads the caller's DOB and
  matches it against each candidate's `dob_hint` to pick the right record.

## One thing to flag to the team (backend, not blocking)

`/smart-query` does fuzzy name matching and returns a **disambiguation** whenever a
first *or* last name collides with other patients (there are many `* Nguyen` and
`* Garcia` records). For identity that's fine — we ask for the DOB. But it also means
"what conditions do I have?" only returns a clean list for **distinctive** names; for
a colliding name the agent can't get a single record back and degrades gracefully
("I can see your records but can't read that out by voice; a nurse will follow up")
instead of re-asking who the caller is.

The clean fix is a small backend endpoint, e.g. `POST /hospital/conditions
{patient_name}` (or letting `/smart-query` accept a `patient_id`), mirroring the
other four `/hospital/*` endpoints. The four core use cases — claim status, cost
estimate, bill explanation, pre-procedure prep — already resolve to a single patient
via `_fuzzy_match_person`, so they work for any name today.
