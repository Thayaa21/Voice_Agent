# Chetan — Agent Intelligence

## Your Role
You are the brain of the conversation. Rishi gives you what the caller said. You figure out who they are, query the knowledge graph, and return a natural spoken response. You handle all conversation state and multi-turn memory.

---

## Architecture

```
Rishi → POST /agent → Your LangGraph state machine → Graph RAG backend (port 8000)
                                                              ↓
                                                     Answer back to Rishi
```

---

## What to Build

### 1. FastAPI server on port 9001

One main endpoint:
```
POST /agent
Input:  {"text": "what caller said", "call_id": "unique-id"}
Output: {"response": "what to say back", "end_call": false}
```

### 2. LangGraph state machine

Use LangGraph to manage the conversation flow. States:

```
[START]
   │
   ▼
verify_identity       ← ask for name + DOB, query graph
   │ (verified)
   ▼
handle_query          ← answer questions using tools
   │ (goodbye / 2 failures)
   ▼
[END]
```

Why LangGraph and not a simple if/else? Because:
- It handles multi-turn memory naturally
- You can add new states without breaking existing ones
- It integrates cleanly with LangChain tool calling

Reference: [LangGraph quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

### 3. Conversation memory

Keep full history per call:
```python
sessions[call_id] = {
    "state": "verify_identity",
    "person": None,
    "history": [],      # list of {"role": "user/assistant", "content": "..."}
    "retries": 0,
}
```

Pass the history to the LLM so it understands context — if the caller already said "my knee surgery" earlier in the call, the agent should remember that.

### 4. The 4 tools

Wire these up as LangChain tools that the agent can call:

**Tool 1: `lookup_patient(query)`**
- Sends caller's name + DOB to `/smart-query`
- Returns patient record or disambiguation options
- Used during identity verification

**Tool 2: `explain_claim_status(person, question)`**
- Queries graph for claim history
- Handles: PAID, DENIED_NO_PA, PENDING_P2P, PARTIAL
- Explains in plain English why a claim was denied or is pending

**Tool 3: `calculate_patient_owes(person, question)`**
- Queries graph for billing details
- Explains deductible, coinsurance, out-of-pocket math in plain English
- e.g. "You've met $800 of your $1,500 deductible. For this procedure, you owe..."

**Tool 4: `get_active_conditions(person)`**
- Returns active conditions and current medications
- Used for pre-procedure questions like "do I need to stop my medication?"

### 5. Identity verification flow

- Send caller text to `/smart-query`
- `type: answer` → save person, move to `handle_query`
- `type: disambiguation` → ask caller to clarify which person they are
- `type: not_found` → ask to repeat (max 2 retries, then `end_call: true`)

### 6. Response generation

Use an LLM (OpenAI or Ollama) to synthesize the graph answer into a natural spoken response. The graph gives you facts — the LLM turns them into something a human wants to hear on the phone.

Keep responses short — no more than 3 sentences. This is a phone call, not an email.

---

## Environment Variables

```
GRAPH_RAG_URL=http://localhost:8000
OPENAI_API_KEY=        # or use Ollama locally
```

---

## Stack

```
fastapi
uvicorn
langgraph
langchain
langchain-openai    # or langchain-community for Ollama
httpx
python-dotenv
```

---

## How to Query the Graph Backend

```python
import httpx

async def query_graph(question: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/smart-query",
            json={"question": question, "max_hops": 3},
            timeout=10.0
        )
        return resp.json()
```

Response types:
```json
{"type": "answer",          "person": "Aiden Garcia", "answer": "...", "has_conflicts": false}
{"type": "disambiguation",  "options": ["Aiden Garcia", "Aiden Hall"]}
{"type": "not_found"}
{"type": "empty"}
```

---

## Test Curl Commands

```bash
# Start your server first
uvicorn agent:app --port 9001 --reload

# Test 1 — identity verification
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "Aiden Garcia March 15 1992", "call_id": "test_001"}'

# Test 2 — follow-up question (same call_id = same session)
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "why was my claim denied", "call_id": "test_001"}'

# Test 3 — billing question
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "how much do I owe", "call_id": "test_001"}'

# Test 4 — conditions
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "what conditions do I have", "call_id": "test_001"}'

# Test 5 — end call
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "bye", "call_id": "test_001"}'
```

---

## Definition of Done

- [ ] `/agent` endpoint running on port 9001
- [ ] LangGraph state machine with `verify_identity` and `handle_query` states
- [ ] All 4 tools implemented and callable
- [ ] Multi-turn memory works — agent remembers context from earlier in the call
- [ ] Test 1-5 all return sensible responses
- [ ] Disambiguation works when multiple patients match
- [ ] After 2 failed identity attempts → `end_call: true`
- [ ] Responses are short and natural (3 sentences max)
