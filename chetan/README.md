# Chetan — Agent Intelligence

## Your Job in One Sentence
Build one HTTP endpoint that receives what the caller said and returns what the agent should say back.

No voice. No Twilio. No database setup. Just Python + HTTP calls.

---

## How It Works

```
Rishi sends you:  {"text": "Aiden Garcia March 15 1992", "call_id": "CA123"}
You return:       {"response": "Found you Aiden Garcia. How can I help?", "end_call": false}
```

You figure out who the caller is by querying Thayaa's graph backend (already running on port 8000).

---

## Step 1 — Set Up Your Environment

```bash
cd chetan
pip install -r requirements.txt
cp ../.env.example .env
```

Your `.env` only needs one line:
```
GRAPH_RAG_URL=http://localhost:8000
```

---

## Step 2 — Build Your Server

Create a file called `agent.py` in the `chetan/` folder.

### The one endpoint you need: `POST /agent`

```python
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# In-memory session store — keyed by call_id
sessions = {}

class AgentRequest(BaseModel):
    text: str
    call_id: str

class AgentResponse(BaseModel):
    response: str
    end_call: bool = False

GRAPH_URL = "http://localhost:8000"

async def query_graph(question: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GRAPH_URL}/smart-query",
            json={"question": question, "max_hops": 3},
            timeout=10.0
        )
        return resp.json()

@app.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest):
    call_id = req.call_id
    text = req.text.strip()

    # Create session if new call
    if call_id not in sessions:
        sessions[call_id] = {"state": "identity_verify", "person": None, "retries": 0}

    session = sessions[call_id]

    # Handle goodbye
    if any(word in text.lower() for word in ["bye", "goodbye", "thank you that's all", "hang up"]):
        del sessions[call_id]
        return AgentResponse(response="Thank you for calling. Goodbye!", end_call=True)

    # STATE: identity_verify
    if session["state"] == "identity_verify":
        result = await query_graph(text)

        if result.get("type") == "answer":
            person = result.get("person", "")
            session["state"] = "answering"
            session["person"] = person
            return AgentResponse(
                response=f"Thank you. I found your records, {person}. How can I help you today?"
            )

        elif result.get("type") == "disambiguation":
            options = result.get("options", [])
            options_str = " or ".join(options[:3])
            return AgentResponse(
                response=f"I found multiple matches. Are you {options_str}?"
            )

        else:
            session["retries"] += 1
            if session["retries"] >= 2:
                del sessions[call_id]
                return AgentResponse(
                    response="I'm unable to find your records. Please call back or visit us in person. Goodbye.",
                    end_call=True
                )
            return AgentResponse(
                response="I couldn't find your records. Could you please repeat your full name and date of birth?"
            )

    # STATE: answering
    elif session["state"] == "answering":
        person = session["person"]
        # Prefix question with person's name for better graph matching
        question = f"{person} {text}"
        result = await query_graph(question)

        answer = result.get("answer", "I'm sorry, I don't have that information.")
        has_conflicts = result.get("has_conflicts", False)

        response = answer
        if has_conflicts:
            response += " Note: there is a discrepancy in your records that our billing team should review."

        return AgentResponse(response=response)

    return AgentResponse(response="I'm sorry, something went wrong. Please try again.")
```

---

## Step 3 — Run Your Server

```bash
uvicorn agent:app --port 9001 --reload
```

---

## Step 4 — Test Without a Phone

Run these curl commands one by one. Each uses the same `call_id` so the session is preserved.

**Test 1 — Identity verification (should find the patient):**
```bash
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "Aiden Garcia March 15 1992", "call_id": "test_001"}'
```
Expected: `"Found your records, Aiden Garcia. How can I help you today?"`

**Test 2 — Ask about conditions (same call):**
```bash
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "what conditions do I have", "call_id": "test_001"}'
```
Expected: answer about their active conditions

**Test 3 — Ask about a claim:**
```bash
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "why was my claim denied", "call_id": "test_001"}'
```
Expected: explanation of denied claim

**Test 4 — End the call:**
```bash
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "bye", "call_id": "test_001"}'
```
Expected: goodbye message with `"end_call": true`

---

## Step 5 — Make Sure the Graph Backend is Running

Before testing, start Thayaa's backend:
```bash
cd graph_rag_backend
uvicorn graph_rag.api.app:app --port 8000 --reload
```

You can verify it's working:
```bash
curl http://localhost:8000/graph/stats
```

---

## Definition of Done

- [ ] `agent.py` running on port 9001
- [ ] Test 1 returns a patient name
- [ ] Test 2 returns condition info
- [ ] Test 3 returns claim info
- [ ] Test 4 returns `end_call: true`
- [ ] Works end-to-end when Rishi calls your endpoint from a real phone call

---

## What the Graph Returns

When you call `POST http://localhost:8000/smart-query`:

```json
// Patient found
{
  "type": "answer",
  "person": "Aiden Garcia",
  "answer": "Aiden Garcia has 3 active conditions including obesity...",
  "has_conflicts": false
}

// Multiple people match
{
  "type": "disambiguation",
  "options": ["Aiden Garcia", "Aiden Hall", "Aiden Kim"]
}

// No match
{
  "type": "not_found"
}
```

Use `answer` directly as your spoken response. Don't rewrite it.

---

## You Do NOT Need To

- Touch Twilio or any voice code (that's Rishi)
- Understand the graph internals (that's Thayaa)
- Build a database or file storage (in-memory dict is fine)
- Handle multiple concurrent users in production (demo is fine with simple dict)
