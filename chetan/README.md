# Chetan — Agent Intelligence

## Your Role
You are the brain. Rishi gives you what the caller said. You figure out who they are, look up their records from the Graph RAG backend, and return a spoken response.

## Stack
- Python FastAPI (agent service, port 9001)
- `httpx` — to call the Graph RAG backend
- `openai` or Ollama — for response generation (optional, graph already gives you the answer)

## The Graph RAG Backend (already running on port 8000)

This is Thayaa's project. It has 45 people with their medical records, insurance, doctor info, etc. You query it like this:

```
POST http://localhost:8000/smart-query
{"question": "what is Mei Lee's diagnosis", "max_hops": 3}
```

Response:
```json
{
  "type": "answer",
  "person": "Mei Lee",
  "answer": "Mei Lee's diagnosis is Type 2 diabetes mellitus. Her doctor is Dr. Emily Chen.",
  "has_conflicts": false
}
```

Response `type` can be:
- `"answer"` — found one person, answer is ready. Use `answer` field directly.
- `"disambiguation"` — multiple people match. Ask caller which one.
- `"not_found"` — no match. Ask them to repeat.

## What to Build

### 1. FastAPI server with one main endpoint

**POST `/agent`**

Input: `{"text": "what caller said", "call_id": "unique-id"}`
Output: `{"response": "text to speak back", "end_call": false}`

### 2. Session/state management (in-memory dict keyed by call_id)

Track each call's state:
- `identity_verify` — waiting for name + DOB
- `answering` — person verified, answering questions
- `end` — call finished

### 3. Logic for each state

**State: `identity_verify`**
- Send the text directly to `/smart-query`
- If `type == "answer"` → save `person` to session, move to `answering` state, reply: "Found you {person}. How can I help?"
- If `type == "disambiguation"` → reply: "I found multiple matches. Are you {option1} or {option2}?"
- If `type == "not_found"` → reply: "I couldn't find your records. Could you repeat your name and date of birth?"
- After 2 failed attempts → `end_call: true`

**State: `answering`**
- Prefix the caller's text with the person's name: `"{person}'s {text}"`
- Send to `/smart-query`
- Return the `answer` field directly as `response`
- If `has_conflicts: true` in the response, add: "Note: there is a date of birth discrepancy in your records."
- If caller says "reschedule", "cancel", "appointment" → reply: "Our scheduling team will call you back within 24 hours."
- If caller says "bye", "goodbye", "thanks, that's all" → `end_call: true`

### 4. Test without Rishi
```bash
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "Mei Lee January 31 1982", "call_id": "test_001"}'

# Follow-up (same call_id keeps context)
curl -X POST http://localhost:9001/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "what is my diagnosis", "call_id": "test_001"}'
```

## .env keys you need
```
GRAPH_RAG_URL=http://localhost:8000
OPENAI_API_KEY=   (optional, the graph backend already synthesizes answers)
```

## Definition of Done
These curl tests should all work:
1. Send name + DOB → get "Found you {name}, how can I help?"
2. Send "what is my diagnosis" → get the diagnosis from the graph
3. Send "what is my doctor" → get the doctor name
4. Send "do I have any conflicts" → mention the DOB mismatch if it exists
5. Send "bye" → get a goodbye message with `end_call: true`
