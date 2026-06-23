"""
Hospital Voice Agent — Port 9001
=================================
Clean, single-definition FastAPI agent for hospital patient calls.

States:
  verify_identity → answering → end_call
"""

import os
import json
import logging
import asyncio
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from events import broadcast, register

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRAPH_URL    = os.getenv("GRAPH_RAG_URL", "http://localhost:8000")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TWILIO_FROM  = os.getenv("TWILIO_PHONE_NUMBER", "")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
sessions: dict[str, dict] = {}


def get_session(call_id: str) -> dict:
    if call_id not in sessions:
        sessions[call_id] = {
            "state":         "verify_identity",
            "person":        None,
            "retries":       0,
            "history":       [],
            "caller_number": "",
        }
    return sessions[call_id]


# ---------------------------------------------------------------------------
# Graph / hospital helpers
# ---------------------------------------------------------------------------

async def smart_query(question: str) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(
                f"{GRAPH_URL}/smart-query",
                json={"question": question, "max_hops": 3},
            )
            return r.json()
        except Exception as e:
            logger.error("smart_query failed: %s", e)
            return {"type": "not_found"}


async def hospital_endpoint(endpoint: str, patient_name: str) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(
                f"{GRAPH_URL}/hospital/{endpoint}",
                json={"patient_name": patient_name},
            )
            return r.json()
        except Exception as e:
            logger.error("hospital/%s failed: %s", endpoint, e)
            return {
                "type": "not_found",
                "answer": "I'm having trouble retrieving that information right now.",
            }


# ---------------------------------------------------------------------------
# Intent / goodbye helpers
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> str:
    """Return one of: claim_status, cost_estimate, pre_procedure, bill_explain, general."""
    t = text.lower()

    claim_kw = ["denied", "denial", "authorization", "prior auth", "p2p",
                "peer to peer", "claim", "rejected", "not covered", "coverage denied"]
    cost_kw  = ["how much", "cost", "owe", "estimate", "out of pocket",
                "deductible", "copay", "coinsurance", "what will i pay"]
    prep_kw  = ["before", "prep", "preparation", "fasting", "fast", "eat",
                "drink", "medication", "stop taking", "blood thinner",
                "arrive", "what time", "what should i"]
    bill_kw  = ["bill", "statement", "charge", "invoice", "why do i owe",
                "balance", "payment", "pay my bill", "received a bill"]

    if any(kw in t for kw in claim_kw): return "claim_status"
    if any(kw in t for kw in cost_kw):  return "cost_estimate"
    if any(kw in t for kw in prep_kw):  return "pre_procedure"
    if any(kw in t for kw in bill_kw):  return "bill_explain"
    return "general"


def is_goodbye(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in [
        "bye", "goodbye", "thank you that's all", "that's all",
        "hang up", "no more questions", "nothing else", "thanks bye",
    ])


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

async def llm_answer(system: str, user: str) -> str:
    if not OPENAI_KEY:
        return "I'm sorry, I'm unable to process that request right now."
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_KEY)
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM failed: %s", e)
        return "I'm having trouble with that right now. Please hold while I connect you to our team."


# ---------------------------------------------------------------------------
# Patient record pre-fetch (parallel)
# ---------------------------------------------------------------------------

async def _fetch_patient_record(person: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r1, r2, r3 = await asyncio.gather(
                client.post(f"{GRAPH_URL}/hospital/claim-status",  json={"patient_name": person}),
                client.post(f"{GRAPH_URL}/hospital/cost-estimate", json={"patient_name": person}),
                client.post(f"{GRAPH_URL}/hospital/pre-procedure", json={"patient_name": person}),
            )
            return {
                "claim_status":  r1.json().get("answer", ""),
                "cost_estimate": r2.json().get("answer", ""),
                "pre_procedure": r3.json().get("answer", ""),
            }
    except Exception as e:
        logger.error("Patient record fetch failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# Main conversation logic
# ---------------------------------------------------------------------------

async def process_turn(call_id: str, text: str) -> dict:
    session = get_session(call_id)
    state   = session["state"]

    await broadcast("transcript", {"call_id": call_id, "text": text, "state": state})
    session["history"].append({"role": "user", "content": text})

    # Goodbye check — works in any state
    if is_goodbye(text):
        person = session.get("person") or ""
        farewell = f"Thank you for calling{', ' + person if person else ''}. Have a great day. Goodbye!"
        session["state"] = "end_call"
        await broadcast("call_ended", {"call_id": call_id, "person": person})
        return _resp(farewell, end_call=True)

    # ── STATE: verify_identity ────────────────────────────────────────────
    if state == "verify_identity":
        words = [w for w in text.strip().split() if len(w) > 1]
        if len(words) < 2:
            reply = "I need your full name to find your records. Please say your first and last name."
            await broadcast("agent_response", {"call_id": call_id, "text": reply, "state": "verify_identity"})
            return _resp(reply)

        await broadcast("searching", {"call_id": call_id, "query": text})
        result = await smart_query(text)
        rtype  = result.get("type", "not_found")

        if rtype == "answer":
            person       = result.get("person", "")
            person_parts = person.lower().split()
            text_lower   = text.lower()
            name_found   = any(part in text_lower for part in person_parts if len(part) > 2)

            if not name_found:
                session["retries"] += 1
                if session["retries"] >= 2:
                    sessions.pop(call_id, None)
                    await broadcast("patient_not_found", {"call_id": call_id})
                    return _resp(
                        "I'm unable to locate your records. Please call back with your full name and date of birth. Goodbye.",
                        end_call=True,
                    )
                reply = "I couldn't find your records with that name. Could you please repeat your full first and last name?"
                await broadcast("agent_response", {"call_id": call_id, "text": reply})
                return _resp(reply)

            session["person"] = person
            session["state"]  = "answering"
            await broadcast("patient_identified", {"call_id": call_id, "person": person})
            patient_data = await _fetch_patient_record(person)
            await broadcast("patient_record", {"call_id": call_id, "person": person, "record": patient_data})
            reply = f"Thank you. I found your records, {person}. How can I help you today?"
            await broadcast("agent_response", {"call_id": call_id, "text": reply})
            return _resp(reply)

        elif rtype == "disambiguation":
            options    = result.get("options", [])
            names      = [o["name"] if isinstance(o, dict) else str(o) for o in options[:3]]
            text_lower = text.lower().strip()

            for name in names:
                if name.lower() in text_lower or text_lower in name.lower():
                    session["person"] = name
                    session["state"]  = "answering"
                    await broadcast("patient_identified", {"call_id": call_id, "person": name})
                    patient_data = await _fetch_patient_record(name)
                    await broadcast("patient_record", {"call_id": call_id, "person": name, "record": patient_data})
                    reply = f"Thank you. I found your records, {name}. How can I help you today?"
                    await broadcast("agent_response", {"call_id": call_id, "text": reply})
                    return _resp(reply)

            names_str = ", or ".join(names)
            reply = f"I found a few patients with that name. Are you {names_str}?"
            await broadcast("agent_response", {"call_id": call_id, "text": reply})
            return _resp(reply)

        else:
            session["retries"] += 1
            if session["retries"] >= 2:
                sessions.pop(call_id, None)
                await broadcast("patient_not_found", {"call_id": call_id})
                return _resp(
                    "I'm unable to locate your records after two attempts. Please call back with your full name and date of birth, or visit us in person. Goodbye.",
                    end_call=True,
                )
            reply = "I couldn't find your records. Could you please repeat your full name and date of birth?"
            await broadcast("agent_response", {"call_id": call_id, "text": reply})
            return _resp(reply)

    # ── STATE: answering ──────────────────────────────────────────────────
    elif state == "answering":
        person = session["person"]
        intent = classify_intent(text)

        await broadcast("tool_called", {"call_id": call_id, "intent": intent, "person": person})

        if intent in ("claim_status", "cost_estimate"):
            reply = (
                f"I understand, {person}. This requires me to contact your insurance company directly. "
                f"I'll call them now and call you back within 15 minutes with a full answer. "
                f"Please keep your phone available. Have a great day."
            )
            await broadcast("agent_response", {"call_id": call_id, "text": reply})
            await broadcast("three_phase_queued", {"call_id": call_id, "person": person, "intent": intent})
            session["state"] = "end_call"

            callback_number = session.get("caller_number") or TWILIO_FROM or "+15313245471"

            from orchestrator import run_three_phase
            asyncio.create_task(
                run_three_phase(
                    patient_name    = person,
                    callback_number = callback_number,
                    inquiry_type    = intent,
                    call_id         = call_id,
                )
            )
            await broadcast("call_ended", {"call_id": call_id, "person": person})
            return _resp(reply, end_call=True)

        if intent == "pre_procedure":
            result = await hospital_endpoint("pre-procedure", person)
            reply  = result.get("answer", "I couldn't find preparation instructions.")
        elif intent == "bill_explain":
            result = await hospital_endpoint("bill-explanation", person)
            reply  = result.get("answer", "I couldn't retrieve your billing details.")
        else:
            q      = f"{person} {text}"
            result = await smart_query(q)
            if result.get("type") == "answer":
                reply = result.get("answer", "I don't have that information.")
            else:
                reply = await llm_answer(
                    f"You are a hospital AI assistant. The verified patient is {person}. Answer concisely in 2 sentences.",
                    text,
                )
            if result.get("has_conflicts"):
                reply += " Note: there is a discrepancy in your records."

        await broadcast("tool_result",    {"call_id": call_id, "intent": intent, "answer": reply})
        await broadcast("agent_response", {"call_id": call_id, "text": reply})
        session["history"].append({"role": "assistant", "content": reply})
        return _resp(reply)

    return _resp("Is there anything else I can help you with?")


def _resp(text: str, end_call: bool = False) -> dict:
    return {"response": text, "end_call": end_call}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Hospital Agent", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    text:          str
    call_id:       str
    caller_number: str = ""


class AgentResponse(BaseModel):
    response: str
    end_call: bool = False


class ResolveRequest(BaseModel):
    patient_name:    str
    inquiry_type:    str
    callback_number: str
    call_id:         str = "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/dashboard")
async def dashboard():
    path = Path(__file__).parent.parent / "dashboard" / "index.html"
    return FileResponse(str(path))


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(req: AgentRequest):
    if req.call_id not in sessions:
        await broadcast("call_started", {"call_id": req.call_id})
    session = get_session(req.call_id)
    if req.caller_number:
        session["caller_number"] = req.caller_number
    result = await process_turn(req.call_id, req.text.strip())
    return AgentResponse(**result)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await register(websocket)


@app.post("/tts")
async def tts_endpoint(body: dict):
    text     = body.get("text", "")
    voice_id = body.get("voice_id", os.getenv("ELEVENLABS_HOSPITAL_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"))
    el_key   = os.getenv("ELEVENLABS_API_KEY", "")

    if not text or not el_key:
        raise HTTPException(status_code=400, detail="Missing text or API key")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key":   el_key,
                    "Content-Type": "application/json",
                    "Accept":       "audio/mpeg",
                },
                json={
                    "text":     text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            )
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"ElevenLabs error: {r.text[:200]}")

            from fastapi.responses import Response as FastResponse
            return FastResponse(
                content=r.content,
                media_type="audio/mpeg",
                headers={"Cache-Control": "no-cache"},
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "service":         "hospital-agent",
        "port":            9001,
        "active_sessions": len(sessions),
    }


@app.delete("/sessions/{call_id}")
def clear_session(call_id: str):
    sessions.pop(call_id, None)
    return {"cleared": call_id}


@app.post("/resolve")
async def resolve_endpoint(req: ResolveRequest):
    from orchestrator import run_three_phase
    logger.info("3-phase resolution triggered: %s → %s", req.patient_name, req.callback_number)
    asyncio.create_task(
        run_three_phase(
            patient_name    = req.patient_name,
            callback_number = req.callback_number,
            inquiry_type    = req.inquiry_type,
            call_id         = req.call_id,
        )
    )
    return {
        "status":   "queued",
        "message":  f"Resolution started for {req.patient_name}. Patient will be called back shortly.",
        "patient":  req.patient_name,
        "inquiry":  req.inquiry_type,
        "callback": req.callback_number,
    }


@app.post("/resolve/sync")
async def resolve_sync_endpoint(req: ResolveRequest):
    from orchestrator import run_three_phase
    result = await run_three_phase(
        patient_name    = req.patient_name,
        callback_number = req.callback_number,
        inquiry_type    = req.inquiry_type,
        call_id         = req.call_id,
    )
    return result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=9001, reload=True)
