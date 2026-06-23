"""
Chetan — Agent Intelligence Service (port 9001)
================================================
The "brain" of the hospital voice agent.

Rishi's voice server (port 8001) sends us what the caller said as plain text.
We decide who the caller is, query Thayaa's Graph RAG backend (port 8000) for
the right facts, and hand back a short, natural sentence to read aloud.

Conversation flow (LangGraph state machine):

    [START]
       │
       ▼
    verify_identity   ← ask for name + DOB, look the caller up in the graph
       │ (matched)
       ▼
    handle_query      ← answer questions by calling the 4 tools
       │ (goodbye  OR  2 failed identity attempts)
       ▼
    [END]

One HTTP call to POST /agent == one turn of the conversation. State for each
call is keyed by `call_id` and kept in a LangGraph checkpointer, so multi-turn
memory "just works" — if the caller mentioned something earlier in the call,
it's still in the history we pass around.

The LLM is optional. If OPENAI_API_KEY (or a local Ollama) is configured we use
it to compress the backend's answer into <= 3 spoken sentences and to read
intent. If neither is available, the agent falls back to deterministic keyword
routing + light sentence-trimming, so the service always runs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, TypedDict

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Load .env if present (no hard dependency on python-dotenv).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

GRAPH_RAG_URL  = os.getenv("GRAPH_RAG_URL", "http://localhost:8000").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_URL     = os.getenv("OLLAMA_URL", "").rstrip("/")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")
HTTP_TIMEOUT   = float(os.getenv("AGENT_HTTP_TIMEOUT", "15"))
MAX_RETRIES    = 2

# A shared sync client (the graph nodes are sync; FastAPI runs them in a pool).
_client = httpx.Client(timeout=HTTP_TIMEOUT)


# ===========================================================================
# Backend access
# ===========================================================================

def _smart_query(question: str) -> dict:
    """Call the Graph RAG /smart-query endpoint (used for identity + conditions)."""
    try:
        r = _client.post(
            f"{GRAPH_RAG_URL}/smart-query",
            json={"question": question, "max_hops": 3},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"type": "error", "answer": f"backend unreachable: {e}"}


def _hospital(endpoint: str, patient_name: str) -> dict:
    """Call one of the spoken-ready /hospital/* endpoints."""
    try:
        r = _client.post(
            f"{GRAPH_RAG_URL}/hospital/{endpoint}",
            json={"patient_name": patient_name},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"type": "error", "answer": f"backend unreachable: {e}"}


# ===========================================================================
# The 4 tools (LangChain tools — callable by the agent)
# ===========================================================================

@tool
def lookup_patient(query: str) -> dict:
    """Identify a caller from their spoken name (and optional date of birth).

    Sends the text to /smart-query. Returns a dict whose "type" is one of:
    answer (matched a single person), disambiguation (several people match),
    suggestion (a fuzzy near-miss), or not_found.
    """
    return _smart_query(query)


@tool
def explain_claim_status(person: str, question: str) -> dict:
    """Explain why a claim was denied or what its status is for a patient."""
    return _hospital("claim-status", person)


@tool
def calculate_patient_owes(person: str, question: str) -> dict:
    """Explain what the patient owes and route estimate vs bill explanation."""
    q = question.lower()
    wants_bill = any(
        k in q
        for k in ["why", "bill", "charge", "charged", "statement", "thought", "covered", "explain"]
    )
    endpoint = "bill-explanation" if wants_bill else "cost-estimate"
    return _hospital(endpoint, person)


@tool
def get_active_conditions(person: str, question: str) -> dict:
    """Answer health questions or pre-procedure prep guidance."""
    q = question.lower()
    wants_prep = any(
        k in q
        for k in ["stop", "fast", "fasting", "prep", "prepare", "before my", "eat", "drink", "instruction"]
    )
    if wants_prep:
        return _hospital("pre-procedure", person)
    return _smart_query(f"what are the active conditions and medications for {person}")


TOOLS = [lookup_patient, explain_claim_status, calculate_patient_owes, get_active_conditions]


# ===========================================================================
# LLM helper (optional — used for synthesis + intent; degrades gracefully)
# ===========================================================================

def _llm_complete(prompt: str, system: str = "", max_tokens: int = 160) -> Optional[str]:
    """Return an LLM completion, or None if no LLM is configured/reachable."""
    if OPENAI_API_KEY:
        try:
            r = _client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
                },
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
    if OLLAMA_URL:
        try:
            r = _client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": (system + "\n\n" + prompt) if system else prompt,
                    "stream": False,
                },
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception:
            return None
    return None


def _first_sentences(text: str, n: int = 3) -> str:
    """Deterministic fallback: keep the first n sentences."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:n]).strip()
    return out or text


def _synthesize(raw_answer: str, question: str, history: list) -> str:
    """Turn the backend's answer into a short, natural spoken reply."""
    if not raw_answer:
        return "I'm sorry, I couldn't find that information right now."
    if len(re.split(r"(?<=[.!?])\s+", raw_answer.strip())) <= 3 and len(raw_answer) < 320:
        return raw_answer.strip()

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Rewrite the FACTS below as a natural, friendly phone reply for the caller. "
        "Maximum THREE short sentences. Keep every number and status exactly as given. "
        "Do not invent anything. Speak directly to the caller.\n\n"
        f"Recent conversation:\n{convo}\n\n"
        f"Caller asked: {question}\n\nFACTS:\n{raw_answer}\n\nReply:"
    )
    polished = _llm_complete(prompt, system="You are a concise, warm hospital phone assistant.")
    return polished or _first_sentences(raw_answer, 3)


# ===========================================================================
# Small NLP helpers
# ===========================================================================

def _parse_dob(text: str) -> Optional[str]:
    """Pull a date of birth out of free text and normalise to YYYY-MM-DD."""
    try:
        from dateutil import parser as dparser
    except Exception:
        dparser = None

    candidates = []
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        candidates.append(f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}")
    if dparser:
        m2 = re.search(
            r"\b([A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
            r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\.?,?\s+\d{4})\b",
            text,
        )
        m3 = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
        for cand in (m2.group(0) if m2 else None, m3.group(0) if m3 else None):
            if not cand:
                continue
            try:
                dt = dparser.parse(cand, dayfirst=False, fuzzy=True)
                candidates.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                pass
    return candidates[0] if candidates else None


GOODBYE_KW = [
    "bye",
    "goodbye",
    "that's all",
    "thats all",
    "that is all",
    "nothing else",
    "no thanks",
    "no thank you",
    "hang up",
    "i'm done",
    "im done",
    "we're done",
    "all set",
    "good bye",
]


def _is_goodbye(text: str) -> bool:
    t = text.lower().strip()
    if t in ("no", "nope", "nah"):
        return True
    return any(k in t for k in GOODBYE_KW)


def _is_affirmative(text: str) -> bool:
    t = text.lower().strip().strip(".!")
    return t in (
        "yes",
        "yeah",
        "yep",
        "correct",
        "that's right",
        "thats right",
        "right",
        "yes please",
        "that's me",
        "thats me",
        "uh huh",
        "sure",
    )


def _route_intent(text: str) -> str:
    """Pick which tool a handle_query turn needs."""
    t = text.lower()
    if any(
        k in t
        for k in [
            "denied",
            "denial",
            "prior auth",
            "authorization",
            "authorisation",
            "claim",
            "p2p",
            "peer to peer",
            "peer-to-peer",
            "approved",
            "appeal",
        ]
    ):
        return "claim"
    if any(
        k in t
        for k in [
            "stop my",
            "stop taking",
            "medication",
            "medicine",
            "drug",
            "pill",
            "prescription",
            "fast",
            "fasting",
            "prep",
            "prepare",
            "before my procedure",
            "before surgery",
            "eat before",
            "drink before",
        ]
    ):
        return "conditions"
    if any(
        k in t
        for k in [
            "how much",
            "owe",
            "cost",
            "bill",
            "balance",
            "charge",
            "charged",
            "out of pocket",
            "out-of-pocket",
            "deductible",
            "coinsurance",
            "pay",
            "price",
        ]
    ):
        return "billing"
    if any(k in t for k in ["condition", "diagnos", "what's wrong", "whats wrong", "health", "history"]):
        return "conditions"
    return "general"


# ===========================================================================
# LangGraph state machine
# ===========================================================================

class AgentState(TypedDict, total=False):
    user_text: str
    call_state: str
    person: Optional[str]
    retries: int
    pending: list
    suggested: Optional[str]
    messages: list
    response: str
    end_call: bool


GREETING_HINT = "your full name and date of birth"


def _greet(name: str) -> str:
    return f"Thank you, {name}. I've found your records. How can I help you today?"


def _resolve_by_dob(text: str, options: list) -> Optional[str]:
    """Resolve a pending disambiguation by DOB if possible."""
    dob = _parse_dob(text)
    if not dob:
        return None
    match = [o for o in options if (o.get("dob_hint") or "") == dob]
    return match[0]["name"] if len(match) == 1 else None


def verify_identity(state: AgentState) -> dict:
    """Identity stage: look the caller up, handle disambiguation / retries."""
    text = state.get("user_text", "")
    retries = state.get("retries", 0)
    pending = state.get("pending", [])
    suggested = state.get("suggested")
    history = state.get("messages", [])

    if _is_goodbye(text):
        return {
            "response": "No problem, take care. Goodbye.",
            "end_call": True,
            "call_state": "done",
            "messages": history + [{"role": "assistant", "content": "No problem, take care. Goodbye."}],
        }

    if suggested and _is_affirmative(text):
        text = suggested
        suggested = None

    if pending:
        person = _resolve_by_dob(text, pending)
        if person:
            msg = _greet(person)
            return {
                "person": person,
                "call_state": "handle",
                "pending": [],
                "suggested": None,
                "response": msg,
                "messages": history + [{"role": "assistant", "content": msg}],
            }

    res = lookup_patient.invoke({"query": text})
    typ = res.get("type")

    if typ == "answer":
        person = res.get("person") or "there"
        msg = _greet(person)
        return {
            "person": person,
            "call_state": "handle",
            "retries": 0,
            "pending": [],
            "suggested": None,
            "response": msg,
            "messages": history + [{"role": "assistant", "content": msg}],
        }

    if typ == "disambiguation":
        opts = res.get("options", []) or []
        person = _resolve_by_dob(text, opts)
        if person:
            msg = _greet(person)
            return {
                "person": person,
                "call_state": "handle",
                "retries": 0,
                "pending": [],
                "suggested": None,
                "response": msg,
                "messages": history + [{"role": "assistant", "content": msg}],
            }
        names = ", ".join(o["name"] for o in opts[:4])
        msg = (
            f"I found more than one patient by that name: {names}. "
            f"Could you confirm your date of birth so I can pull the right record?"
        )
        return {
            "call_state": "verify",
            "pending": opts,
            "suggested": None,
            "response": msg,
            "messages": history + [{"role": "assistant", "content": msg}],
        }

    if typ == "suggestion":
        retries += 1
        if retries >= MAX_RETRIES:
            msg = (
                "I'm sorry, I wasn't able to verify your identity. "
                "Please call back with your full name and date of birth, "
                "or hold for a representative. Goodbye."
            )
            return {
                "retries": retries,
                "call_state": "done",
                "end_call": True,
                "response": msg,
                "messages": history + [{"role": "assistant", "content": msg}],
            }
        sugg = (res.get("suggestions") or [None])[0]
        msg = (
            f"I'm not sure I caught that. Did you mean {sugg}?"
            if sugg
            else "I'm sorry, I didn't catch your name. Could you repeat it?"
        )
        return {
            "call_state": "verify",
            "suggested": sugg,
            "pending": [],
            "retries": retries,
            "response": msg,
            "messages": history + [{"role": "assistant", "content": msg}],
        }

    retries += 1
    if retries >= MAX_RETRIES:
        msg = (
            "I'm sorry, I still couldn't find your records after a couple of tries. "
            "Please call back with your name and date of birth, or hold for a representative. Goodbye."
        )
        return {
            "retries": retries,
            "call_state": "done",
            "end_call": True,
            "response": msg,
            "messages": history + [{"role": "assistant", "content": msg}],
        }
    msg = f"I couldn't find a match. Could you please tell me {GREETING_HINT} again, speaking slowly?"
    return {
        "retries": retries,
        "call_state": "verify",
        "response": msg,
        "messages": history + [{"role": "assistant", "content": msg}],
    }


def handle_query(state: AgentState) -> dict:
    """Question stage: route to a tool, synthesise a short spoken reply."""
    text = state.get("user_text", "")
    person = state.get("person") or ""
    history = state.get("messages", [])

    if _is_goodbye(text):
        msg = f"You're welcome, {person.split()[0] if person else 'and'}. Take care. Goodbye."
        msg = re.sub(r"\s+,", ",", msg)
        return {
            "response": msg,
            "end_call": True,
            "call_state": "done",
            "messages": history + [{"role": "assistant", "content": msg}],
        }

    intent = _route_intent(text)
    if intent == "claim":
        res = explain_claim_status.invoke({"person": person, "question": text})
    elif intent == "billing":
        res = calculate_patient_owes.invoke({"person": person, "question": text})
    elif intent == "conditions":
        res = get_active_conditions.invoke({"person": person, "question": text})
    else:
        res = _smart_query(f"{text} for {person}")

    typ = res.get("type") if isinstance(res, dict) else None
    raw = res.get("answer", "") if isinstance(res, dict) else str(res)

    first = person.split()[0] if person else "there"
    if typ in ("disambiguation", "suggestion"):
        reply = (
            f"I can see your records, {first}, but I'm not able to read that "
            f"particular detail out by voice right now. I'll have a member of "
            f"our team follow up with you."
        )
    elif typ in ("not_found", "empty", "error") or not raw:
        reply = (
            "I'm sorry, I don't have that information on file at the moment. "
            "Our team can look into it and follow up if you'd like."
        )
    else:
        reply = _synthesize(raw, text, history)
    return {
        "response": reply,
        "call_state": "handle",
        "end_call": False,
        "messages": history + [{"role": "assistant", "content": reply}],
    }


def _entry_router(state: AgentState) -> str:
    """Decide which node handles this turn based on persisted call_state."""
    return "handle_query" if state.get("call_state") == "handle" else "verify_identity"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("verify_identity", verify_identity)
    g.add_node("handle_query", handle_query)
    g.add_conditional_edges(
        START,
        _entry_router,
        {"verify_identity": "verify_identity", "handle_query": "handle_query"},
    )
    g.add_edge("verify_identity", END)
    g.add_edge("handle_query", END)
    return g.compile(checkpointer=MemorySaver())


GRAPH = build_graph()


# ===========================================================================
# FastAPI surface
# ===========================================================================

app = FastAPI(title="Chetan — Agent Intelligence", version="1.0.0")


class AgentRequest(BaseModel):
    text: str
    call_id: str


class AgentResponse(BaseModel):
    response: str
    end_call: bool = False


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "agent-intelligence",
        "graph_rag": GRAPH_RAG_URL,
        "llm": "openai" if OPENAI_API_KEY else ("ollama" if OLLAMA_URL else "deterministic"),
    }


@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest):
    cfg = {"configurable": {"thread_id": req.call_id}}
    snap = GRAPH.get_state(cfg)
    history = (snap.values.get("messages") if snap and snap.values else None) or []
    history = history + [{"role": "user", "content": req.text}]

    out = GRAPH.invoke({"user_text": req.text, "messages": history}, config=cfg)
    return AgentResponse(
        response=out.get("response", "I'm sorry, could you repeat that?"),
        end_call=bool(out.get("end_call", False)),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent:app", host="0.0.0.0", port=9001, reload=False)
