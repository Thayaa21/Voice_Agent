"""
3-Phase Call Orchestrator
==========================
Phase 1: Patient calls → agent collects inquiry → call ends
Phase 2: AI calls insurance → gets denial details (with live events)
Phase 3: AI calls patient back with full answer
"""

import os
import asyncio
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

INSURANCE_URL = os.getenv("INSURANCE_AGENT_URL", "http://localhost:8002")
GRAPH_URL     = os.getenv("GRAPH_RAG_URL", "http://localhost:8000")
TWILIO_SID    = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM   = os.getenv("TWILIO_PHONE_NUMBER", "")


async def run_three_phase(
    patient_name:    str,
    callback_number: str,
    inquiry_type:    str,
    call_id:         str,
) -> dict:
    from events import broadcast
    logger.info("[3-PHASE] Starting for %s (type=%s)", patient_name, inquiry_type)

    # ── Phase 2: Call insurance ───────────────────────────────────────────
    await broadcast("insurance_calling", {
        "call_id": call_id,
        "patient_name": patient_name,
        "message": "📞 Dialing Pacific Shield Insurance...",
    })
    await asyncio.sleep(1)

    await broadcast("insurance_ivr", {
        "call_id": call_id,
        "message": "🔢 Navigating IVR: 'For claims, press 1' → pressing 1",
        "dtmf": "1",
    })
    await asyncio.sleep(1)

    await broadcast("insurance_hold", {
        "call_id": call_id,
        "message": "🎵 On hold... 'All agents are busy. Your call is important to us.'",
    })

    insurance_result = {}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{INSURANCE_URL}/inquiry",
                json={
                    "patient_name":  patient_name,
                    "inquiry_type":  inquiry_type,
                    "simulate_hold": True,
                    "hold_seconds":  3,
                }
            )
            insurance_result = r.json()

        await broadcast("insurance_connected", {
            "call_id": call_id,
            "message": "☎️ Insurance agent answered: 'Thank you for holding. This is Pacific Shield Claims. How can I assist you?'",
        })
        await asyncio.sleep(0.5)

        await broadcast("insurance_response", {
            "call_id":        call_id,
            "agent_response": insurance_result.get("agent_response", ""),
            "claim_status":   insurance_result.get("claim_status", ""),
            "summary":        insurance_result.get("summary", ""),
        })
        logger.info("[3-PHASE] Insurance: status=%s", insurance_result.get("claim_status"))

    except Exception as e:
        logger.error("[3-PHASE] Insurance call failed: %s", e)
        insurance_result = {
            "agent_response": "Unable to reach insurance company.",
            "claim_status":   "unknown",
        }

    # ── Phase 2b: Get hospital endpoint answer ────────────────────────────
    hospital_answer = ""
    try:
        endpoint_map = {
            "claim_status":  "claim-status",
            "cost_estimate": "cost-estimate",
            "pre_procedure": "pre-procedure",
            "bill_explain":  "bill-explanation",
        }
        ep = endpoint_map.get(inquiry_type, "claim-status")
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{GRAPH_URL}/hospital/{ep}",
                                  json={"patient_name": patient_name})
            hospital_answer = r.json().get("answer", "")
    except Exception as e:
        logger.error("[3-PHASE] Hospital endpoint failed: %s", e)

    # ── Build callback message ────────────────────────────────────────────
    insurance_response = insurance_result.get("agent_response", "")
    claim_status       = insurance_result.get("claim_status", "unknown")
    callback_message   = _build_callback_message(
        patient_name, inquiry_type, insurance_response, hospital_answer, claim_status
    )

    # ── Phase 3: Call patient back ────────────────────────────────────────
    await broadcast("callback_initiated", {
        "call_id":          call_id,
        "patient_name":     patient_name,
        "callback_number":  callback_number,
        "callback_message": callback_message,
        "claim_status":     claim_status,
    })

    callback_result = await _call_patient_back(callback_number, callback_message)

    return {
        "patient_name":       patient_name,
        "inquiry_type":       inquiry_type,
        "claim_status":       claim_status,
        "insurance_response": insurance_response,
        "callback_message":   callback_message,
        "callback_result":    callback_result,
        "phases_completed":   3,
    }


def _build_callback_message(patient_name, inquiry_type, insurance_response,
                             hospital_answer, claim_status) -> str:
    intro = (
        f"Hello, this is the hospital's automated assistant calling back for {patient_name}. "
        f"I spoke with Pacific Shield Insurance on your behalf and I have the information you need. "
    )
    body = hospital_answer or insurance_response
    if claim_status == "DENIED_NO_PA" and insurance_response:
        body = body.rstrip() + (
            " I confirmed with the insurance company that you have 30 days to file an appeal, "
            "and your doctor can request a Peer-to-Peer clinical review. "
            "Your care coordinator will follow up with you within 24 hours."
        )
    return intro + body + " Is there anything else you'd like to know?"


async def _call_patient_back(callback_number: str, message: str) -> dict:
    if not TWILIO_SID or not TWILIO_TOKEN or not TWILIO_FROM:
        logger.info("[3-PHASE] Twilio not configured — logged only")
        return {"status": "logged_only", "message": message}
    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse
        twiml = VoiceResponse()
        twiml.say(message, voice="Polly.Joanna")
        twiml.say("Thank you for calling. Goodbye.", voice="Polly.Joanna")
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        call   = client.calls.create(twiml=str(twiml), to=callback_number, from_=TWILIO_FROM)
        logger.info("[3-PHASE] Callback: CallSid=%s", call.sid)
        return {"status": "initiated", "call_sid": call.sid}
    except Exception as e:
        logger.error("[3-PHASE] Callback failed: %s", e)
        return {"status": "failed", "error": str(e)}
