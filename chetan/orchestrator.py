"""
3-Phase Call Orchestrator
==========================
Manages the full async 3-phase resolution flow:

  Phase 1: Patient calls → agent collects inquiry → call ends
  Phase 2: Agent calls insurance → gets denial details
  Phase 3: Agent calls patient back with full answer

This runs as a background task triggered after Phase 1.
Uses Twilio outbound call API for Phases 2 and 3.
"""

import os
import asyncio
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

INSURANCE_URL      = os.getenv("INSURANCE_AGENT_URL", "http://localhost:8002")
GRAPH_URL          = os.getenv("GRAPH_RAG_URL", "http://localhost:8000")
TWILIO_SID         = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN       = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM        = os.getenv("TWILIO_PHONE_NUMBER", "")


async def run_three_phase(
    patient_name:    str,
    callback_number: str,
    inquiry_type:    str,
    call_id:         str,
) -> dict:
    """
    Execute the full 3-phase resolution flow.

    Args:
        patient_name:    verified patient display name
        callback_number: patient's phone number to call back
        inquiry_type:    "claim_status" | "cost_estimate" | "pre_procedure" | "bill_explain"
        call_id:         original Twilio CallSid for logging

    Returns:
        dict with phase results and final callback message
    """
    logger.info("[3-PHASE] Starting for %s (type=%s)", patient_name, inquiry_type)

    # ── Phase 2: Call insurance agent ────────────────────────────────────
    logger.info("[3-PHASE] Phase 2: Calling insurance agent for %s", patient_name)

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
            logger.info("[3-PHASE] Insurance response: status=%s",
                        insurance_result.get("claim_status"))
    except Exception as e:
        logger.error("[3-PHASE] Insurance call failed: %s", e)
        insurance_result = {
            "agent_response": "We were unable to reach the insurance company. Please try again later.",
            "claim_status":   "unknown",
        }

    # ── Phase 2b: Also get full hospital endpoint data ────────────────────
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
            r = await client.post(
                f"{GRAPH_URL}/hospital/{ep}",
                json={"patient_name": patient_name}
            )
            hospital_answer = r.json().get("answer", "")
    except Exception as e:
        logger.error("[3-PHASE] Hospital endpoint failed: %s", e)

    # ── Build callback message ────────────────────────────────────────────
    insurance_response = insurance_result.get("agent_response", "")
    claim_status       = insurance_result.get("claim_status", "unknown")

    callback_message = _build_callback_message(
        patient_name      = patient_name,
        inquiry_type      = inquiry_type,
        insurance_response = insurance_response,
        hospital_answer   = hospital_answer,
        claim_status      = claim_status,
    )

    logger.info("[3-PHASE] Callback message built: %s...", callback_message[:100])

    # ── Phase 3: Call patient back ────────────────────────────────────────
    logger.info("[3-PHASE] Phase 3: Calling patient back at %s", callback_number)

    callback_result = await _call_patient_back(callback_number, callback_message)

    return {
        "patient_name":      patient_name,
        "inquiry_type":      inquiry_type,
        "claim_status":      claim_status,
        "insurance_response": insurance_response,
        "callback_message":  callback_message,
        "callback_result":   callback_result,
        "phases_completed":  3,
    }


def _build_callback_message(
    patient_name:      str,
    inquiry_type:      str,
    insurance_response: str,
    hospital_answer:   str,
    claim_status:      str,
) -> str:
    """Combine insurance and hospital data into a natural callback message."""

    intro = (
        f"Hello, this is the hospital's automated assistant calling back for {patient_name}. "
        f"I spoke with Pacific Shield Insurance on your behalf and I have the information you need. "
    )

    # Use hospital answer as the primary response (it's already spoken-ready)
    # Append key insurance details
    if hospital_answer:
        body = hospital_answer
    else:
        body = insurance_response

    # Add insurance agent specific details if claim was denied
    if claim_status == "DENIED_NO_PA" and insurance_response:
        appeal_note = (
            " I confirmed with the insurance company that you have 30 days to file an appeal, "
            "and your doctor can request a Peer-to-Peer clinical review. "
            "Your care coordinator will follow up with you within 24 hours."
        )
        body = body.rstrip() + appeal_note

    closing = " Is there anything else you'd like to know?"

    return intro + body + closing


async def _call_patient_back(callback_number: str, message: str) -> dict:
    """
    Make an outbound Twilio call to the patient with the callback message.
    Falls back to logging if Twilio credentials are not set.
    """
    if not TWILIO_SID or not TWILIO_TOKEN or not TWILIO_FROM:
        logger.info("[3-PHASE] Twilio not configured — callback message logged only")
        logger.info("[3-PHASE] Would call %s with: %s", callback_number, message[:200])
        return {"status": "logged_only", "message": message}

    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse, Say

        # Build TwiML for callback
        twiml = VoiceResponse()
        twiml.say(message, voice="Polly.Joanna")
        twiml.say(
            "Thank you for calling. Goodbye.",
            voice="Polly.Joanna"
        )

        # Make the call
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        call   = client.calls.create(
            twiml = str(twiml),
            to    = callback_number,
            from_ = TWILIO_FROM,
        )

        logger.info("[3-PHASE] Callback initiated: CallSid=%s to=%s", call.sid, callback_number)
        return {"status": "initiated", "call_sid": call.sid}

    except Exception as e:
        logger.error("[3-PHASE] Callback failed: %s", e)
        return {"status": "failed", "error": str(e)}
