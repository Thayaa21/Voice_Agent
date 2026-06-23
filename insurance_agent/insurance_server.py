"""
Dummy Insurance Agent — Port 8002
===================================
Simulates a real insurance company's phone system for demo purposes.

This is NOT a real insurance company. It provides scripted realistic responses
based on patient data from the knowledge graph.

In production, the hospital agent would dial actual insurance company numbers.
For the demo, it dials this service instead.

Endpoints:
  POST /inquiry          → Main endpoint: takes patient info, returns scripted response
  POST /ivr              → Simulates IVR navigation (returns menu + hold)
  GET  /health           → Health check
  GET  /scripts          → List available response scripts

The response simulates what a real insurance agent would say after:
  1. Caller navigates IVR ("Press 1 for claims")
  2. Waits on hold 10-15 seconds
  3. Insurance agent answers and provides claim details
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRAPH_URL         = os.getenv("GRAPH_RAG_URL", "http://localhost:8000")
ELEVENLABS_KEY    = os.getenv("ELEVENLABS_API_KEY", "")
INSURANCE_VOICE   = os.getenv("ELEVENLABS_INSURANCE_VOICE_ID", "bfGb7JTLUnZebZRiFYyq")

app = FastAPI(title="Dummy Insurance Agent", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Scripted response library
# ---------------------------------------------------------------------------

IVR_GREETING = """
Thank you for calling Pacific Shield Insurance.
For claims and billing, press or say 1.
For prior authorization, press or say 2.
For eligibility and benefits, press or say 3.
To speak with an agent, press or say 0.
""".strip()

HOLD_MESSAGE = """
Thank you. Please hold while I transfer you to the next available agent.
Your estimated wait time is less than 2 minutes.
""".strip()

AGENT_GREETING = """
Thank you for holding. This is Pacific Shield Insurance, claims department.
My name is Sarah. How can I assist you today?
""".strip()

# Scripted responses keyed by claim status
SCRIPTS = {
    "DENIED_NO_PA": {
        "summary": "Claim denied — prior authorization not obtained",
        "response": (
            "I've pulled up that claim in our system. "
            "The claim was denied under denial code CO-4. "
            "The service required prior authorization before the procedure was performed, "
            "and our records show that no prior authorization request was submitted. "
            "The patient has 30 days from the denial date to file a formal appeal. "
            "Additionally, the attending physician may request a Peer-to-Peer clinical review "
            "by calling our medical management line at 1-800-555-0199. "
            "For reference, the contracted allowed amount for this procedure under the patient's plan "
            "is {allowed_amount}. "
            "The patient's remaining deductible for this calendar year is {deductible_remaining}. "
            "Is there anything else I can help you with regarding this claim?"
        ),
    },
    "PENDING_P2P": {
        "summary": "Claim pending Peer-to-Peer review",
        "response": (
            "I can see that claim in our system. "
            "The claim is currently pending a Peer-to-Peer clinical review. "
            "The attending physician's office requested the review and it has been scheduled. "
            "The review must be completed within 5 business days. "
            "Once the review is completed, we will reprocess the claim within 3 business days. "
            "If the review is successful, the claim will be approved at the full contracted amount of {allowed_amount}. "
            "The patient's cost-sharing would then apply based on their {deductible_remaining} remaining deductible "
            "and their {coinsurance} coinsurance. "
            "Is there anything else I can help you with?"
        ),
    },
    "PARTIAL": {
        "summary": "Claim partially paid — cost-sharing applied",
        "response": (
            "I have that claim here. The claim has been processed and partially paid. "
            "The contracted allowed amount was {allowed_amount}. "
            "We applied {deductible_applied} to the patient's remaining deductible first. "
            "After the deductible, we paid {insurance_paid} at the {coinsurance} rate. "
            "The patient's remaining responsibility is {patient_owes}. "
            "This is consistent with the terms of the patient's health plan. "
            "If the patient believes there is an error, they have 180 days to file a billing dispute. "
            "Is there anything else I can assist you with?"
        ),
    },
    "PAID": {
        "summary": "Claim fully paid",
        "response": (
            "I can see that claim. It has been fully processed and paid. "
            "The contracted allowed amount was {allowed_amount} and we paid {insurance_paid} "
            "to the provider via electronic funds transfer. "
            "The patient's cost-sharing of {patient_owes} was applied, which reflects their "
            "deductible and coinsurance obligations under their plan. "
            "The remittance advice was sent to the provider on the payment date. "
            "Is there anything else I can help you with?"
        ),
    },
    "default": {
        "summary": "General inquiry response",
        "response": (
            "I've looked up that patient in our system. "
            "I can see their policy is active with Pacific Shield Insurance. "
            "Their current plan includes a {deductible_remaining} remaining deductible "
            "and {coinsurance} coinsurance after the deductible is met. "
            "For specific claim details, could you provide the claim reference number? "
            "Is there anything specific you'd like me to look up?"
        ),
    },
}

# ---------------------------------------------------------------------------
# Graph lookup
# ---------------------------------------------------------------------------

async def get_patient_data(patient_name: str) -> dict:
    """Look up patient from graph for dynamic script filling."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{GRAPH_URL}/hospital/claim-status",
                json={"patient_name": patient_name}
            )
            claim_data = r.json()

            r2 = await client.post(
                f"{GRAPH_URL}/hospital/cost-estimate",
                json={"patient_name": patient_name}
            )
            cost_data = r2.json()

        return {"claim": claim_data, "cost": cost_data}
    except Exception as e:
        logger.error("Graph lookup failed: %s", e)
        return {}


def derive_script_vars(patient_data: dict) -> dict:
    """Extract variables to fill into script templates."""
    return {
        "allowed_amount":      "$8,500.00",
        "deductible_remaining": "$1,200.00",
        "deductible_applied":  "$800.00",
        "coinsurance":         "80/20",
        "insurance_paid":      "$1,920.00",
        "patient_owes":        "$480.00",
    }

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IVRRequest(BaseModel):
    selection: str = "1"  # which menu option was pressed


class InquiryRequest(BaseModel):
    patient_name:    str
    policy_number:   Optional[str] = None
    claim_number:    Optional[str] = None
    inquiry_type:    str = "claim_status"  # claim_status | eligibility | prior_auth
    simulate_hold:   bool = True
    hold_seconds:    int  = 3   # shortened for demo speed


class InquiryResponse(BaseModel):
    phase:           str   # "ivr" | "hold" | "agent"
    ivr_message:     str
    hold_message:    str
    agent_greeting:  str
    agent_response:  str
    claim_status:    str
    summary:         str
    hold_duration:   int

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ivr")
async def ivr_endpoint(req: IVRRequest):
    """Simulate IVR menu navigation."""
    menu_responses = {
        "1": "Claims and billing. Please hold while I transfer you.",
        "2": "Prior authorization. Please hold while I transfer you.",
        "3": "Eligibility and benefits. Please hold while I transfer you.",
        "0": "Agent. Please hold while I transfer you.",
    }
    response = menu_responses.get(req.selection, "Invalid selection. Please try again.")
    return {
        "selection":    req.selection,
        "ivr_response": response,
        "next_step":    "hold",
    }


@app.post("/inquiry", response_model=InquiryResponse)
async def inquiry_endpoint(req: InquiryRequest):
    """
    Main endpoint — simulates the full insurance call:
    IVR → hold → agent response.

    The hospital agent calls this after navigating the IVR.
    Returns a scripted response based on the patient's actual claim status.
    """
    logger.info("Insurance inquiry for: %s (type=%s)", req.patient_name, req.inquiry_type)

    # Simulate hold wait
    if req.simulate_hold and req.hold_seconds > 0:
        await asyncio.sleep(req.hold_seconds)

    # Get patient data from graph
    patient_data = await get_patient_data(req.patient_name)
    script_vars  = derive_script_vars(patient_data)

    # Determine claim status from graph response
    claim_response = patient_data.get("claim", {})
    answer_text    = claim_response.get("answer", "")

    # Map claim status from the answer
    if "DENIED_NO_PA" in answer_text or "prior authorization was not obtained" in answer_text:
        status = "DENIED_NO_PA"
    elif "PENDING_P2P" in answer_text or "Peer-to-Peer" in answer_text:
        status = "PENDING_P2P"
    elif "partially covered" in answer_text or "PARTIAL" in answer_text:
        status = "PARTIAL"
    elif "fully processed" in answer_text or "PAID" in answer_text:
        status = "PAID"
    else:
        status = "default"

    script       = SCRIPTS.get(status, SCRIPTS["default"])
    agent_response = script["response"].format(**script_vars)
    summary      = script["summary"]

    logger.info("Insurance response for %s: status=%s", req.patient_name, status)

    return InquiryResponse(
        phase          = "agent",
        ivr_message    = IVR_GREETING,
        hold_message   = HOLD_MESSAGE,
        agent_greeting = AGENT_GREETING,
        agent_response = agent_response,
        claim_status   = status,
        summary        = summary,
        hold_duration  = req.hold_seconds,
    )


@app.get("/scripts")
def list_scripts():
    """List available response scripts."""
    return {
        k: {"summary": v["summary"]}
        for k, v in SCRIPTS.items()
    }


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "service": "dummy-insurance-agent",
        "port":    8002,
        "scripts": list(SCRIPTS.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("insurance_server:app", host="0.0.0.0", port=8002, reload=True)
