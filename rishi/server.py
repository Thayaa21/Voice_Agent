"""
Voice Infrastructure Server — Port 8001
=========================================
Handles inbound Twilio calls, STT via Deepgram, TTS via ElevenLabs.

Endpoints:
  POST /incoming-call  → TwiML webhook for Twilio (call received)
  POST /process-speech → TwiML webhook (speech captured by Twilio Gather)
  POST /test           → Test endpoint (no phone needed)
  GET  /health         → Health check

Flow:
  1. Patient calls Twilio number
  2. Twilio POSTs to /incoming-call
  3. Server returns TwiML with <Gather> to capture speech
  4. Patient speaks, Twilio transcribes and POSTs to /process-speech
  5. Server forwards to Agent (port 9001)
  6. Agent returns response text
  7. Server returns TwiML with <Say> (ElevenLabs audio or Twilio built-in)
  8. Loop until end_call=true
"""

import os
import logging
import httpx
from pathlib import Path
from fastapi import FastAPI, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AGENT_URL        = os.getenv("AGENT_URL", "http://localhost:9001")
ELEVENLABS_KEY   = os.getenv("ELEVENLABS_API_KEY", "")
HOSPITAL_VOICE   = os.getenv("ELEVENLABS_HOSPITAL_VOICE_ID", "nf4MCGNSdM0hxM95ZBQR")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")

app = FastAPI(title="Voice Infrastructure", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------

async def tts_elevenlabs(text: str) -> bytes | None:
    """Generate audio from ElevenLabs. Returns MP3 bytes or None on failure."""
    if not ELEVENLABS_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{HOSPITAL_VOICE}",
                headers={
                    "xi-api-key": ELEVENLABS_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                }
            )
            if r.status_code == 200:
                return r.content
    except Exception as e:
        logger.error("ElevenLabs TTS failed: %s", e)
    return None


def twiml_say(text: str, loop: bool = True) -> str:
    """Build TwiML response using Twilio's built-in <Say>."""
    gather_action = "/process-speech"
    if loop:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="{gather_action}" timeout="5" speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">{_escape(text)}</Say>
    </Gather>
    <Say voice="Polly.Joanna">I didn't catch that. Could you please repeat?</Say>
    <Redirect>/incoming-call</Redirect>
</Response>"""
    else:
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{_escape(text)}</Say>
    <Hangup/>
</Response>"""


def _escape(text: str) -> str:
    """Escape XML special characters for TwiML."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))

# ---------------------------------------------------------------------------
# Agent call
# ---------------------------------------------------------------------------

async def call_agent(text: str, call_id: str) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.post(
                f"{AGENT_URL}/agent",
                json={"text": text, "call_id": call_id}
            )
            return r.json()
        except Exception as e:
            logger.error("Agent call failed: %s", e)
            return {"response": "I'm experiencing technical difficulties. Please hold.", "end_call": False}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/incoming-call")
async def incoming_call():
    """Twilio webhook — called when patient dials the hospital number."""
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/process-speech" timeout="5" speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">
            Hello, thank you for calling. I'm the hospital's automated assistant.
            To get started, please say your full name and date of birth.
        </Say>
    </Gather>
    <Say voice="Polly.Joanna">I didn't hear anything. Please call back and try again.</Say>
    <Hangup/>
</Response>"""
    return FastAPIResponse(content=twiml, media_type="application/xml")


@app.post("/process-speech")
async def process_speech(
    SpeechResult: str = Form(default=""),
    CallSid:      str = Form(default="unknown"),
    Confidence:   str = Form(default="0"),
):
    """Twilio webhook — receives transcribed speech, forwards to agent."""
    logger.info("CallSid=%s Speech='%s' Confidence=%s", CallSid, SpeechResult, Confidence)

    if not SpeechResult.strip():
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/process-speech" timeout="5" speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">I didn't catch that. Could you please repeat?</Say>
    </Gather>
</Response>"""
        return FastAPIResponse(content=twiml, media_type="application/xml")

    # Call agent
    result   = await call_agent(SpeechResult, CallSid)
    response = result.get("response", "I'm sorry, something went wrong.")
    end_call = result.get("end_call", False)

    logger.info("Agent response: '%s' end_call=%s", response[:100], end_call)

    twiml = twiml_say(response, loop=not end_call)
    return FastAPIResponse(content=twiml, media_type="application/xml")


class TestRequest(BaseModel):
    text:    str
    call_id: str = "test_call_001"


@app.post("/test")
async def test_endpoint(req: TestRequest):
    """Test the full pipeline without a real phone call."""
    result = await call_agent(req.text, req.call_id)
    return {
        "input":    req.text,
        "response": result.get("response"),
        "end_call": result.get("end_call", False),
    }


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "service":        "voice-infrastructure",
        "port":           8001,
        "agent_url":      AGENT_URL,
        "elevenlabs_tts": bool(ELEVENLABS_KEY),
        "twilio_sid":     TWILIO_ACCOUNT_SID[:8] + "..." if TWILIO_ACCOUNT_SID else "not set",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
