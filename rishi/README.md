# Rishi — Voice Infrastructure

## Your Job in One Sentence
Get a phone call to forward speech to an endpoint and speak back the reply.

That's it. No agent logic. No database. No graph. Just: voice in → text to Chetan → text back → voice out.

---

## Step 1 — Create Accounts (do this first, ~15 min)

1. **Twilio** — [twilio.com](https://twilio.com)
   - Sign up for free
   - Go to Console → Phone Numbers → Get a Number (pick any US number)
   - Note down: Account SID, Auth Token, Phone Number

2. **Deepgram** — [deepgram.com](https://deepgram.com)
   - Sign up for free ($200 free credit)
   - Create a project → API Keys → Create a Key
   - Note down: API Key

3. **ElevenLabs** — [elevenlabs.io](https://elevenlabs.io) *(optional — Twilio's built-in voice works fine for now)*
   - Sign up for free
   - Profile → API Keys
   - Note down: API Key

---

## Step 2 — Set Up Your Environment

```bash
cd rishi
pip install -r requirements.txt

# Create your .env file
cp ../.env.example .env
```

Fill in `.env`:
```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
DEEPGRAM_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # optional
AGENT_URL=http://localhost:9001
```

---

## Step 3 — Build Your Server

Create a file called `server.py` in the `rishi/` folder.

You need **two endpoints**:

### `POST /incoming-call`
Twilio calls this when someone dials your number.
Return TwiML XML that greets the caller and listens for speech.

```python
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.post("/incoming-call")
async def incoming_call():
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Gather input="speech" action="/process-speech" timeout="5" speechTimeout="auto">
            <Say>Hello, thank you for calling. Please say your name and date of birth.</Say>
        </Gather>
    </Response>"""
    return Response(content=twiml, media_type="application/xml")
```

### `POST /process-speech`
Twilio sends you what the caller said. Forward it to Chetan, speak the reply.

```python
import httpx
from fastapi import Form

@app.post("/process-speech")
async def process_speech(SpeechResult: str = Form(""), CallSid: str = Form("")):
    # Forward to Chetan's agent
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:9001/agent",
            json={"text": SpeechResult, "call_id": CallSid},
            timeout=10.0
        )
    data = resp.json()
    response_text = data.get("response", "I'm sorry, I didn't understand that.")
    end_call = data.get("end_call", False)

    # Speak the reply back
    say_tag = f"<Say>{response_text}</Say>"
    if end_call:
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>{say_tag}<Hangup/></Response>"""
    else:
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Gather input="speech" action="/process-speech" timeout="5" speechTimeout="auto">
                {say_tag}
            </Gather>
        </Response>"""
    return Response(content=twiml, media_type="application/xml")
```

### `POST /test` *(optional but useful)*
Test without a real phone call.

```python
@app.post("/test")
async def test(body: dict):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:9001/agent",
            json={"text": body.get("text", ""), "call_id": "test_001"},
            timeout=10.0
        )
    return resp.json()
```

---

## Step 4 — Run Your Server

```bash
uvicorn server:app --port 8001 --reload
```

---

## Step 5 — Expose It to Twilio with ngrok

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8001
```

Copy the HTTPS URL it gives you (looks like `https://abc123.ngrok.io`).

Go to [Twilio Console](https://console.twilio.com) → Phone Numbers → Manage → your number → Voice webhook:
- Set **"A call comes in"** to: `https://abc123.ngrok.io/incoming-call`
- Method: HTTP POST
- Save

---

## Step 6 — Test It

**Without a phone (test endpoint):**
```bash
curl -X POST http://localhost:8001/test \
  -H "Content-Type: application/json" \
  -d '{"text": "Aiden Garcia, March 15 1992"}'
```

**With a real phone:**
- Call your Twilio number
- Say "Aiden Garcia, March 15 1992"
- You should hear a response about that patient

---

## Definition of Done

- [ ] Twilio account set up, phone number obtained
- [ ] Deepgram account set up
- [ ] `server.py` running on port 8001
- [ ] ngrok tunnel active, URL set in Twilio console
- [ ] `/test` endpoint returns a real response from Chetan
- [ ] Real phone call goes through end to end

---

## How It Connects to Chetan

You call his endpoint:
```
POST http://localhost:9001/agent
{"text": "what caller said", "call_id": "twilio-call-sid"}
```

He gives you back:
```json
{"response": "text to speak", "end_call": false}
```

That's all you need to know about his side. Don't touch his code.
