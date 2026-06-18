# Rishi — Voice Infrastructure

## Your Role
You handle everything phone-related. When someone calls, you receive it, convert speech to text, pass it to Chetan's agent, and convert the reply back to speech.

## Accounts to Create (do this first)
1. **Twilio** — twilio.com → get a free phone number
2. **Deepgram** — deepgram.com → get API key (STT)
3. **ElevenLabs** — elevenlabs.io → get API key (TTS, optional — Twilio's built-in voice works too)

## Stack
- Python FastAPI (webhook server, port 8001)
- `twilio` SDK — receive calls, return TwiML
- `deepgram-sdk` — speech to text (or use Twilio's built-in)
- `elevenlabs` — text to speech (optional upgrade)
- `ngrok` — expose local server to Twilio during dev

## What to Build

### 1. FastAPI server with two endpoints

**POST `/incoming-call`**
- Twilio calls this when someone dials your number
- Return TwiML that says: "Hello, please say your name and date of birth"
- Use `<Gather input="speech" action="/process-speech">` to capture what they say

**POST `/process-speech`**
- Twilio sends the transcribed speech as `SpeechResult` form field
- Forward it to Chetan's agent: `POST http://localhost:9001/agent` with `{"text": SpeechResult, "call_id": CallSid}`
- Read the `response` field from Chetan's reply
- Return TwiML that speaks the response back using `<Say>` or ElevenLabs audio
- If `end_call: true` in Chetan's reply → add `<Hangup/>`

### 2. Test endpoint (no phone needed)
**POST `/test`** — directly call Chetan's agent and return what he says. Use this to test without a real phone call.

### 3. Ngrok tunnel for Twilio
Run `ngrok http 8001` → copy the HTTPS URL → paste into Twilio console as the webhook for your phone number.

## Interface with Chetan
You call his endpoint:
```
POST http://localhost:9001/agent
{"text": "what caller said", "call_id": "unique-id"}
```
He returns:
```json
{"response": "agent speaks this", "end_call": false}
```
That's all you need to know about his side.

## .env keys you need
```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
ELEVENLABS_API_KEY=   (optional)
AGENT_URL=http://localhost:9001
```

## Definition of Done
- Someone calls the Twilio number
- They say "Mei Lee, January 31 1982"
- They hear a response about Mei Lee's records
- They can ask a follow-up question and hear another answer
