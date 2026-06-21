# Rishi — Voice Infrastructure

## Your Role
You own the entire real-time voice pipeline. This is the most latency-sensitive part of the system — every millisecond matters. A caller should never wait more than 2 seconds for a response.

---

## Architecture You Need to Build

```
Caller (phone)
    │
    ▼ Twilio Media Stream (WebSocket, raw PCM audio)
    │
    ▼ Deepgram Nova-2 (real-time STT via WebSocket)
    │ sub-100ms transcription
    ▼
FastAPI server (port 8001)
    │
    ▼ POST /agent on Chetan's service (port 9001)
    │
    ▼ ElevenLabs TTS (streaming audio generation)
    │
    ▼ Back to Twilio → Caller hears response
```

The key difference from a basic implementation: **WebSocket streaming throughout**, not request/response. You never buffer a full audio clip before processing.

---

## Accounts to Set Up

1. **Twilio** — [twilio.com](https://twilio.com)
   - Free trial account
   - Get a US phone number
   - Enable Media Streams on your number

2. **Deepgram** — [deepgram.com](https://deepgram.com)
   - Free account ($200 credit)
   - Use **Nova-2** model — best accuracy for names and medical terms
   - Use the WebSocket streaming API, not the REST API

3. **ElevenLabs** — [elevenlabs.io](https://elevenlabs.io)
   - Free account
   - Pick a voice that sounds professional and calm (e.g. "Rachel" or "Bella")
   - Use streaming TTS endpoint for low latency

---

## What to Build

### 1. Twilio WebSocket Media Stream handler

When a call comes in, Twilio can stream raw PCM audio to your WebSocket endpoint in real time. This is much lower latency than the `<Gather>` approach.

- `POST /incoming-call` → return TwiML that connects to your WebSocket
- `WS /media-stream` → WebSocket endpoint that receives raw audio from Twilio

Reference: [Twilio Media Streams docs](https://www.twilio.com/docs/voice/media-streams)

### 2. Deepgram real-time transcription

Pipe the raw PCM audio from Twilio directly into Deepgram's WebSocket STT. When Deepgram returns a `is_final: true` transcript, that's your trigger to send text to Chetan.

Key settings to use:
- `model=nova-2`
- `language=en-US`
- `punctuate=true`
- `endpointing=300` (ms of silence before finalizing)

Reference: [Deepgram streaming docs](https://developers.deepgram.com/docs/getting-started-with-live-streaming-audio)

### 3. Call Chetan's agent

When you have a final transcript:
```python
POST http://localhost:9001/agent
{"text": "transcript text", "call_id": "twilio-call-sid"}
```

He returns:
```json
{"response": "text to speak", "end_call": false}
```

### 4. ElevenLabs streaming TTS

Convert Chetan's text response to audio using ElevenLabs streaming endpoint. Stream the audio chunks back to Twilio as they arrive — don't wait for the full audio to generate.

Reference: [ElevenLabs streaming TTS](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-as-stream)

### 5. Graceful error handling

Handle these cases cleanly:
- Chetan's service is down → say "I'm having technical difficulties, please hold"
- STT returns empty or noise → don't forward to Chetan, prompt caller to repeat
- Call drops mid-stream → clean up WebSocket connections, release session

### 6. Test endpoint (no phone needed)

```
POST /test
{"text": "Aiden Garcia March 15 1992"}
```
Calls Chetan directly and returns the spoken response text. Use this for development.

### 7. Latency logging

Log timestamps at each stage so you can measure:
- Time from speech end to STT final transcript
- Time from STT to Chetan response
- Time from Chetan response to first TTS audio byte
- Total round-trip time

Target: under 2 seconds total.

---

## Environment Variables

```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=    # get from ElevenLabs dashboard
AGENT_URL=http://localhost:9001
```

---

## Stack

```
fastapi
uvicorn
twilio
deepgram-sdk
elevenlabs
websockets
httpx
python-dotenv
```

---

## Definition of Done

- [ ] Real phone call received by your server via Twilio Media Stream WebSocket
- [ ] Audio streamed to Deepgram in real time, transcript returned under 500ms
- [ ] Transcript sent to Chetan, response received
- [ ] ElevenLabs TTS audio streamed back to caller
- [ ] Full round-trip under 2 seconds
- [ ] Latency logged at each stage
- [ ] `/test` endpoint works without a real phone call
- [ ] Handles dropped calls and service errors gracefully

---

## Interface Contract with Chetan

You send:
```json
POST http://localhost:9001/agent
{"text": "caller's words", "call_id": "CA-twilio-sid"}
```

You receive:
```json
{"response": "text to speak back", "end_call": false}
```

If `end_call` is `true`, speak the response then hang up.
