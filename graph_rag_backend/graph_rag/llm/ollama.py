"""
Ollama Provider
===============

TEACHING NOTES
--------------
Ollama is a tool that runs LLMs locally on your machine.
You install it, pull a model (e.g. `ollama pull llama3`), and it exposes
a simple HTTP API at http://localhost:11434.

Why HTTP and not a Python library?
    Ollama's Python SDK exists but is thin — the HTTP API is stable and
    well-documented. Using `requests` directly makes it easy to debug
    (you can test with curl) and avoids an extra dependency.

Two Ollama endpoints we use:
    POST /api/generate   — simple prompt → completion (no history)
    POST /api/chat       — conversation with message history

Streaming:
    Ollama streams responses by default (token by token, like watching
    ChatGPT type). We set "stream": false to get the full response at once,
    which is simpler to handle.

Temperature = 0.0:
    For extraction tasks (classify this doc, extract these entities) we
    want deterministic output — same input → same output every time.
    Temperature 0.0 turns off randomness.
"""

import json
import logging

import requests

from .provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

# How long to wait for Ollama to respond (seconds)
# Generation can take 10-30s for large prompts on a laptop
REQUEST_TIMEOUT = 120


class OllamaProvider(LLMProvider):
    """
    Calls the local Ollama HTTP API.

    Prerequisites:
        1. Install Ollama: https://ollama.com
        2. Start it: `ollama serve`
        3. Pull a model: `ollama pull llama3`
        4. Verify: curl http://localhost:11434/api/tags

    No API key needed. Everything runs on your machine.
    Your document text NEVER leaves your computer.
    """

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        self._model = model
        self._host  = host.rstrip("/")
        logger.debug("OllamaProvider initialized: model=%s host=%s", model, host)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        """
        POST /api/generate — single prompt, single response.

        Request body:
            {
                "model": "llama3",
                "prompt": "What type of document is this? ...",
                "stream": false,
                "options": {"temperature": 0.0}
            }

        Response body (when stream=false):
            {
                "model": "llama3",
                "response": "BIRTH_CERTIFICATE",
                "done": true,
                ...
            }
        """
        url     = f"{self._host}/api/generate"
        payload = {
            "model":   self._model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self._host}. "
                f"Start it with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise LLMProviderError(
                f"Ollama request timed out after {REQUEST_TIMEOUT}s. "
                f"The model may be loading. Try again or use a smaller model."
            )
        except requests.exceptions.HTTPError as e:
            raise LLMProviderError(f"Ollama HTTP error: {e}. Response: {resp.text}")

        data = resp.json()

        # Ollama returns {"error": "..."} for model not found etc.
        if "error" in data:
            raise LLMProviderError(
                f"Ollama error: {data['error']}. "
                f"Make sure the model is pulled: ollama pull {self._model}"
            )

        return data.get("response", "").strip()

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """
        POST /api/chat — conversation with message history.

        messages format:
            [
                {"role": "system",    "content": "You extract entities..."},
                {"role": "user",      "content": "Extract from: ..."},
            ]

        Response body (when stream=false):
            {
                "message": {"role": "assistant", "content": "..."},
                "done": true
            }
        """
        url     = f"{self._host}/api/chat"
        payload = {
            "model":    self._model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self._host}. "
                f"Start it with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise LLMProviderError(
                f"Ollama chat request timed out after {REQUEST_TIMEOUT}s."
            )
        except requests.exceptions.HTTPError as e:
            raise LLMProviderError(f"Ollama HTTP error: {e}. Response: {resp.text}")

        data = resp.json()

        if "error" in data:
            raise LLMProviderError(f"Ollama error: {data['error']}")

        # Response is nested: {"message": {"role": "assistant", "content": "..."}}
        return data.get("message", {}).get("content", "").strip()

    def is_available(self) -> bool:
        """
        Quick health check — returns True if Ollama is running.
        Useful in tests and startup checks.
        """
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[str]:
        """
        Returns the names of all models currently pulled locally.
        e.g. ["llama3:latest", "mistral:latest"]
        """
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []
