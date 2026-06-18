"""
OpenAI Provider
===============

TEACHING NOTES
--------------
OpenAI's API is the "cloud" option. Instead of running a model locally,
you send your prompt to OpenAI's servers and pay per token.

Tokens:
    LLMs don't process words — they process "tokens" (roughly 3/4 of a word).
    "Hello world" ≈ 2 tokens. A full birth certificate ≈ 200–400 tokens.
    OpenAI charges ~$0.15 per 1M input tokens for gpt-4o-mini.
    For our use case (a few dozen documents) the cost is negligible.

Why gpt-4o-mini as default?
    It's cheap, fast, and very capable for structured extraction tasks.
    gpt-4o is more powerful but costs ~10x more. For document classification
    and entity extraction, mini is more than sufficient.

API Key security:
    NEVER put the API key in source code.
    We read it from the OPENAI_API_KEY environment variable only.
    The .env file (in .gitignore) stores it locally.
    On GitHub, you'd use a GitHub Secret / environment variable.

openai Python SDK:
    OpenAI provides an official Python package. We use it here because
    it handles retries, rate limiting, and error parsing for us.
    Install: pip install openai>=1.0
"""

import logging
import os
from typing import Optional

from .provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Calls the OpenAI API using the official openai Python SDK.

    Prerequisites:
        1. pip install openai
        2. Set OPENAI_API_KEY in your .env file
        3. Optionally set OPENAI_MODEL (default: gpt-4o-mini)

    Pricing reference (as of 2024):
        gpt-4o-mini  — $0.15/1M input tokens  (recommended for this project)
        gpt-4o       — $2.50/1M input tokens
        gpt-3.5-turbo— $0.50/1M input tokens
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self._model = model

        # Lazy import — only error if someone actually tries to use OpenAI
        try:
            import openai as _openai
        except ImportError:
            raise LLMProviderError(
                "The 'openai' package is not installed. "
                "Run: pip install openai>=1.0"
            )

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMProviderError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )

        # Create a client instance (SDK >= 1.0 style)
        self._client = _openai.OpenAI(api_key=key)
        logger.debug("OpenAIProvider initialized: model=%s", model)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Wraps the prompt in a single user message and calls chat completions.

        OpenAI doesn't have a separate "complete" endpoint anymore (the old
        Completions API is deprecated). Instead we wrap single prompts as
        chat messages — the result is the same.
        """
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """
        POST to /v1/chat/completions via the openai SDK.

        messages format (same as Ollama):
            [
                {"role": "system",    "content": "You are a document classifier."},
                {"role": "user",      "content": "What type is this document?"},
            ]
        """
        try:
            import openai

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )

            # SDK returns a structured object — extract the text
            return response.choices[0].message.content.strip()

        except Exception as e:
            # openai.AuthenticationError, RateLimitError, APIConnectionError etc.
            error_type = type(e).__name__
            raise LLMProviderError(
                f"OpenAI API error ({error_type}): {e}. "
                f"Check your OPENAI_API_KEY and network connection."
            ) from e
