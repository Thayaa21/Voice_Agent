"""
LLM Provider — Abstract Base Class + Factory
=============================================

TEACHING NOTES
--------------
Abstract Base Class (ABC):
    An ABC is a class you can't instantiate directly. It defines a contract —
    a set of methods that every subclass MUST implement.

    Think of it like a job description. "LLMProvider" says:
    "Anyone who calls themselves an LLM provider MUST have complete() and chat()."
    Ollama and OpenAI each implement those methods their own way.

    If a subclass forgets to implement a required method, Python raises
    TypeError at import time — not at runtime when it's too late.

Factory function:
    create_llm_provider() reads the LLM_PROVIDER environment variable and
    returns the right concrete class. The rest of the pipeline just calls
    create_llm_provider() once and never thinks about which LLM is active.

Why environment variables?
    You never hardcode API keys or config in source code — they'd be
    committed to git and exposed. Environment variables stay on your machine.
    .env files (loaded by python-dotenv) make this convenient locally.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMProviderError(Exception):
    """
    Raised when the LLM provider is unreachable or misconfigured.

    The message always includes:
    - Which provider failed (ollama / openai)
    - What the user should do to fix it
    """
    pass


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Every component that needs an LLM receives an LLMProvider.
    It doesn't know (or care) whether it's Ollama or OpenAI underneath.

    Two methods to implement:
        complete(prompt)         — single-turn: send a prompt, get a string back
        chat(messages)           — multi-turn: send a conversation history, get a reply

    Why two methods?
        complete() is simpler — just a string in, string out.
        chat() is for when you need to maintain context (system message + history).
        The DocumentClassifier uses complete(). The QueryEngine uses chat().
    """

    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send a single prompt and get a text response.

        Args:
            prompt      — the full prompt string
            temperature — 0.0 = deterministic/focused, 1.0 = creative/random
                          We default to 0.0 because extraction needs to be
                          consistent, not creative.

        Returns:
            The model's response as a plain string.

        Raises:
            LLMProviderError — if the provider is unreachable or returns an error
        """
        ...

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """
        Send a list of messages and get a reply.

        messages format (OpenAI standard, also used by Ollama):
            [
                {"role": "system",    "content": "You are a document classifier."},
                {"role": "user",      "content": "What type of document is this?"},
                {"role": "assistant", "content": "BIRTH_CERTIFICATE"},
                {"role": "user",      "content": "Extract all entities..."},
            ]

        Returns:
            The model's reply as a plain string.

        Raises:
            LLMProviderError — if the provider is unreachable or returns an error
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        The identifier of the model being used.
        e.g. "llama3", "mistral", "gpt-4o", "gpt-3.5-turbo"
        Stored on every extracted Entity as `extractor_model`.
        """
        ...


def create_llm_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMProvider:
    """
    Factory function — reads config and returns the right LLM provider.

    Priority order for provider selection:
        1. Explicit `provider` argument (for testing / programmatic use)
        2. LLM_PROVIDER environment variable
        3. Default: "ollama"

    Priority order for model selection:
        1. Explicit `model` argument
        2. OLLAMA_MODEL or OPENAI_MODEL environment variable
        3. Provider-specific default

    Usage:
        # Default (uses env vars or falls back to Ollama)
        llm = create_llm_provider()

        # Explicit
        llm = create_llm_provider(provider="openai", model="gpt-4o")

        # In tests — pass a mock
        llm = MockLLMProvider()

    Raises:
        LLMProviderError — if LLM_PROVIDER is set to an unknown value
        LLMProviderError — if openai is selected but OPENAI_API_KEY is missing
    """
    # Lazy imports so that missing packages only error when actually used
    from .ollama import OllamaProvider
    from .openai_provider import OpenAIProvider

    selected = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()

    if selected == "ollama":
        ollama_model = model or os.getenv("OLLAMA_MODEL", "llama3")
        ollama_host  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(model=ollama_model, host=ollama_host)

    elif selected == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Add it to your .env file: OPENAI_API_KEY=sk-..."
            )
        openai_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIProvider(model=openai_model, api_key=api_key)

    else:
        raise LLMProviderError(
            f"Unknown LLM_PROVIDER='{selected}'. "
            f"Valid values are: 'ollama', 'openai'."
        )
