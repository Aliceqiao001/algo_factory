"""Unified LLM client: OpenAI-compatible wrapper supporting SiliconFlow and other providers.

Configure via environment variables:
    ANTHROPIC_API_KEY  — API key (reused for any provider)
    LLM_BASE_URL       — provider base URL (default: SiliconFlow)
    LLM_MODEL          — model name      (default: Qwen2.5-72B-Instruct)
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL    = "Qwen/Qwen2.5-72B-Instruct"


def create_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at the configured provider."""
    api_key  = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", _DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    client: OpenAI,
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> str:
    """Send a system + user message and return the assistant's text reply.

    Parameters
    ----------
    client:   OpenAI client instance from :func:`create_client`.
    system:   System prompt (role instructions).
    user:     User message (the actual task).
    max_tokens: Maximum tokens in the response.
    """
    model = os.getenv("LLM_MODEL", _DEFAULT_MODEL)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content
