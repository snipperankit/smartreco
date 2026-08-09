"""Single choke point for every LLM / embedding call.

The hackathon mandates that all AI traffic go through Mesh (an OpenAI-compatible
gateway). Nothing else in the codebase talks to a model directly, so this file
is the one place to swap models or add retries/telemetry.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

from openai import OpenAI, RateLimitError

from app.config import settings

_log = logging.getLogger("smartreco.mesh")


@lru_cache
def _client() -> OpenAI:
    return OpenAI(
        base_url=settings.mesh_base_url,
        api_key=settings.mesh_api_key,
        max_retries=0,  # we handle model fallback ourselves
    )


class AllModelsExhaustedError(Exception):
    """Raised when every available model is rate-limited."""


def chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 700) -> str:
    """Primary model (tencent/hy3 — reasoning). Falls back to fast model on 429."""
    budget = max_tokens
    try:
        for _ in range(2):
            resp = _client().chat.completions.create(
                model=settings.mesh_chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=budget,
            )
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if content or choice.finish_reason != "length":
                return content
            budget = max(budget * 4, 2000)
        return content
    except RateLimitError:
        _log.warning("Primary model rate-limited, falling back to %s", settings.mesh_fast_model)
        try:
            return _call_model(settings.mesh_fast_model, messages, temperature, max_tokens)
        except RateLimitError:
            raise AllModelsExhaustedError("All free models rate-limited for today")


def chat_fast(messages: list[dict], temperature: float = 0.4, max_tokens: int = 300) -> str:
    """Fast utility model (minimax/m2-her). Falls back to primary on 429."""
    try:
        return _call_model(settings.mesh_fast_model, messages, temperature, max_tokens)
    except RateLimitError:
        _log.warning("Fast model rate-limited, falling back to %s", settings.mesh_chat_model)
        try:
            return _call_model(settings.mesh_chat_model, messages, temperature, max_tokens)
        except RateLimitError:
            raise AllModelsExhaustedError("All free models rate-limited for today")


def _call_model(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    resp = _client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_stream(messages: list[dict], temperature: float = 0.7, max_tokens: int = 700):
    """Streaming chat completion.

    Yields text chunks as they arrive. Cuts perceived latency from
    ~1500ms (wait for full response) to ~200ms (first token). At 100k
    users this doesn't change server load but dramatically improves
    UX and is the single biggest latency win available.
    """
    stream = _client().chat.completions.create(
        model=settings.mesh_chat_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# --- Embeddings -------------------------------------------------------------
# Cache identical strings within a process run so re-embedding the same product
# description (e.g. edit that didn't change text) is free.
_embed_cache: dict[str, list[float]] = {}


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed(text: str) -> list[float]:
    k = _key(text)
    if k in _embed_cache:
        return _embed_cache[k]
    resp = _client().embeddings.create(model=settings.mesh_embed_model, input=text)
    vec = resp.data[0].embedding
    _embed_cache[k] = vec
    return vec


def embed_many(texts: list[str]) -> list[list[float]]:
    """Batched embedding call — one HTTP round trip for the whole seed catalog."""
    if not texts:
        return []
    resp = _client().embeddings.create(model=settings.mesh_embed_model, input=texts)
    # OpenAI SDK preserves input order in resp.data.
    ordered = sorted(resp.data, key=lambda d: d.index)
    vecs = [d.embedding for d in ordered]
    for t, v in zip(texts, vecs):
        _embed_cache[_key(t)] = v
    return vecs
