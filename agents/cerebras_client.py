"""Gemma 4 (gemma-4-31b) on Cerebras — the single model powering every agent.

Design goals:
  * One client, used by all 5 agents, so the whole system is "Gemma 4 on Cerebras."
  * Text + vision (image_url / base64 data URIs) in one call path.
  * Strict structured outputs (json_schema) so agents hand typed data to each other.
  * reasoning_effort passthrough (off by default per Cerebras docs).
  * Captures `time_info` + usage on every call -> this is the SPEED story for the demo.
  * MOCK MODE: develop the full pipeline before hackathon API access opens (10:30am PT).
    Auto-enabled when CEREBRAS_API_KEY is absent, or forced with GEMMA_MOCK=1.

Swap-to-real is a no-op: fill CEREBRAS_API_KEY in .env and mock turns itself off.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    # Load code/.env explicitly so the key is found even when the Webots controller
    # runs from its own working directory (controllers/drone_agent/).
    _ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_ENV)
except Exception:
    pass

# The hackathon model. Override with GEMMA_MODEL while gemma-4-31b preview is not yet
# enabled (e.g. GEMMA_MODEL=gpt-oss-120b to exercise the live text pipeline early).
MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-31b")


@dataclass
class GemmaResult:
    """Everything an agent needs back, plus the timing we put on screen."""
    content: str
    parsed: Optional[dict]          # populated when a schema was requested
    latency_s: float                # wall-clock round trip (client-side)
    time_info: dict = field(default_factory=dict)   # Cerebras server timing
    usage: dict = field(default_factory=dict)
    provider: str = "cerebras"
    model: str = MODEL


def image_content(path_or_url: str) -> dict:
    """Build an OpenAI-style image content block for Gemma 4 vision.

    Accepts a local file path (-> base64 data URI) or an http(s) URL (passed through).
    """
    if path_or_url.startswith(("http://", "https://", "data:")):
        url = path_or_url
    elif os.path.exists(path_or_url):
        mime = mimetypes.guess_type(path_or_url)[0] or "image/jpeg"
        with open(path_or_url, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        url = f"data:{mime};base64,{b64}"
    else:
        # Missing local file: fine during mock dev (live camera frames don't exist yet).
        # A real run with a real frame path will always hit the branch above.
        url = _PLACEHOLDER_PNG
    return {"type": "image_url", "image_url": {"url": url}}


# 1x1 transparent PNG, used only as a stand-in when a local image path is absent.
_PLACEHOLDER_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


class GemmaClient:
    def __init__(self, mock: Optional[bool] = None):
        self.api_key = os.environ.get("CEREBRAS_API_KEY")
        forced = os.environ.get("GEMMA_MOCK", "").strip() in ("1", "true", "True")
        self.mock = forced if mock is None else mock
        if mock is None and not forced:
            self.mock = not self.api_key

        self._client = None
        if not self.mock:
            # Lazy import so the package isn't required during mock-mode dev.
            from cerebras.cloud.sdk import Cerebras
            self._client = Cerebras(api_key=self.api_key)

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict],
        *,
        schema: Optional[dict] = None,
        reasoning_effort: str = "none",
        tools: Optional[list] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        mock_response: Optional[dict | str] = None,
    ) -> GemmaResult:
        """One call path for every agent.

        `schema` -> Cerebras strict json_schema response_format; result.parsed is the dict.
        `mock_response` -> what to return in mock mode (dict for schema calls, str otherwise).
        """
        if self.mock:
            return self._mock(schema, mock_response)

        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs["tools"] = tools
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}

        t0 = time.perf_counter()
        resp = self._call_with_retry(kwargs)
        latency = time.perf_counter() - t0

        msg = resp.choices[0].message
        content = msg.content or ""
        parsed = None
        if schema:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None

        time_info = _to_dict(getattr(resp, "time_info", None))
        usage = _to_dict(getattr(resp, "usage", None))
        return GemmaResult(content, parsed, latency, time_info, usage)

    def _call_with_retry(self, kwargs, attempts: int = 6):
        """Same Gemma call, retried with backoff on TRANSIENT failures (DNS/connection
        drops, timeouts, rate limits, 5xx). This bridges the network-not-up-yet race at
        sim launch — a single blip used to crash the whole drone controller. Non-transient
        errors (bad request, auth) re-raise immediately. Still 100% gemma-4-31b; no mock fallback."""
        delay = 1.0
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                name = type(e).__name__
                transient = any(k in name for k in (
                    "Connection", "Timeout", "RateLimit", "InternalServer", "APIStatus"))
                if not transient or i == attempts - 1:
                    raise
                print(f"[gemma] {name} on Cerebras call (attempt {i+1}/{attempts}) — "
                      f"retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 20.0)

    # ------------------------------------------------------------------ #
    def _mock(self, schema, mock_response) -> GemmaResult:
        # Simulate Cerebras-class latency so the dev pipeline feels real.
        time.sleep(0.15)
        if schema and mock_response is None:
            mock_response = {}  # caller should supply realistic mocks per agent
        if isinstance(mock_response, dict):
            content = json.dumps(mock_response)
            parsed = mock_response
        else:
            content = mock_response or "[mock] Gemma 4 response"
            parsed = None
        return GemmaResult(
            content=content,
            parsed=parsed,
            latency_s=0.15,
            time_info={"queue_time": 0.0, "inference_time": 0.15, "mock": True},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            provider="mock",
        )


def _to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}


# Shared singleton most agents import.
gemma = GemmaClient()
