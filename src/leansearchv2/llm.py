"""Async LLM client for OpenAI-compatible and Anthropic Vertex endpoints.

Used by reasoning mode (decompose / filter / judge) and the prove task
(prover + query generator). Each named profile under `llm.<name>` in
config.yaml supplies its own base_url, api key, model, and optional
sampling defaults; consumers pick a profile by name. Mapping roles to
profiles lives in `reasoning.*_llm` and `prove.*_llm`.
"""

from __future__ import annotations

import inspect
import os
from contextvars import ContextVar, Token
from typing import Any, Callable

from openai import AsyncOpenAI

from .config import get


def _profile_config(profile: str) -> dict[str, Any]:
    cfg = get(f"LLM_{profile.upper()}", "llm", profile, default=None)
    if not isinstance(cfg, dict):
        raise RuntimeError(
            f"llm profile {profile!r} not defined in config (expected `llm.{profile}` mapping)"
        )
    return cfg


class LLMClient:
    def __init__(
        self,
        profile: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        cfg = _profile_config(profile)
        self.profile = profile
        self.provider = str(cfg.get("provider", "openai")).lower()
        base_url = base_url or cfg.get("base_url")
        if self.provider == "openai" and api_key is None:
            if cfg.get("api_key") not in (None, ""):
                api_key = str(cfg["api_key"])
            else:
                key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
                api_key = os.environ.get(str(key_env))
                if api_key is None:
                    raise RuntimeError(
                        f"LLM api key not set for profile {profile!r}: env var {key_env} is missing"
                    )
        self.model = model or cfg.get("model")
        self.timeout = float(timeout if timeout is not None else cfg.get("timeout_s", 120))
        max_retries = int(cfg.get("max_retries", 3))
        self._temperature_default = cfg.get("temperature")
        self._temperature_configured = (
            "temperature" in cfg and cfg.get("temperature") is not None
        )
        self._top_p = cfg.get("top_p")
        self._max_tokens_default = cfg.get("max_tokens")
        self._thinking = cfg.get("thinking")
        self._output_config = cfg.get("output_config")
        self._thinking_level = str(cfg.get("thinking_level", "HIGH")).upper()
        self._seed = int(cfg.get("seed", 20260806))
        self._usage = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        self._usage_observer: ContextVar[Callable[[dict[str, int]], None] | None] = (
            ContextVar(f"llm_usage_observer_{id(self)}", default=None)
        )
        if self.provider == "openai":
            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=self.timeout,
                max_retries=max_retries,
            )
        elif self.provider in ("anthropic_vertex", "vertex_anthropic"):
            try:
                import google.auth
                from anthropic import AsyncAnthropicVertex
            except ImportError as e:
                raise RuntimeError(
                    "Anthropic Vertex support is not installed. Run "
                    "`pip install 'anthropic[vertex]'`."
                ) from e
            project_id = cfg.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                credentials, detected_project = google.auth.default()
                project_id = detected_project or getattr(credentials, "quota_project_id", None)
            if not project_id:
                raise RuntimeError(
                    f"Google Cloud project not set for LLM profile {profile!r}; "
                    "set llm.<profile>.project_id or GOOGLE_CLOUD_PROJECT"
                )
            region = str(cfg.get("region", "global"))
            self._client = AsyncAnthropicVertex(
                project_id=str(project_id),
                region=region,
                timeout=self.timeout,
                max_retries=max_retries,
            )
        elif self.provider in ("gemini_vertex", "vertex_gemini"):
            from .gemini_client import GeminiVertexToolClient

            project_id = cfg.get("project_id") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise RuntimeError(
                    f"Google Cloud project not set for LLM profile {profile!r}; "
                    "set llm.<profile>.project_id or GOOGLE_CLOUD_PROJECT"
                )
            if self._thinking_level not in {"LOW", "MEDIUM", "HIGH"}:
                raise RuntimeError(
                    f"llm.{profile}.thinking_level must be LOW, MEDIUM, or HIGH"
                )
            enable_google_search = cfg.get("enable_google_search", False)
            if type(enable_google_search) is not bool:
                raise RuntimeError(
                    f"llm.{profile}.enable_google_search must be a bool"
                )
            self._client = GeminiVertexToolClient(
                project_id=str(project_id),
                region=str(cfg.get("region", "global")),
                model=str(self.model),
                timeout_s=self.timeout,
                operation_retries=max_retries,
                thinking_level=self._thinking_level,
                enable_google_search=enable_google_search,
                credentials_file=cfg.get("credentials_file"),
            )
        else:
            raise RuntimeError(
                f"unsupported LLM provider {self.provider!r} for profile {profile!r}"
            )

    def usage_snapshot(self) -> dict[str, int]:
        """Return cumulative, response-reported token usage for this client."""
        return dict(self._usage)

    def set_usage_observer(
        self, observer: Callable[[dict[str, int]], None]
    ) -> Token[Callable[[dict[str, int]], None] | None]:
        """Attach a task-local observer for exact per-response usage."""

        return self._usage_observer.set(observer)

    def reset_usage_observer(
        self, token: Token[Callable[[dict[str, int]], None] | None]
    ) -> None:
        self._usage_observer.reset(token)

    async def aclose(self) -> None:
        """Close the provider client when it owns reusable network resources."""
        closer = getattr(self._client, "aclose", None)
        if closer is None:
            closer = getattr(self._client, "close", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        recorded = {
            "requests": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        fields = {
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "output_tokens": ("output_tokens", "completion_tokens"),
            "cache_creation_input_tokens": ("cache_creation_input_tokens",),
            "cache_read_input_tokens": ("cache_read_input_tokens",),
        }
        for target, candidates in fields.items():
            value = 0
            for field in candidates:
                found = (
                    usage.get(field)
                    if isinstance(usage, dict)
                    else getattr(usage, field, None)
                )
                if found is not None:
                    value = int(found)
                    break
            recorded[target] = value
        for key, value in recorded.items():
            self._usage[key] += value
        observer_var = getattr(self, "_usage_observer", None)
        observer = observer_var.get() if observer_var is not None else None
        if observer is not None:
            observer(dict(recorded))

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if temperature is None:
            temperature = self._temperature_default if self._temperature_default is not None else 0.0
        if max_tokens is None:
            max_tokens = self._max_tokens_default
        if self.provider == "openai":
            kwargs: dict[str, Any] = {}
            if self._top_p is not None:
                kwargs["top_p"] = self._top_p
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            resp = await self._client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
            self._record_usage(getattr(resp, "usage", None))
            return resp.choices[0].message.content or ""

        if self.provider in ("gemini_vertex", "vertex_gemini"):
            if max_tokens is None:
                raise RuntimeError(
                    f"llm.{self.profile}.max_tokens is required for Gemini Vertex"
                )
            completion = await self._client.complete(
                messages,
                tools=[],
                sampling={
                    # The legacy premise prompts request temperature=0, but
                    # Gemini evaluations should retain the model default unless
                    # the Gemini profile explicitly configures a temperature.
                    "temperature": (
                        float(temperature) if self._temperature_configured else None
                    ),
                    "top_p": float(self._top_p if self._top_p is not None else 1.0),
                    "max_tokens": int(max_tokens),
                    "thinking_level": self._thinking_level,
                    "tool_choice": "none",
                },
                seed=self._seed,
            )
            self._record_usage(completion.usage)
            return completion.content

        if max_tokens is None:
            raise RuntimeError(
                f"llm.{self.profile}.max_tokens is required for Anthropic Vertex"
            )
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(str(message.get("content", "")))
            else:
                anthropic_messages.append(message)
        selected_model = str(model or self.model)
        vertex_kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        # Claude Sonnet 5 rejects the legacy sampling controls instead of
        # silently ignoring them.  The prove pipeline historically passes
        # temperature=0 explicitly, so omit it at the provider boundary for
        # this model and use Sonnet 5's server-side default behavior.
        sonnet5 = selected_model == "claude-sonnet-5" or selected_model.startswith(
            "claude-sonnet-5@"
        )
        if not sonnet5:
            vertex_kwargs["temperature"] = temperature
        if system_parts:
            vertex_kwargs["system"] = "\n\n".join(system_parts)
        thinking = getattr(self, "_thinking", None)
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise RuntimeError(
                    f"llm.{self.profile}.thinking must be a mapping when provided"
                )
            vertex_kwargs["thinking"] = dict(thinking)
        output_config = getattr(self, "_output_config", None)
        if output_config is not None:
            if not isinstance(output_config, dict):
                raise RuntimeError(
                    f"llm.{self.profile}.output_config must be a mapping when provided"
                )
            vertex_kwargs["output_config"] = dict(output_config)
        # Claude 4.5 accepts temperature or top_p, but not both. The prove
        # pipeline explicitly fixes temperature=0, so temperature wins here.
        if not sonnet5 and temperature is None and self._top_p is not None:
            vertex_kwargs["top_p"] = self._top_p
        resp = await self._client.messages.create(**vertex_kwargs)
        self._record_usage(getattr(resp, "usage", None))
        return "".join(
            str(getattr(block, "text", ""))
            for block in getattr(resp, "content", [])
            if getattr(block, "type", None) == "text"
        )
