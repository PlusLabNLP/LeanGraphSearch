"""Gemini Vertex client used by the paper model adapter."""


from __future__ import annotations

import asyncio

import copy

import json

import time

import uuid

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any

import httpx

@dataclass(frozen=True)
class NativeToolCall:
    call_id: str
    name: str
    arguments: str

    def as_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

@dataclass(frozen=True)
class ModelCompletion:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: tuple[NativeToolCall, ...] = ()
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    operation_retries: int = 0

def _gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the evaluator's OpenAI-style tools to Vertex Gemini."""

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool definition must contain a function object")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("tool function parameters must be a JSON schema object")
        declarations.append(
            {
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": parameters,
            }
        )
    return [{"functionDeclarations": declarations}]

def _gemini_tool_config(value: Any, tools: list[dict[str, Any]]) -> dict[str, Any]:
    names = [str(tool["function"]["name"]) for tool in tools]
    mode = "AUTO"
    allowed: list[str] | None = None
    if value in (None, "auto"):
        mode = "AUTO"
    elif value in ("required", "any"):
        mode = "ANY"
        allowed = names
    elif value == "none":
        mode = "NONE"
    elif isinstance(value, dict):
        choice_type = value.get("type")
        if choice_type == "function":
            function = value.get("function") or {}
            if not isinstance(function, dict) or not function.get("name"):
                raise ValueError("forced function tool choice requires a name")
            mode = "ANY"
            allowed = [str(function["name"])]
        elif choice_type in {"auto", "any", "none"}:
            mode = {"auto": "AUTO", "any": "ANY", "none": "NONE"}[choice_type]
            if mode == "ANY":
                allowed = names
        elif choice_type == "tool" and value.get("name"):
            mode = "ANY"
            allowed = [str(value["name"])]
        else:
            raise ValueError(f"unsupported tool choice for Gemini: {value!r}")
    else:
        raise ValueError(f"unsupported tool choice for Gemini: {value!r}")
    function_calling: dict[str, Any] = {"mode": mode}
    if allowed is not None:
        function_calling["allowedFunctionNames"] = allowed
    return {"functionCallingConfig": function_calling}

class GeminiVertexToolClient:
    """Gemini generateContent tool client authenticated by Google Cloud ADC.

    Gemini 3 requires the model's thought signatures to be replayed exactly
    after a function call.  The evaluator stores provider-neutral messages,
    so this client retains the original model parts behind host-unique call
    IDs and restores them when constructing the next request.
    """

    # Vertex can return 499/CANCELLED when the managed generation operation is
    # interrupted upstream even though the caller is still awaiting it. Treat
    # that provider-side cancellation like the other retryable operation
    # failures; the evaluator's independent hard deadline remains authoritative.
    _TRANSIENT_STATUSES = {408, 429, 499, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        project_id: str,
        region: str,
        model: str,
        timeout_s: float = 3600.0,
        operation_retries: int = 0,
        thinking_level: str = "HIGH",
        enable_google_search: bool = False,
        credentials_file: str | None = None,
        client: httpx.AsyncClient | None = None,
        credentials: Any | None = None,
    ) -> None:
        if type(operation_retries) is not int or operation_retries < 0:
            raise ValueError("operation_retries must be a non-negative integer")
        normalized_thinking = thinking_level.upper()
        if normalized_thinking not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("thinking_level must be LOW, MEDIUM, or HIGH")
        self.project_id = project_id
        self.region = region
        self.model = model
        self.timeout_s = float(timeout_s)
        self.operation_retries = operation_retries
        self.thinking_level = normalized_thinking
        if type(enable_google_search) is not bool:
            raise TypeError("enable_google_search must be a bool")
        self.enable_google_search = enable_google_search
        self._client = client
        self._owns_client = client is None
        if credentials is None:
            if credentials_file:
                from google.oauth2.credentials import Credentials

                credentials = Credentials.from_authorized_user_file(
                    Path(credentials_file),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            else:
                import google.auth

                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
        self._credentials = credentials
        self._auth_lock = asyncio.Lock()
        self._parts_by_call_id: dict[str, list[dict[str, Any]]] = {}
        self._name_by_call_id: dict[str, str] = {}
        self._provider_call_id_by_call_id: dict[str, str] = {}

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def _headers(self) -> dict[str, str]:
        async with self._auth_lock:
            if not getattr(self._credentials, "valid", False):
                from google.auth.transport.requests import Request

                await asyncio.to_thread(self._credentials.refresh, Request())
            token = str(getattr(self._credentials, "token", "") or "")
            if not token:
                raise RuntimeError("Google Cloud ADC did not provide an access token")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            quota_project = getattr(self._credentials, "quota_project_id", None)
            if quota_project:
                headers["x-goog-user-project"] = str(quota_project)
            return headers

    def _history(
        self, messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        def append(role: str, parts: list[dict[str, Any]]) -> None:
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(str(message.get("content") or ""))
                continue
            if role == "user":
                append("user", [{"text": str(message.get("content") or "")}])
                continue
            if role == "assistant":
                raw_calls = message.get("tool_calls") or []
                restored: list[dict[str, Any]] | None = None
                for raw_call in raw_calls:
                    call_id = str(raw_call.get("id") or "")
                    if call_id in self._parts_by_call_id:
                        restored = copy.deepcopy(self._parts_by_call_id[call_id])
                        break
                if restored is not None:
                    append("model", restored)
                    continue
                parts: list[dict[str, Any]] = []
                if message.get("content"):
                    parts.append({"text": str(message["content"])})
                for raw_call in raw_calls:
                    function = raw_call.get("function") or {}
                    raw_arguments = function.get("arguments", "{}")
                    try:
                        arguments = (
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else raw_arguments
                        )
                    except json.JSONDecodeError:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    parts.append(
                        {
                            "functionCall": {
                                "name": str(function.get("name") or ""),
                                "args": arguments,
                            }
                        }
                    )
                append("model", parts or [{"text": ""}])
                continue
            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                name = self._name_by_call_id.get(call_id, "")
                raw_content = message.get("content") or ""
                try:
                    response = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                except json.JSONDecodeError:
                    response = {"result": str(raw_content)}
                if not isinstance(response, dict):
                    response = {"result": response}
                function_response = {
                    "name": name,
                    "response": response,
                }
                provider_call_id = self._provider_call_id_by_call_id.get(call_id, "")
                if provider_call_id:
                    function_response["id"] = provider_call_id
                append("user", [{"functionResponse": function_response}])
                continue
            raise ValueError(f"unsupported conversation role: {role!r}")
        system = None
        if system_parts:
            system = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return system, contents

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sampling: dict[str, Any],
        seed: int,
        *,
        deadline: float | None = None,
    ) -> ModelCompletion:
        system, contents = self._history(messages)
        generation: dict[str, Any] = {
            "topP": float(sampling["top_p"]),
            "seed": int(seed),
            "thinkingConfig": {
                "thinkingLevel": str(
                    sampling.get("thinking_level") or self.thinking_level
                ).upper(),
                "includeThoughts": True,
            },
        }
        if sampling.get("temperature") is not None:
            generation["temperature"] = float(sampling["temperature"])
        max_tokens = sampling.get("max_tokens")
        if max_tokens is not None:
            generation["maxOutputTokens"] = int(max_tokens)
        if sampling.get("top_k") is not None:
            generation["topK"] = int(sampling["top_k"])
        gemini_tools = _gemini_tools(tools) if tools else []
        tool_config = (
            _gemini_tool_config(sampling.get("tool_choice", "auto"), tools)
            if tools
            else None
        )
        if self.enable_google_search:
            gemini_tools.insert(0, {"googleSearch": {}})
            if tool_config is None:
                tool_config = {}
            tool_config["includeServerSideToolInvocations"] = True
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation,
        }
        if gemini_tools:
            body["tools"] = gemini_tools
        if tool_config:
            body["toolConfig"] = tool_config
        if system is not None:
            body["systemInstruction"] = system
        endpoint = (
            "https://aiplatform.googleapis.com/v1/projects/"
            f"{self.project_id}/locations/{self.region}/publishers/google/models/"
            f"{self.model}:generateContent"
        )
        started = time.perf_counter()
        response: httpx.Response | None = None
        retries_used = 0
        for attempt in range(self.operation_retries + 1):
            request_timeout = self.timeout_s
            if deadline is not None:
                remaining_s = deadline - time.perf_counter()
                if remaining_s <= 0:
                    raise asyncio.TimeoutError
                request_timeout = min(request_timeout, remaining_s)
            try:
                response = await self._http_client().post(
                    endpoint,
                    json=body,
                    headers=await self._headers(),
                    timeout=request_timeout,
                )
                if response.status_code not in self._TRANSIENT_STATUSES:
                    response.raise_for_status()
                    break
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in self._TRANSIENT_STATUSES:
                    excerpt = exc.response.text[:4000]
                    raise httpx.HTTPStatusError(
                        f"{exc}; response body: {excerpt}",
                        request=exc.request,
                        response=exc.response,
                    ) from exc
                if attempt >= self.operation_retries:
                    raise
                retries_used += 1
                await asyncio.sleep(min(2**attempt, 4))
            except (httpx.TimeoutException, httpx.TransportError):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise asyncio.TimeoutError
                if attempt >= self.operation_retries:
                    raise
                retries_used += 1
                await asyncio.sleep(min(2**attempt, 4))
        assert response is not None
        payload = response.json()
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Gemini response has no candidates: {str(payload)[:2000]}")
        candidate = candidates[0]
        content = candidate.get("content") or {}
        raw_parts = content.get("parts") or []
        if not isinstance(raw_parts, list):
            raise ValueError("Gemini response content parts must be a list")
        calls: list[NativeToolCall] = []
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                continue
            if "text" in raw_part:
                if raw_part.get("thought"):
                    thinking_parts.append(str(raw_part.get("text") or ""))
                else:
                    text_parts.append(str(raw_part.get("text") or ""))
            function_call = raw_part.get("functionCall")
            if isinstance(function_call, dict):
                name = str(function_call.get("name") or "")
                arguments = function_call.get("args") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                provider_call_id = str(function_call.get("id") or "")
                call_id = provider_call_id or f"gemini_{uuid.uuid4().hex}"
                self._parts_by_call_id[call_id] = copy.deepcopy(raw_parts)
                self._name_by_call_id[call_id] = name
                self._provider_call_id_by_call_id[call_id] = provider_call_id
                calls.append(
                    NativeToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    )
                )
        usage_raw = payload.get("usageMetadata") or {}
        prompt_tokens = int(usage_raw.get("promptTokenCount", 0) or 0)
        candidate_tokens = int(usage_raw.get("candidatesTokenCount", 0) or 0)
        thinking_tokens = int(usage_raw.get("thoughtsTokenCount", 0) or 0)
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": candidate_tokens + thinking_tokens,
            "candidate_tokens": candidate_tokens,
            "thinking_tokens": thinking_tokens,
        }
        cached_tokens = int(usage_raw.get("cachedContentTokenCount", 0) or 0)
        if cached_tokens:
            usage["cache_read_input_tokens"] = cached_tokens
        web_search_queries: list[str] = []
        server_tool_calls: list[dict[str, Any]] = []
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                continue
            tool_call = raw_part.get("toolCall")
            if not isinstance(tool_call, dict):
                continue
            tool_type = str(tool_call.get("toolType") or "")
            arguments = tool_call.get("args") or {}
            queries = arguments.get("queries") if isinstance(arguments, dict) else None
            normalized_queries = (
                [str(query) for query in queries]
                if isinstance(queries, list)
                else []
            )
            server_tool_calls.append(
                {
                    "id": str(tool_call.get("id") or ""),
                    "tool_type": tool_type,
                    "queries": normalized_queries,
                }
            )
            if tool_type.startswith("GOOGLE_SEARCH"):
                web_search_queries.extend(normalized_queries)
        grounding_metadata = candidate.get("groundingMetadata")
        if not web_search_queries and isinstance(grounding_metadata, dict):
            grounded_queries = grounding_metadata.get("webSearchQueries")
            if isinstance(grounded_queries, list):
                web_search_queries.extend(str(query) for query in grounded_queries)
        usage["web_search_queries"] = len(web_search_queries)
        provider_metadata = {
            "google_search_enabled": self.enable_google_search,
            "google_search_queries": web_search_queries,
            "server_tool_calls": server_tool_calls,
        }
        if isinstance(grounding_metadata, dict):
            grounding_audit: dict[str, Any] = {
                "webSearchQueries": list(grounding_metadata.get("webSearchQueries") or [])
            }
            chunks = grounding_metadata.get("groundingChunks")
            if isinstance(chunks, list):
                grounding_audit["groundingChunks"] = [
                    {
                        key: value
                        for key, value in chunk.items()
                        if key in {"web", "retrievedContext"}
                    }
                    for chunk in chunks
                    if isinstance(chunk, dict)
                ]
            provider_metadata["grounding_metadata"] = grounding_audit
        return ModelCompletion(
            content="".join(text_parts),
            reasoning_content="".join(thinking_parts),
            tool_calls=tuple(calls),
            finish_reason=str(candidate.get("finishReason") or ""),
            usage=usage,
            provider_metadata=provider_metadata,
            elapsed_s=time.perf_counter() - started,
            operation_retries=retries_used,
        )
