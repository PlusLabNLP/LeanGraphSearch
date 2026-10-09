import asyncio
from types import SimpleNamespace

from leansearchv2.llm import LLMClient


def test_record_usage_supports_anthropic_fields() -> None:
    client = LLMClient.__new__(LLMClient)
    client._usage = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    client._record_usage(
        SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=20,
        )
    )

    assert client.usage_snapshot() == {
        "requests": 1,
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_creation_input_tokens": 10,
        "cache_read_input_tokens": 20,
    }


def test_record_usage_supports_openai_fields() -> None:
    client = LLMClient.__new__(LLMClient)
    client._usage = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    client._record_usage(SimpleNamespace(prompt_tokens=9, completion_tokens=4))

    assert client.usage_snapshot()["input_tokens"] == 9
    assert client.usage_snapshot()["output_tokens"] == 4


def test_record_usage_supports_gemini_mapping() -> None:
    client = LLMClient.__new__(LLMClient)
    client._usage = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    client._record_usage(
        {
            "input_tokens": 90,
            "output_tokens": 45,
            "cache_read_input_tokens": 12,
            "thinking_tokens": 30,
        }
    )

    assert client.usage_snapshot() == {
        "requests": 1,
        "input_tokens": 90,
        "output_tokens": 45,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 12,
    }


def test_gemini_vertex_chat_forwards_high_thinking_without_tools() -> None:
    captured: dict = {}

    class _GeminiClient:
        async def complete(self, messages, tools, sampling, seed):
            captured.update(
                messages=messages,
                tools=tools,
                sampling=sampling,
                seed=seed,
            )
            return SimpleNamespace(
                content="answer",
                usage={"input_tokens": 7, "output_tokens": 11},
            )

    client = LLMClient.__new__(LLMClient)
    client.provider = "gemini_vertex"
    client.profile = "gemini31pro"
    client.model = "gemini-3.1-pro-preview-customtools"
    client._temperature_default = 0.0
    client._temperature_configured = False
    client._top_p = 1.0
    client._max_tokens_default = 65536
    client._thinking_level = "HIGH"
    client._seed = 20260806
    client._client = _GeminiClient()
    client._usage = {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    result = asyncio.run(client.chat([{"role": "user", "content": "test"}]))

    assert result == "answer"
    assert captured["tools"] == []
    assert captured["sampling"]["thinking_level"] == "HIGH"
    assert captured["sampling"]["temperature"] is None
    assert captured["sampling"]["tool_choice"] == "none"
    assert captured["sampling"]["max_tokens"] == 65536
    assert captured["seed"] == 20260806
    assert client.usage_snapshot()["output_tokens"] == 11


def test_llm_client_aclose_supports_async_provider_clients() -> None:
    closed = False

    class _Client:
        async def aclose(self):
            nonlocal closed
            closed = True

    client = LLMClient.__new__(LLMClient)
    client._client = _Client()

    asyncio.run(client.aclose())

    assert closed


def test_sonnet5_vertex_omits_deprecated_temperature() -> None:
    captured: dict = {}

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[], usage=None)

    client = LLMClient.__new__(LLMClient)
    client.provider = "anthropic_vertex"
    client.profile = "sonnet5"
    client.model = "claude-sonnet-5"
    client._temperature_default = 0.0
    client._top_p = None
    client._max_tokens_default = 1024
    client._client = SimpleNamespace(messages=_Messages())

    asyncio.run(client.chat([{"role": "user", "content": "test"}], temperature=0.0))

    assert captured["model"] == "claude-sonnet-5"
    assert "temperature" not in captured


def test_sonnet5_vertex_forwards_thinking_and_effort_configuration() -> None:
    captured: dict = {}

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[], usage=None)

    client = LLMClient.__new__(LLMClient)
    client.provider = "anthropic_vertex"
    client.profile = "sonnet5"
    client.model = "claude-sonnet-5"
    client._temperature_default = 0.0
    client._top_p = None
    client._max_tokens_default = 1024
    client._thinking = {"type": "adaptive", "display": "omitted"}
    client._output_config = {"effort": "high"}
    client._client = SimpleNamespace(messages=_Messages())

    asyncio.run(client.chat([{"role": "user", "content": "test"}]))

    assert captured["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert captured["output_config"] == {"effort": "high"}


def test_sonnet45_vertex_keeps_temperature() -> None:
    captured: dict = {}

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[], usage=None)

    client = LLMClient.__new__(LLMClient)
    client.provider = "anthropic_vertex"
    client.profile = "sonnet45"
    client.model = "claude-sonnet-4-5@20250929"
    client._temperature_default = 0.0
    client._top_p = None
    client._max_tokens_default = 1024
    client._client = SimpleNamespace(messages=_Messages())

    asyncio.run(client.chat([{"role": "user", "content": "test"}], temperature=0.0))

    assert captured["temperature"] == 0.0
