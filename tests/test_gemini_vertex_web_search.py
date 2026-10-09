from __future__ import annotations

import json
import unittest

import httpx

from leansearchv2.gemini_client import GeminiVertexToolClient


class _Credentials:
    valid = True
    token = "test-token"
    quota_project_id = None


class _Client:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    async def post(self, url, *, json, headers, timeout):
        self.requests.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return httpx.Response(200, json=self.payload, request=httpx.Request("POST", url))


class GeminiVertexWebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_reasoning_request_omits_all_tools_and_web_search(self):
        payload = {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [
                    {"thought": True, "text": "internal reasoning"},
                    {"text": "final answer"},
                ]},
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 5,
            },
        }
        transport = _Client(payload)
        model = GeminiVertexToolClient(
            project_id="project",
            region="global",
            model="gemini-3.1-pro-preview-customtools",
            thinking_level="HIGH",
            enable_google_search=False,
            client=transport,
            credentials=_Credentials(),
        )

        result = await model.complete(
            [{"role": "user", "content": "plan a proof"}],
            [],
            {
                "temperature": None,
                "top_p": 1.0,
                "max_tokens": 65536,
                "thinking_level": "HIGH",
                "tool_choice": "none",
            },
            20260806,
        )

        request = transport.requests[0]["json"]
        self.assertNotIn("tools", request)
        self.assertNotIn("toolConfig", request)
        self.assertNotIn("temperature", request["generationConfig"])
        self.assertEqual(
            request["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "HIGH",
        )
        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "internal reasoning")
        self.assertEqual(result.usage["output_tokens"], 7)
        self.assertEqual(result.usage["web_search_queries"], 0)
        self.assertFalse(result.provider_metadata["google_search_enabled"])

    async def test_combines_google_search_with_functions_and_records_queries(self):
        payload = {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [
                    {
                        "thoughtSignature": "encrypted-search",
                        "toolCall": {
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "id": "search-1",
                            "args": {"queries": ["Mathlib Nat gcd documentation"]},
                        },
                    },
                    {
                        "thoughtSignature": "encrypted-response",
                        "toolResponse": {
                            "toolType": "GOOGLE_SEARCH_WEB",
                            "id": "search-1",
                            "response": {"search_suggestions": "..."},
                        },
                    },
                    {
                        "thoughtSignature": "encrypted-function",
                        "functionCall": {
                            "name": "compile_lean",
                            "id": "function-1",
                            "args": {"proof": "by simp"},
                        },
                    },
                ]},
                "groundingMetadata": {"webSearchQueries": ["Mathlib Nat gcd documentation"]},
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 7,
            },
        }
        transport = _Client(payload)
        model = GeminiVertexToolClient(
            project_id="project",
            region="global",
            model="gemini-3.1-pro-preview-customtools",
            enable_google_search=True,
            client=transport,
            credentials=_Credentials(),
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "compile_lean",
                "description": "Compile Lean",
                "parameters": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                },
            },
        }]
        result = await model.complete(
            [{"role": "user", "content": "prove it"}],
            tools,
            {"temperature": 0.0, "top_p": 1.0, "tool_choice": "required"},
            1,
        )

        request = transport.requests[0]["json"]
        self.assertEqual(request["tools"][0], {"googleSearch": {}})
        self.assertIn("functionDeclarations", request["tools"][1])
        self.assertTrue(request["toolConfig"]["includeServerSideToolInvocations"])
        self.assertEqual(result.tool_calls[0].call_id, "function-1")
        self.assertEqual(result.usage["web_search_queries"], 1)
        self.assertEqual(
            result.provider_metadata["google_search_queries"],
            ["Mathlib Nat gcd documentation"],
        )

        model._history([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [result.tool_calls[0].as_openai_dict()],
            },
            {
                "role": "tool",
                "tool_call_id": "function-1",
                "content": json.dumps({"success": True}),
            },
        ])
        _, history = model._history([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [result.tool_calls[0].as_openai_dict()],
            },
            {
                "role": "tool",
                "tool_call_id": "function-1",
                "content": json.dumps({"success": True}),
            },
        ])
        response = history[-1]["parts"][0]["functionResponse"]
        self.assertEqual(response["id"], "function-1")
        self.assertEqual(history[0]["parts"], payload["candidates"][0]["content"]["parts"])


if __name__ == "__main__":
    unittest.main()
