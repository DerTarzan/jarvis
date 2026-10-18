"""Cloud-Provider: OpenAI-kompatibler HTTP-Adapter und vorbereitete Erweiterungspunkte."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from jarvis.ai.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    ProviderConfigError,
    ProviderError,
    ProviderNotImplementedError,
    ProviderStatus,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolSpec,
)
from jarvis.config import ProviderConfig

log = logging.getLogger(__name__)


class _PreparedCloudProvider(LLMProvider):
    is_cloud = True
    implemented = False

    def __init__(self, cfg: ProviderConfig) -> None:
        self._cfg = cfg

    @property
    def model(self) -> str:
        return self._cfg.model

    def _not_implemented(self) -> ProviderNotImplementedError:
        return ProviderNotImplementedError(
            f"Der Provider '{self.name}' ist nur als Erweiterungspunkt vorbereitet und noch "
            "nicht implementiert. Wähle 'ollama' oder implementiere den Adapter."
        )

    async def chat(self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None) -> LLMResponse:
        raise self._not_implemented()

    async def check(self) -> ProviderStatus:
        return ProviderStatus(False, str(self._not_implemented()))


class OpenAIProvider(_PreparedCloudProvider):
    name = "openai"


class AnthropicProvider(_PreparedCloudProvider):
    name = "anthropic"


class GeminiProvider(_PreparedCloudProvider):
    name = "gemini"


class OpenAICompatibleProvider(_PreparedCloudProvider):
    name = "openai_compatible"
    implemented = True
    supports_tools = True

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(cfg)
        self._headers = {"Authorization": f"Bearer {cfg.api_key()}"} if cfg.api_key() else {}
        self._client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/") + "/",
            headers=self._headers,
            timeout=httpx.Timeout(cfg.timeout_s, connect=10.0),
        )

    @staticmethod
    def _messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        result = []
        for message in messages:
            if message.role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id or f"call_{message.tool_name or 'tool'}",
                    }
                )
                continue
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id or f"call_{call.name}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in message.tool_calls
                ]
            result.append(item)
        return result

    def _payload(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": self._messages(messages),
            "stream": stream,
        }
        if self._cfg.options:
            payload.update(self._cfg.options)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        return payload

    @staticmethod
    def _response(response: httpx.Response) -> LLMResponse:
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    id=str(raw.get("id") or ""),
                )
            )
        return LLMResponse(message.get("content") or "", calls, data.get("model", ""))

    def _check_response(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            detail = str(response.json().get("error", response.text))
        except (ValueError, AttributeError):
            detail = response.text
        raise ProviderError(
            f"OpenAI-kompatible API antwortete mit HTTP {response.status_code}: {detail[:500]}"
        )

    @staticmethod
    def _transport_error(exc: httpx.HTTPError, timeout_s: float) -> Exception:
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(
                f"OpenAI-kompatible API antwortete nicht innerhalb von {timeout_s:.0f} s."
            )
        return ProviderUnavailableError(f"OpenAI-kompatible API ist nicht erreichbar: {exc}")

    async def chat(self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None) -> LLMResponse:
        try:
            response = await self._client.post(
                "chat/completions", json=self._payload(messages, tools, stream=False), headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, self._cfg.timeout_s) from exc
        self._check_response(response)
        try:
            return self._response(response)
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderConfigError(f"Ungültige Antwort von OpenAI-kompatibler API: {exc}") from exc

    async def stream(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, tools, stream=True)
        try:
            async with self._client.stream(
                "POST", "chat/completions", json=payload, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._check_response(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choice = (chunk.get("choices") or [{}])[0]
                    text = (choice.get("delta") or {}).get("content")
                    if isinstance(text, str) and text:
                        yield text
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, self._cfg.timeout_s) from exc
        except (ValueError, TypeError) as exc:
            raise ProviderConfigError(f"Ungültiger Streaming-Chunk der API: {exc}") from exc

    async def stream_response(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        on_chunk: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> LLMResponse:
        payload = self._payload(messages, tools, stream=True)
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        try:
            async with self._client.stream(
                "POST", "chat/completions", json=payload, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._check_response(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        content.append(text)
                        if on_chunk is not None:
                            result = on_chunk(text)
                            if result is not None:
                                await result
                    for raw in delta.get("tool_calls") or []:
                        index = int(raw.get("index", 0))
                        call = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        call["id"] += raw.get("id", "")
                        function = raw.get("function") or {}
                        call["name"] += function.get("name", "")
                        call["arguments"] += function.get("arguments", "")
        except httpx.HTTPError as exc:
            raise self._transport_error(exc, self._cfg.timeout_s) from exc
        except (ValueError, TypeError) as exc:
            raise ProviderConfigError(f"Ungültiger Streaming-Chunk der API: {exc}") from exc

        tool_calls = []
        for item in calls.values():
            try:
                args = json.loads(item["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise ProviderConfigError("Die API lieferte ungültige Tool-Argumente.") from exc
            tool_calls.append(
                ToolCall(item["name"], args if isinstance(args, dict) else {}, item["id"])
            )
        return LLMResponse("".join(content).strip(), tool_calls, self._cfg.model)

    async def check(self) -> ProviderStatus:
        try:
            response = await self._client.get("models", headers=self._headers)
        except httpx.HTTPError as exc:
            return ProviderStatus(False, str(self._transport_error(exc, self._cfg.timeout_s)))
        if response.status_code >= 400:
            return ProviderStatus(False, f"API-Status {response.status_code}: {response.text[:300]}")
        try:
            models = [item.get("id", "") for item in response.json().get("data", [])]
        except (ValueError, AttributeError, TypeError) as exc:
            return ProviderStatus(False, f"Ungültige Modellliste: {exc}")
        return ProviderStatus(True, f"OpenAI-kompatible API erreichbar, Modell '{self._cfg.model}'.", models)

    async def aclose(self) -> None:
        await self._client.aclose()
