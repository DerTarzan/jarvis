"""Ollama-Adapter (lokal). Kapselt die Ollama-spezifische REST-API."""

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
    ModelNotFoundError,
    ProviderError,
    ProviderStatus,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolSpec,
    ToolsUnsupportedError,
)
from jarvis.config import ProviderConfig

log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"
    is_cloud = False

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None) -> None:
        self._cfg = cfg
        self.supports_tools = True
        self._client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_s, connect=5.0),
        )

    @property
    def model(self) -> str:
        return self._cfg.model

    # -- Konvertierung -------------------------------------------------
    @staticmethod
    def _convert_message(msg: ChatMessage) -> dict[str, Any]:
        out: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in msg.tool_calls
            ]
        if msg.role == "tool" and msg.tool_name:
            out["tool_name"] = msg.tool_name
        return out

    def _payload(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [self._convert_message(m) for m in messages],
            "stream": stream,
        }
        if self._cfg.options:
            payload["options"] = self._cfg.options
        if tools and self.supports_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        return payload

    # -- Fehlerübersetzung ---------------------------------------------
    def _raise_for_response(self, response: httpx.Response, had_tools: bool) -> None:
        if response.status_code < 400:
            return
        try:
            detail = str(response.json().get("error", response.text))
        except ValueError:
            detail = response.text
        if response.status_code == 404 or "not found" in detail.lower():
            raise ModelNotFoundError(
                f"Modell '{self._cfg.model}' ist in Ollama nicht installiert. "
                f"Installiere es mit: ollama pull {self._cfg.model}"
            )
        if had_tools and "does not support tools" in detail.lower():
            raise ToolsUnsupportedError(detail)
        raise ProviderError(f"Ollama-Fehler {response.status_code}: {detail}")

    def _translate_transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
            return ProviderUnavailableError(
                f"Ollama ist unter {self._cfg.base_url} nicht erreichbar. "
                "Starte die Ollama-App bzw. 'ollama serve'."
            )
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(
                f"Ollama hat nicht innerhalb von {self._cfg.timeout_s:.0f} s geantwortet "
                "(Modell lädt evtl. noch). Erhöhe ai.providers.ollama.timeout_s."
            )
        return ProviderError(f"Netzwerkfehler bei Ollama: {exc}")

    # -- Schnittstelle ----------------------------------------------------
    async def chat(self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None) -> LLMResponse:
        try:
            return await self._chat_once(messages, tools)
        except ToolsUnsupportedError:
            log.warning("Modell %s unterstützt keine Tools – weiter ohne Tool-Calling", self._cfg.model)
            self.supports_tools = False
            return await self._chat_once(messages, None)

    async def _chat_once(self, messages: list[ChatMessage], tools: list[ToolSpec] | None) -> LLMResponse:
        payload = self._payload(messages, tools, stream=False)
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise self._translate_transport_error(exc) from exc
        self._raise_for_response(response, had_tools="tools" in payload)
        data = response.json()
        message = data.get("message", {})
        calls = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            fn = raw.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=str(i)))
        return LLMResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=calls,
            model=data.get("model", self._cfg.model),
        )

    async def stream(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, None, stream=True)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_response(response, had_tools=False)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise self._translate_transport_error(exc) from exc

    async def stream_response(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        on_chunk: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> LLMResponse:
        payload = self._payload(messages, tools, stream=True)
        content: list[str] = []
        calls: list[ToolCall] = []
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_response(response, had_tools="tools" in payload)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    text = message.get("content", "")
                    if text:
                        content.append(text)
                        if on_chunk is not None:
                            result = on_chunk(text)
                            if result is not None:
                                await result
                    for index, raw in enumerate(message.get("tool_calls") or []):
                        function = raw.get("function", {})
                        args = function.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except ValueError:
                                args = {}
                        calls.append(
                            ToolCall(
                                name=function.get("name", ""),
                                arguments=args if isinstance(args, dict) else {},
                                id=str(raw.get("id") or index),
                            )
                        )
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise self._translate_transport_error(exc) from exc
        return LLMResponse(
            content="".join(content).strip(),
            tool_calls=calls,
            model=self._cfg.model,
        )

    async def check(self) -> ProviderStatus:
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return ProviderStatus(False, str(self._translate_transport_error(exc)))
        names = [m.get("name", "") for m in response.json().get("models", [])]
        wanted = self._cfg.model
        candidates = {wanted, f"{wanted}:latest"} if ":" not in wanted else {wanted}
        if candidates & set(names):
            return ProviderStatus(True, f"Ollama bereit, Modell '{wanted}' vorhanden.", names)
        return ProviderStatus(
            False,
            f"Ollama läuft, aber Modell '{wanted}' fehlt. Installiere es: ollama pull {wanted}",
            names,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
