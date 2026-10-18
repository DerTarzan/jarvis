"""Provider-unabhängige Schnittstelle für Sprachmodelle.

Der Rest der Anwendung kennt nur diese Typen – nie anbieterspezifische Details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

Role = Literal["system", "user", "assistant", "tool"]


class ProviderError(Exception):
    """Basisklasse; Meldung ist für Endnutzer verständlich formuliert."""


class ProviderConfigError(ProviderError):
    """Ungültige oder unvollständige Provider-Konfiguration (z. B. fehlender API-Schlüssel)."""


class ProviderNotImplementedError(ProviderError):
    """Provider ist nur als Erweiterungspunkt vorbereitet."""


class ProviderUnavailableError(ProviderError):
    """Server/Dienst nicht erreichbar."""


class ProviderTimeoutError(ProviderError):
    """Zeitüberschreitung bei der Anfrage."""


class ModelNotFoundError(ProviderError):
    """Modell ist beim Provider nicht installiert/verfügbar."""


class ToolsUnsupportedError(ProviderError):
    """Das Modell unterstützt kein natives Tool-Calling."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """Beschreibung eines Tools für das Modell (JSON-Schema für die Parameter)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ChatMessage:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None  # nur bei role == "tool"
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [{"name": c.name, "arguments": c.arguments, "id": c.id} for c in self.tool_calls],
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=[
                ToolCall(c["name"], c.get("arguments", {}), c.get("id", ""))
                for c in data.get("tool_calls", [])
            ],
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""


@dataclass
class ProviderStatus:
    ok: bool
    message: str
    models: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Gemeinsame Schnittstelle aller Sprachmodell-Anbieter."""

    name: ClassVar[str]
    is_cloud: ClassVar[bool] = False
    #: False bei Platzhaltern, die nur als Erweiterungspunkt existieren.
    implemented: ClassVar[bool] = True
    supports_tools: bool = False

    @property
    @abstractmethod
    def model(self) -> str:
        """Konfigurierter Modellname."""

    @abstractmethod
    async def chat(self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None) -> LLMResponse:
        """Erzeugt eine vollständige Antwort. Wirft ProviderError-Unterklassen."""

    async def stream(
        self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[str]:
        """Liefert Textteile. Standard: ganze Antwort als ein Teil."""
        response = await self.chat(messages, tools)
        yield response.content

    async def stream_response(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        on_chunk: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> LLMResponse:
        """Erzeugt eine vollständige Antwort und gibt Textteile währenddessen weiter."""
        response = await self.chat(messages, tools)
        if response.content and on_chunk is not None:
            result = on_chunk(response.content)
            if result is not None:
                await result
        return response

    @abstractmethod
    async def check(self) -> ProviderStatus:
        """Prüft Erreichbarkeit und Modellverfügbarkeit, ohne Nutzdaten zu senden."""

    async def aclose(self) -> None:
        """Gibt Verbindungen frei."""
