"""Zentrale, provider-unabhängige Tool-Registry."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from jarvis.ai.base import ToolSpec

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


class ToolRegistryError(Exception):
    """Ungültige Registrierung (Name doppelt, Schema ungültig, ...)."""


@dataclass(frozen=True)
class Tool:
    """Ein vom Modell aufrufbares Werkzeug.

    ``handler`` wird mit den validierten Argumenten als Keyword-Argumente aufgerufen und
    darf synchron oder ``async`` sein. Rückgabe: JSON-serialisierbar oder Text.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )
    requires_confirmation: bool = False
    timeout_s: float | None = None
    source: str = "core"

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not _NAME_RE.match(tool.name):
            raise ToolRegistryError(
                f"Ungültiger Tool-Name '{tool.name}' (erlaubt: Buchstaben, Ziffern, _; max. 64)."
            )
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool.name}' ist bereits registriert (Quelle: {self._tools[tool.name].source})."
            )
        if not tool.description.strip():
            raise ToolRegistryError(f"Tool '{tool.name}' braucht eine Beschreibung für das Modell.")
        if tool.parameters.get("type") != "object":
            raise ToolRegistryError(f"Tool '{tool.name}': parameters muss type 'object' haben.")
        try:
            Draft202012Validator.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ToolRegistryError(f"Tool '{tool.name}': ungültiges Schema: {exc.message}") from exc
        self._tools[tool.name] = tool

    def unregister_source(self, source: str) -> list[str]:
        """Entfernt alle Tools einer Quelle (z. B. beim Entladen eines Plugins)."""
        names = [n for n, t in self._tools.items() if t.source == source]
        for name in names:
            del self._tools[name]
        return names

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self, disabled: set[str] | None = None) -> list[ToolSpec]:
        skip = disabled or set()
        return [t.spec() for t in self._tools.values() if t.name not in skip]
