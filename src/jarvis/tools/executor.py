"""Kontrollierte Tool-Ausführung: Validierung, Bestätigung, Timeout, Fehlerisolierung."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from jarvis.ai.base import ToolCall
from jarvis.tools.registry import Tool, ToolRegistry

log = logging.getLogger(__name__)

ConfirmCallback = Callable[[Tool, dict[str, Any]], Awaitable[bool]]
MAX_RESULT_CHARS = 4000


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: str  # für das Modell bestimmt (Ergebnis oder Fehlertext)


class ToolExecutor:
    """Führt ausschließlich registrierte Tools aus – niemals freien Code oder Shell-Befehle."""

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout_s: float = 30.0,
        disabled: set[str] | None = None,
    ) -> None:
        self._registry = registry
        self._timeout = default_timeout_s
        self._disabled = disabled or set()

    async def execute(self, call: ToolCall, confirm: ConfirmCallback | None = None) -> ToolResult:
        tool = self._registry.get(call.name)
        if tool is None or call.name in self._disabled:
            return ToolResult(call.name, False, f"Unbekanntes oder deaktiviertes Tool '{call.name}'.")

        args = call.arguments if isinstance(call.arguments, dict) else {}
        errors = sorted(Draft202012Validator(tool.parameters).iter_errors(args), key=lambda e: list(e.path))
        if errors:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e.path) or 'Argumente'}: {e.message}" for e in errors
            )
            return ToolResult(call.name, False, f"Ungültige Parameter: {detail}")

        if tool.requires_confirmation:
            approved = False
            if confirm is not None:
                try:
                    approved = await confirm(tool, args)
                except Exception:
                    log.exception("Bestätigungsabfrage für '%s' fehlgeschlagen", tool.name)
            if not approved:
                return ToolResult(call.name, False, "Aktion wurde vom Benutzer nicht bestätigt.")

        timeout = tool.timeout_s or self._timeout
        try:
            if inspect.iscoroutinefunction(tool.handler):
                raw = await asyncio.wait_for(tool.handler(**args), timeout)
            else:
                raw = await asyncio.wait_for(asyncio.to_thread(tool.handler, **args), timeout)
        except TimeoutError:
            log.warning("Tool '%s' Timeout nach %.0f s", tool.name, timeout)
            return ToolResult(call.name, False, f"Zeitüberschreitung nach {timeout:.0f} s.")
        except Exception as exc:
            log.exception("Tool '%s' ist fehlgeschlagen", tool.name)
            return ToolResult(call.name, False, f"Tool-Fehler: {type(exc).__name__}: {exc}")

        return ToolResult(call.name, True, _serialize(raw))


def _serialize(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + " …(gekürzt)"
    return text
