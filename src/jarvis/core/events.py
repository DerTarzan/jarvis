"""Minimaler Event-Bus für lose Kopplung zwischen Kern, Plugins und Oberfläche."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# Bekannte Ereignisse (Konstanten verhindern Tippfehler)
STATE_CHANGED = "state_changed"  # state: str
USER_TEXT = "user_text"  # text: str, source: "text" | "voice"
ASSISTANT_TEXT = "assistant_text"  # text: str
TOOL_CALLED = "tool_called"  # name: str, ok: bool
ERROR = "error"  # message: str
INFO = "info"  # message: str (Statushinweis, kein Fehler)

Handler = Callable[..., None]


class EventBus:
    """Synchroner Bus. Fehler in Handlern werden geloggt und nie weitergereicht."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event: str, handler: Handler) -> Callable[[], None]:
        self._handlers[event].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[event]:
                self._handlers[event].remove(handler)

        return unsubscribe

    def emit(self, event: str, **data: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(**data)
            except Exception:
                log.exception("Event-Handler für '%s' ist fehlgeschlagen", event)
