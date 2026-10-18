"""Gesprächslogik: Nutzertext → LLM → (Tools →)* Antwort. Kennt weder Audio noch UI."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from jarvis.ai.base import LLMProvider
from jarvis.conversation.manager import ConversationManager
from jarvis.core import events as ev
from jarvis.core.events import EventBus
from jarvis.tools.executor import ConfirmCallback, ToolExecutor
from jarvis.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

TOOL_LIMIT_REPLY = (
    "Ich konnte die Anfrage nach mehreren Werkzeugaufrufen nicht abschließen. "
    "Bitte formuliere sie etwas konkreter."
)


class Assistant:
    def __init__(
        self,
        provider: LLMProvider,
        conversation: ConversationManager,
        registry: ToolRegistry,
        executor: ToolExecutor,
        events: EventBus,
        tools_enabled: bool = True,
        max_tool_rounds: int = 4,
        disabled_tools: set[str] | None = None,
    ) -> None:
        self.provider = provider
        self.conversation = conversation
        self._registry = registry
        self._executor = executor
        self._events = events
        self._tools_enabled = tools_enabled
        self._max_rounds = max_tool_rounds
        self._disabled = disabled_tools or set()

    async def respond(
        self,
        text: str,
        confirm: ConfirmCallback | None = None,
        on_chunk: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str:
        """Verarbeitet eine Nutzeräußerung und gibt die Antwort zurück.

        Bei Provider-Fehlern wird der Gesprächszustand zurückgesetzt und die Ausnahme
        (ProviderError) weitergereicht; die Oberfläche zeigt sie verständlich an.
        """
        mark = self.conversation.mark()
        self.conversation.add_user(text)
        try:
            for _ in range(self._max_rounds + 1):
                specs = (
                    self._registry.specs(self._disabled)
                    if self._tools_enabled and self.provider.supports_tools
                    else None
                )
                response = await self.provider.stream_response(
                    self.conversation.get_messages(), specs or None, on_chunk
                )

                if not response.tool_calls:
                    answer = response.content or "(keine Antwort erhalten)"
                    self.conversation.add_assistant(answer)
                    return answer

                self.conversation.add_assistant(response.content, response.tool_calls)
                for call in response.tool_calls:
                    result = await self._executor.execute(call, confirm)
                    log.info("Tool %s → %s", call.name, "ok" if result.ok else "Fehler")
                    self._events.emit(ev.TOOL_CALLED, name=call.name, ok=result.ok)
                    self.conversation.add_tool_result(call.name, result.content, call.id or None)

            self.conversation.add_assistant(TOOL_LIMIT_REPLY)
            return TOOL_LIMIT_REPLY
        except BaseException:
            self.conversation.rollback(mark)
            raise
