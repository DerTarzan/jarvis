"""Gesprächsverwaltung: Rollen, begrenzter Kontext, optionale Persistenz."""

from __future__ import annotations

from jarvis.ai.base import ChatMessage, ToolCall
from jarvis.conversation.memory import InMemoryStore, MemoryStore


class ConversationManager:
    """Hält den Verlauf einer Sitzung und liefert das Kontextfenster für das Modell."""

    def __init__(
        self,
        system_prompt: str,
        max_messages: int = 20,
        store: MemoryStore | None = None,
        session: str = "default",
    ) -> None:
        if max_messages < 2:
            raise ValueError("max_messages muss mindestens 2 sein")
        self._system_prompt = system_prompt
        self._max = max_messages
        self._store = store or InMemoryStore()
        self._session = session
        self._messages: list[ChatMessage] = self._store.load(session)

    # -- Schreiben -----------------------------------------------------
    def add_user(self, text: str) -> None:
        self._add(ChatMessage("user", text))

    def add_assistant(self, text: str, tool_calls: list[ToolCall] | None = None) -> None:
        self._add(ChatMessage("assistant", text, tool_calls=list(tool_calls or [])))

    def add_tool_result(self, tool_name: str, content: str, call_id: str | None = None) -> None:
        self._add(ChatMessage("tool", content, tool_name=tool_name, tool_call_id=call_id))

    def _add(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._store.append(self._session, message)

    # -- Lesen ---------------------------------------------------------
    def get_messages(self) -> list[ChatMessage]:
        """System-Prompt plus die letzten ``max_messages`` Nachrichten.

        Das Fenster beginnt immer mit einer Benutzernachricht, damit keine verwaisten
        Tool-Ergebnisse oder Assistentenantworten am Anfang stehen. Passt die laufende Runde
        (Frage + Tool-Aufrufe) nicht ins Limit, wird sie vollständig behalten.
        """
        start = max(0, len(self._messages) - self._max)
        forward = start
        while forward < len(self._messages) and self._messages[forward].role != "user":
            forward += 1
        if forward < len(self._messages):
            start = forward
        else:
            # Fenster enthält keine Benutzernachricht (lange Tool-Kette): lieber etwas länger
            # als die laufende Frage zu verlieren – zurück bis zur zugehörigen Benutzernachricht.
            while start > 0 and self._messages[start].role != "user":
                start -= 1
        return [ChatMessage("system", self._system_prompt), *self._messages[start:]]

    def __len__(self) -> int:
        return len(self._messages)

    # -- Rücksetzen ----------------------------------------------------
    def mark(self) -> int:
        """Merkt sich den aktuellen Stand, um einen fehlgeschlagenen Zug zurückzunehmen."""
        return len(self._messages)

    def rollback(self, mark: int) -> None:
        """Entfernt Nachrichten nach ``mark`` aus dem Speicher des Prozesses.

        Hinweis: Persistierte Zeilen bleiben bestehen; für Fehlerfälle akzeptabel, da der
        Verlauf nur ergänzt wird. Bei aktiver Persistenz wird die Sitzung daher neu geschrieben.
        """
        del self._messages[mark:]
        if not isinstance(self._store, InMemoryStore):
            self._store.clear(self._session)
            for message in self._messages:
                self._store.append(self._session, message)

    def reset(self) -> None:
        """Beendet die Sitzung und beginnt ein neues Gespräch."""
        self._messages.clear()
        self._store.clear(self._session)
