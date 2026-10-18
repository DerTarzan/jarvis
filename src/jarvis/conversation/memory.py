"""Speicherkomponenten: Sitzungsverlauf (kurzfristig) und Langzeitgedächtnis-Schnittstelle."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from jarvis.ai.base import ChatMessage


class MemoryStore(ABC):
    """Austauschbarer Speicher für den Gesprächsverlauf einer Sitzung."""

    @abstractmethod
    def load(self, session: str) -> list[ChatMessage]: ...

    @abstractmethod
    def append(self, session: str, message: ChatMessage) -> None: ...

    @abstractmethod
    def clear(self, session: str) -> None: ...

    def close(self) -> None:  # noqa: B027 - optionaler Hook
        """Gibt Ressourcen frei."""


class InMemoryStore(MemoryStore):
    """Standard: nichts wird auf die Festplatte geschrieben."""

    def __init__(self) -> None:
        self._data: dict[str, list[ChatMessage]] = {}

    def load(self, session: str) -> list[ChatMessage]:
        return list(self._data.get(session, []))

    def append(self, session: str, message: ChatMessage) -> None:
        self._data.setdefault(session, []).append(message)

    def clear(self, session: str) -> None:
        self._data.pop(session, None)


class LongTermMemory(ABC):
    """Schnittstelle für späteres Langzeitgedächtnis (Fakten, Notizen)."""

    @abstractmethod
    def remember(self, text: str) -> int: ...

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[str]: ...

    @abstractmethod
    def forget(self, fact_id: int) -> bool: ...


class SQLiteMemory(MemoryStore, LongTermMemory):
    """Lokale SQLite-Datenbank: Sitzungsverlauf plus einfache Faktenablage (LIKE-Suche)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, "
                "payload TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL)"
            )

    def load(self, session: str) -> list[ChatMessage]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM messages WHERE session = ? ORDER BY id", (session,)
            ).fetchall()
        return [ChatMessage.from_dict(json.loads(r[0])) for r in rows]

    def append(self, session: str, message: ChatMessage) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (session, payload) VALUES (?, ?)",
                (session, json.dumps(message.to_dict(), ensure_ascii=False)),
            )

    def clear(self, session: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM messages WHERE session = ?", (session,))

    def remember(self, text: str) -> int:
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO facts (text) VALUES (?)", (text,))
            return int(cur.lastrowid or 0)

    def search(self, query: str, limit: int = 5) -> list[str]:
        like = f"%{query}%"
        with self._lock:
            rows = self._db.execute(
                "SELECT text FROM facts WHERE text LIKE ? ORDER BY id DESC LIMIT ?", (like, limit)
            ).fetchall()
        return [r[0] for r in rows]

    def forget(self, fact_id: int) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._db.close()
