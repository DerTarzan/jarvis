"""Anwendungskern: baut alle Komponenten zusammen und steuert den Lebenszyklus."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from jarvis.ai.base import LLMProvider, ProviderError, ProviderStatus
from jarvis.ai.manager import create_provider
from jarvis.audio.errors import AudioError, STTError, TTSError, WakeWordError
from jarvis.audio.voice import VoiceIO
from jarvis.audio.wake_word import NullWakeWord, TranscriptWakeWord, create_wake_word
from jarvis.config.loader import Settings
from jarvis.conversation.manager import ConversationManager
from jarvis.conversation.memory import InMemoryStore, LongTermMemory, MemoryStore, SQLiteMemory
from jarvis.core import events as ev
from jarvis.core.assistant import Assistant
from jarvis.core.events import EventBus
from jarvis.core.handsfree import HandsFreeLoop
from jarvis.plugins.manager import PluginManager, PluginStatus
from jarvis.tools.builtin import register_builtin_tools
from jarvis.tools.executor import ConfirmCallback, ToolExecutor
from jarvis.tools.registry import Tool, ToolRegistry

log = logging.getLogger(__name__)

STATE_READY = "bereit"
STATE_LISTENING = "hört zu"
STATE_THINKING = "denkt nach"
STATE_SPEAKING = "spricht"


@dataclass
class StartupReport:
    provider: ProviderStatus
    plugins: list[PluginStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class Application:
    """Hält alle Komponenten. Die Oberfläche spricht ausschließlich mit dieser Klasse."""

    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider | None = None,
        voice: VoiceIO | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.settings = settings
        self.events = EventBus()
        self.state = STATE_READY
        self.speak_enabled = settings.general.speak_replies

        # Provider: Fehler (nicht implementiert/Schlüssel fehlt/...) gehen an den Aufrufer.
        self.provider = provider or create_provider(settings.ai.provider, settings.active_provider)

        self.registry = ToolRegistry()
        if settings.tools.enabled:
            register_builtin_tools(self.registry)
        self.plugins = PluginManager(
            directory=settings.resolve(settings.plugins.directory),
            registry=self.registry,
            events=self.events,
            data_dir=settings.resolve("data"),
            settings=settings.plugins.settings,
        )
        disabled = set(settings.tools.disabled)
        self.executor = ToolExecutor(self.registry, settings.tools.timeout_s, disabled)

        self._store = store or self._create_store()
        prompt = settings.general.system_prompt.replace("{name}", settings.general.assistant_name)
        self._long_term_memory = (
            self._store
            if isinstance(self._store, LongTermMemory) and settings.memory.engine != "none"
            else None
        )
        if settings.tools.enabled and self._long_term_memory is not None:
            self._register_memory_tools()
            prompt += (
                "\nDu hast ein lokales Langzeitgedächtnis. Nutze search_memory, wenn gespeicherte "
                "persönliche Fakten hilfreich sind. Verwende remember_fact nur, wenn der Nutzer "
                "ausdrücklich bittet, etwas zu merken. Löschen erfordert immer Bestätigung."
            )
        self.conversation = ConversationManager(
            prompt,
            settings.conversation.max_messages,
            self._store if settings.conversation.persist else InMemoryStore(),
            settings.conversation.session_name,
        )
        self.assistant = Assistant(
            self.provider,
            self.conversation,
            self.registry,
            self.executor,
            self.events,
            tools_enabled=settings.tools.enabled,
            max_tool_rounds=settings.tools.max_tool_rounds,
            disabled_tools=disabled,
        )
        self.voice = voice or VoiceIO(settings)
        self._turn_lock = asyncio.Lock()  # eine Äußerung nach der anderen (Tastatur + Stimme)
        self._hf_task: asyncio.Task[None] | None = None
        self._hf_stop = threading.Event()

        # Wake Word: Fehler sind nie fatal, verhindern aber Hands-free (siehe _handsfree_problem)
        self.wake_word_error = ""
        self.wake = TranscriptWakeWord(settings.wake_word)
        try:
            self.wake_detector = create_wake_word(
                settings.wake_word, settings.resolve(settings.wake_word.model_dir)
            )
        except WakeWordError as exc:
            self.wake_word_error = str(exc)
            self.wake_detector = NullWakeWord()
        mode = settings.general.activation_mode
        if mode == "handsfree" and not self.wake_word_error:
            self.wake_word_error = self._handsfree_problem()

    def _create_store(self) -> MemoryStore:
        s = self.settings
        if s.memory.engine == "sqlite" or s.conversation.persist:
            return SQLiteMemory(s.resolve(s.memory.path))
        return InMemoryStore()

    def _register_memory_tools(self) -> None:
        assert self._long_term_memory is not None
        self.registry.register(
            Tool(
                name="remember_fact",
                description=(
                    "Speichert eine kurze persönliche Information dauerhaft, wenn der Nutzer darum bittet."
                ),
                parameters={
                    "type": "object",
                    "properties": {"fact": {"type": "string", "minLength": 1, "maxLength": 500}},
                    "required": ["fact"],
                    "additionalProperties": False,
                },
                handler=lambda fact: {"id": self._long_term_memory.remember(fact)},
            )
        )
        self.registry.register(
            Tool(
                name="search_memory",
                description="Sucht im lokalen Langzeitgedächtnis nach gespeicherten persönlichen Fakten.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=lambda query: {"fakten": self._long_term_memory.search(query)},
            )
        )
        self.registry.register(
            Tool(
                name="forget_memory",
                description="Löscht einen gespeicherten Fakt anhand seiner ID. Immer vorher bestätigen.",
                parameters={
                    "type": "object",
                    "properties": {"fact_id": {"type": "integer", "minimum": 1}},
                    "required": ["fact_id"],
                    "additionalProperties": False,
                },
                handler=lambda fact_id: {"gelöscht": self._long_term_memory.forget(fact_id)},
                requires_confirmation=True,
            )
        )

    # -- Lebenszyklus ------------------------------------------------------
    async def startup(self) -> StartupReport:
        provider_status = await self.provider.check()
        plugin_status = self.plugins.load_all(self.settings.plugins.enabled)
        warnings = []
        if self.provider.is_cloud:
            warnings.append(
                f"Cloud-Provider '{self.provider.name}' aktiv: Gesprächstexte werden an einen "
                "externen Dienst übertragen (Kosten- und Datenschutzrisiko)."
            )
        if self.wake_word_error:
            warnings.append(self.wake_word_error)
        for status in plugin_status:
            if not status.loaded:
                warnings.append(f"Plugin '{status.name}' nicht geladen: {status.error}")
        return StartupReport(provider_status, plugin_status, warnings)

    # -- Hands-free ---------------------------------------------------------
    def _handsfree_problem(self) -> str:
        """Grund, warum Hands-free nicht starten kann, sonst ''."""
        wake = self.settings.wake_word
        if not (wake.enabled and wake.engine in ("whisper", "openwakeword")):
            return (
                'Hands-free braucht wake_word.enabled: true und wake_word.engine: "whisper" '
                'oder "openwakeword" in der Konfiguration.'
            )
        if not self.voice.can_listen:
            return f"Hands-free nicht möglich: Spracheingabe nicht verfügbar ({self.voice.stt_status})."
        return ""

    @property
    def handsfree_active(self) -> bool:
        return self._hf_task is not None and not self._hf_task.done()

    async def start_handsfree(self) -> bool:
        """Startet das dauerhafte Zuhören im Hintergrund. False (mit Fehlermeldung) bei Problemen."""
        if self.handsfree_active:
            return True
        problem = self._handsfree_problem()
        if problem:
            self.report_error(problem)
            return False
        self._hf_stop = threading.Event()
        self._hf_task = asyncio.create_task(HandsFreeLoop(self, self._hf_stop).run(), name="handsfree")
        return True

    async def stop_handsfree(self) -> None:
        """Beendet das Zuhören und wartet kurz auf das Ende der Mikrofonaufnahme."""
        task = self._hf_task
        if task is None:
            return
        self._hf_stop.set()
        self.voice.stop_speaking()
        try:
            await asyncio.wait_for(task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            log.warning("Hands-free-Schleife musste abgebrochen werden")
        except Exception:
            log.exception("Fehler beim Beenden von Hands-free")
        self._hf_task = None

    async def shutdown(self) -> None:
        """Beendet alles kontrolliert; mehrfach aufrufbar."""
        log.info("Beende Jarvis …")
        await self.stop_handsfree()
        for step in (self.voice.close, self.plugins.unload_all, self._store.close):
            try:
                step()
            except Exception:
                log.exception("Fehler beim Beenden (%s)", getattr(step, "__qualname__", step))
        try:
            await self.provider.aclose()
        except Exception:
            log.exception("Fehler beim Schließen des Providers")

    # -- Zustand -----------------------------------------------------------
    def set_state(self, state: str) -> None:
        self.state = state
        self.events.emit(ev.STATE_CHANGED, state=state)

    def report_error(self, message: str) -> None:
        log.error(message)
        self.events.emit(ev.ERROR, message=message)

    _set_state = set_state
    _error = report_error

    # -- Verarbeitung ------------------------------------------------------
    async def process(
        self,
        text: str,
        source: str = "text",
        confirm: ConfirmCallback | None = None,
        speak: bool | None = None,
    ) -> str | None:
        """Text → Antwort. Gibt None zurück, wenn ein Fehler gemeldet wurde.

        Äußerungen (Tastatur und Stimme) werden nacheinander abgearbeitet.
        """
        async with self._turn_lock:
            return await self._process_locked(text, source, confirm, speak)

    async def say(self, text: str) -> None:
        """Zeigt und spricht einen festen Text (ohne Sprachmodell), z. B. eine Bestätigung."""
        async with self._turn_lock:
            self.events.emit(ev.ASSISTANT_TEXT, text=text)
            if self.speak_enabled and self.voice.can_speak:
                await self._speak(text)

    async def acknowledge(self, phrase: str) -> None:
        """Hörbare Bestätigung nach dem Wake Word: Satz (Standard „Ja?“) oder kurzer Signalton."""
        if not self.voice.can_speak:
            self.events.emit(ev.INFO, message="Ja?")
            return
        try:
            if phrase.strip():
                await self.voice.speak(phrase)
            else:
                await self.voice.beep()
        except Exception as exc:  # Bestätigung ist optional – nie den Dialog abbrechen
            log.warning("Bestätigung fehlgeschlagen: %s", exc)

    async def _speak(self, text: str) -> None:
        self.set_state(STATE_SPEAKING)
        try:
            await self.voice.speak(text)
        except (AudioError, TTSError) as exc:
            self.report_error(f"Sprachausgabe fehlgeschlagen: {exc}")
        except Exception as exc:  # Audio-Stack darf den Assistenten nie beenden
            log.exception("Unerwarteter Fehler bei der Sprachausgabe")
            self.report_error(f"Sprachausgabe fehlgeschlagen: {type(exc).__name__}: {exc}")
        finally:
            self.set_state(STATE_READY)

    async def _speak_stream(self, chunks) -> None:
        self.set_state(STATE_SPEAKING)
        try:
            await self.voice.speak_stream(chunks)
        except (AudioError, TTSError) as exc:
            self.report_error(f"Sprachausgabe fehlgeschlagen: {exc}")
        except Exception as exc:
            log.exception("Unerwarteter Fehler bei der Sprachausgabe")
            self.report_error(f"Sprachausgabe fehlgeschlagen: {type(exc).__name__}: {exc}")

    async def _process_locked(
        self,
        text: str,
        source: str,
        confirm: ConfirmCallback | None,
        speak: bool | None,
    ) -> str | None:
        self.events.emit(ev.USER_TEXT, text=text, source=source)
        self._set_state(STATE_THINKING)
        stream_speech = (
            (self.speak_enabled if speak is None else speak)
            and self.voice.can_speak
            and callable(getattr(self.voice, "speak_stream", None))
        )
        chunks: asyncio.Queue[str | None] | None = asyncio.Queue() if stream_speech else None
        speech_task: asyncio.Task[None] | None = None
        if chunks is not None:
            async def stream_chunks():
                while (chunk := await chunks.get()) is not None:
                    yield chunk

            speech_task = asyncio.create_task(self._speak_stream(stream_chunks()))

        async def enqueue_chunk(chunk: str) -> None:
            if chunks is not None:
                chunks.put_nowait(chunk)

        try:
            reply = await self.assistant.respond(text, confirm, enqueue_chunk if chunks else None)
        except ProviderError as exc:
            self._error(str(exc))
            return None
        except Exception as exc:  # unerwarteter Fehler: melden, aber Assistent läuft weiter
            log.exception("Unerwarteter Fehler bei der Verarbeitung")
            self._error(f"Unerwarteter Fehler: {type(exc).__name__}: {exc}")
            return None
        finally:
            if chunks is not None and speech_task is not None:
                chunks.put_nowait(None)
                await speech_task
            self._set_state(STATE_READY)
        self.events.emit(ev.ASSISTANT_TEXT, text=reply)

        if not stream_speech and (self.speak_enabled if speak is None else speak) and self.voice.can_speak:
            await self._speak(reply)
        return reply

    async def listen(self, stop: threading.Event, use_silence_detection: bool = False) -> str:
        """Nimmt auf und erkennt Text. Fehler werden gemeldet, Rückgabe dann ''."""
        self._set_state(STATE_LISTENING)
        try:
            text = await self.voice.listen(stop, use_silence_detection)
        except (AudioError, STTError) as exc:
            self._error(str(exc))
            return ""
        except Exception as exc:
            log.exception("Unerwarteter Fehler bei der Spracheingabe")
            self._error(f"Spracheingabe fehlgeschlagen: {type(exc).__name__}: {exc}")
            return ""
        finally:
            self._set_state(STATE_READY)
        if not text:
            self._error("Es wurde keine Sprache erkannt.")
        return text

    def stop_speaking(self) -> None:
        self.voice.stop_speaking()

    def reset_conversation(self) -> None:
        self.conversation.reset()
