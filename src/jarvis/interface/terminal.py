"""Terminaloberfläche. Spricht nur mit ``Application`` – enthält keine KI-/Audiologik."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from rich.console import Console
from rich.control import Control
from rich.panel import Panel
from rich.segment import ControlType
from rich.table import Table
from rich.text import Text

from jarvis.audio.errors import AudioError
from jarvis.audio.microphone import list_devices
from jarvis.core import events as ev
from jarvis.core.application import Application, StartupReport
from jarvis.tools.registry import Tool

log = logging.getLogger(__name__)

HELP = (
    "Hands-free (Standard): einfach „Jarvis, <Frage>“ sagen – kein Befehl nötig.\n"
    "  Nach einer Antwort kannst du ein paar Sekunden ohne „Jarvis“ weitersprechen.\n"
    "  „Danke“ oder „Stopp“ beendet das Gespräch.\n"
    "Eingabe + Enter: Text senden (geht jederzeit parallel)\n"
    "Enter (leer): Sprachaufnahme starten/beenden (nur Modus push_to_talk)\n"
    "/listen  Hands-free ein/aus         /voice  eine Aufnahme per Enter (Push-to-Talk)\n"
    "/mute    Sprachausgabe an/aus      /stop   Sprachausgabe abbrechen\n"
    "/status  Status anzeigen           /tools  Tools        /plugins  Plugins\n"
    "/devices Audiogeräte               /reset  neues Gespräch\n"
    "/quit    beenden (auch Strg+C / Strg+D)"
)


class TerminalUI:
    def __init__(self, app: Application, console: Console | None = None) -> None:
        self.app = app
        self.console = console or Console()
        self._pending: asyncio.Future[str] | None = None
        self._speech_line_visible = False
        app.events.subscribe(ev.USER_TEXT, self._on_user)
        app.events.subscribe(ev.ASSISTANT_TEXT, self._on_assistant)
        app.events.subscribe(ev.ERROR, self._on_error)
        app.events.subscribe(ev.INFO, self._on_info)
        app.events.subscribe(ev.STATE_CHANGED, self._on_state)
        app.events.subscribe(ev.TOOL_CALLED, self._on_tool)

    # -- Ereignisse --------------------------------------------------------
    def _on_user(self, text: str, source: str, **_: Any) -> None:
        if source == "voice":
            self._clear_speech_line()
            self.console.print(Text(f"Du: {text}", style="dim", no_wrap=True, overflow="ellipsis"), end="\r")
            self._speech_line_visible = True

    def _on_assistant(self, text: str, **_: Any) -> None:
        self._clear_speech_line()
        self.console.print(
            Text.assemble((f"{self.app.settings.general.assistant_name}: ", "bold green"), text)
        )

    def _on_error(self, message: str, **_: Any) -> None:
        self._clear_speech_line()
        self.console.print(Text.assemble(("Fehler: ", "bold red"), message))

    def _on_info(self, message: str, **_: Any) -> None:
        self._clear_speech_line()
        self.console.print(Text(f"· {message}", style="cyan"))

    def _on_state(self, state: str, **_: Any) -> None:
        if state in ("hört zu", "denkt nach"):
            if state == "denkt nach" and self._speech_line_visible:
                return
            self._clear_speech_line()
            self.console.print(Text(f"· {state} …", style="dim"))

    def _on_tool(self, name: str, ok: bool, **_: Any) -> None:
        if self._speech_line_visible:
            return
        self._clear_speech_line()
        style = "dim" if ok else "yellow"
        self.console.print(Text(f"· Tool {name}: {'ok' if ok else 'fehlgeschlagen'}", style=style))

    def _clear_speech_line(self) -> None:
        if self._speech_line_visible:
            self.console.control(Control.move_to_column(0), Control((ControlType.ERASE_IN_LINE, 2)))
            self._speech_line_visible = False

    # -- Eingabe (daemon-Thread, damit Strg+C sauber beendet) --------------
    def _start_read(self, prompt: str) -> asyncio.Future[str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        def deliver(value: str | None, exc: BaseException | None) -> None:
            if future.done():
                return
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(value or "")

        def worker() -> None:
            try:
                value = input(prompt)
            except BaseException as exc:  # EOFError etc. an die Event-Loop weiterreichen
                loop.call_soon_threadsafe(deliver, None, exc)
            else:
                loop.call_soon_threadsafe(deliver, value, None)

        threading.Thread(target=worker, name="terminal-input", daemon=True).start()
        return future

    async def _read(self, prompt: str) -> str:
        future = self._pending or self._start_read(prompt)
        self._pending = future
        try:
            return await future
        finally:
            self._pending = None

    async def _confirm(self, tool: Tool, args: dict[str, Any]) -> bool:
        self.console.print(Text(f"Jarvis möchte '{tool.name}' mit {args} ausführen.", style="bold yellow"))
        answer = await self._read("Bestätigen? [j/N]: ")
        return answer.strip().lower() in ("j", "ja", "y", "yes")

    # -- Anzeige ------------------------------------------------------------
    def show_banner(self, report: StartupReport) -> None:
        app, s = self.app, self.app.settings
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Zustand", app.state)
        cloud = "  [bold red]CLOUD – Texte verlassen den Rechner[/bold red]" if app.provider.is_cloud else ""
        table.add_row("LLM", f"{app.provider.name} / {app.provider.model}{cloud}")
        llm_ok = "[green]bereit[/green]" if report.provider.ok else "[red]nicht bereit[/red]"
        table.add_row("LLM-Status", f"{llm_ok} – {Text(report.provider.message).plain}")
        table.add_row("STT", Text(app.voice.stt_status).plain)
        table.add_row("TTS", Text(app.voice.tts_status).plain)
        plugins = ", ".join(p.name for p in report.plugins if p.loaded) or "keine"
        table.add_row("Plugins", plugins)
        table.add_row("Tools", ", ".join(t.name for t in app.registry.all()) or "keine")
        mode = s.general.activation_mode
        if mode == "handsfree":
            mode = f"handsfree – Wake Word „{s.wake_word.phrase.capitalize()}“"
        table.add_row("Aktivierung", mode)
        self.console.print(Panel(table, title=f"{s.general.assistant_name} {self._version()}", expand=False))
        for warning in report.warnings:
            self.console.print(Text(f"Hinweis: {warning}", style="yellow"))
        self.console.print(Text("Tippe /help für Befehle.", style="dim"))

    @staticmethod
    def _version() -> str:
        from jarvis import __version__

        return f"v{__version__}"

    def _show_status(self) -> None:
        app = self.app
        app.voice.refresh_status()
        self.console.print(Text(f"Zustand: {app.state}"))
        self.console.print(Text(f"Hands-free: {'aktiv' if app.handsfree_active else 'aus'}"))
        self.console.print(Text(f"LLM: {app.provider.name} / {app.provider.model}"))
        self.console.print(Text(f"STT: {app.voice.stt_status}"))
        self.console.print(Text(f"TTS: {app.voice.tts_status}"))
        self.console.print(Text(f"Sprachausgabe: {'an' if app.speak_enabled else 'aus'}"))

    # -- Hauptschleife -------------------------------------------------------
    async def run(self) -> None:
        report = await self.app.startup()
        self.show_banner(report)
        mode = self.app.settings.general.activation_mode
        push_to_talk = mode == "push_to_talk"
        if mode == "handsfree":
            await self.app.start_handsfree()  # läuft im Hintergrund, Tastatur bleibt nutzbar
        while True:
            try:
                line = (await self._read("Du: ")).strip()
            except EOFError:
                break
            try:
                if line.startswith("/"):
                    if not await self._command(line):
                        break
                elif line:
                    await self.app.process(line, source="text", confirm=self._confirm)
                elif push_to_talk:
                    await self._voice_turn()
            except EOFError:
                break
            except Exception as exc:  # ein fehlerhafter Zug darf die Sitzung nicht beenden
                log.exception("Fehler in der Hauptschleife")
                self._on_error(f"Unerwarteter Fehler: {type(exc).__name__}: {exc}")

    async def _command(self, line: str) -> bool:
        """Verarbeitet einen Befehl. False = Programm beenden."""
        cmd = line.split()[0].lower()
        app = self.app
        if cmd in ("/quit", "/exit", "/q"):
            return False
        if cmd == "/help":
            self.console.print(Text(HELP))
        elif cmd == "/status":
            self._show_status()
        elif cmd == "/tools":
            for tool in app.registry.all():
                flag = " (Bestätigung nötig)" if tool.requires_confirmation else ""
                self.console.print(Text(f"{tool.name} [{tool.source}]{flag}: {tool.description}"))
        elif cmd == "/plugins":
            for st in app.plugins.statuses:
                state = "geladen" if st.loaded else f"FEHLER: {st.error}"
                self.console.print(Text(f"{st.name}: {state} {st.description}"))
            if not app.plugins.statuses:
                self.console.print(Text("Keine Plugins aktiviert."))
        elif cmd == "/listen":
            if app.handsfree_active:
                await app.stop_handsfree()
            else:
                await app.start_handsfree()
        elif cmd == "/voice":
            if app.handsfree_active:
                self.console.print(Text("Hands-free ist aktiv – sprich einfach. /listen schaltet es aus."))
            else:
                await self._voice_turn()
        elif cmd == "/mute":
            app.speak_enabled = not app.speak_enabled
            self.console.print(Text(f"Sprachausgabe {'an' if app.speak_enabled else 'aus'}."))
        elif cmd == "/stop":
            app.stop_speaking()
        elif cmd == "/reset":
            app.reset_conversation()
            self.console.print(Text("Neues Gespräch begonnen."))
        elif cmd == "/devices":
            try:
                for entry in list_devices():
                    self.console.print(Text(entry))
            except AudioError as exc:
                self._on_error(str(exc))
        else:
            self.console.print(Text(f"Unbekannter Befehl {cmd}. /help zeigt alle Befehle."))
        return True

    async def _voice_turn(self) -> None:
        """Push-to-Talk: Enter beendet die Aufnahme."""
        app = self.app
        if not app.voice.can_listen:
            self._on_error(f"Spracheingabe nicht verfügbar ({app.voice.stt_status}).")
            return
        app.stop_speaking()  # eigene Ausgabe nie als Eingabe aufnehmen
        stop = threading.Event()
        self.console.print(Text("● Aufnahme läuft – Enter zum Beenden.", style="bold red"))
        listen_task = asyncio.create_task(app.listen(stop))
        enter = self._pending or self._start_read("")
        self._pending = enter
        try:
            await asyncio.wait({listen_task, enter}, return_when=asyncio.FIRST_COMPLETED)
            stop.set()
            text = await listen_task
        except BaseException:
            stop.set()
            listen_task.cancel()
            raise
        if enter.done():
            self._pending = None
            enter.exception()  # EOF o. Ä. nicht als unbehandelt melden
        if text:
            await app.process(text, source="voice", confirm=self._confirm)
