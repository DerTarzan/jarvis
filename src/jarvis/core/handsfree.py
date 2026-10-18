"""Hands-free-Schleife („wie Alexa“): dauerhaft lokal zuhören, auf das Wake Word reagieren,
Befehl ausführen, danach kurz ohne Wake Word weiter zuhören (Folgegespräch).

Ablauf pro Äußerung:
  1. Auf Sprache warten (``VoiceIO.capture``) – im Leerlauf nur Lautstärkemessung.
  2. Lokal in Text umwandeln (STT).
  3. Wake Word am Anfang? Nein → verwerfen (Text wird nicht gespeichert und nicht geloggt).
     Ja, mit Befehl → direkt ausführen („Jarvis, wie spät ist es?“).
     Ja, ohne Befehl → kurze Bestätigung („Ja?“), dann auf den Befehl warten.
  4. Antwort sprechen; danach ``followup_window_s`` Sekunden ohne Wake Word weiter zuhören.

Gehört wird nie während Jarvis spricht (kein Echo-Problem, aber auch kein Unterbrechen per Stimme).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from jarvis.audio.errors import AudioError, STTError
from jarvis.core import events as ev
from jarvis.tools.registry import Tool

if TYPE_CHECKING:
    from jarvis.core.application import Application

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
RETRY_DELAY_S = 1.0  # Pause nach einem Mikrofon-/STT-Fehler, bevor es erneut versucht wird

STATE_WAITING = "wartet auf Wake-Word"
STATE_FOLLOWUP = "hört weiter zu"

# Lokale Abschlussformeln: beenden das Folgegespräch, ohne das Sprachmodell zu fragen.
CLOSING_PHRASES: dict[str, str] = {
    "danke": "Gern geschehen.",
    "danke dir": "Gern geschehen.",
    "vielen dank": "Gern geschehen.",
    "okay danke": "Gern geschehen.",
    "ok danke": "Gern geschehen.",
    "das wars": "In Ordnung.",
    "das wars danke": "Gern geschehen.",
    "stopp": "In Ordnung.",
    "stop": "In Ordnung.",
    "abbrechen": "In Ordnung.",
    "vergiss es": "In Ordnung.",
}


def closing_reply(command: str) -> str | None:
    """Antwort auf eine Abschlussformel (z. B. „Danke“) oder ``None``."""
    key = re.sub(r"[^\w\s]", "", command.lower().replace("’", "").replace("'", "")).strip()
    return CLOSING_PHRASES.get(re.sub(r"\s+", " ", key))


class HandsFreeLoop:
    def __init__(self, app: Application, stop: threading.Event) -> None:
        self.app = app
        self._stop = stop
        self._cfg = app.settings.wake_word
        self._pause_after_speech = app.settings.audio.post_speech_pause_s

    def _info(self, message: str) -> None:
        self.app.events.emit(ev.INFO, message=message)

    async def run(self) -> None:
        """Läuft, bis ``stop`` gesetzt wird oder das Mikrofon dauerhaft ausfällt. Wirft nie."""
        try:
            if not await self._prepare():
                return
            await self._loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Hands-free-Schleife abgebrochen")
            self.app.report_error(f"Hands-free beendet wegen Fehler: {type(exc).__name__}: {exc}")
        finally:
            self.app.set_state("bereit")
            self._info("Hands-free beendet.")

    async def _prepare(self) -> bool:
        app, voice = self.app, self.app.voice
        self._info("Lade Spracherkennung und Stimme … (beim ersten Mal Download)")
        try:
            warning = await voice.preload()
        except (STTError, AudioError) as exc:
            app.report_error(f"Hands-free nicht möglich: {exc}")
            return False
        if self._cfg.engine == "openwakeword":
            try:
                await asyncio.to_thread(app.wake_detector.preload)
            except Exception as exc:
                app.report_error(f"openWakeWord nicht verfügbar: {exc}")
                return False
        if warning:
            app.report_error(warning)
        if app.settings.audio.calibrate_noise:
            self._info("Messe Hintergrundrauschen – bitte kurz still sein …")
            try:
                await voice.calibrate(self._stop)
            except AudioError as exc:
                app.report_error(f"Hands-free nicht möglich: {exc}")
                return False
        phrase = self._cfg.phrase.capitalize()
        self._info(f"Hands-free aktiv – sag „{phrase}“ und deine Frage. (/listen schaltet es aus)")
        return True

    async def _loop(self) -> None:
        app, voice, wake = self.app, self.app.voice, self.app.wake
        failures = 0
        followup_until = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            in_followup = now < followup_until
            app.set_state(STATE_FOLLOWUP if in_followup else STATE_WAITING)
            try:
                audio = await voice.capture(self._stop, followup_until - now if in_followup else None)
                if audio is None:
                    continue  # Folgefenster abgelaufen oder gestoppt
                wake_detected = False
                if not in_followup and self._cfg.engine == "openwakeword":
                    wake_detected = await asyncio.to_thread(app.wake_detector.process, audio)
                text = await voice.transcribe(audio)
                failures = 0
            except (AudioError, STTError) as exc:
                failures += 1
                app.report_error(str(exc))
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    app.report_error(
                        "Mikrofon/Spracherkennung fällt wiederholt aus – Hands-free wird beendet."
                    )
                    return
                await asyncio.sleep(RETRY_DELAY_S)
                continue
            if not text:
                continue
            if (
                not in_followup
                and self._cfg.engine == "openwakeword"
                and not wake_detected
                and wake.match(text) is None
            ):
                # Das mitgelieferte Modell erkennt „Hey Jarvis“, nicht zuverlässig nur „Jarvis“.
                # Whisper dient daher als gezielter Fallback für verpasste Modell-Treffer.
                log.debug("Wake-Word-Modell ohne Treffer; Transkript enthält kein Wake Word")
                continue

            matched = wake.match(text)
            if matched is None and not in_followup and self._cfg.engine != "openwakeword":
                log.debug("Äußerung ohne Wake Word verworfen")
                continue
            command = (text if matched is None else matched).strip()
            if not command:
                command = await self._ask_for_command()
                if not command:
                    continue

            if not await self._handle(command):
                followup_until = 0.0
                continue
            followup_until = time.monotonic() + self._cfg.followup_window_s
            await asyncio.sleep(self._pause_after_speech)

    async def _ask_for_command(self) -> str:
        """Nur das Wake Word wurde gesagt: bestätigen und auf den Befehl warten."""
        app, voice = self.app, self.app.voice
        await app.acknowledge(self._cfg.ack_phrase)
        await asyncio.sleep(self._pause_after_speech)
        app.set_state("hört zu")
        try:
            audio = await voice.capture(self._stop, self._cfg.listen_timeout_s)
            if audio is None:
                return ""
            text = await voice.transcribe(audio)
        except (AudioError, STTError) as exc:
            app.report_error(str(exc))
            return ""
        matched = app.wake.match(text)
        return (text if matched is None else matched).strip()

    async def _handle(self, command: str) -> bool:
        """Führt einen Befehl aus. False = Gespräch beenden (kein Folgefenster)."""
        farewell = closing_reply(command)
        if farewell is not None:
            self.app.events.emit(ev.USER_TEXT, text=command, source="voice")
            await self.app.say(farewell)
            return False
        interrupt_stop = threading.Event()
        interrupt_task = None
        if self._cfg.engine in ("openwakeword", "whisper") and self.app.voice.can_listen:
            interrupt_task = asyncio.create_task(self._wait_for_interrupt(interrupt_stop))
        try:
            reply = await self.app.process(command, source="voice", confirm=self._confirm_tool)
        finally:
            interrupt_stop.set()
        interrupted_text = await interrupt_task if interrupt_task is not None else None
        if interrupted_text:
            matched = self.app.wake.match(interrupted_text)
            next_command = (interrupted_text if matched is None else matched).strip()
            if not next_command:
                next_command = await self._ask_for_command()
            if next_command:
                reply = await self.app.process(next_command, source="voice", confirm=self._confirm_tool)
        return reply is not None

    async def _wait_for_interrupt(self, stop: threading.Event) -> str | None:
        """Während der Antwort auf ein echtes Audio-Wake-Word warten und TTS abbrechen."""
        app, voice = self.app, self.app.voice
        while not stop.is_set():
            try:
                audio = await voice.capture(stop)
                if audio is None or stop.is_set():
                    return None
                if self._cfg.engine == "openwakeword":
                    detected = await asyncio.to_thread(app.wake_detector.process, audio)
                    text = await voice.transcribe(audio) if detected else ""
                else:
                    text = await voice.transcribe(audio)
                    detected = app.wake.match(text) is not None
                if detected:
                    app.voice.stop_speaking()
                    return text
            except (AudioError, STTError) as exc:
                if not stop.is_set():
                    app.report_error(f"Sprachunterbrechung fehlgeschlagen: {exc}")
                return None
            except Exception:
                if not stop.is_set():
                    log.exception("Sprachunterbrechung fehlgeschlagen")
                return None
        return None

    async def _confirm_tool(self, tool: Tool, args: dict[str, Any]) -> bool:
        """Fragt bei bestätigungspflichtigen Tools per Sprache nach."""
        app, voice = self.app, self.app.voice
        if not voice.can_listen:
            self._info(f"Aktion '{tool.name}' abgelehnt: keine Spracheingabe für Bestätigung verfügbar.")
            return False

        details = json.dumps(args, ensure_ascii=False)
        # self._info(f"Bestätigung erforderlich: {tool.name} {details}")
        if voice.can_speak:
            await voice.speak(f"Soll ich {tool.name} mit {details} ausführen? Bitte sag Ja oder Nein.")
        await asyncio.sleep(self._pause_after_speech)
        app.set_state("hört zu")
        try:
            audio = await voice.capture(self._stop, self._cfg.listen_timeout_s)
            if audio is None:
                return False
            answer = (await voice.transcribe(audio)).casefold()
        except (AudioError, STTError) as exc:
            app.report_error(f"Bestätigung nicht erkannt: {exc}")
            return False
        finally:
            app.set_state("denkt nach")

        words = set(re.findall(r"\w+", answer))
        if words & {"nein", "abbrechen", "stopp", "stop"}:
            return False
        return bool(words & {"ja", "genau", "richtig", "bestätige", "bestätigen"})
