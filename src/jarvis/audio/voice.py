"""Koordiniert Mikrofon, STT und TTS zu einer Sprach-Ein-/Ausgabe (ohne LLM-Wissen)."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator
from typing import Any

from jarvis.audio.errors import AudioError, STTError, TTSError
from jarvis.audio.microphone import Recorder
from jarvis.audio.stt import STTEngine, create_stt
from jarvis.audio.tts import Speaker, create_tts
from jarvis.config.loader import Settings

log = logging.getLogger(__name__)


class VoiceIO:
    """Jede Komponente fällt einzeln aus: Fehler beim Aufbau werden als Status gemeldet."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.stt: STTEngine | None = None
        self.speaker: Speaker | None = None
        self.recorder: Recorder | None = None
        self.stt_status = "nicht initialisiert"
        self.tts_status = "nicht initialisiert"
        self._engine: Any = None
        self._build()

    def _build(self) -> None:
        s = self._settings
        try:
            self.stt = create_stt(s.stt, s.resolve(s.stt.model_dir))
            self.recorder = Recorder(
                device=s.audio.input_device,
                sample_rate=s.audio.sample_rate,
                silence_threshold=s.audio.silence_threshold,
                silence_duration_s=s.audio.silence_duration_s,
                max_record_s=s.audio.max_record_s,
            )
            self.stt_status = self.stt.status()
        except STTError as exc:
            self.stt_status = f"Fehler: {exc}"
            log.error("STT-Aufbau fehlgeschlagen: %s", exc)
        try:
            self._engine = create_tts(s.tts, s.resolve(s.tts.model_dir))
            self.speaker = Speaker(self._engine, s.audio.output_device, s.tts.sentence_pause_s)
            self.tts_status = self._engine.status()
        except TTSError as exc:
            self.tts_status = f"Fehler: {exc}"
            log.error("TTS-Aufbau fehlgeschlagen: %s", exc)

    def refresh_status(self) -> None:
        if self.stt is not None:
            self.stt_status = self.stt.status()
        if self._engine is not None:
            self.tts_status = self._engine.status()

    @property
    def can_listen(self) -> bool:
        return self.stt is not None and self.recorder is not None

    @property
    def can_speak(self) -> bool:
        return self.speaker is not None

    # -- Vorbereitung ----------------------------------------------------------
    async def preload(self) -> str | None:
        """Lädt STT-Modell und Stimme vorab (kann beim ersten Mal Downloads auslösen).

        Returns:
            Warntext, falls nur die Sprachausgabe nicht bereit ist; sonst ``None``.

        Raises:
            STTError: Spracherkennung nicht nutzbar (dann ist Hands-free nicht möglich).
        """
        if not self.can_listen:
            raise STTError(f"Spracheingabe nicht verfügbar ({self.stt_status}).")
        assert self.stt is not None
        try:
            await asyncio.to_thread(self.stt.preload)
        finally:
            self.refresh_status()
        warning = None
        if self._engine is not None:
            try:
                await asyncio.to_thread(self._engine.preload)
            except TTSError as exc:
                self.speaker = None
                self.tts_status = f"Fehler: {exc}"
                warning = f"Sprachausgabe nicht verfügbar, Antworten erscheinen nur als Text: {exc}"
            finally:
                if self.speaker is not None:
                    self.refresh_status()
        return warning

    async def calibrate(self, stop: threading.Event, seconds: float = 1.0) -> float:
        """Misst Hintergrundrauschen (Mikrofon). Raises AudioError."""
        if self.recorder is None:
            raise AudioError("Kein Mikrofon konfiguriert.")
        return await asyncio.to_thread(self.recorder.calibrate, stop, seconds)

    # -- Eingabe -----------------------------------------------------------------
    async def listen(self, stop: threading.Event, use_silence_detection: bool = False) -> str:
        """Nimmt auf (bis ``stop`` gesetzt oder Stille) und erkennt den Text (Push-to-Talk).

        Raises:
            AudioError, STTError: mit nutzerverständlicher Meldung.
        """
        if not self.can_listen:
            raise STTError(f"Spracheingabe nicht verfügbar ({self.stt_status}).")
        assert self.recorder is not None and self.stt is not None
        audio = await asyncio.to_thread(self.recorder.record, stop, use_silence_detection)
        text = await asyncio.to_thread(self.stt.transcribe, audio, self.recorder.sample_rate)
        self.refresh_status()
        return text

    async def capture(self, stop: threading.Event, wait_timeout_s: float | None = None) -> Any | None:
        """Wartet auf eine Äußerung und nimmt sie auf (Hands-free). ``None`` bei Stop/Timeout."""
        if self.recorder is None:
            raise AudioError("Kein Mikrofon konfiguriert.")
        return await asyncio.to_thread(self.recorder.capture_utterance, stop, wait_timeout_s)

    async def transcribe(self, audio: Any) -> str:
        """Erkennt Text in zuvor aufgenommenem Audio."""
        if self.stt is None or self.recorder is None:
            raise STTError(f"Spracheingabe nicht verfügbar ({self.stt_status}).")
        text = await asyncio.to_thread(self.stt.transcribe, audio, self.recorder.sample_rate)
        self.refresh_status()
        return text

    # -- Ausgabe ---------------------------------------------------------------
    async def speak(self, text: str) -> None:
        if self.speaker is None:
            raise TTSError(f"Sprachausgabe nicht verfügbar ({self.tts_status}).")
        try:
            await asyncio.to_thread(self.speaker.speak, text)
        finally:
            self.refresh_status()

    async def speak_stream(self, chunks: AsyncIterator[str]) -> None:
        """Synthesisiert und spricht vollständige Sätze, sobald sie aus dem LLM eintreffen."""
        buffer = ""
        async for chunk in chunks:
            buffer += chunk
            parts = re.split(r"(?<=[.!?…])\s+", buffer)
            buffer = parts.pop()
            for sentence in parts:
                if sentence.strip():
                    await self.speak(sentence)
        if buffer.strip():
            await self.speak(buffer)

    async def beep(self) -> None:
        if self.speaker is None:
            raise TTSError(f"Sprachausgabe nicht verfügbar ({self.tts_status}).")
        await asyncio.to_thread(self.speaker.beep)

    def stop_speaking(self) -> None:
        if self.speaker is not None:
            self.speaker.stop()

    def close(self) -> None:
        """Beendet laufende Aus-/Eingabe und gibt Audioressourcen frei."""
        self.stop_speaking()


__all__ = ["AudioError", "VoiceIO"]
