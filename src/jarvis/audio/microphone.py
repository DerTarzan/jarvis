"""Mikrofonaufnahme und Erkennung des Äußerungsendes (ohne Spracherkennung)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from jarvis.audio.errors import AudioError

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

MIN_RECORDING_S = 0.3
BLOCK_S = 0.05
PREROLL_S = 0.4  # Audio vor dem erkannten Sprechbeginn, damit der Wortanfang nicht fehlt
MIN_SPEECH_S = 0.25  # kürzere Geräusche (Klicken, Husten) werden ignoriert


def threshold_from_ambient(
    base: float, ambient_rms: float, factor: float = 3.0, ceiling: float = 0.2
) -> float:
    """Sprachschwelle aus gemessenem Hintergrundrauschen: mindestens ``base``, höchstens ``ceiling``."""
    return min(max(base, ambient_rms * factor), ceiling)


def _import_audio() -> tuple[Any, Any]:
    try:
        import numpy
        import sounddevice
    except ImportError as exc:
        raise AudioError("Audio-Pakete fehlen. Installiere sie mit: pip install -e .[audio]") from exc
    except OSError as exc:  # sounddevice wirft OSError, wenn die PortAudio-Bibliothek fehlt
        raise AudioError(f"Audio-Backend (PortAudio) nicht verfügbar: {exc}") from exc
    return numpy, sounddevice


def list_devices() -> list[str]:
    """Lesbare Liste aller Audiogeräte (für die Konfiguration von input_device/output_device)."""
    _, sd = _import_audio()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise AudioError(f"Audiogeräte konnten nicht abgefragt werden: {exc}") from exc
    return [
        f"[{i}] {d['name']} (Eingang: {d['max_input_channels']}, Ausgang: {d['max_output_channels']})"
        for i, d in enumerate(devices)
    ]


class SilenceEndpointer:
    """Erkennt das Ende einer Äußerung anhand der Lautstärke (RMS) pro Block.

    Erst nach erkanntem Sprechbeginn zählt Stille. ``feed`` gibt True zurück, sobald
    ``silence_duration_s`` Stille nach Sprache vergangen ist.
    """

    def __init__(self, threshold: float, silence_duration_s: float, block_s: float = BLOCK_S) -> None:
        self._threshold = threshold
        self._needed_blocks = max(1, round(silence_duration_s / block_s))
        self._silent_blocks = 0
        self.speech_started = False

    def feed(self, rms: float) -> bool:
        if rms >= self._threshold:
            self.speech_started = True
            self._silent_blocks = 0
            return False
        if self.speech_started:
            self._silent_blocks += 1
            return self._silent_blocks >= self._needed_blocks
        return False


class Recorder:
    """Nimmt Mono-Audio (float32, [-1, 1]) vom konfigurierten Mikrofon auf."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16000,
        silence_threshold: float = 0.012,
        silence_duration_s: float = 1.0,
        max_record_s: float = 30.0,
    ) -> None:
        self._device = device
        self.sample_rate = sample_rate
        self._base_threshold = silence_threshold
        self._threshold = silence_threshold
        self._silence_duration = silence_duration_s
        self._max_s = max_record_s

    def record(self, stop: threading.Event, use_silence_detection: bool = False) -> np.ndarray:
        """Nimmt auf, bis ``stop`` gesetzt wird (Push-to-Talk), bei Stille oder max. Dauer.

        Blockierend – aus async-Code via ``asyncio.to_thread`` aufrufen.

        Raises:
            AudioError: kein Mikrofon, Gerätefehler oder leere Aufnahme.
        """
        np, sd = _import_audio()
        frames: list[Any] = []
        endpointer = SilenceEndpointer(self._threshold, self._silence_duration)
        blocksize = int(self.sample_rate * BLOCK_S)
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                device=self._device,
            ) as stream:
                started = time.monotonic()
                while not stop.is_set():
                    if time.monotonic() - started > self._max_s:
                        log.info("Maximale Aufnahmedauer erreicht")
                        break
                    block, overflowed = stream.read(blocksize)
                    if overflowed:
                        log.warning("Audio-Puffer übergelaufen")
                    mono = block[:, 0].copy()
                    frames.append(mono)
                    rms = float(np.sqrt(np.mean(np.square(mono))))
                    if use_silence_detection and endpointer.feed(rms):
                        break
        except sd.PortAudioError as exc:
            raise AudioError(
                f"Mikrofon nicht nutzbar ({exc}). Prüfe audio.input_device und die "
                "Windows-Datenschutzeinstellungen für das Mikrofon."
            ) from exc
        except Exception as exc:
            raise AudioError(f"Aufnahmefehler: {exc}") from exc

        if not frames:
            raise AudioError("Aufnahme leer.")
        audio = np.concatenate(frames)
        if len(audio) / self.sample_rate < MIN_RECORDING_S:
            raise AudioError("Aufnahme zu kurz.")
        if use_silence_detection and not endpointer.speech_started:
            raise AudioError("Keine Sprache erkannt.")
        return audio

    def _open_stream(self, sd: Any, blocksize: int) -> Any:
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=self._device,
        )

    @staticmethod
    def _wrap_device_error(exc: Exception, sd: Any) -> AudioError:
        if isinstance(exc, sd.PortAudioError):
            return AudioError(
                f"Mikrofon nicht nutzbar ({exc}). Prüfe audio.input_device und die "
                "Windows-Datenschutzeinstellungen für das Mikrofon."
            )
        return AudioError(f"Aufnahmefehler: {exc}")

    def calibrate(self, stop: threading.Event, seconds: float = 1.0) -> float:
        """Misst das Hintergrundrauschen und setzt die Sprachschwelle passend. Blockierend.

        Returns:
            Die neue Schwelle (RMS).
        """
        np, sd = _import_audio()
        blocksize = int(self.sample_rate * BLOCK_S)
        levels: list[float] = []
        try:
            with self._open_stream(sd, blocksize) as stream:
                while not stop.is_set() and len(levels) < round(seconds / BLOCK_S):
                    block, _ = stream.read(blocksize)
                    levels.append(float(np.sqrt(np.mean(np.square(block[:, 0])))))
        except Exception as exc:
            raise self._wrap_device_error(exc, sd) from exc
        if levels:
            ambient = float(np.percentile(levels, 90))
            self._threshold = threshold_from_ambient(self._base_threshold, ambient)
            log.info("Rauschpegel %.4f → Sprachschwelle %.4f", ambient, self._threshold)
        return self._threshold

    def capture_utterance(
        self,
        stop: threading.Event,
        wait_timeout_s: float | None = None,
        min_speech_s: float = MIN_SPEECH_S,
    ) -> np.ndarray | None:
        """Wartet auf Sprache und nimmt genau eine Äußerung auf (Ende per Stille). Blockierend.

        Im Wartezustand ist die CPU-Last minimal (nur Lautstärkemessung pro 50-ms-Block).
        Kurze Geräusche unter ``min_speech_s`` werden verworfen; ein kurzer Vorlauf (Pre-Roll)
        verhindert abgeschnittene Wortanfänge.

        Returns:
            Das Audio (float32, Mono) oder ``None``, wenn ``stop`` gesetzt wurde oder innerhalb
            von ``wait_timeout_s`` keine Sprache begann.

        Raises:
            AudioError: Mikrofon/Backend nicht nutzbar.
        """
        np, sd = _import_audio()
        blocksize = int(self.sample_rate * BLOCK_S)
        needed_silent = max(1, round(self._silence_duration / BLOCK_S))
        preroll: deque[Any] = deque(maxlen=max(1, round(PREROLL_S / BLOCK_S)))
        frames: list[Any] = []
        speech_blocks = silent_blocks = 0
        started = completed = False
        began_at = 0.0
        deadline = None if wait_timeout_s is None else time.monotonic() + wait_timeout_s
        try:
            with self._open_stream(sd, blocksize) as stream:
                while not stop.is_set():
                    now = time.monotonic()
                    if not started and deadline is not None and now >= deadline:
                        return None
                    block, overflowed = stream.read(blocksize)
                    if overflowed:
                        log.warning("Audio-Puffer übergelaufen")
                    mono = block[:, 0].copy()
                    loud = float(np.sqrt(np.mean(np.square(mono)))) >= self._threshold
                    if not started:
                        if loud:
                            started, began_at = True, now
                            frames = [*preroll, mono]
                            speech_blocks, silent_blocks = 1, 0
                        else:
                            preroll.append(mono)
                        continue
                    frames.append(mono)
                    if loud:
                        speech_blocks += 1
                        silent_blocks = 0
                    else:
                        silent_blocks += 1
                    if now - began_at > self._max_s:
                        completed = True
                        break
                    if silent_blocks >= needed_silent:
                        if speech_blocks * BLOCK_S < min_speech_s:  # nur ein Geräusch: weiter warten
                            started = False
                            frames = []
                            speech_blocks = silent_blocks = 0
                            preroll.clear()
                            continue
                        completed = True
                        break
        except Exception as exc:
            raise self._wrap_device_error(exc, sd) from exc
        if not completed or not frames:
            return None
        return np.concatenate(frames)
