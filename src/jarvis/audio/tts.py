"""Text-to-Speech (Synthese) und Wiedergabe – getrennt und austauschbar."""

from __future__ import annotations

import logging
import queue
import re
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.audio.errors import AudioError, TTSError
from jarvis.config.loader import TTSConfig

log = logging.getLogger(__name__)


@dataclass
class AudioClip:
    samples: Any  # numpy float32, Mono
    sample_rate: int


def prepare_text_for_speech(text: str) -> str:
    """Entfernt Markdown-Reste und Emojis, die schlecht vorgelesen werden."""
    text = re.sub(r"```.*?```", " Codeblock. ", text, flags=re.DOTALL)
    text = re.sub(r"[`*_#>~]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Grobe Satztrennung, damit die Wiedergabe schon vor dem Ende der Synthese beginnt."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p]


class TTSEngine(ABC):
    name: str

    @abstractmethod
    def synthesize(self, text: str) -> AudioClip:
        """Erzeugt Audio für einen kurzen Text. Blockierend."""

    @abstractmethod
    def status(self) -> str: ...

    def preload(self) -> None:  # noqa: B027 - optionaler Hook
        """Lädt Stimme vorab (blockierend), damit die erste Ausgabe nicht verzögert ist."""


class PiperTTS(TTSEngine):
    """Piper (lokal, ONNX). Stimme standardmäßig 'de_DE-thorsten-high'.

    Die Stimmdateien (.onnx + .onnx.json) werden beim ersten Gebrauch einmalig nach
    ``tts.model_dir`` heruntergeladen.
    """

    name = "piper"

    def __init__(self, cfg: TTSConfig, model_dir: Path) -> None:
        self._cfg = cfg
        self._dir = model_dir
        self._voice: Any = None

    @property
    def _onnx_path(self) -> Path:
        return self._dir / f"{self._cfg.voice}.onnx"

    def _ensure_voice_files(self) -> None:
        if self._onnx_path.is_file() and self._onnx_path.with_suffix(".onnx.json").is_file():
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        log.info("Lade Piper-Stimme '%s' herunter …", self._cfg.voice)
        cmd = [
            sys.executable,
            "-m",
            "piper.download_voices",
            self._cfg.voice,
            "--download-dir",
            str(self._dir),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        except (subprocess.SubprocessError, OSError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise TTSError(
                f"Stimme '{self._cfg.voice}' nicht vorhanden und Download fehlgeschlagen: "
                f"{detail.strip()[-300:]}"
            ) from exc
        if not self._onnx_path.is_file():
            raise TTSError(f"Stimmdatei nach Download nicht gefunden: {self._onnx_path}")

    def _ensure_voice(self) -> Any:
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise TTSError("piper-tts fehlt. Installiere es mit: pip install -e .[audio]") from exc
        self._ensure_voice_files()
        try:
            self._voice = PiperVoice.load(str(self._onnx_path))
        except Exception as exc:
            raise TTSError(f"Stimme '{self._cfg.voice}' konnte nicht geladen werden: {exc}") from exc
        return self._voice

    def preload(self) -> None:
        self._ensure_voice()

    def synthesize(self, text: str) -> AudioClip:
        import numpy as np

        voice = self._ensure_voice()
        try:
            from piper import SynthesisConfig

            syn = SynthesisConfig(length_scale=1.0 / self._cfg.speed, volume=self._cfg.volume)
            chunks = list(voice.synthesize(text, syn_config=syn))
        except Exception as exc:
            raise TTSError(f"Sprachsynthese fehlgeschlagen: {exc}") from exc
        if not chunks:
            raise TTSError("Sprachsynthese lieferte kein Audio.")
        parts = []
        for chunk in chunks:
            arr = getattr(chunk, "audio_float_array", None)
            if arr is None:
                arr = chunk.audio_int16_array.astype(np.float32) / 32768.0
            parts.append(np.asarray(arr, dtype=np.float32))
        return AudioClip(np.concatenate(parts), int(chunks[0].sample_rate))

    def status(self) -> str:
        state = "geladen" if self._voice is not None else "lädt beim ersten Gebrauch"
        return f"{self.name} ({self._cfg.voice}) – {state}"


def create_tts(cfg: TTSConfig, model_dir: Path) -> TTSEngine:
    """Fabrik. Weitere Engines hier registrieren."""
    if cfg.engine == "piper":
        return PiperTTS(cfg, model_dir)
    raise TTSError(f"Unbekannte TTS-Engine '{cfg.engine}'. Verfügbar: piper")


class Speaker:
    """Spricht Text satzweise: Synthese und Wiedergabe laufen parallel, unterbrechbar."""

    def __init__(
        self, engine: TTSEngine, output_device: int | str | None = None, pause_s: float = 0.2
    ) -> None:
        self._engine = engine
        self._device = output_device
        self._pause_s = pause_s
        self._stop = threading.Event()
        self._cache: dict[str, AudioClip] = {}  # kurze, wiederkehrende Sätze ("Ja?") sofort parat

    def _synthesize_cached(self, sentence: str) -> AudioClip:
        clip = self._cache.get(sentence)
        if clip is None:
            clip = self._engine.synthesize(sentence)
            if len(sentence) <= 40 and len(self._cache) < 32:
                self._cache[sentence] = clip
        return clip

    def beep(self) -> None:
        """Kurzer Bestätigungston (blockierend). Fehler werden als AudioError gemeldet."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("Audio-Pakete fehlen. Installiere: pip install -e .[audio]") from exc
        except OSError as exc:
            raise AudioError(f"Audio-Backend (PortAudio) nicht verfügbar: {exc}") from exc
        rate = 24000
        t = np.linspace(0, 0.12, int(rate * 0.12), endpoint=False)
        tone = (0.2 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)
        try:
            sd.play(tone, rate, device=self._device)
            sd.wait()
        except sd.PortAudioError as exc:
            raise AudioError(f"Audioausgabe nicht möglich: {exc}") from exc

    def stop(self) -> None:
        """Bricht laufende Sprachausgabe ab (threadsicher)."""
        self._stop.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # Audio-Stack evtl. nicht verfügbar – dann gibt es nichts zu stoppen
            log.debug("sd.stop() nicht möglich", exc_info=True)

    def speak(self, text: str) -> None:
        """Blockierend (aus async via ``asyncio.to_thread`` aufrufen)."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("Audio-Pakete fehlen. Installiere: pip install -e .[audio]") from exc
        except OSError as exc:  # sounddevice wirft OSError, wenn die PortAudio-Bibliothek fehlt
            raise AudioError(f"Audio-Backend (PortAudio) nicht verfügbar: {exc}") from exc

        sentences = split_sentences(prepare_text_for_speech(text))
        if not sentences:
            return
        self._stop.clear()
        clips: queue.Queue[AudioClip | Exception | None] = queue.Queue(maxsize=2)

        def producer() -> None:
            try:
                for sentence in sentences:
                    if self._stop.is_set():
                        break
                    clips.put(self._synthesize_cached(sentence))
            except Exception as exc:
                clips.put(exc)
            finally:
                clips.put(None)

        thread = threading.Thread(target=producer, name="tts-producer", daemon=True)
        thread.start()
        try:
            while not self._stop.is_set():
                item = clips.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    if isinstance(item, TTSError):
                        raise item
                    raise TTSError(str(item)) from item
                pause = np.zeros(int(item.sample_rate * self._pause_s), dtype=np.float32)
                data = np.concatenate([item.samples, pause])
                try:
                    sd.play(data, item.sample_rate, device=self._device)
                    while sd.get_stream().active and not self._stop.is_set():
                        sd.sleep(30)
                except sd.PortAudioError as exc:
                    raise AudioError(f"Audioausgabe nicht möglich: {exc}") from exc
        finally:
            self._stop.set()  # Producer beenden, falls vorzeitig verlassen
            try:
                while True:
                    clips.get_nowait()
            except queue.Empty:
                pass
            thread.join(timeout=2)
