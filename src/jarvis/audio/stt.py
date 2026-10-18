"""Speech-to-Text hinter einer austauschbaren Schnittstelle."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from jarvis.audio.errors import STTError
from jarvis.config.loader import STTConfig

log = logging.getLogger(__name__)

# Typische Whisper-Halluzinationen bei Stille/Rauschen (kleine, erweiterbare Liste).
_HALLUCINATIONS = (
    "untertitel der amara.org-community",
    "untertitel im auftrag des zdf",
    "vielen dank fürs zuschauen",
    "danke fürs zuschauen",
    "copyright wdr",
)


def clean_transcript(text: str) -> str:
    """Normalisiert Leerraum und verwirft bekannte Halluzinationen."""
    text = re.sub(r"\s+", " ", text).strip()
    lowered = text.lower().strip(" .!")
    if any(lowered == h or lowered.startswith(h) for h in _HALLUCINATIONS):
        log.info("Whisper-Halluzination verworfen: %r", text)
        return ""
    return text


class STTEngine(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio: Any, sample_rate: int) -> str:
        """Wandelt Mono-float32-Audio (16 kHz erwartet) in Text. Blockierend."""

    @abstractmethod
    def status(self) -> str:
        """Kurzer Statustext für die Oberfläche."""

    def preload(self) -> None:  # noqa: B027 - optionaler Hook
        """Lädt Modelle vorab (blockierend), damit die erste Erkennung nicht verzögert ist."""


class FasterWhisperSTT(STTEngine):
    """faster-whisper (CTranslate2). Das Modell wird beim ersten Gebrauch geladen.

    Der erste Start lädt die Modellgewichte einmalig aus dem Internet (Hugging Face) nach
    ``stt.model_dir``; danach läuft alles offline.
    """

    name = "faster-whisper"

    def __init__(self, cfg: STTConfig, model_dir: Path) -> None:
        self._cfg = cfg
        self._model_dir = model_dir
        self._model: Any = None
        self._loaded = False

    def preload(self) -> None:
        self._ensure_model()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError("faster-whisper fehlt. Installiere es mit: pip install -e .[audio]") from exc
        self._model_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "Lade Whisper-Modell '%s' (%s/%s) …", self._cfg.model, self._cfg.device, self._cfg.compute_type
        )
        try:
            self._model = WhisperModel(
                self._cfg.model,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
                download_root=str(self._model_dir),
            )
        except Exception as exc:
            raise STTError(
                f"Whisper-Modell '{self._cfg.model}' konnte nicht geladen werden: {exc}. "
                "Beim ersten Start ist eine Internetverbindung für den Download nötig."
            ) from exc
        self._loaded = True
        return self._model

    def transcribe(self, audio: Any, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise STTError(f"Whisper erwartet 16 kHz, erhalten: {sample_rate} Hz.")
        model = self._ensure_model()
        try:
            segments, _info = model.transcribe(
                audio,
                language=self._cfg.language,
                beam_size=self._cfg.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments)
        except Exception as exc:
            raise STTError(f"Spracherkennung fehlgeschlagen: {exc}") from exc
        return clean_transcript(text)

    def status(self) -> str:
        state = "geladen" if self._loaded else "lädt beim ersten Gebrauch"
        return f"{self.name} ({self._cfg.model}, {self._cfg.device}) – {state}"


def create_stt(cfg: STTConfig, model_dir: Path) -> STTEngine:
    """Fabrik. Weitere Engines hier registrieren."""
    if cfg.engine == "faster-whisper":
        return FasterWhisperSTT(cfg, model_dir)
    raise STTError(f"Unbekannte STT-Engine '{cfg.engine}'. Verfügbar: faster-whisper")
