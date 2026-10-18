"""Wake-Word-Erkennung per lokalem Audio-Modell oder Whisper-Transkript."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from jarvis.audio.errors import WakeWordError
from jarvis.config.loader import WakeWordConfig

# Wake Word muss unter den ersten Wörtern stehen ("Jarvis, …", "Hey Jarvis, …"), damit ein
# Satz wie "Ich habe gestern Jarvis gebaut" nicht auslöst.
MAX_PREFIX_WORDS = 2

# Schreibweisen, die Whisper bei "Jarvis" erfahrungsgemäß liefern kann (erweiterbar per Config).
_JARVIS_VARIANTS = frozenset(
    {"jarvis", "dscharvis", "tscharvis", "scharvis", "charvis", "jarwis", "jarvies", "jarvice", "jarvus"}
)


def _normalize(word: str) -> str:
    return re.sub(r"[^\wäöüß]", "", word.lower())


class TranscriptWakeWord:
    """Sucht das Wake Word am Anfang eines erkannten Textes (tolerant gegenüber Schreibweisen)."""

    def __init__(self, cfg: WakeWordConfig) -> None:
        self._phrase = _normalize(cfg.phrase)
        if not self._phrase:
            raise WakeWordError("wake_word.phrase darf nicht leer sein.")
        variants = {_normalize(v) for v in cfg.variants} - {""}
        if self._phrase == "jarvis":
            variants |= _JARVIS_VARIANTS
        self._variants = variants | {self._phrase}
        self._threshold = cfg.match_threshold

    def _is_wake(self, word: str) -> bool:
        return word in self._variants or SequenceMatcher(None, word, self._phrase).ratio() >= self._threshold

    def match(self, text: str) -> str | None:
        """Prüft den Text auf das Wake Word.

        Returns:
            ``None`` – kein Wake Word (Äußerung ignorieren);
            ``""`` – nur das Wake Word („Jarvis“), Befehl folgt separat;
            sonst der Befehl hinter dem Wake Word.
        """
        tokens = text.split()
        for index, token in enumerate(tokens[:MAX_PREFIX_WORDS]):
            word = _normalize(token)
            if word and self._is_wake(word):
                return " ".join(tokens[index + 1 :]).lstrip(" ,.:;!?-–—").strip()
        return None


class WakeWordDetector(ABC):
    """Schnittstelle für audio-basierte Wake-Word-Detektoren."""

    name: str

    @abstractmethod
    def process(self, frame: Any) -> bool:
        """Verarbeitet eine Mono-float32-Aufnahme mit 16 kHz."""

    def reset(self) -> None:  # noqa: B027 - optionaler Hook
        """Setzt internen Zustand zurück (nach jeder Aktivierung)."""


class NullWakeWord(WakeWordDetector):
    """Platzhalter, der nie auslöst."""

    name = "none"

    def process(self, frame: Any) -> bool:
        return False


class OpenWakeWordDetector(WakeWordDetector):
    """openWakeWord-Detektor für vollständige Aufnahmen."""

    name = "openwakeword"

    def __init__(self, cfg: WakeWordConfig, model_dir: Path | None = None) -> None:
        self._model_name = cfg.model
        self._threshold = cfg.threshold
        self._model_dir = model_dir or Path(cfg.model_dir)
        self._model: Any = None

    def preload(self) -> None:
        if self._model is not None:
            return
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise WakeWordError(
                "openWakeWord fehlt. Installiere es mit: pip install -e .[wake-word]"
            ) from exc
        try:
            self._model_dir.mkdir(parents=True, exist_ok=True)
            download_models(
                model_names=[self._model_name],
                target_directory=str(self._model_dir),
            )
            self._model = Model(
                wakeword_models=[str(self._model_dir / f"{self._model_name}_v0.1.onnx")],
                inference_framework="onnx",
                melspec_model_path=str(self._model_dir / "melspectrogram.onnx"),
                embedding_model_path=str(self._model_dir / "embedding_model.onnx"),
            )
        except Exception as exc:
            raise WakeWordError(
                f"openWakeWord-Modell '{self._model_name}' konnte nicht geladen werden: {exc}"
            ) from exc

    def process(self, frame: Any) -> bool:
        if self._model is None:
            self.preload()
        try:
            import numpy as np

            self._model.reset()
            samples = np.asarray(frame, dtype=np.float32).reshape(-1)
            if not len(samples):
                return False
            pcm = np.clip(samples * 32768, -32768, 32767).astype(np.int16)
            detected = False
            for offset in range(0, len(pcm), 1280):
                chunk = pcm[offset : offset + 1280]
                if len(chunk) < 1280:
                    chunk = np.pad(chunk, (0, 1280 - len(chunk)))
                scores = self._model.predict(chunk)
                detected = detected or any(float(score) >= self._threshold for score in scores.values())
            return detected
        except Exception as exc:
            raise WakeWordError(f"openWakeWord-Auswertung fehlgeschlagen: {exc}") from exc

    def detected(self, frame: Any) -> bool:
        return self.process(frame)

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()


def create_wake_word(cfg: WakeWordConfig, model_dir: Path | None = None) -> WakeWordDetector:
    """Fabrik für audio-basierte Detektoren."""
    if not cfg.enabled or cfg.engine in ("none", "whisper"):
        return NullWakeWord()
    if cfg.engine == "openwakeword":
        return OpenWakeWordDetector(cfg, model_dir)
    raise WakeWordError(
        f"Unbekannte Wake-Word-Engine '{cfg.engine}'. Verfügbar: whisper, openwakeword, none."
    )
