"""Laden und Validieren der zentralen Konfiguration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

DEFAULT_SYSTEM_PROMPT = (
    "Du bist {name}, ein persönlicher KI-Assistent. Antworte auf Deutsch, ruhig, präzise und "
    "höflich, mit trockenem Understatement. Halte Antworten kurz (meist 1-3 Sätze), da sie "
    "vorgelesen werden. Verwende keine Markdown-Formatierung, keine Listen mit Sternchen und "
    "keine Emojis. Wenn dir Informationen fehlen, frage kurz nach statt zu raten. Nutze "
    "bereitgestellte Werkzeuge nur, wenn sie für die Frage wirklich nötig sind."
)


class ConfigError(Exception):
    """Ungültige oder fehlende Konfiguration (mit verständlicher Meldung)."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneralConfig(_Section):
    assistant_name: str = "Jarvis"
    language: str = "de"
    # text: nur Tastatur | push_to_talk: Enter startet/stoppt Aufnahme |
    # handsfree: dauerhaft zuhören, Aktivierung per Wake Word (wie Alexa)
    activation_mode: Literal["text", "push_to_talk", "handsfree"] = "text"
    speak_replies: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class ProviderConfig(_Section):
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = Field(default=120.0, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)

    def api_key(self) -> str | None:
        """Liest den Schlüssel aus der Umgebung (nie aus der Konfigdatei)."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None


def _default_providers() -> dict[str, ProviderConfig]:
    return {
        "ollama": ProviderConfig(
            enabled=True,
            base_url="http://localhost:11434",
            model="llama3.1:8b",
            options={"temperature": 0.6, "num_ctx": 4096},
        ),
        "openai": ProviderConfig(api_key_env="OPENAI_API_KEY"),
        "anthropic": ProviderConfig(api_key_env="ANTHROPIC_API_KEY"),
        "gemini": ProviderConfig(api_key_env="GEMINI_API_KEY"),
        "openai_compatible": ProviderConfig(api_key_env="OPENAI_COMPATIBLE_API_KEY"),
    }


class AIConfig(_Section):
    provider: str = "ollama"
    providers: dict[str, ProviderConfig] = Field(default_factory=_default_providers)

    @model_validator(mode="after")
    def _check_selection(self) -> AIConfig:
        if self.provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "(keine)"
            raise ValueError(
                f"Provider '{self.provider}' ist nicht unter ai.providers definiert. Bekannt: {known}"
            )
        if not self.providers[self.provider].enabled:
            raise ValueError(
                f"Provider '{self.provider}' ist ausgewählt, aber 'enabled: false'. "
                "Aktiviere ihn bewusst oder wähle einen anderen Provider."
            )
        return self


class STTConfig(_Section):
    engine: str = "faster-whisper"
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "de"
    beam_size: int = Field(default=5, ge=1)
    model_dir: str = "models/stt"


class TTSConfig(_Section):
    engine: str = "piper"
    voice: str = "de_DE-thorsten-high"
    model_dir: str = "models/tts"
    speed: float = Field(default=0.95, gt=0.25, le=3.0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    sentence_pause_s: float = Field(default=0.2, ge=0.0, le=3.0)


class WakeWordConfig(_Section):
    enabled: bool = False
    # "whisper": Wake Word wird im lokal erkannten Text gesucht; "openwakeword": Audio-Modell
    # "none": aus.
    engine: str = "none"
    phrase: str = "jarvis"
    model: str = "hey_jarvis"
    model_dir: str = "models/wake_word"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Zusätzliche Schreibweisen, die Whisper für das Wake Word liefern könnte
    variants: list[str] = Field(default_factory=list)
    # Ähnlichkeitsschwelle (0.5-1.0) für unscharfen Vergleich; höher = strenger
    match_threshold: float = Field(default=0.75, ge=0.5, le=1.0)
    # So lange (Sekunden) nach einer Antwort kann man ohne Wake Word weitersprechen; 0 = aus
    followup_window_s: float = Field(default=8.0, ge=0.0, le=60.0)
    # Bestätigung nach dem Wake Word allein ("Jarvis"); leer = kurzer Signalton
    ack_phrase: str = "Ja?"
    # So lange wartet Jarvis nach der Bestätigung auf den Befehl
    listen_timeout_s: float = Field(default=6.0, gt=0, le=60.0)


class AudioConfig(_Section):
    input_device: int | str | None = None
    output_device: int | str | None = None
    sample_rate: int = Field(default=16000, ge=8000)
    silence_threshold: float = Field(default=0.012, gt=0)
    silence_duration_s: float = Field(default=1.0, gt=0)
    max_record_s: float = Field(default=30.0, gt=0)
    # Hintergrundrauschen beim Hands-free-Start messen und Schwelle anpassen
    calibrate_noise: bool = True
    # Pause nach der Sprachausgabe, bevor wieder zugehört wird (gegen Nachhall/Echo)
    post_speech_pause_s: float = Field(default=0.5, ge=0.0, le=5.0)


class ConversationConfig(_Section):
    max_messages: int = Field(default=20, ge=2)
    session_name: str = "default"
    persist: bool = False


class MemoryConfig(_Section):
    engine: Literal["none", "sqlite"] = "none"
    path: str = "data/memory.db"


class PluginsConfig(_Section):
    directory: str = "plugins"
    enabled: list[str] = Field(default_factory=list)
    settings: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ToolsConfig(_Section):
    enabled: bool = True
    max_tool_rounds: int = Field(default=4, ge=1, le=10)
    timeout_s: float = Field(default=30.0, gt=0)
    disabled: list[str] = Field(default_factory=list)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str | None = "logs/jarvis.log"
    json_format: bool = Field(default=False, alias="json")
    console: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Settings(_Section):
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def resolve(self, path: str) -> Path:
        """Löst einen Konfigurationspfad relativ zum Projektordner auf."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else self._base_dir / p

    @property
    def active_provider(self) -> ProviderConfig:
        return self.ai.providers[self.ai.provider]


def _format_validation_error(err: ValidationError) -> str:
    lines = []
    for item in err.errors():
        loc = ".".join(str(part) for part in item["loc"]) or "(Wurzel)"
        msg = item["msg"].removeprefix("Value error, ")
        lines.append(f"  - {loc}: {msg}")
    return "Ungültige Konfiguration:\n" + "\n".join(lines)


def load_config(path: str | Path | None = None, *, load_env: bool = True) -> Settings:
    """Lädt, validiert und gibt die Einstellungen zurück.

    Reihenfolge für den Pfad: Argument, Umgebungsvariable JARVIS_CONFIG, ./config/config.yaml.
    Raises:
        ConfigError: Datei fehlt, YAML kaputt oder Werte ungültig.
    """
    cfg_path = Path(path or os.environ.get("JARVIS_CONFIG") or "config/config.yaml").expanduser()
    if not cfg_path.is_file():
        raise ConfigError(
            f"Konfigurationsdatei nicht gefunden: {cfg_path}. "
            "Starte Jarvis im Projektordner oder setze JARVIS_CONFIG."
        )
    cfg_path = cfg_path.resolve()
    base_dir = cfg_path.parent.parent if cfg_path.parent.name == "config" else cfg_path.parent

    if load_env:
        load_dotenv(base_dir / ".env", override=False)

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML-Syntaxfehler in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} muss ein YAML-Mapping auf oberster Ebene enthalten.")

    try:
        settings = Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    settings._base_dir = base_dir
    return settings
