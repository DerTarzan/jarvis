"""Erzeugt den konfigurierten Provider. Kein stiller Fallback auf andere Anbieter."""

from __future__ import annotations

from jarvis.ai.base import LLMProvider, ProviderConfigError, ProviderNotImplementedError
from jarvis.ai.providers.cloud import (
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from jarvis.ai.providers.ollama import OllamaProvider
from jarvis.config import ProviderConfig

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """Erweiterungspunkt: eigenen Provider unter einem Konfigurationsnamen registrieren.

    Der Konstruktor erhält die zugehörige ProviderConfig.
    """
    _PROVIDERS[name] = cls


def create_provider(name: str, cfg: ProviderConfig) -> LLMProvider:
    """Baut den Provider ``name`` aus ``cfg``.

    Raises:
        ProviderConfigError: deaktiviert, unbekannt, Modell/URL/Schlüssel fehlt.
        ProviderNotImplementedError: nur vorbereiteter Erweiterungspunkt.
    """
    if not cfg.enabled:
        raise ProviderConfigError(f"Provider '{name}' ist deaktiviert (enabled: false).")
    cls = _PROVIDERS.get(name)
    if cls is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise ProviderConfigError(
            f"Unbekannter Provider '{name}'. Verfügbar: {known}. "
            "Eigene Provider per register_provider() registrieren."
        )
    if not cls.implemented:
        raise ProviderNotImplementedError(
            f"Der Provider '{name}' ist nur als Erweiterungspunkt vorbereitet und noch nicht "
            "implementiert. Setze ai.provider auf 'ollama' oder implementiere den Adapter."
        )
    if not cfg.model:
        raise ProviderConfigError(f"Für Provider '{name}' ist kein Modell (model) konfiguriert.")
    if name in ("ollama", "openai_compatible") and not cfg.base_url:
        raise ProviderConfigError(f"Für Provider '{name}' fehlt base_url.")
    if cls.is_cloud and not cfg.api_key():
        raise ProviderConfigError(
            f"API-Schlüssel für '{name}' fehlt: Umgebungsvariable "
            f"'{cfg.api_key_env or '(api_key_env nicht gesetzt)'}' ist nicht gesetzt."
        )
    return cls(cfg)  # type: ignore[call-arg]
