"""Plugin-Schnittstelle.

WICHTIG: Das Plugin-System ist KEINE Sicherheits-Sandbox. Ein Plugin ist normaler Python-Code
und läuft mit allen Rechten des Jarvis-Prozesses. Installiere nur Plugins, denen du vertraust.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jarvis.core.events import EventBus
from jarvis.tools.registry import Tool, ToolRegistry


class PluginContext:
    """Kontrollierte Schnittstelle des Kerns für ein Plugin."""

    def __init__(
        self,
        plugin_name: str,
        config: dict[str, Any],
        events: EventBus,
        registry: ToolRegistry,
        data_dir: Path,
    ) -> None:
        self.plugin_name = plugin_name
        self.config = config
        self.events = events
        self.data_dir = data_dir
        self.log = logging.getLogger(f"jarvis.plugin.{plugin_name}")
        self._registry = registry

    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
        requires_confirmation: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        """Registriert ein Tool, das dem Plugin zugeordnet wird (wird beim Entladen entfernt)."""
        kwargs: dict[str, Any] = {}
        if parameters is not None:
            kwargs["parameters"] = parameters
        self._registry.register(
            Tool(
                name=name,
                description=description,
                handler=handler,
                requires_confirmation=requires_confirmation,
                timeout_s=timeout_s,
                source=f"plugin:{self.plugin_name}",
                **kwargs,
            )
        )


class BasePlugin(ABC):
    """Basisklasse. Die Datei ``plugins/<name>/plugin.py`` muss eine Unterklasse ``Plugin`` definieren."""

    description: str = ""
    version: str = "0.1.0"

    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def setup(self) -> None:
        """Registriert Tools/Event-Handler. Ausnahmen führen nur zum Verwerfen dieses Plugins."""

    def teardown(self) -> None:  # noqa: B027 - optionaler Hook
        """Räumt Ressourcen auf (optional)."""
