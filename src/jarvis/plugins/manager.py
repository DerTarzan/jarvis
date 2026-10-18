"""Entdecken, Laden und Entladen von Plugins aus einem Verzeichnis."""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.core.events import EventBus
from jarvis.plugins.base import BasePlugin, PluginContext
from jarvis.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


@dataclass
class PluginStatus:
    name: str
    loaded: bool
    description: str = ""
    error: str = ""
    tools: tuple[str, ...] = ()


class PluginManager:
    """Lädt aktivierte Plugins; Fehler eines Plugins betreffen nur dieses Plugin."""

    def __init__(
        self,
        directory: Path,
        registry: ToolRegistry,
        events: EventBus,
        data_dir: Path,
        settings: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._dir = directory
        self._registry = registry
        self._events = events
        self._data_dir = data_dir
        self._settings = settings or {}
        self._loaded: dict[str, BasePlugin] = {}
        self.statuses: list[PluginStatus] = []

    def discover(self) -> list[str]:
        """Namen aller Plugin-Ordner (mit plugin.py), unabhängig von der Aktivierung."""
        if not self._dir.is_dir():
            return []
        return sorted(
            p.name
            for p in self._dir.iterdir()
            if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "plugin.py").is_file()
        )

    def load_all(self, enabled: list[str]) -> list[PluginStatus]:
        self.statuses = [self.load(name) for name in enabled]
        return self.statuses

    def load(self, name: str) -> PluginStatus:
        status = self._load(name)
        if not status.loaded:
            log.error("Plugin '%s' nicht geladen: %s", name, status.error)
        else:
            log.info("Plugin '%s' geladen (%d Tools)", name, len(status.tools))
        return status

    def _load(self, name: str) -> PluginStatus:
        if name in self._loaded:
            return PluginStatus(name, False, error="bereits geladen")
        path = self._dir / name / "plugin.py"
        if not path.is_file():
            return PluginStatus(name, False, error=f"{path} nicht gefunden")

        source = f"plugin:{name}"
        try:
            module_name = f"jarvis_plugins.{name}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("Modul-Spezifikation konnte nicht erstellt werden")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            plugin_cls = getattr(module, "Plugin", None)
            if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, BasePlugin)):
                raise TypeError("plugin.py muss eine Klasse 'Plugin(BasePlugin)' definieren")

            ctx = PluginContext(
                plugin_name=name,
                config=dict(self._settings.get(name, {})),
                events=self._events,
                registry=self._registry,
                data_dir=self._data_dir / "plugins" / name,
            )
            plugin = plugin_cls(ctx)
            plugin.setup()
        except Exception as exc:
            self._registry.unregister_source(source)
            sys.modules.pop(f"jarvis_plugins.{name}", None)
            log.debug("Plugin-Ladefehler", exc_info=True)
            return PluginStatus(name, False, error=f"{type(exc).__name__}: {exc}")

        self._loaded[name] = plugin
        tools = tuple(t.name for t in self._registry.all() if t.source == source)
        return PluginStatus(name, True, description=plugin.description, tools=tools)

    def unload(self, name: str) -> bool:
        plugin = self._loaded.pop(name, None)
        if plugin is None:
            return False
        try:
            plugin.teardown()
        except Exception:
            log.exception("teardown von Plugin '%s' fehlgeschlagen", name)
        self._registry.unregister_source(f"plugin:{name}")
        sys.modules.pop(f"jarvis_plugins.{name}", None)
        return True

    def unload_all(self) -> None:
        for name in list(self._loaded):
            self.unload(name)

    @property
    def loaded_names(self) -> list[str]:
        return list(self._loaded)
