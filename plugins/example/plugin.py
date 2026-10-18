"""Beispiel-Plugin: zeigt Tool-Registrierung, Konfiguration, Logging und Events.

Kopiere diesen Ordner nach plugins/<neuer_name>/, passe die Klasse an und trage den Ordnernamen
unter plugins.enabled in config/config.yaml ein. Siehe docs/adding_plugins.md.
"""

from __future__ import annotations

import random
from typing import Any

from jarvis.plugins import BasePlugin


class Plugin(BasePlugin):
    description = "Beispiel: würfelt eine Zufallszahl"
    version = "0.1.0"

    def setup(self) -> None:
        self.max_sides = int(self.ctx.config.get("max_sides", 100))
        self.ctx.register_tool(
            name="roll_dice",
            description="Würfelt mit einem Würfel mit der angegebenen Seitenzahl und gibt die "
            "gewürfelte Zahl zurück.",
            parameters={
                "type": "object",
                "properties": {
                    "sides": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": self.max_sides,
                        "description": "Anzahl der Seiten des Würfels",
                    }
                },
                "additionalProperties": False,
            },
            handler=self.roll,
        )
        self.ctx.log.info("Beispiel-Plugin bereit (max_sides=%d)", self.max_sides)

    def roll(self, sides: int = 6) -> dict[str, Any]:
        return {"seiten": sides, "ergebnis": random.randint(1, sides)}
