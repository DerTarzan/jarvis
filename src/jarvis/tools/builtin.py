"""Eingebaute, ungefährliche Tools (nur lesend)."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from typing import Any

from jarvis.tools.registry import Tool, ToolRegistry


def system_info() -> dict[str, Any]:
    """Liest grundlegende Systemdaten (nur lesend, keine Prozesse oder Dateien)."""
    info: dict[str, Any] = {
        "betriebssystem": f"{platform.system()} {platform.release()}",
        "architektur": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_kerne": os.cpu_count(),
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_gesamt_gb"] = round(mem.total / 1024**3, 1)
        info["ram_frei_gb"] = round(mem.available / 1024**3, 1)
        info["cpu_auslastung_prozent"] = psutil.cpu_percent(interval=0.2)
    except ImportError:  # psutil ist optional für dieses Tool
        info["hinweis"] = "psutil nicht installiert – RAM/CPU-Auslastung nicht verfügbar."
    return info


def current_time() -> dict[str, str]:
    now = datetime.now().astimezone()
    return {
        "datum": now.strftime("%d.%m.%Y"),
        "uhrzeit": now.strftime("%H:%M:%S"),
        "zeitzone": str(now.tzinfo),
    }


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="system_info",
            description="Liefert aktuelle Systeminformationen des Computers (Betriebssystem, "
            "CPU-Kerne, Arbeitsspeicher, CPU-Auslastung).",
            handler=system_info,
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="Liefert aktuelles Datum und aktuelle Uhrzeit.",
            handler=current_time,
        )
    )
