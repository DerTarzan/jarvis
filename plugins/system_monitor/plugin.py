"""Liefert lokale Systemauslastung als lesendes Jarvis-Tool."""

from __future__ import annotations

from typing import Any

from jarvis.plugins import BasePlugin


class Plugin(BasePlugin):
    description = "Liefert CPU-, RAM-, Datenträger- und Temperaturwerte dieses Computers"
    version = "0.1.0"

    def setup(self) -> None:
        self.ctx.register_tool(
            name="system_monitor",
            description="Zeigt CPU-, RAM-, Datenträger- und verfügbare Temperaturauslastung des Computers.",
            handler=self.monitor,
            timeout_s=10,
        )

    @staticmethod
    def monitor() -> dict[str, Any]:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("psutil fehlt. Installiere Jarvis mit pip install -e .") from exc

        memory = psutil.virtual_memory()
        disks = []
        for partition in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(partition.mountpoint)
            except OSError:
                continue
            disks.append(
                {
                    "mount": partition.mountpoint,
                    "gesamt_gb": round(usage.total / 1024**3, 1),
                    "frei_gb": round(usage.free / 1024**3, 1),
                    "auslastung_prozent": usage.percent,
                }
            )
        result: dict[str, Any] = {
            "cpu_prozent": psutil.cpu_percent(interval=0.2),
            "ram_gesamt_gb": round(memory.total / 1024**3, 1),
            "ram_verwendet_prozent": memory.percent,
            "datentraeger": disks,
        }
        try:
            temperatures = psutil.sensors_temperatures()
        except (AttributeError, NotImplementedError, OSError):
            temperatures = {}
        if temperatures:
            result["temperaturen_celsius"] = {
                name: [sensor.current for sensor in sensors]
                for name, sensors in temperatures.items()
                if sensors
            }
        return result
