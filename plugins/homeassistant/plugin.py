"""Steuert Home-Assistant-Lampen über die REST-API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from jarvis.plugins import BasePlugin

_COLORS: dict[str, tuple[int, int, int]] = {
    "rot": (255, 0, 0),
    "grün": (0, 255, 0),
    "blau": (0, 0, 255),
    "gelb": (255, 255, 0),
    "orange": (255, 128, 0),
    "lila": (128, 0, 255),
    "violett": (128, 0, 255),
    "rosa": (255, 64, 160),
    "pink": (255, 64, 160),
    "weiß": (255, 255, 255),
    "warmweiß": (255, 180, 100),
    "türkis": (0, 220, 200),
}
_COLOR_SCHEMA = {
    "type": "string",
    "enum": list(_COLORS),
    "description": "Gewünschte Lichtfarbe; nur bei Aktion 'an'.",
}


class Plugin(BasePlugin):
    description = "Steuert Home-Assistant-Lampen, Helligkeit und Farben"
    version = "0.1.0"

    def setup(self) -> None:
        self._base_url = str(self.ctx.config.get("base_url", "")).rstrip("/")
        self._token_env = str(self.ctx.config.get("token_env", "HOME_ASSISTANT_TOKEN"))
        self._validate_base_url()
        self._token = os.environ.get(self._token_env, "").strip()
        self.ctx.register_tool(
            name="homeassistant_list_lights",
            description="Listet Home-Assistant-Lampen mit Name, Entity-ID und aktuellem Zustand auf.",
            handler=self.list_lights,
            timeout_s=15,
        )
        self.ctx.register_tool(
            name="homeassistant_control_light",
            description=(
                "Schaltet eine Home-Assistant-Lampe an oder aus und kann beim Einschalten "
                "Helligkeit (1-100 Prozent) oder eine Farbe setzen. Ziel muss eine einzelne Lampe sein."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                        "description": "Exakte Entity-ID (z. B. light.wohnzimmer) oder exakter Lampenname.",
                    },
                    "action": {"type": "string", "enum": ["an", "aus"]},
                    "brightness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Optionale Helligkeit in Prozent, nur beim Einschalten.",
                    },
                    "color": _COLOR_SCHEMA,
                },
                "required": ["entity", "action"],
                "additionalProperties": False,
            },
            handler=self.control_light,
            timeout_s=15,
        )

    def _validate_base_url(self) -> None:
        parsed = urlsplit(self._base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError(
                "Home-Assistant base_url muss eine HTTP(S)-Adresse ohne Zugangsdaten sein "
                "(z. B. http://homeassistant.local:8123)."
            )

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        if not self._token:
            raise RuntimeError(
                f"Home-Assistant-Token fehlt. Lege ihn als Umgebungsvariable "
                f"{self._token_env} fest (z. B. in der Projektdatei .env)."
            )
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self._base_url}/api/{path}",
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                body = response.read()
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Home Assistant hat den Zugriff verweigert. Prüfe den Long-Lived-Access-Token."
                ) from exc
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Home-Assistant-API antwortete mit HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise RuntimeError(f"Home Assistant unter {self._base_url} ist nicht erreichbar: {exc}") from exc
        if not body:
            return None
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Home Assistant lieferte eine ungültige JSON-Antwort.") from exc

    def _lights(self) -> list[dict[str, Any]]:
        states = self._request("states")
        if not isinstance(states, list):
            raise RuntimeError("Home Assistant lieferte keine gültige Zustandsliste.")
        return [
            state
            for state in states
            if isinstance(state, dict)
            and isinstance(state.get("entity_id"), str)
            and state["entity_id"].startswith("light.")
        ]

    def list_lights(self) -> list[dict[str, str]]:
        """Gibt die Lampenliste aus Home Assistant zurück."""
        return [
            {
                "entity": state["entity_id"],
                "name": str(state.get("attributes", {}).get("friendly_name", state["entity_id"])),
                "state": str(state.get("state", "unbekannt")),
            }
            for state in self._lights()
        ]

    def control_light(
        self,
        entity: str,
        action: str,
        brightness: int | None = None,
        color: str | None = None,
    ) -> dict[str, str]:
        """Steuert nur eine einzelne Lampe."""
        if action not in ("an", "aus"):
            raise ValueError("Aktion muss 'an' oder 'aus' sein.")
        if action == "aus" and (brightness is not None or color is not None):
            raise ValueError("Helligkeit und Farbe können nur beim Einschalten gesetzt werden.")
        if color is not None and color not in _COLORS:
            raise ValueError(f"Unbekannte Farbe. Verfügbar: {', '.join(_COLORS)}.")

        lights = self._lights()
        matches = [
            light
            for light in lights
            if light["entity_id"].casefold() == entity.casefold()
            or str(light.get("attributes", {}).get("friendly_name", "")).casefold() == entity.casefold()
        ]
        if not matches:
            raise ValueError(f"Keine Lampe mit der Entity-ID oder dem Namen '{entity}' gefunden.")
        if len(matches) > 1:
            choices = ", ".join(light["entity_id"] for light in matches)
            raise ValueError(f"Der Name '{entity}' ist nicht eindeutig. Verwende eine Entity-ID: {choices}.")

        light = matches[0]
        service_data: dict[str, Any] = {"entity_id": light["entity_id"]}
        if brightness is not None:
            service_data["brightness_pct"] = brightness
        if color is not None:
            service_data["rgb_color"] = list(_COLORS[color])
        service = "turn_on" if action == "an" else "turn_off"
        self._request(f"services/light/{service}", service_data)
        name = str(light.get("attributes", {}).get("friendly_name", light["entity_id"]))
        return {"lampe": name, "aktion": action, "ergebnis": "erfolgreich"}