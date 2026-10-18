"""Lesende Infrastrukturübersicht und bestätigungspflichtige Docker-/Proxmox-Aktionen."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from jarvis.plugins import BasePlugin

_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class Plugin(BasePlugin):
    description = "Listet Docker-Container und Proxmox-VMs auf; Start/Stop nur nach Bestätigung"
    version = "0.1.0"

    def setup(self) -> None:
        self._docker_command = str(self.ctx.config.get("docker_command", "docker"))
        self._proxmox_url = str(self.ctx.config.get("proxmox_url", "")).rstrip("/")
        self._proxmox_token_env = str(self.ctx.config.get("proxmox_token_env", "PROXMOX_API_TOKEN"))
        if self._proxmox_url:
            parsed = urlsplit(self._proxmox_url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError("proxmox_url muss eine HTTPS-Adresse ohne Zugangsdaten sein.")

        self.ctx.register_tool(
            name="docker_list_containers",
            description="Listet lokale Docker-Container mit Name, Image und Status auf.",
            handler=self.list_containers,
            timeout_s=15,
        )
        self.ctx.register_tool(
            name="docker_control_container",
            description="Startet oder stoppt einen Docker-Container anhand des exakten Namens oder der ID.",
            parameters={
                "type": "object",
                "properties": {
                    "container": {"type": "string", "minLength": 1, "maxLength": 128},
                    "action": {"type": "string", "enum": ["start", "stop"]},
                },
                "required": ["container", "action"],
                "additionalProperties": False,
            },
            handler=self.control_container,
            requires_confirmation=True,
            timeout_s=30,
        )
        self.ctx.register_tool(
            name="proxmox_list_vms",
            description="Listet Proxmox-VMs und LXC-Container mit Name, Node, Typ und Status auf.",
            handler=self.list_vms,
            timeout_s=15,
        )
        self.ctx.register_tool(
            name="proxmox_control_vm",
            description="Startet oder fährt eine Proxmox-VM bzw. einen LXC-Container herunter.",
            parameters={
                "type": "object",
                "properties": {
                    "node": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
                    "vmid": {"type": "integer", "minimum": 100, "maximum": 999999999},
                    "type": {"type": "string", "enum": ["qemu", "lxc"]},
                    "action": {"type": "string", "enum": ["start", "shutdown"]},
                },
                "required": ["node", "vmid", "type", "action"],
                "additionalProperties": False,
            },
            handler=self.control_vm,
            requires_confirmation=True,
            timeout_s=30,
        )

    def _docker(self, *args: str) -> str:
        try:
            result = subprocess.run(
                [self._docker_command, *args],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Docker CLI konnte nicht ausgeführt werden: {exc}") from exc
        if result.returncode:
            detail = result.stderr.strip()[:500] or f"Exit-Code {result.returncode}"
            raise RuntimeError(f"Docker-Befehl fehlgeschlagen: {detail}")
        return result.stdout

    def _containers(self) -> list[dict[str, Any]]:
        output = self._docker("ps", "-a", "--format", "{{json .}}")
        containers = []
        for line in output.splitlines():
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Docker lieferte eine ungültige Containerliste.") from exc
        return containers

    def list_containers(self) -> list[dict[str, str]]:
        return [
            {
                "id": str(item.get("ID", "")),
                "name": str(item.get("Names", "")),
                "image": str(item.get("Image", "")),
                "status": str(item.get("Status", "")),
            }
            for item in self._containers()
        ]

    def control_container(self, container: str, action: str) -> dict[str, str]:
        if action not in ("start", "stop") or not _TARGET_RE.fullmatch(container):
            raise ValueError("Ungültiger Containername oder Aktion.")
        matches = [
            item
            for item in self._containers()
            if container in (str(item.get("ID", "")), str(item.get("Names", "")))
        ]
        if len(matches) != 1:
            raise ValueError("Container nicht gefunden oder Auswahl nicht eindeutig.")
        target = str(matches[0]["ID"])
        output = self._docker(action, target)
        return {
            "container": str(matches[0].get("Names", target)),
            "aktion": action,
            "ergebnis": output.strip(),
        }

    def _proxmox(self, path: str, post: bool = False) -> Any:
        token = os.environ.get(self._proxmox_token_env, "").strip()
        if not self._proxmox_url:
            raise RuntimeError("proxmox_url ist in plugins.settings.docker_proxmox nicht konfiguriert.")
        if not token or "!" not in token:
            raise RuntimeError(
                f"Proxmox-API-Token fehlt oder ist ungültig. Setze {self._proxmox_token_env}."
            )
        request = Request(
            f"{self._proxmox_url}/api2/json/{path}",
            data=b"" if post else None,
            headers={"Authorization": f"PVEAPIToken={token}"},
            method="POST" if post else "GET",
        )
        try:
            with urlopen(request, timeout=10) as response:
                body = response.read()
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError("Proxmox hat den Zugriff verweigert. Prüfe Token und ACL-Rechte.") from exc
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Proxmox-API antwortete mit HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise RuntimeError(f"Proxmox unter {self._proxmox_url} ist nicht erreichbar: {exc}") from exc
        try:
            return json.loads(body).get("data")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
            raise RuntimeError("Proxmox lieferte eine ungültige API-Antwort.") from exc

    def list_vms(self) -> list[dict[str, Any]]:
        resources = self._proxmox("cluster/resources?type=vm")
        if not isinstance(resources, list):
            raise RuntimeError("Proxmox lieferte keine gültige VM-Liste.")
        return [
            {
                "vmid": item.get("vmid"),
                "node": item.get("node"),
                "type": item.get("type"),
                "name": item.get("name", ""),
                "status": item.get("status", "unbekannt"),
            }
            for item in resources
            if isinstance(item, dict)
        ]

    def control_vm(self, node: str, vmid: int, type: str, action: str) -> dict[str, str]:
        if (
            not _TARGET_RE.fullmatch(node)
            or type not in ("qemu", "lxc")
            or action not in ("start", "shutdown")
        ):
            raise ValueError("Ungültiger Proxmox-Node, Typ oder Aktion.")
        path = f"nodes/{quote(node)}/{type}/{vmid}/status/{action}"
        result = self._proxmox(path, post=True)
        return {"node": node, "vmid": str(vmid), "aktion": action, "task": str(result or "")}
