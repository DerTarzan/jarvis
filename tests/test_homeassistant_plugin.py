import json
import sys

from jarvis.ai import ToolCall
from jarvis.core.events import EventBus
from jarvis.plugins import PluginManager
from jarvis.tools import ToolExecutor, ToolRegistry


def load_plugin(project_root, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME_ASSISTANT_TOKEN", "test-token")
    registry = ToolRegistry()
    manager = PluginManager(
        project_root / "plugins",
        registry,
        EventBus(),
        tmp_path / "data",
        {"homeassistant": {"base_url": "http://ha.test:8123"}},
    )
    status = manager.load("homeassistant")
    assert status.loaded, status.error
    return manager, registry, sys.modules["jarvis_plugins.homeassistant"]


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


async def test_light_control_runs_without_confirmation_and_sends_rgb_and_brightness(
    project_root, tmp_path, monkeypatch
):
    _, registry, module = load_plugin(project_root, tmp_path, monkeypatch)
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if request.method == "GET":
            return FakeResponse(
                json.dumps(
                    [
                        {
                            "entity_id": "light.living_room",
                            "state": "off",
                            "attributes": {"friendly_name": "Wohnzimmer"},
                        },
                        {"entity_id": "switch.coffee", "state": "on", "attributes": {}},
                    ]
                ).encode()
            )
        return FakeResponse(b"[]")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    executor = ToolExecutor(registry)
    call = ToolCall(
        "homeassistant_control_light",
        {"entity": "Wohnzimmer", "action": "an", "brightness": 40, "color": "blau"},
    )

    assert not registry.get("homeassistant_control_light").requires_confirmation
    success = await executor.execute(call)
    assert success.ok
    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].full_url.endswith("/api/services/light/turn_on")
    assert json.loads(requests[1].data) == {
        "entity_id": "light.living_room",
        "brightness_pct": 40,
        "rgb_color": [0, 0, 255],
    }
    assert requests[0].get_header("Authorization") == "Bearer test-token"


async def test_light_list_excludes_non_lights_and_ambiguous_names_are_rejected(
    project_root, tmp_path, monkeypatch
):
    _, registry, module = load_plugin(project_root, tmp_path, monkeypatch)

    def fake_urlopen(request, timeout):
        return FakeResponse(
            json.dumps(
                [
                    {"entity_id": "light.one", "state": "on", "attributes": {"friendly_name": "Decke"}},
                    {"entity_id": "light.two", "state": "off", "attributes": {"friendly_name": "Decke"}},
                    {"entity_id": "switch.one", "state": "on", "attributes": {"friendly_name": "Decke"}},
                ]
            ).encode()
        )

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    executor = ToolExecutor(registry)
    lights = await executor.execute(ToolCall("homeassistant_list_lights", {}))
    assert lights.ok
    assert "switch.one" not in lights.content

    async def confirm(tool, args):
        return True

    ambiguous = await executor.execute(
        ToolCall("homeassistant_control_light", {"entity": "Decke", "action": "aus"}), confirm
    )
    assert not ambiguous.ok and "nicht eindeutig" in ambiguous.content


def test_plugin_requires_valid_http_base_url(project_root, tmp_path):
    registry = ToolRegistry()
    manager = PluginManager(
        project_root / "plugins",
        registry,
        EventBus(),
        tmp_path / "data",
        {"homeassistant": {"base_url": "https://user:secret@ha.test"}},
    )
    status = manager.load("homeassistant")
    assert not status.loaded
    assert "ohne Zugangsdaten" in status.error
