import sys
from types import SimpleNamespace

from jarvis.ai import ToolCall
from jarvis.core.events import EventBus
from jarvis.plugins import PluginManager
from jarvis.tools import ToolExecutor, ToolRegistry


def load_plugin(project_root, tmp_path, name, config=None):
    registry = ToolRegistry()
    manager = PluginManager(
        project_root / "plugins",
        registry,
        EventBus(),
        tmp_path / "data",
        {name: config or {}},
    )
    status = manager.load(name)
    assert status.loaded, status.error
    return registry, sys.modules[f"jarvis_plugins.{name}"]


async def test_docker_actions_are_confirmation_gated_and_use_argument_lists(
    project_root, tmp_path, monkeypatch
):
    registry, module = load_plugin(project_root, tmp_path, "docker_proxmox")
    commands = []

    def fake_run(args, **kwargs):
        commands.append(args)
        if args[1:3] == ["ps", "-a"]:
            return SimpleNamespace(
                returncode=0,
                stdout='{"ID":"abc123","Names":"web","Image":"nginx","Status":"Up"}\n',
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="abc123", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    executor = ToolExecutor(registry)

    denied = await executor.execute(
        ToolCall("docker_control_container", {"container": "web", "action": "stop"})
    )
    assert not denied.ok and len(commands) == 0

    async def confirm(tool, args):
        return True

    result = await executor.execute(
        ToolCall("docker_control_container", {"container": "web", "action": "stop"}), confirm
    )
    assert result.ok
    assert commands[-1] == ["docker", "stop", "abc123"]


async def test_proxmox_control_uses_token_and_confirmed_exact_vm_target(
    project_root, tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_PROXMOX_TOKEN", "user@pam!jarvis=secret")
    registry, module = load_plugin(
        project_root,
        tmp_path,
        "docker_proxmox",
        {"proxmox_url": "https://pve.test:8006", "proxmox_token_env": "TEST_PROXMOX_TOKEN"},
    )
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data":"UPID:task"}'

    def fake_urlopen(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    executor = ToolExecutor(registry)

    async def confirm(tool, args):
        return True

    result = await executor.execute(
        ToolCall(
            "proxmox_control_vm",
            {"node": "pve1", "vmid": 101, "type": "qemu", "action": "start"},
        ),
        confirm,
    )
    assert result.ok
    assert requests[0].method == "POST"
    assert requests[0].full_url.endswith("/nodes/pve1/qemu/101/status/start")
    assert requests[0].get_header("Authorization") == "PVEAPIToken=user@pam!jarvis=secret"


def test_system_monitor_plugin_registers_read_only_tool(project_root, tmp_path):
    registry, _ = load_plugin(project_root, tmp_path, "system_monitor")
    assert registry.get("system_monitor") is not None
    assert not registry.get("system_monitor").requires_confirmation
