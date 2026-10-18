import textwrap
from pathlib import Path

from jarvis.ai import ToolCall
from jarvis.core.events import EventBus
from jarvis.plugins import PluginManager
from jarvis.tools import ToolExecutor, ToolRegistry


def manager(directory: Path, tmp_path: Path, settings=None):
    reg = ToolRegistry()
    return PluginManager(directory, reg, EventBus(), tmp_path / "data", settings), reg


def make_plugin(root: Path, name: str, body: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "plugin.py").write_text(textwrap.dedent(body), encoding="utf-8")


async def test_example_plugin_loads_and_tool_works(project_root, tmp_path):
    mgr, reg = manager(project_root / "plugins", tmp_path, {"example": {"max_sides": 20}})
    statuses = mgr.load_all(["example"])
    assert statuses[0].loaded and statuses[0].tools == ("roll_dice",)

    ex = ToolExecutor(reg)
    ok = await ex.execute(ToolCall("roll_dice", {"sides": 10}))
    assert ok.ok and '"seiten": 10' in ok.content
    # Plugin-Konfiguration begrenzt das Schema (max_sides=20)
    assert not (await ex.execute(ToolCall("roll_dice", {"sides": 50}))).ok


def test_unload_removes_tools(project_root, tmp_path):
    mgr, reg = manager(project_root / "plugins", tmp_path)
    mgr.load_all(["example"])
    assert reg.get("roll_dice") is not None
    mgr.unload_all()
    assert reg.get("roll_dice") is None and mgr.loaded_names == []


def test_discover_lists_plugin_folders(project_root, tmp_path):
    mgr, _ = manager(project_root / "plugins", tmp_path)
    assert "example" in mgr.discover()


def test_broken_plugins_do_not_affect_others(tmp_path):
    plugins = tmp_path / "plugins"
    make_plugin(plugins, "syntax", "def (:\n")
    make_plugin(plugins, "noclass", "x = 1\n")
    make_plugin(
        plugins,
        "setupfail",
        """
        from jarvis.plugins import BasePlugin
        class Plugin(BasePlugin):
            def setup(self):
                self.ctx.register_tool("half_registered", "x", lambda: 1)
                raise RuntimeError("setup kaputt")
        """,
    )
    make_plugin(
        plugins,
        "good",
        """
        from jarvis.plugins import BasePlugin
        class Plugin(BasePlugin):
            def setup(self):
                self.ctx.register_tool("good_tool", "ok", lambda: 1)
        """,
    )
    mgr, reg = manager(plugins, tmp_path)
    by_name = {s.name: s for s in mgr.load_all(["syntax", "noclass", "setupfail", "good", "missing"])}

    assert by_name["good"].loaded
    assert not by_name["syntax"].loaded and "SyntaxError" in by_name["syntax"].error
    assert "Plugin(BasePlugin)" in by_name["noclass"].error
    assert "setup kaputt" in by_name["setupfail"].error
    assert "nicht gefunden" in by_name["missing"].error
    assert reg.get("half_registered") is None  # unvollständige Registrierung aufgeräumt
    assert reg.get("good_tool") is not None


def test_duplicate_tool_name_between_plugins_fails_second_only(tmp_path):
    plugins = tmp_path / "plugins"
    body = """
        from jarvis.plugins import BasePlugin
        class Plugin(BasePlugin):
            def setup(self):
                self.ctx.register_tool("same_name", "x", lambda: 1)
        """
    make_plugin(plugins, "one", body)
    make_plugin(plugins, "two", body)
    mgr, _ = manager(plugins, tmp_path)
    one, two = mgr.load_all(["one", "two"])
    assert one.loaded and not two.loaded and "bereits registriert" in two.error
