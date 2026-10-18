import asyncio

import pytest

from jarvis.ai import ToolCall
from jarvis.tools import Tool, ToolExecutor, ToolRegistry, ToolRegistryError
from jarvis.tools.builtin import register_builtin_tools

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def make(tool: Tool, **kwargs) -> ToolExecutor:
    reg = ToolRegistry()
    reg.register(tool)
    return ToolExecutor(reg, **kwargs)


def test_registry_rejects_duplicates_bad_names_and_schemas():
    reg = ToolRegistry()
    reg.register(Tool("ok_tool", "beschreibung", lambda: 1))
    with pytest.raises(ToolRegistryError, match="bereits registriert"):
        reg.register(Tool("ok_tool", "x", lambda: 1))
    with pytest.raises(ToolRegistryError, match="Ungültiger Tool-Name"):
        reg.register(Tool("bad name!", "x", lambda: 1))
    with pytest.raises(ToolRegistryError, match="Beschreibung"):
        reg.register(Tool("empty_desc", " ", lambda: 1))
    with pytest.raises(ToolRegistryError, match="Schema"):
        reg.register(
            Tool(
                "bad_schema",
                "x",
                lambda: 1,
                parameters={"type": "object", "properties": {"a": {"type": "nope"}}},
            )
        )


def test_specs_and_unregister_source():
    reg = ToolRegistry()
    reg.register(Tool("a_tool", "A", lambda: 1, source="plugin:p"))
    reg.register(Tool("b_tool", "B", lambda: 1))
    assert {s.name for s in reg.specs()} == {"a_tool", "b_tool"}
    assert {s.name for s in reg.specs({"b_tool"})} == {"a_tool"}
    assert reg.unregister_source("plugin:p") == ["a_tool"]
    assert reg.get("a_tool") is None


async def test_execute_valid_sync_and_async_handlers():
    ex = make(Tool("add", "addiert", lambda a, b: a + b, ADD_SCHEMA))
    result = await ex.execute(ToolCall("add", {"a": 2, "b": 3}))
    assert result.ok and result.content == "5"

    async def aadd(a, b):
        return {"sum": a + b}

    ex = make(Tool("add", "addiert", aadd, ADD_SCHEMA))
    assert (await ex.execute(ToolCall("add", {"a": 1, "b": 1}))).content == '{"sum": 2}'


@pytest.mark.parametrize("args", [{"a": 1}, {"a": "x", "b": 1}, {"a": 1, "b": 2, "c": 3}, {}])
async def test_invalid_parameters_are_rejected_without_calling_handler(args):
    called = []
    ex = make(Tool("add", "addiert", lambda a, b: called.append(1), ADD_SCHEMA))
    result = await ex.execute(ToolCall("add", args))
    assert not result.ok and "Ungültige Parameter" in result.content
    assert called == []


async def test_unknown_and_disabled_tool():
    ex = make(Tool("t", "x", lambda: 1), disabled={"t"})
    assert not (await ex.execute(ToolCall("t", {}))).ok
    assert not (await ex.execute(ToolCall("nope", {}))).ok


async def test_handler_exception_is_contained():
    def boom():
        raise RuntimeError("kaputt")

    result = await make(Tool("boom", "x", boom)).execute(ToolCall("boom", {}))
    assert not result.ok and "kaputt" in result.content


async def test_timeout():
    async def slow():
        await asyncio.sleep(1)

    ex = make(Tool("slow", "x", slow, timeout_s=0.05))
    result = await ex.execute(ToolCall("slow", {}))
    assert not result.ok and "Zeitüberschreitung" in result.content


async def test_confirmation_required():
    ran = []
    ex = make(Tool("danger", "x", lambda: ran.append(1) or "done", requires_confirmation=True))

    assert not (await ex.execute(ToolCall("danger", {}))).ok  # ohne Callback: verweigert

    async def no(tool, args):
        return False

    async def yes(tool, args):
        return True

    assert not (await ex.execute(ToolCall("danger", {}), no)).ok
    assert ran == []
    assert (await ex.execute(ToolCall("danger", {}), yes)).ok
    assert ran == [1]


async def test_result_is_truncated():
    ex = make(Tool("big", "x", lambda: "x" * 10000))
    assert len((await ex.execute(ToolCall("big", {}))).content) < 4100


@pytest.mark.parametrize("value", ["Zeit\u00a0\udce4", {"zone": "Zeit\u00a0\udce4"}])
async def test_tool_result_replaces_invalid_unicode_surrogates(value):
    result = await make(Tool("unicode", "x", lambda: value)).execute(ToolCall("unicode", {}))
    assert result.ok
    assert "\udce4" not in result.content
    assert "\ufffd" in result.content


async def test_builtin_system_info_really_works():
    reg = ToolRegistry()
    register_builtin_tools(reg)
    result = await ToolExecutor(reg).execute(ToolCall("system_info", {}))
    assert result.ok
    assert "betriebssystem" in result.content and "cpu_kerne" in result.content
    assert (await ToolExecutor(reg).execute(ToolCall("current_time", {}))).ok
