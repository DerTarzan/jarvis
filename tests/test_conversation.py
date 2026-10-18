import pytest

from jarvis.ai import ToolCall
from jarvis.conversation import ConversationManager, SQLiteMemory


def test_roles_and_system_prompt_first():
    conv = ConversationManager("SYS", max_messages=10)
    conv.add_user("Hi")
    conv.add_assistant("Hallo")
    roles = [m.role for m in conv.get_messages()]
    assert roles == ["system", "user", "assistant"]
    assert conv.get_messages()[0].content == "SYS"


def test_context_is_limited_and_starts_with_user():
    conv = ConversationManager("SYS", max_messages=3)
    for i in range(4):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    msgs = conv.get_messages()
    assert msgs[1].role == "user"
    assert len(msgs) - 1 <= 3
    assert len(conv) == 8  # nichts geht verloren, nur das Fenster wird begrenzt


def test_orphan_tool_messages_are_not_left_at_window_start():
    conv = ConversationManager("SYS", max_messages=2)
    conv.add_user("u")
    conv.add_assistant("", [ToolCall("t", {})])
    conv.add_tool_result("t", "r")
    conv.add_assistant("fertig")
    msgs = conv.get_messages()
    # Limit 2 reicht nicht für die Runde – sie bleibt komplett, beginnend mit der Frage
    assert [m.role for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    conv.add_user("nächste Frage")
    conv.add_assistant("Antwort")
    assert [m.role for m in conv.get_messages()] == ["system", "user", "assistant"]


def test_rollback_and_reset():
    conv = ConversationManager("SYS")
    conv.add_user("a")
    mark = conv.mark()
    conv.add_user("b")
    conv.rollback(mark)
    assert len(conv) == 1
    conv.reset()
    assert len(conv) == 0


def test_invalid_max_messages():
    with pytest.raises(ValueError):
        ConversationManager("SYS", max_messages=1)


def test_sqlite_persistence_roundtrip(tmp_path):
    db = tmp_path / "m.db"
    store = SQLiteMemory(db)
    conv = ConversationManager("SYS", store=store, session="s1")
    conv.add_user("Frage")
    conv.add_assistant("", [ToolCall("t", {"x": 1})])
    conv.add_tool_result("t", "ok")
    store.close()

    store2 = SQLiteMemory(db)
    conv2 = ConversationManager("SYS", store=store2, session="s1")
    assert len(conv2) == 3
    assert conv2.get_messages()[2].tool_calls[0].arguments == {"x": 1}
    assert len(ConversationManager("SYS", store=store2, session="other")) == 0
    store2.close()


def test_sqlite_rollback_rewrites_session(tmp_path):
    store = SQLiteMemory(tmp_path / "m.db")
    conv = ConversationManager("SYS", store=store, session="s")
    conv.add_user("a")
    mark = conv.mark()
    conv.add_user("b")
    conv.rollback(mark)
    assert len(store.load("s")) == 1
    store.close()


def test_long_term_facts(tmp_path):
    store = SQLiteMemory(tmp_path / "m.db")
    fact_id = store.remember("Lieblingsfarbe ist Blau")
    assert store.search("Blau") == ["Lieblingsfarbe ist Blau"]
    assert store.forget(fact_id) is True
    assert store.search("Blau") == []
    store.close()
