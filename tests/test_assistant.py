import pytest

from conftest import FakeProvider, tool_call
from jarvis.ai import LLMResponse, ProviderUnavailableError
from jarvis.conversation import ConversationManager
from jarvis.core import events as ev
from jarvis.core.application import Application
from jarvis.core.assistant import TOOL_LIMIT_REPLY, Assistant
from jarvis.core.events import EventBus
from jarvis.tools import Tool, ToolExecutor, ToolRegistry


def build(provider, tools=None, rounds=3):
    reg = ToolRegistry()
    for t in tools or []:
        reg.register(t)
    bus = EventBus()
    conv = ConversationManager("SYS")
    assistant = Assistant(provider, conv, reg, ToolExecutor(reg), bus, max_tool_rounds=rounds)
    return assistant, conv, bus


async def test_plain_answer_and_context_is_kept():
    provider = FakeProvider([LLMResponse("Eins"), LLMResponse("Zwei")])
    assistant, conv, _ = build(provider)
    assert await assistant.respond("a") == "Eins"
    assert await assistant.respond("b") == "Zwei"
    sent = provider.calls[1][0]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]


async def test_tool_loop_executes_tool_and_feeds_result_back():
    tool = Tool("zwei", "gibt zwei", lambda: 2)
    provider = FakeProvider([LLMResponse("", [tool_call("zwei")]), LLMResponse("Das Ergebnis ist 2.")])
    assistant, conv, bus = build(provider, [tool])
    seen = []
    bus.subscribe(ev.TOOL_CALLED, lambda name, ok: seen.append((name, ok)))

    assert await assistant.respond("rechne") == "Das Ergebnis ist 2."
    assert seen == [("zwei", True)]
    second_call_msgs = provider.calls[1][0]
    assert second_call_msgs[-1].role == "tool" and second_call_msgs[-1].content == "2"
    assert provider.calls[0][1] is not None  # Tool-Spezifikationen wurden mitgesendet


async def test_failing_tool_result_is_reported_to_model_not_raised():
    def boom():
        raise RuntimeError("kaputt")

    provider = FakeProvider([LLMResponse("", [tool_call("boom")]), LLMResponse("Das hat nicht geklappt.")])
    assistant, _, _ = build(provider, [Tool("boom", "x", boom)])
    assert await assistant.respond("los") == "Das hat nicht geklappt."
    assert "kaputt" in provider.calls[1][0][-1].content


async def test_endless_tool_calls_are_stopped():
    provider = FakeProvider([LLMResponse("", [tool_call("zwei")])])
    assistant, _, _ = build(provider, [Tool("zwei", "x", lambda: 2)], rounds=2)
    assert await assistant.respond("x") == TOOL_LIMIT_REPLY
    assert len(provider.calls) == 3


async def test_provider_error_rolls_back_conversation():
    provider = FakeProvider([ProviderUnavailableError("weg")])
    assistant, conv, _ = build(provider)
    with pytest.raises(ProviderUnavailableError):
        await assistant.respond("hallo")
    assert len(conv) == 0


async def test_no_tools_sent_when_provider_lacks_support():
    provider = FakeProvider([LLMResponse("ok")])
    provider.supports_tools = False
    assistant, _, _ = build(provider, [Tool("zwei", "x", lambda: 2)])
    await assistant.respond("x")
    assert provider.calls[0][1] is None


# --- Application (mit Fake-Provider und Fake-Stimme) -----------------------
class FakeVoice:
    can_speak = True
    can_listen = True
    stt_status = tts_status = "fake"

    def __init__(self, text="gesprochene Frage"):
        self.spoken: list[str] = []
        self.text = text

    async def speak(self, text):
        self.spoken.append(text)

    async def listen(self, stop, use_silence_detection=False):
        return self.text

    def stop_speaking(self): ...
    def refresh_status(self): ...
    def close(self): ...


async def test_application_voice_chain_text_to_speech(settings):
    voice = FakeVoice()
    app = Application(settings, provider=FakeProvider([LLMResponse("Guten Abend.")]), voice=voice)
    events = []
    app.events.subscribe(ev.ASSISTANT_TEXT, lambda text: events.append(text))

    import threading

    heard = await app.listen(threading.Event())
    reply = await app.process(heard, source="voice")

    assert heard == "gesprochene Frage"
    assert reply == "Guten Abend." and voice.spoken == ["Guten Abend."]
    assert events == ["Guten Abend."] and app.state == "bereit"
    await app.shutdown()


async def test_application_reports_provider_error_without_crashing(settings):
    app = Application(
        settings, provider=FakeProvider([ProviderUnavailableError("Ollama weg")]), voice=FakeVoice()
    )
    errors = []
    app.events.subscribe(ev.ERROR, lambda message: errors.append(message))
    assert await app.process("hi") is None
    assert errors == ["Ollama weg"] and app.state == "bereit"
    await app.shutdown()


async def test_application_registers_usable_long_term_memory_tools(settings):
    settings.memory.engine = "sqlite"
    settings.memory.path = "data/facts.db"
    app = Application(settings, provider=FakeProvider(), voice=FakeVoice())
    assert app.registry.get("remember_fact") is not None
    assert app.registry.get("search_memory") is not None
    assert app.registry.get("forget_memory").requires_confirmation

    saved = await app.executor.execute(
        tool_call("remember_fact", fact="Lieblingsfarbe ist türkis")
    )
    assert saved.ok
    found = await app.executor.execute(tool_call("search_memory", query="türkis"))
    assert found.ok and "Lieblingsfarbe ist türkis" in found.content

    denied = await app.executor.execute(tool_call("forget_memory", fact_id=1))
    assert not denied.ok and "nicht bestätigt" in denied.content

    async def confirm(tool, args):
        return True

    removed = await app.executor.execute(tool_call("forget_memory", fact_id=1), confirm)
    assert removed.ok and "true" in removed.content
    await app.shutdown()


async def test_application_startup_loads_plugins_and_warns(settings, project_root):
    settings.plugins.directory = str(project_root / "plugins")
    settings.plugins.enabled = ["example", "missing_plugin"]
    app = Application(settings, provider=FakeProvider(), voice=FakeVoice())
    report = await app.startup()
    assert report.provider.ok
    assert [p.loaded for p in report.plugins] == [True, False]
    assert any("missing_plugin" in w for w in report.warnings)
    assert app.registry.get("roll_dice") is not None
    await app.shutdown()
    assert app.registry.get("roll_dice") is None


async def test_speech_errors_do_not_lose_the_answer(settings):
    from jarvis.audio.errors import TTSError

    class BrokenVoice(FakeVoice):
        async def speak(self, text):
            raise TTSError("keine Stimme")

    app = Application(settings, provider=FakeProvider([LLMResponse("Antwort")]), voice=BrokenVoice())
    errors = []
    app.events.subscribe(ev.ERROR, lambda message: errors.append(message))
    assert await app.process("hi") == "Antwort"
    assert any("keine Stimme" in e for e in errors)
    await app.shutdown()


def test_cloud_provider_in_config_is_refused_at_startup(settings):
    from jarvis.ai import ProviderNotImplementedError

    settings.ai.provider = "openai"
    settings.ai.providers["openai"].enabled = True
    settings.ai.providers["openai"].model = "x"
    with pytest.raises(ProviderNotImplementedError):
        Application(settings, voice=FakeVoice())


async def test_unexpected_audio_exception_does_not_kill_the_assistant(settings):
    class ExplodingVoice(FakeVoice):
        async def speak(self, text):
            raise OSError("PortAudio library not found")

        async def listen(self, stop, use_silence_detection=False):
            raise OSError("kein Mikrofon-Backend")

    import threading

    app = Application(settings, provider=FakeProvider([LLMResponse("Antwort")]), voice=ExplodingVoice())
    errors = []
    app.events.subscribe(ev.ERROR, lambda message: errors.append(message))
    assert await app.process("hi") == "Antwort"
    assert await app.listen(threading.Event()) == ""
    assert len(errors) == 2 and app.state == "bereit"
    await app.shutdown()
