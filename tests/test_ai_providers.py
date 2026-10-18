import json

import httpx
import pytest

from jarvis.ai import (
    ChatMessage,
    ModelNotFoundError,
    ProviderConfigError,
    ProviderNotImplementedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
    ToolSpec,
    create_provider,
)
from jarvis.ai.providers.ollama import OllamaProvider
from jarvis.config import ProviderConfig

CFG = ProviderConfig(enabled=True, base_url="http://ollama.test", model="llama3.1:8b")
MSGS = [ChatMessage("system", "s"), ChatMessage("user", "Hallo")]
TOOL = ToolSpec("system_info", "Info", {"type": "object", "properties": {}})


def provider(handler) -> OllamaProvider:
    client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaProvider(CFG, client)


async def test_chat_returns_text_and_sends_options():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"model": "llama3.1:8b", "message": {"role": "assistant", "content": " Guten Tag. "}}
        )

    p = provider(handler)
    response = await p.chat(MSGS)
    assert response.content == "Guten Tag."
    assert seen["stream"] is False and seen["model"] == "llama3.1:8b"
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


async def test_tool_calls_are_parsed_and_tools_sent():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "system_info", "arguments": {}}}],
                }
            },
        )

    response = await provider(handler).chat(MSGS, [TOOL])
    assert response.tool_calls[0].name == "system_info"
    assert seen["tools"][0]["function"]["name"] == "system_info"


async def test_tool_result_message_conversion():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    msgs = [*MSGS, ChatMessage("tool", "42", tool_name="system_info")]
    await provider(handler).chat(msgs)
    assert seen["messages"][-1] == {"role": "tool", "content": "42", "tool_name": "system_info"}


async def test_server_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailableError, match="nicht erreichbar"):
        await provider(handler).chat(MSGS)


async def test_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        await provider(handler).chat(MSGS)


async def test_model_missing_mentions_pull():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'llama3.1:8b' not found"})

    with pytest.raises(ModelNotFoundError, match="ollama pull llama3.1:8b"):
        await provider(handler).chat(MSGS)


async def test_model_without_tool_support_falls_back_controlled():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append("tools" in body)
        if "tools" in body:
            return httpx.Response(400, json={"error": "registry.ollama.ai/x does not support tools"})
        return httpx.Response(200, json={"message": {"content": "ohne Tools"}})

    p = provider(handler)
    response = await p.chat(MSGS, [TOOL])
    assert response.content == "ohne Tools"
    assert calls == [True, False]
    assert p.supports_tools is False


async def test_check_reports_installed_and_missing_model():
    def handler_ok(request):
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})

    def handler_missing(request):
        return httpx.Response(200, json={"models": [{"name": "other:latest"}]})

    assert (await provider(handler_ok).check()).ok
    status = await provider(handler_missing).check()
    assert not status.ok and "ollama pull" in status.message


async def test_stream_yields_chunks():
    lines = [{"message": {"content": "Hal"}}, {"message": {"content": "lo"}, "done": True}]

    def handler(request):
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    chunks = [c async for c in provider(handler).stream(MSGS)]
    assert "".join(chunks) == "Hallo"


# --- Manager -----------------------------------------------------------------
def test_manager_builds_ollama():
    p = create_provider("ollama", CFG)
    assert p.name == "ollama" and not p.is_cloud


def test_manager_rejects_disabled():
    with pytest.raises(ProviderConfigError, match="deaktiviert"):
        create_provider("ollama", ProviderConfig(enabled=False, base_url="x", model="m"))


@pytest.mark.parametrize("name", ["openai", "anthropic", "gemini"])
def test_cloud_providers_are_not_implemented(name):
    cfg = ProviderConfig(enabled=True, model="m", base_url="http://x", api_key_env="K")
    with pytest.raises(ProviderNotImplementedError, match="nicht implementiert"):
        create_provider(name, cfg)


async def test_openai_compatible_chat_and_tool_call_with_api_key(monkeypatch):
    from jarvis.ai.providers.cloud import OpenAICompatibleProvider

    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "content": "Fertig",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "system_info",
                                        "arguments": '{"x": 1}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.test/v1/",
        transport=httpx.MockTransport(handler),
    )
    cfg = ProviderConfig(
        enabled=True,
        base_url="https://api.test/v1",
        model="test-model",
        api_key_env="TEST_LLM_KEY",
    )
    llm = OpenAICompatibleProvider(cfg, client)
    response = await llm.chat(MSGS, [TOOL])
    await llm.aclose()

    assert response.content == "Fertig"
    assert response.tool_calls == [ToolCall("system_info", {"x": 1}, "call-1")]
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["tools"][0]["function"]["name"] == "system_info"


async def test_openai_compatible_streams_tokens_and_tool_calls():
    from jarvis.ai.providers.cloud import OpenAICompatibleProvider

    chunks = [
        {"choices": [{"delta": {"content": "Guten "}}]},
        {
            "choices": [
                {
                    "delta": {
                        "content": "Tag.",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "system_info", "arguments": '{"x":1}'},
                            }
                        ],
                    }
                }
            ]
        },
    ]

    def handler(request):
        data = "\n".join(f"data: {json.dumps(chunk)}" for chunk in chunks) + "\n\ndata: [DONE]\n\n"
        return httpx.Response(200, text=data)

    client = httpx.AsyncClient(
        base_url="https://api.test/v1/",
        transport=httpx.MockTransport(handler),
    )
    cfg = ProviderConfig(enabled=True, base_url="https://api.test/v1", model="test-model")
    llm = OpenAICompatibleProvider(cfg, client)
    tokens = []

    async def collect(text):
        tokens.append(text)

    response = await llm.stream_response(MSGS, [TOOL], collect)
    streamed = [part async for part in llm.stream(MSGS)]
    await llm.aclose()

    assert "".join(tokens) == "Guten Tag."
    assert streamed == ["Guten ", "Tag."]
    assert response.content == "Guten Tag."
    assert response.tool_calls == [ToolCall("system_info", {"x": 1}, "call-1")]


def test_unknown_provider():
    with pytest.raises(ProviderConfigError, match="Unbekannter Provider"):
        create_provider("foo", ProviderConfig(enabled=True, model="m"))


def test_missing_model_and_url():
    with pytest.raises(ProviderConfigError, match="kein Modell"):
        create_provider("ollama", ProviderConfig(enabled=True, base_url="http://x"))
    with pytest.raises(ProviderConfigError, match="base_url"):
        create_provider("ollama", ProviderConfig(enabled=True, model="m"))
