from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.ai.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    ProviderStatus,
    ToolCall,
    ToolSpec,
)
from jarvis.config import Settings

ROOT = Path(__file__).resolve().parent.parent


class FakeProvider(LLMProvider):
    """Skriptbarer Provider ohne Netzwerk."""

    name = "fake"
    supports_tools = True

    def __init__(self, responses: list[LLMResponse | Exception] | None = None) -> None:
        self.responses = list(responses or [LLMResponse("Hallo!")])
        self.calls: list[tuple[list[ChatMessage], list[ToolSpec] | None]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    async def chat(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def check(self):
        return ProviderStatus(True, "fake ok")


def tool_call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s._base_dir = tmp_path
    return s


@pytest.fixture
def project_root() -> Path:
    return ROOT
