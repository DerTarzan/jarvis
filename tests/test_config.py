from pathlib import Path

import pytest

from jarvis.config import ConfigError, load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config" / "config.yaml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_config_is_valid(project_root):
    settings = load_config(project_root / "config" / "config.yaml", load_env=False)
    assert settings.ai.provider == "ollama"
    assert settings.active_provider.model
    assert settings.base_dir == project_root


def test_empty_file_uses_defaults(tmp_path):
    settings = load_config(write(tmp_path, ""), load_env=False)
    assert settings.ai.provider == "ollama"
    assert settings.resolve("data") == tmp_path / "data"


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="nicht gefunden"):
        load_config(tmp_path / "nope.yaml", load_env=False)


def test_yaml_syntax_error(tmp_path):
    with pytest.raises(ConfigError, match="YAML"):
        load_config(write(tmp_path, "a: [unclosed"), load_env=False)


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="tts.speeed"):
        load_config(write(tmp_path, "tts:\n  speeed: 1.0\n"), load_env=False)


def test_selected_provider_must_be_enabled(tmp_path):
    cfg = "ai:\n  provider: openai\n"
    with pytest.raises(ConfigError, match="enabled: false"):
        load_config(write(tmp_path, cfg), load_env=False)


def test_unknown_provider_selected(tmp_path):
    with pytest.raises(ConfigError, match="nicht unter ai.providers"):
        load_config(write(tmp_path, "ai:\n  provider: foo\n"), load_env=False)


def test_value_range_validation(tmp_path):
    with pytest.raises(ConfigError, match="conversation.max_messages"):
        load_config(write(tmp_path, "conversation:\n  max_messages: 1\n"), load_env=False)


def test_api_key_comes_from_environment_only(monkeypatch, tmp_path):
    settings = load_config(write(tmp_path, ""), load_env=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert settings.ai.providers["openai"].api_key() is None
    monkeypatch.setenv("OPENAI_API_KEY", "abc")
    assert settings.ai.providers["openai"].api_key() == "abc"
