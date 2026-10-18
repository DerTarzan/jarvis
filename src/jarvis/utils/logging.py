"""Logging-Setup: Datei-Log, optional JSON, mit Schwärzung von Geheimnissen."""

from __future__ import annotations

import json
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(
        r"(?i)(api[_-]?key|token|authorization|bearer)(['\"]?\s*[:=]\s*['\"]?(?:bearer\s+)?)"
        r"[^\s'\",;]+"
    ),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
]


def redact(text: str) -> str:
    """Ersetzt typische Schlüsselmuster durch ***."""
    text = _SECRET_PATTERNS[0].sub("sk-***", text)
    text = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)
    return _SECRET_PATTERNS[2].sub("AIza***", text)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
    console: bool = False,
) -> None:
    """Konfiguriert den Root-Logger 'jarvis'. Mehrfachaufruf ist sicher."""
    logger = logging.getLogger("jarvis")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = (
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handlers: list[logging.Handler] = []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"))
    if console:
        handlers.append(logging.StreamHandler())
    if not handlers:
        handlers.append(logging.NullHandler())
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
