"""Einstiegspunkt: ``jarvis`` bzw. ``python -m jarvis``."""

from __future__ import annotations

import argparse
import asyncio
import sys

from jarvis import __version__
from jarvis.ai.base import ProviderError
from jarvis.audio.errors import AudioError
from jarvis.audio.microphone import list_devices
from jarvis.config import ConfigError, load_config
from jarvis.core.application import Application
from jarvis.interface.terminal import TerminalUI
from jarvis.utils.logging import setup_logging


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jarvis", description="Lokaler KI-Sprachassistent")
    parser.add_argument("--config", help="Pfad zur config.yaml (Standard: config/config.yaml)")
    parser.add_argument(
        "--check", action="store_true", help="Konfiguration, LLM-Server und Plugins prüfen und beenden"
    )
    parser.add_argument("--list-devices", action="store_true", help="Audiogeräte auflisten")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    return parser.parse_args(argv)


async def _run(app: Application, check_only: bool) -> int:
    try:
        if check_only:
            report = await app.startup()
            print(f"LLM:     {'OK' if report.provider.ok else 'FEHLER'} – {report.provider.message}")
            print(f"STT:     {app.voice.stt_status}")
            print(f"TTS:     {app.voice.tts_status}")
            for plugin in report.plugins:
                print(f"Plugin:  {plugin.name}: {'geladen' if plugin.loaded else plugin.error}")
            for warning in report.warnings:
                print(f"Hinweis: {warning}")
            return 0 if report.provider.ok else 1
        await TerminalUI(app).run()
        return 0
    finally:
        await app.shutdown()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.list_devices:
        try:
            print("\n".join(list_devices()))
        except AudioError as exc:
            sys.exit(f"Fehler: {exc}")
        return

    try:
        settings = load_config(args.config)
    except ConfigError as exc:
        sys.exit(f"Konfigurationsfehler: {exc}")

    log_file = settings.resolve(settings.logging.file) if settings.logging.file else None
    setup_logging(settings.logging.level, log_file, settings.logging.json_format, settings.logging.console)

    try:
        app = Application(settings)
    except ProviderError as exc:
        sys.exit(f"Provider-Fehler: {exc}")

    try:
        code = asyncio.run(_run(app, args.check))
    except KeyboardInterrupt:
        print("\nAuf Wiedersehen.")
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
