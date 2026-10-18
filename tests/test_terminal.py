from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from jarvis.interface.terminal import TerminalUI


def test_voice_transcript_uses_one_dimmed_replaceable_line():
    output = StringIO()
    ui = TerminalUI.__new__(TerminalUI)
    ui.console = Console(file=output, force_terminal=True, width=40, color_system="standard")
    ui._speech_line_visible = False

    ui._on_user("Jarvis wie geht's dir?", source="voice")
    first_line = output.getvalue()
    assert "Du: Jarvis wie geht's dir?" in first_line
    assert "\x1b[2m" in first_line
    assert not first_line.endswith("\n")

    ui._on_user("Es geht mir gut.", source="voice")
    replaced_line = output.getvalue()[len(first_line) :]
    assert "\x1b[2K" in replaced_line
    assert "Du: Es geht mir gut." in replaced_line
    assert not replaced_line.endswith("\n")

    before_progress = output.getvalue()
    ui._on_state("denkt nach")
    ui._on_tool("current_time", True)
    assert output.getvalue() == before_progress

    ui.app = SimpleNamespace(settings=SimpleNamespace(general=SimpleNamespace(assistant_name="Jarvis")))
    ui._on_assistant("Mir geht's gut.")
    response = output.getvalue()[len(before_progress) :]
    assert "\x1b[2K" in response
    assert "Jarvis: " in response and "Mir geht's gut." in response
