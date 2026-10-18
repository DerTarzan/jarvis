"""Tests für Hands-free: Wake-Word-Erkennung, Äußerungserkennung (mit simuliertem Mikrofon)
und die komplette Dialogschleife (mit simulierter Spracherkennung und Sprachausgabe)."""

import asyncio
import threading
import time

import numpy as np
import pytest

from conftest import FakeProvider
from jarvis.audio import microphone
from jarvis.audio.errors import AudioError, WakeWordError
from jarvis.audio.microphone import Recorder, threshold_from_ambient
from jarvis.audio.wake_word import TranscriptWakeWord
from jarvis.config.loader import WakeWordConfig
from jarvis.core import events as ev
from jarvis.core import handsfree
from jarvis.core.application import Application
from jarvis.core.handsfree import closing_reply
from jarvis.tools.registry import Tool


# --- Wake-Word-Erkennung --------------------------------------------------------------
@pytest.fixture
def wake():
    return TranscriptWakeWord(WakeWordConfig(enabled=True, engine="whisper"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Jarvis, wie spät ist es?", "wie spät ist es?"),
        ("jarvis wie spät ist es", "wie spät ist es"),
        ("Hey Jarvis, mach das Licht an.", "mach das Licht an."),
        ("Okay Jarvis was ist 2 plus 2?", "was ist 2 plus 2?"),
        ("Jarvis.", ""),
        ("Jarvis", ""),
        ("Dscharvis, wie geht's?", "wie geht's?"),  # typische Whisper-Schreibweise
        ("Charvis was läuft?", "was läuft?"),
        ("Jarwis, Zeit?", "Zeit?"),  # unscharf, aber ähnlich genug
        ("Wie spät ist es?", None),
        ("Ich habe gestern Jarvis gebaut", None),  # Wake Word nicht am Anfang
        ("Das ist Jarvis", None),
        ("Javascript ist eine Sprache", None),
        ("", None),
    ],
)
def test_wake_word_matching(wake, text, expected):
    assert wake.match(text) == expected


def test_custom_phrase_and_variants():
    custom = TranscriptWakeWord(
        WakeWordConfig(enabled=True, engine="whisper", phrase="computer", variants=["kompjuter"])
    )
    assert custom.match("Computer, Licht an") == "Licht an"
    assert custom.match("Kompjuter Licht an") == "Licht an"
    assert custom.match("Jarvis, Licht an") is None  # Jarvis-Varianten gelten nur für 'jarvis'


def test_empty_phrase_is_rejected():
    with pytest.raises(WakeWordError):
        TranscriptWakeWord(WakeWordConfig(enabled=True, engine="whisper", phrase=" ! "))


@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("Danke!", "Gern geschehen."),
        ("Okay, danke.", "Gern geschehen."),
        ("Stopp", "In Ordnung."),
        ("Danke schön für die Hilfe", None),
        ("Wie spät ist es", None),
    ],
)
def test_closing_phrases(text, reply):
    assert closing_reply(text) == reply


# --- Äußerungserkennung mit simuliertem Mikrofon -----------------------------------------
class FakeStream:
    def __init__(self, levels):
        self.levels = levels

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames):
        time.sleep(0.001)
        level = self.levels.pop(0) if self.levels else 0.0
        return np.full((frames, 1), level, dtype=np.float32), False


class FakeSD:
    class PortAudioError(Exception):
        pass

    def __init__(self, levels=(), error=None):
        self.levels = list(levels)
        self.error = error

    def InputStream(self, **kwargs):  # noqa: N802 - Name wie in sounddevice
        if self.error:
            raise self.error
        return FakeStream(self.levels)


BLOCK = 800  # 16 kHz * 50 ms


def recorder_with(monkeypatch, sd):
    monkeypatch.setattr(microphone, "_import_audio", lambda: (np, sd))
    return Recorder(sample_rate=16000, silence_threshold=0.012, silence_duration_s=0.2, max_record_s=5)


def test_capture_returns_one_utterance_with_preroll(monkeypatch):
    levels = [0.0] * 5 + [0.1] * 10 + [0.0] * 10
    audio = recorder_with(monkeypatch, FakeSD(levels)).capture_utterance(threading.Event())
    assert audio is not None
    assert audio[: 5 * BLOCK].max() == 0.0  # Vorlauf vor dem Sprechbeginn ist enthalten
    assert int((audio > 0.05).sum()) == 10 * BLOCK  # komplette Sprache, nichts abgeschnitten
    assert len(audio) == (5 + 10 + 4) * BLOCK  # endet nach 4 Stille-Blöcken (0,2 s)


def test_short_noise_is_ignored_and_waiting_continues(monkeypatch):
    levels = [0.1] * 2 + [0.0] * 10 + [0.1] * 10 + [0.0] * 10  # Klick, dann echte Sprache
    audio = recorder_with(monkeypatch, FakeSD(levels)).capture_utterance(threading.Event())
    assert audio is not None
    assert int((audio > 0.05).sum()) == 10 * BLOCK  # nur die Sprache, der Klick wurde verworfen


def test_capture_times_out_without_speech(monkeypatch):
    rec = recorder_with(monkeypatch, FakeSD([]))
    assert rec.capture_utterance(threading.Event(), wait_timeout_s=0.05) is None


def test_capture_returns_none_when_stopped(monkeypatch):
    stop = threading.Event()
    stop.set()
    assert recorder_with(monkeypatch, FakeSD([0.1] * 20)).capture_utterance(stop) is None


def test_capture_device_error_becomes_audio_error(monkeypatch):
    sd = FakeSD(error=FakeSD.PortAudioError("kein Gerät"))
    with pytest.raises(AudioError, match="Mikrofon nicht nutzbar"):
        recorder_with(monkeypatch, sd).capture_utterance(threading.Event())


def test_calibration_raises_threshold_in_noisy_rooms(monkeypatch):
    rec = recorder_with(monkeypatch, FakeSD([0.01] * 40))
    assert rec.calibrate(threading.Event(), seconds=1.0) == pytest.approx(0.03)
    # Ein lauter Raum wird begrenzt, damit Sprache erkennbar bleibt
    assert threshold_from_ambient(0.012, 0.5) == 0.2
    assert threshold_from_ambient(0.012, 0.001) == 0.012  # leiser Raum: Grundschwelle bleibt


# --- Komplette Dialogschleife -------------------------------------------------------------
class FakeHFVoice:
    """Simuliert Mikrofon+STT (Skript aus Texten) und Sprachausgabe."""

    can_listen = can_speak = True
    stt_status = tts_status = "fake"

    def __init__(self, script):
        self.script = list(script)  # str = gesprochene Äußerung, None = Timeout, Exception = Fehler
        self.spoken: list[str] = []
        self.waits: list[float | None] = []
        self.beeps = 0

    async def preload(self):
        return None

    async def calibrate(self, stop, seconds=1.0):
        return 0.01

    async def capture(self, stop, wait_timeout_s=None):
        self.waits.append(wait_timeout_s)
        if not self.script:
            stop.set()  # Skript zu Ende → Hands-free beenden
            return None
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def transcribe(self, audio):
        return audio

    async def speak(self, text):
        self.spoken.append(text)

    async def beep(self):
        self.beeps += 1

    def stop_speaking(self): ...
    def refresh_status(self): ...
    def close(self): ...


async def run_handsfree(settings, script, provider=None, followup_s=60, engine="whisper", detector=None):
    settings.general.activation_mode = "handsfree"
    settings.wake_word.enabled = True
    settings.wake_word.engine = engine
    settings.wake_word.followup_window_s = followup_s
    settings.audio.post_speech_pause_s = 0
    voice = FakeHFVoice(script)
    provider = provider or FakeProvider()
    app = Application(settings, provider=provider, voice=voice)
    if detector is not None:
        app.wake_detector = detector
    events = {"user": [], "assistant": [], "errors": [], "info": []}
    app.events.subscribe(ev.USER_TEXT, lambda text, source: events["user"].append((text, source)))
    app.events.subscribe(ev.ASSISTANT_TEXT, lambda text: events["assistant"].append(text))
    app.events.subscribe(ev.ERROR, lambda message: events["errors"].append(message))
    app.events.subscribe(ev.INFO, lambda message: events["info"].append(message))
    assert await app.start_handsfree()
    await asyncio.wait_for(app._hf_task, timeout=5)
    await app.shutdown()
    return app, voice, provider, events


def user_messages(provider):
    return [m.content for m in provider.calls[-1][0] if m.role == "user"]


async def test_wake_word_with_command_runs_without_ack_and_ignores_other_speech(settings):
    script = ["Das Wetter ist heute schön", "Jarvis, wie spät ist es?"]
    _, voice, provider, events = await run_handsfree(settings, script)
    assert len(provider.calls) == 1  # nur die Äußerung mit Wake Word erreicht das Modell
    assert user_messages(provider) == ["wie spät ist es?"]
    assert voice.spoken == ["Hallo!"] and voice.beeps == 0  # keine Bestätigung nötig
    assert events["user"] == [("wie spät ist es?", "voice")]
    assert events["errors"] == []
    # Erst nach der Antwort läuft das Folgefenster (Wartezeit begrenzt), vorher nicht
    assert voice.waits[:2] == [None, None] and voice.waits[2] is not None and voice.waits[2] <= 60


async def test_openwakeword_falls_back_to_transcript_for_jarvis_phrase(settings):
    class MissedWakeWord:
        def preload(self): ...

        def process(self, audio):
            return False

    _, _, provider, _ = await run_handsfree(
        settings,
        ["Das Wetter ist heute schön", "Jarvis, wie spät ist es?"],
        engine="openwakeword",
        detector=MissedWakeWord(),
    )

    assert len(provider.calls) == 1
    assert user_messages(provider) == ["wie spät ist es?"]


async def test_wake_word_alone_is_acknowledged_then_command_is_taken(settings):
    _, voice, provider, _ = await run_handsfree(settings, ["Jarvis", "was ist zwei plus zwei?"])
    assert voice.spoken == ["Ja?", "Hallo!"]
    assert user_messages(provider) == ["was ist zwei plus zwei?"]
    assert voice.waits[1] == settings.wake_word.listen_timeout_s


async def test_ack_can_be_a_beep(settings):
    settings.wake_word.ack_phrase = ""
    _, voice, _, _ = await run_handsfree(settings, ["Jarvis", "Hallo"])
    assert voice.beeps == 1 and voice.spoken == ["Hallo!"]


async def test_wake_word_alone_then_silence_does_nothing(settings):
    _, voice, provider, _ = await run_handsfree(settings, ["Jarvis", None])
    assert provider.calls == [] and voice.spoken == ["Ja?"]


async def test_followup_needs_no_wake_word(settings):
    provider = FakeProvider()
    _, voice, provider, _ = await run_handsfree(settings, ["Jarvis, eins", "und zwei?"], provider)
    assert len(provider.calls) == 2
    assert user_messages(provider) == ["eins", "und zwei?"]  # derselbe Gesprächskontext
    assert voice.spoken == ["Hallo!", "Hallo!"]


async def test_followup_expiry_requires_wake_word_again(settings):
    # Folgefenster 0 s = sofort abgelaufen → "ohne Wake Word" wird ignoriert, "Jarvis, zwei" läuft
    script = ["Jarvis, eins", "ohne Wake Word", "Jarvis, zwei"]
    _, _, provider, _ = await run_handsfree(settings, script, followup_s=0)
    assert user_messages(provider) == ["eins", "zwei"]


async def test_thanks_ends_conversation_without_asking_the_model(settings):
    _, voice, provider, events = await run_handsfree(settings, ["Jarvis, eins", "Danke"])
    assert len(provider.calls) == 1
    assert voice.spoken == ["Hallo!", "Gern geschehen."]
    assert voice.waits[-1] is None  # kein Folgefenster mehr nach „Danke“
    assert events["assistant"][-1] == "Gern geschehen."


@pytest.mark.parametrize(("answer", "expected"), [("Ja", True), ("Nein", False), ("Ja, aber nein", False)])
async def test_handsfree_tool_confirmation_is_explicit_and_voice_based(settings, answer, expected):
    settings.general.activation_mode = "handsfree"
    settings.wake_word.enabled = True
    settings.wake_word.engine = "whisper"
    settings.audio.post_speech_pause_s = 0
    voice = FakeHFVoice([answer])
    app = Application(settings, provider=FakeProvider(), voice=voice)
    loop = handsfree.HandsFreeLoop(app, threading.Event())

    tool = Tool("homeassistant_control_light", "steuert eine Lampe", lambda: None)
    confirmed = await loop._confirm_tool(tool, {})

    assert confirmed is expected
    assert "Bitte sag Ja oder Nein" in voice.spoken[0]
    assert voice.waits == [settings.wake_word.listen_timeout_s]
    await app.shutdown()


async def test_openwakeword_can_interrupt_speech_and_transcribe_followup(settings):
    settings.wake_word.enabled = True
    settings.wake_word.engine = "openwakeword"

    class InterruptVoice(FakeHFVoice):
        def __init__(self):
            super().__init__([])
            self.stopped = False

        async def capture(self, stop, wait_timeout_s=None):
            return "captured-audio"

        async def transcribe(self, audio):
            return "Jarvis, mach das Licht an"

        def stop_speaking(self):
            self.stopped = True

    class Detector:
        def process(self, audio):
            return True

    app = Application(settings, provider=FakeProvider(), voice=InterruptVoice())
    app.wake_detector = Detector()
    loop = handsfree.HandsFreeLoop(app, threading.Event())

    assert await loop._wait_for_interrupt(threading.Event()) == "Jarvis, mach das Licht an"
    assert app.voice.stopped
    await app.shutdown()


async def test_transcript_wake_word_can_interrupt_speech(settings):
    settings.wake_word.enabled = True
    settings.wake_word.engine = "whisper"

    class InterruptVoice(FakeHFVoice):
        def __init__(self):
            super().__init__([])
            self.stopped = False
            self.transcripts = iter(["kein Wake Word", "Jarvis, stopp"])

        async def capture(self, stop, wait_timeout_s=None):
            return "captured-audio"

        async def transcribe(self, audio):
            return next(self.transcripts)

        def stop_speaking(self):
            self.stopped = True

    app = Application(settings, provider=FakeProvider(), voice=InterruptVoice())
    loop = handsfree.HandsFreeLoop(app, threading.Event())

    assert await loop._wait_for_interrupt(threading.Event()) == "Jarvis, stopp"
    assert app.voice.stopped
    await app.shutdown()


async def test_repeated_microphone_failures_stop_handsfree_cleanly(settings, monkeypatch):
    monkeypatch.setattr(handsfree, "RETRY_DELAY_S", 0)
    script = [AudioError("Mikrofon weg")] * 3
    app, _, provider, events = await run_handsfree(settings, script)
    assert events["errors"][:3] == ["Mikrofon weg"] * 3
    assert "wird beendet" in events["errors"][3]
    assert provider.calls == [] and not app.handsfree_active


async def test_a_single_microphone_glitch_does_not_stop_handsfree(settings, monkeypatch):
    monkeypatch.setattr(handsfree, "RETRY_DELAY_S", 0)
    _, _, provider, events = await run_handsfree(settings, [AudioError("kurz weg"), "Jarvis, hallo"])
    assert user_messages(provider) == ["hallo"] and events["errors"] == ["kurz weg"]


async def test_provider_error_does_not_end_the_dialog(settings):
    from jarvis.ai import LLMResponse, ProviderUnavailableError

    provider = FakeProvider([ProviderUnavailableError("Ollama weg"), LLMResponse("Wieder da.")])
    _, voice, provider, events = await run_handsfree(settings, ["Jarvis, eins", "Jarvis, zwei"], provider)
    assert events["errors"] == ["Ollama weg"]
    assert voice.spoken == ["Wieder da."]


async def test_handsfree_refuses_to_start_without_wake_word_config(settings):
    settings.general.activation_mode = "handsfree"
    app = Application(settings, provider=FakeProvider(), voice=FakeHFVoice([]))
    errors = []
    app.events.subscribe(ev.ERROR, lambda message: errors.append(message))
    assert await app.start_handsfree() is False
    assert "wake_word.enabled" in errors[0]
    report = await app.startup()
    assert any("Hands-free braucht" in w for w in report.warnings)
    await app.shutdown()


async def test_stop_handsfree_ends_the_background_task(settings):
    settings.general.activation_mode = "handsfree"
    settings.wake_word.enabled = True
    settings.wake_word.engine = "whisper"

    class BlockingVoice(FakeHFVoice):
        async def capture(self, stop, wait_timeout_s=None):
            while not stop.is_set():
                await asyncio.sleep(0.01)
            return None

    app = Application(settings, provider=FakeProvider(), voice=BlockingVoice([]))
    await app.start_handsfree()
    await asyncio.sleep(0.05)
    assert app.handsfree_active
    await app.stop_handsfree()
    assert not app.handsfree_active
    await app.shutdown()
