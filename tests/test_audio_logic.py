"""Tests der Audio-Logik ohne Mikrofon, Modelle oder Lautsprecher."""

import pytest

from jarvis.audio.errors import STTError, TTSError
from jarvis.audio.microphone import SilenceEndpointer
from jarvis.audio.stt import clean_transcript, create_stt
from jarvis.audio.tts import PiperTTS, create_tts, prepare_text_for_speech, split_sentences
from jarvis.audio.voice import VoiceIO
from jarvis.audio.wake_word import NullWakeWord, OpenWakeWordDetector, create_wake_word
from jarvis.config.loader import STTConfig, TTSConfig, WakeWordConfig
from jarvis.utils.logging import redact


def test_endpointer_waits_for_speech_then_silence():
    ep = SilenceEndpointer(threshold=0.1, silence_duration_s=0.2, block_s=0.05)  # 4 Blöcke
    assert [ep.feed(0.0) for _ in range(10)] == [False] * 10  # Stille vor Sprache zählt nicht
    assert ep.speech_started is False
    assert ep.feed(0.5) is False and ep.speech_started
    assert [ep.feed(0.0) for _ in range(4)] == [False, False, False, True]


def test_endpointer_resets_silence_counter_on_new_speech():
    ep = SilenceEndpointer(0.1, 0.2, 0.05)
    ep.feed(0.5)
    ep.feed(0.0)
    ep.feed(0.0)
    ep.feed(0.5)  # Sprechpause war kürzer als die Endschwelle
    assert [ep.feed(0.0) for _ in range(3)] == [False, False, False]
    assert ep.feed(0.0) is True


def test_clean_transcript_filters_hallucinations_and_whitespace():
    assert clean_transcript("  Wie   spät ist es? ") == "Wie spät ist es?"
    assert clean_transcript("Untertitel der Amara.org-Community") == ""
    assert clean_transcript("") == ""


def test_prepare_text_for_speech_strips_markdown():
    text = "**Hallo** `Welt`! Siehe [Link](http://x.de) 😀\n```\ncode\n```"
    out = prepare_text_for_speech(text)
    assert "*" not in out and "`" not in out and "http" not in out and "😀" not in out
    assert "Hallo Welt" in out and "Link" in out


def test_split_sentences():
    assert split_sentences("Eins. Zwei! Drei? Vier") == ["Eins.", "Zwei!", "Drei?", "Vier"]
    assert split_sentences("") == []


async def test_voice_speaks_complete_sentences_as_stream_chunks_arrive():
    voice = VoiceIO.__new__(VoiceIO)
    spoken = []

    async def record(text):
        spoken.append(text)

    voice.speak = record

    async def chunks():
        yield "Hallo. "
        yield "Wie geht"
        yield " es dir?"

    await VoiceIO.speak_stream(voice, chunks())
    assert spoken == ["Hallo.", "Wie geht es dir?"]


def test_engine_factories_reject_unknown_engines(tmp_path):
    with pytest.raises(STTError, match="Unbekannte STT-Engine"):
        create_stt(STTConfig(engine="x"), tmp_path)
    with pytest.raises(TTSError, match="Unbekannte TTS-Engine"):
        create_tts(TTSConfig(engine="x"), tmp_path)


def test_engines_are_lazy_and_report_status(tmp_path):
    stt = create_stt(STTConfig(), tmp_path)
    tts = create_tts(TTSConfig(), tmp_path)
    assert "lädt beim ersten Gebrauch" in stt.status()
    assert "de_DE-thorsten-high" in tts.status()
    assert isinstance(tts, PiperTTS)


def test_stt_requires_16khz(tmp_path):
    with pytest.raises(STTError, match="16 kHz"):
        create_stt(STTConfig(), tmp_path).transcribe([0.0], 44100)


def test_wake_word_interface():
    assert isinstance(create_wake_word(WakeWordConfig()), NullWakeWord)
    assert NullWakeWord().process([0.0]) is False
    detector = create_wake_word(WakeWordConfig(enabled=True, engine="openwakeword"))
    assert isinstance(detector, OpenWakeWordDetector)


def test_openwakeword_uses_model_threshold_and_reset(monkeypatch):
    np = pytest.importorskip("numpy")
    detector = OpenWakeWordDetector(WakeWordConfig(enabled=True, engine="openwakeword", threshold=0.7))

    class Model:
        reset_called = False

        def predict(self, frame):
            assert frame.dtype == np.int16 and len(frame) == 1280
            return {"hey_jarvis": 0.8}

        def reset(self):
            self.reset_called = True

    model = Model()
    detector._model = model
    assert detector.process(np.ones(1280, dtype=np.float32) * 0.2)
    detector.reset()
    assert model.reset_called


def test_voice_io_builds_without_audio_packages(settings):
    voice = VoiceIO(settings)  # darf nichts laden und nicht abstürzen
    assert voice.can_listen and voice.can_speak
    voice.close()


def test_secrets_are_redacted_in_logs():
    assert "sk-abcdefgh12345678" not in redact("key sk-abcdefgh12345678 end")
    assert "geheim" not in redact("api_key=geheim123 und mehr")
    assert "geheim" not in redact("Authorization: Bearer geheim123")
    assert redact("normaler Text") == "normaler Text"


def _block_sounddevice_with_oserror(monkeypatch):
    """Simuliert eine fehlende PortAudio-Bibliothek (sounddevice wirft dann OSError beim Import)."""
    import builtins
    import sys

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_portaudio_becomes_audio_error_not_crash(monkeypatch):
    from jarvis.audio.errors import AudioError
    from jarvis.audio.microphone import Recorder, list_devices
    from jarvis.audio.tts import AudioClip, Speaker, TTSEngine

    class Dummy(TTSEngine):
        name = "dummy"

        def synthesize(self, text):
            return AudioClip([0.0], 22050)

        def status(self):
            return "dummy"

    _block_sounddevice_with_oserror(monkeypatch)
    with pytest.raises(AudioError, match="PortAudio"):
        list_devices()
    with pytest.raises(AudioError, match="PortAudio"):
        Recorder().record(__import__("threading").Event())
    with pytest.raises(AudioError, match="PortAudio"):
        Speaker(Dummy()).speak("Hallo.")
    Speaker(Dummy()).stop()  # darf nie werfen
