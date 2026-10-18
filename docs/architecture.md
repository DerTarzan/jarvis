# Architektur

## Überblick

```
                 ┌─────────────────────── interface/terminal.py (UI) ───────────────────────┐
                 │  nutzt nur Application + Events; enthält keine KI-/Audio-/Plugin-Logik    │
                 └───────────────────────────────┬───────────────────────────────────────────┘
                                                 │
                      core/application.py  (baut alles zusammen, Lebenszyklus, Zustände)
        ┌────────────┬───────────────┬───────────┴──────────┬──────────────┬─────────────────┐
   core/assistant   audio/voice      plugins/manager     tools/registry   conversation/     core/events
   (LLM-Schleife)   (VoiceIO)                              + executor      manager + memory  (EventBus)
        │              │  │  │             │                   ▲
   ai/base.py     microphone stt tts       └── lädt plugins/<name>/plugin.py, registriert Tools
   (LLMProvider)   (+ wake_word-Schnittstelle)
        │
   ai/manager.py → providers/ollama.py
                   providers/cloud.py (OpenAI-kompatibel + Erweiterungspunkte)
```

Abhängigkeitsrichtung: `interface → core → {ai, audio, conversation, plugins, tools}`; `config`
und `utils` sind Blätter. Es gibt keine zyklischen Importe; Module kommunizieren über
`Application` und den `EventBus`.

## Datenfluss einer gesprochenen Frage

1. UI: Aufnahme starten (`Application.listen`) → `VoiceIO.listen`.
2. `Recorder.record` (Thread, `sounddevice`) nimmt auf, bis Enter bzw. Stille/Maximaldauer.
3. `STTEngine.transcribe` (Thread) liefert Text; Halluzinationsfilter in `clean_transcript`.
4. `Application.process` zeigt den Text (Event `user_text`), ruft `Assistant.respond`.
5. `Assistant`: Kontext aus `ConversationManager` → `LLMProvider.stream_response`. Fordert das Modell Tools
   an, führt `ToolExecutor` sie aus (Validierung, ggf. Bestätigung, Timeout), die Ergebnisse gehen
   als `tool`-Nachrichten zurück an das Modell (max. `tools.max_tool_rounds` Runden).
6. Antwort wird während der Generierung satzweise von `Speaker` gesprochen und danach angezeigt (Event `assistant_text`)
   (Synthese und Wiedergabe parallel, per `/stop` unterbrechbar).
7. Zurück zu „bereit“.

**Fehlerisolierung:** Provider-Fehler (`ProviderError`-Familie) werden mit verständlicher Meldung
über das Event `error` gezeigt, der Gesprächszustand wird zurückgerollt. Audio-/STT-/TTS-Fehler
betreffen nur den jeweiligen Schritt. Plugin-Fehler betreffen nur das jeweilige Plugin. Zusätzlich
fängt `Application` unerwartete Ausnahmen ab, damit die Sitzung weiterläuft.

## Erweiterungspunkte

### Neuen LLM-Provider hinzufügen

1. Klasse von `jarvis.ai.base.LLMProvider` ableiten: `name`, `is_cloud`, `model`, `chat()`, `check()`
   (optional `stream()`, `aclose()`); `supports_tools` setzen, wenn natives Tool-Calling möglich ist.
2. Fehler in die `ProviderError`-Unterklassen übersetzen (nicht erreichbar, Timeout, Modell fehlt …).
3. Mit `register_provider("name", Klasse)` (in `ai/manager.py`) bzw. im Dictionary `_PROVIDERS`
   registrieren; Konfig unter `ai.providers.<name>` ergänzen.
4. Tests nach dem Muster in `tests/test_ai_providers.py` (httpx `MockTransport`).

`openai_compatible` ist implementiert (HTTP Chat Completions, Tools und SSE-Streaming); die anderen
Cloud-Klassen setzen `implemented = False`. Ein stiller Wechsel zwischen Providern existiert nicht.

### Neue STT-/TTS-Engine

`STTEngine` bzw. `TTSEngine` ableiten und in `create_stt` / `create_tts` eintragen. Engine-Name
kommt aus `stt.engine` / `tts.engine`. Das LLM bleibt davon unberührt.

### Wake Word

Implementiert ist der Hands-free-Modus (`core/handsfree.py`): `Recorder.capture_utterance` schneidet
per Lautstärke eine Äußerung aus. Standardmäßig prüft `TranscriptWakeWord` das lokale
`FasterWhisperSTT`-Transkript auf die konfigurierte Phrase. Die optionale openWakeWord-Engine nutzt
zusätzlich das Audiomodell; das mitgelieferte Modell ist auf „Hey Jarvis“ trainiert und nicht auf
das allein gesprochene „Jarvis“.
Danach folgen Bestätigung, Befehl, Antwort und ein Folgefenster ohne Wake Word. Die Schleife läuft
als Hintergrund-Task der `Application`; Tastatur und Stimme teilen sich ein Turn-Lock.
Während der Sprachausgabe lauscht eine zweite Capture-Aufgabe auf das Wake Word und kann TTS
unterbrechen. Akustisches Echo kann Fehlaktivierungen verursachen.

### Tools

`Tool(name, description, handler, parameters, requires_confirmation, timeout_s)`. `parameters`
ist ein JSON-Schema (`type: object`); Parameter werden vor jedem Aufruf validiert, der Handler
läuft in einem Thread (sync) bzw. als Coroutine mit Timeout. Gefährliche Aktionen:
`requires_confirmation=True` – ohne Bestätigung der UI wird nie ausgeführt. Das Modell kann nur
registrierte Tools nutzen; Shell- oder Python-Ausführung durch das Modell existiert nicht.
Provider ohne Tool-Calling erhalten keine Tool-Spezifikationen.

### Gedächtnis

`MemoryStore` (Sitzungsverlauf) und `LongTermMemory` (Fakten) sind Schnittstellen;
`SQLiteMemory` implementiert beide lokal (LIKE-Suche). Bei `memory.engine: sqlite` registriert
Jarvis `remember_fact`, `search_memory` und das bestätigungspflichtige `forget_memory`. Fakten
werden nur auf ausdrückliche Bitte gespeichert; der Gesprächsverlauf bleibt separat konfigurierbar.

## Konfigurationsbereiche

`general`, `ai`, `stt`, `tts`, `wake_word`, `audio`, `conversation`, `memory`, `plugins`, `tools`,
`logging` (siehe `config/config.yaml`). Pfade sind relativ zum Projektordner.

## Sicherheitsgrenzen

Plugins sind keine Sandbox (siehe `docs/adding_plugins.md`). Zugangsdaten nur über Umgebung;
`utils/logging.py` schwärzt Schlüsselmuster. Cloud-Provider zeigen beim Start eine
Datenschutzwarnung (`StartupReport.warnings`).
