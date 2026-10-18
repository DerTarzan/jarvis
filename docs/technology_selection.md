# Technologieauswahl (Entscheidungsvorlage, Stand 2026-10-17)

**Was belegt ist und was nicht:** Aussagen mit „Quelle“ stammen aus Recherche am 2026-10-17.
Alle anderen stammen aus allgemeinem Vorwissen und sind vor produktiver Nutzung zu prüfen
(insbesondere Lizenzen). **Es wurden keine eigenen Benchmarks durchgeführt; Qualitäts- und
Geschwindigkeitsaussagen sind Erwartungen, keine Messwerte.**

## Hardware-Annahmen (nicht in der Entwicklungsumgebung gemessen)

Die Entwicklungsumgebung ist eine Cloud-Sandbox (Linux, kein Mikrofon, keine GPU) – eine
Bestandsaufnahme deines PCs war dort nicht möglich. Angenommen wird dein Desktop laut deinen
früheren Angaben: Windows 11, Ryzen 5 7500F, 32 GB DDR5, AMD Radeon RX 7800 XT (16 GB VRAM).
Falls das nicht stimmt, genügt es, Modell und `stt.device` in `config.yaml` anzupassen.
`jarvis --list-devices` listet die Audiogeräte.

## LLM / Inferenzserver

| Option | Bewertung |
|---|---|
| **Ollama (gewählt)** | Einfache Installation unter Windows, REST-API mit Tool-Calling, Modellverwaltung per `ollama pull`. Laut Ollama-Dokumentation (Quelle: docs.ollama.com/gpu) ist die RX 7800 XT als unterstützte AMD-GPU gelistet, unter Windows über ROCm. Ob die Beschleunigung auf deinem System tatsächlich greift, ist ungetestet (`ollama ps` zeigt es). |
| llama.cpp direkt (Vulkan/ROCm) | Mehr Kontrolle, mehr Wartungsaufwand; als zweiter lokaler Adapter später möglich. |
| LM Studio | Gute GUI, OpenAI-kompatible API; als `openai_compatible`-Adapter später denkbar. |

**Modell:** `llama3.1:8b` (Llama-Richtung wie gewünscht). Ein 8B-Modell in 4-Bit passt nach
Erfahrungswerten komfortabel in 16 GB VRAM; die Deutschqualität ist brauchbar, aber nicht
herausragend. Das Modell ist nur ein Konfigurationswert – andere (z. B. Qwen, Gemma) lassen sich
ohne Codeänderung eintragen. Lizenz: Llama-Community-Lizenz (Vorwissen) – für private Nutzung
unkritisch, bei kommerzieller Nutzung Bedingungen prüfen.

## Speech-to-Text

| Option | Bewertung |
|---|---|
| **faster-whisper (gewählt)** | Whisper-Modelle über CTranslate2, MIT-Lizenz (Vorwissen), gute Deutschqualität zu erwarten, eingebaute VAD-Filterung, Modelle werden lokal gecacht. Nachteil: GPU-Beschleunigung gibt es nur für NVIDIA/CUDA – auf deiner AMD-Karte läuft es auf der CPU (`int8`). Für kurze Sprachbefehle ist das mit `small` voraussichtlich ausreichend schnell; ungemessen. |
| whisper.cpp / pywhispercpp | Alternative mit Vulkan-Option, evtl. besser für AMD-GPU; als zusätzliche `STTEngine` leicht ergänzbar (`create_stt`). |
| Vosk | Klein und schnell, aber nach Erwartung schlechtere Erkennung von Umgangssprache. |
| Cloud-STT | Ausgeschlossen (Vorgabe: lokal). |

## Text-to-Speech und Stimme

| Option | Bewertung |
|---|---|
| **Piper + `de_DE-thorsten-high` (gewählt)** | Lokal, schnell (ONNX, CPU), Python-Paket `piper-tts`. Die Thorsten-Stimme für Piper ist laut Hugging-Face-Modellkarte MIT-lizenziert (Quelle: huggingface.co/Thorsten-Voice/Piper). Die Piper-Projektseite weist darauf hin, dass jede Stimme eine Model Card mit Lizenzangaben hat (Quelle: Piper-README) – bei anderen Stimmen prüfen. |
| Coqui XTTS (Stimmklonen) | Könnte eine J.A.R.V.I.S.-ähnlichere Stimme erlauben, aber: schwerer, und das Klonen einer geschützten Originalstimme ist ohne Berechtigung ausgeschlossen. Lizenz des Modells (nach Vorwissen nichtkommerziell) vor Einsatz prüfen. |
| edge-tts | Klingt gut, ist aber ein Cloud-Dienst – widerspricht der Offline-Vorgabe. |
| Windows SAPI (pyttsx3) | Offline, aber mechanisch. |

**Ehrliche Einschätzung zur Stimme:** Thorsten (high) ist eine ruhige, klare, natürlich wirkende
deutsche Männerstimme mit eher mittlerer Lage. Sie ist **keine** J.A.R.V.I.S.-Imitation und klingt
**nicht britisch**; eine solche deutsche Stimme habe ich nicht gefunden. Ich habe keine Hörproben
angehört (kein Audio in der Umgebung) – bitte selbst anhören und ggf. andere Piper-Stimmen oder
`tts.speed` ausprobieren. Der Austausch ist ein Konfigurationswert bzw. eine neue `TTSEngine`.

## Wake Word

Standard ist das Whisper-Transkript: ohne zusätzliche Abhängigkeit, frei wählbare Phrase, aber
pro erkannter Sprache eine Transkription (CPU-Last) und Erkennungsfehler bei „Jarvis“ möglich.
openWakeWord ist optional verfügbar; das vortrainierte Modell erkennt „Hey Jarvis“ und passt daher
nicht zur standardmäßigen Einzelphrase „Jarvis“. Push-to-Talk bleibt als Rückfall.

## Weitere Entscheidungen

| Thema | Wahl | Begründung |
|---|---|---|
| Konfiguration | YAML + Pydantic v2 | Validierung mit lesbaren Fehlern, `extra="forbid"` fängt Tippfehler ab |
| Geheimnisse | Umgebungsvariablen / `.env` (python-dotenv) | Nie in YAML oder Code |
| HTTP | httpx (async) | Timeouts, Streaming, testbar mit `MockTransport` |
| Tool-Schemas | JSON Schema (jsonschema) | Provider-neutral, strikte Validierung vor Ausführung |
| Plugins | Ordner mit `plugin.py` (importlib) | Einfach, ohne Packaging; **keine Sandbox** |
| Gedächtnis | In-Memory (Standard), SQLite optional | Datenschutzfreundlicher Default, austauschbar |
| Nebenläufigkeit | asyncio; blockierende Audio-/STT-/TTS-Aufrufe via `asyncio.to_thread` | Klare Trennung sync/async |
| Terminal-UI | rich | Lesbare Statusanzeige, kleine Abhängigkeit |
| Tests/Qualität | pytest + pytest-asyncio, ruff (Lint + Format) | Schnell, ein Werkzeug für beides |

## Bekannte Einschränkungen

* STT läuft auf der CPU (siehe oben).
* Wake Word nur per Transkript, kein Unterbrechen per Stimme; keine Cloud-Provider (nur Erweiterungspunkte).
* Sprachausgabe satzweise; Streaming direkt aus dem LLM ist noch nicht verdrahtet.
* Erster Start benötigt Internet für Modell-Downloads.
* Audio, Whisper, Piper und echtes Ollama sind noch nicht auf echter Hardware getestet.
