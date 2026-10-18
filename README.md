# Jarvis – lokaler, modularer KI-Sprachassistent

Jarvis hört dir zu (lokale Spracherkennung), antwortet mit einem lokalen Sprachmodell und spricht
die Antwort mit einer lokalen Stimme aus. Standardmäßig läuft **alles auf deinem Rechner**:
keine Cloud-Konten, keine laufenden Kosten, keine Übertragung von Gesprächen oder Aufnahmen.

| Baustein | Standard | Cloud nötig? |
|---|---|---|
| Sprachmodell (LLM) | Ollama + `llama3.1:8b` | nein (nur Download des Modells) |
| Spracherkennung (STT) | faster-whisper, Modell `small`, CPU | nein (nur Download beim ersten Start) |
| Sprachausgabe (TTS) | Piper, Stimme `de_DE-thorsten-high` | nein (nur Download beim ersten Start) |
| Wake Word / Hands-free | lokale Whisper-STT, Phrase „Jarvis“ | nein |
| Cloud-LLMs | OpenAI-kompatibler Chat-Completions-Adapter | ja |

> **Ehrlicher Teststatus (Version 0.2):** Die automatisierten Tests (115) laufen grün und prüfen
> Konfiguration, Provider-Logik (gegen simulierte Ollama-Antworten), Tools, Plugins, Gespräch,
> Fehlerbehandlung, die reine Audio-Logik sowie die Hands-free-Schleife (Wake Word, Folgefenster,
> „Danke“, Fehlerabbruch) mit *simuliertem* Mikrofon, Spracherkennung und Sprachausgabe. Ein Ende-zu-Ende-Lauf der Textkette gegen einen
> *simulierten* Ollama-Server war erfolgreich. **Nicht getestet** – weil in der Entwicklungsumgebung
> weder Mikrofon, Lautsprecher, GPU noch echte Modelle verfügbar waren: echtes Ollama-Modell,
> Mikrofonaufnahme, faster-whisper, Piper-Stimme und Audiowiedergabe. Siehe
> [Offene manuelle Tests](#offene-manuelle-tests).

## Installation (Windows 11, PowerShell)

Voraussetzungen: Python 3.11 oder neuer (empfohlen 3.12), Internet für die einmaligen Downloads.

```powershell
# 1. Ollama installieren (Installer von https://ollama.com/download) und ein Modell laden
ollama pull llama3.1:8b

# 2. Projekt einrichten (im Projektordner)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # bei Skript-Fehler einmalig: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[audio,dev,wake-word]"

# 3. Prüfen
jarvis --check
jarvis --list-devices

# 4. Starten
jarvis
```

### Optionale Integrationen

* **Wake Word:** standardmäßig sucht die lokale Whisper-STT nach „Jarvis“ am Anfang der Äußerung.
  Das funktioniert mit dem einzelnen Wort; das openWakeWord-Modell `hey_jarvis` ist nur für
  „Hey Jarvis“ trainiert. openWakeWord bleibt optional über `wake_word.engine: "openwakeword"`.
* **OpenAI-kompatible APIs:** `ai.provider: openai_compatible`, `base_url`, `model` und
  `api_key_env` konfigurieren; den Schlüssel nur in `.env` hinterlegen.
* **Home Assistant:** Lampen auflisten/steuern, Helligkeit und Farbe setzen. `HOME_ASSISTANT_TOKEN`
  und die Instanz-URL konfigurieren. Änderungen werden bestätigt.
* **Systemmonitor:** lokale CPU-, RAM- und Datenträgerwerte.
* **Docker/Proxmox:** Container und VMs auflisten; Start-/Stop-Aktionen werden bestätigt.
  Proxmox-URL und API-Token bleiben lokale Konfiguration.
* **Langzeitgedächtnis:** SQLite-Faktenablage ist lokal aktiv. Jarvis kann auf ausdrückliche Bitte
  Fakten speichern oder suchen; Löschen verlangt Bestätigung.

Linux/macOS: gleiche Schritte, Aktivierung mit `source .venv/bin/activate`. Unter Linux wird
zusätzlich die System-Bibliothek PortAudio benötigt (`sudo apt install libportaudio2`).

### Was beim ersten Start heruntergeladen wird (einmalig, danach offline)

| Was | Wann | Wohin | Größe (ungefähr) |
|---|---|---|---|
| `llama3.1:8b` | `ollama pull` | Ollama-Modellordner | einige GB |
| Whisper-Modell `small` | erste Spracheingabe | `models/stt/` | rund 0,5 GB |
| Piper-Stimme `de_DE-thorsten-high` | erste Sprachausgabe | `models/tts/` | zig MB |

Der Ordner `models/` ist in `.gitignore`. Nach den Downloads läuft Jarvis ohne Netzwerk.

## Bedienung (Hands-free, wie Alexa)

Standardmodus ist `handsfree`: Nach dem Start (`jarvis`) hört Jarvis dauerhaft zu. Du musst nichts
tippen und keine Taste drücken:

* **„Jarvis, wie spät ist es?“** – Wake Word und Frage in einem Satz; Jarvis antwortet gesprochen.
* **„Jarvis“** allein – Jarvis antwortet „Ja?“ (oder Signalton) und wartet ~6 s auf deine Frage.
* Danach ~8 s **Folgegespräch ohne Wake Word** („und morgen?“). **„Danke“**, „Stopp“, „Das war's“
  beenden es sofort (lokal, ohne Sprachmodell).
* `/listen` schaltet das Zuhören im Terminal an/aus. Alle Einstellungen: Abschnitt `wake_word` und
  `audio` in `config/config.yaml`.

So funktioniert es und das sind die Grenzen:

* Das Mikrofon und Whisper werden lokal ausgewertet; Audio bleibt im Arbeitsspeicher und wird nie
  gespeichert. Im Leerlauf wird Sprache lokal transkribiert und nur bei „Jarvis“ am Anfang verarbeitet.
* Während Jarvis spricht, kann „Jarvis“ die Wiedergabe unterbrechen; Echo kann je nach
  Mikrofon/Lautsprecher dennoch Fehlaktivierungen auslösen.
* Bestätigungspflichtige Aktionen werden im Hands-free-Modus per Sprache mit „Ja“ oder „Nein“
  bestätigt; ohne klare Zustimmung werden sie abgelehnt.
* Beim Start wird das Hintergrundrauschen gemessen (~1 s still sein), Modelle werden vorgeladen.

## Bedienung per Tastatur

Beim Start prüft Jarvis den LLM-Server und zeigt eine Statusübersicht. Eingabe in der Zeile `Du:`:

| Eingabe | Wirkung |
|---|---|
| Text + Enter | Frage senden (Antwort wird angezeigt und – wenn TTS verfügbar – vorgelesen) |
| leere Zeile (Modus `push_to_talk`) oder `/voice` | Aufnahme startet; **Enter beendet** sie, dann wird erkannt und beantwortet |
| `/mute` · `/stop` | Sprachausgabe an/aus · laufende Ausgabe abbrechen |
| `/status` · `/tools` · `/plugins` · `/devices` | Informationen |
| `/reset` | neues Gespräch |
| `/quit`, Strg+C, Strg+D | beenden (Audio wird freigegeben) |

Aktivierung in `config/config.yaml`: `general.activation_mode: "push_to_talk"` macht die leere
Zeile zum Aufnahmestart. Jarvis nimmt während der eigenen Sprachausgabe nicht auf (Aufnahme
startet erst nach Enter und bricht laufende Ausgabe ab), daher keine Rückkopplung.

## Konfiguration

Alles steht in `config/config.yaml` und wird beim Start validiert (unbekannte Schlüssel,
falsche Werte und deaktivierte Provider erzeugen eine verständliche Fehlermeldung).

### Modell wechseln (lokal)

```powershell
ollama pull qwen2.5:7b            # Beispiel
```
```yaml
ai:
  providers:
    ollama:
      model: "qwen2.5:7b"
```
Modellnamen sind frei konfigurierbar. Für 16 GB VRAM sind 7–8B-Modelle in 4-Bit-Quantisierung
unproblematisch; größere Modelle auf eigene Verantwortung testen. Unterstützt ein Modell kein
Tool-Calling, schaltet Jarvis kontrolliert auf Betrieb ohne Tools um (Log-Hinweis).

### Stimme ändern

```yaml
tts:
  voice: "de_DE-thorsten-high"   # andere Piper-Stimme, z. B. de_DE-... (Namen siehe Piper-Stimmenliste)
  speed: 0.95                    # <1 langsamer, >1 schneller
  volume: 1.0
  sentence_pause_s: 0.2
```
Eine neue Stimme wird beim nächsten Sprechen automatisch nach `models/tts/` geladen. Hörproben
findest du auf der Piper-Projektseite. **Hinweis:** Die Standardstimme ist eine ruhige, klare
deutsche Männerstimme – keine J.A.R.V.I.S.-Kopie und nicht britisch (siehe
[docs/technology_selection.md](docs/technology_selection.md)). Prüfe die Lizenz (Model Card)
jeder Stimme, bevor du sie anders als privat nutzt.

### Mikrofon und Lautsprecher

`jarvis --list-devices` zeigt Indizes. Dann in `audio.input_device` / `audio.output_device` den Index
(Zahl) oder Namen eintragen; `null` = Systemstandard. Schlägt der Zugriff fehl, prüfe unter Windows
*Einstellungen → Datenschutz → Mikrofon*.

### Provider wechseln

Der aktive Provider steht in `ai.provider`. Ein Wechsel braucht keine Codeänderung, aber
**es gibt keinen stillen Fallback**: Ist der gewählte Provider deaktiviert, nicht implementiert,
ohne Modell/URL/Schlüssel oder nicht erreichbar, meldet Jarvis das und startet bzw. antwortet nicht
heimlich über einen anderen Anbieter.

| Provider | Status in v0.1 |
|---|---|
| `ollama` | **funktioniert** (lokal) |
| `openai_compatible` | **funktioniert** (Chat Completions, Tool-Calling und Streaming) |
| `openai`, `anthropic`, `gemini` | nur Erweiterungspunkte |

STT, TTS und LLM sind getrennte Konfigurationsbereiche: Ein späterer Cloud-LLM ändert nichts an
der lokalen Spracherkennung und Sprachausgabe.

### OpenAI-kompatible API sicher einrichten

* API-Schlüssel **nie** in `config.yaml` oder Quellcode: `cp .env.example .env` (bzw. Datei kopieren)
  und dort eintragen; die Konfiguration verweist nur auf den Variablennamen (`api_key_env`).
  `.env` ist in `.gitignore`.
* Bei Cloud-Providern zeigt Jarvis einen deutlichen Hinweis: Gesprächstexte verlassen den Rechner,
  es können Kosten entstehen. Audioaufnahmen werden nie an Cloud-Dienste gesendet (STT bleibt lokal).
* Schlüssel und Tokens werden aus Logs geschwärzt (Muster-basiert, kein Ersatz für Sorgfalt).

## Datenschutz und Sicherheit

* **Lokal:** LLM (Ollama), STT, TTS, Gesprächsverlauf. Standardmäßig wird nichts auf die Festplatte
  geschrieben außer Logs (`logs/jarvis.log`, ohne Gesprächsinhalte auf INFO-Stufe) und den
  heruntergeladenen Modellen. Aufnahmen bleiben im Arbeitsspeicher und werden nicht gespeichert.
* **Netzwerkzugriff nur bei der Einrichtung:** `pip install`, `ollama pull`, Whisper- und Stimmen-Download.
  Danach keiner. Jarvis öffnet keine Netzwerkdienste.
* **Tools:** Das Modell kann nur registrierte Tools mit schema-validierten Parametern aufrufen –
  nie freien Python-Code oder Shell-Befehle. Tools mit `requires_confirmation=True` fragen vorher
  nach (ohne Bestätigung wird nichts ausgeführt).
* **Plugins sind keine Sandbox:** Ein Plugin ist normaler Python-Code mit allen Rechten des
  Jarvis-Prozesses. Aktiviere nur Plugins, deren Code du kennst.
* Verlauf speichern (optional): `conversation.persist: true` legt ihn in lokaler SQLite
  (`memory.path`) ab.

## Fehlerdiagnose

| Meldung / Symptom | Ursache und Abhilfe |
|---|---|
| „Ollama ist unter … nicht erreichbar“ | Ollama-App starten bzw. `ollama serve`; URL in `ai.providers.ollama.base_url` prüfen. |
| „Modell … nicht installiert“ | `ollama pull <modell>` (Name wie in der Config). |
| Zeitüberschreitung beim LLM | Erste Anfrage lädt das Modell; `timeout_s` erhöhen. |
| „Audio-Pakete fehlen“ | `pip install -e ".[audio]"` im aktivierten venv. |
| „Audio-Backend (PortAudio) nicht verfügbar“ | Windows: Paket `sounddevice` neu installieren; Linux: `libportaudio2`. |
| „Mikrofon nicht nutzbar“ / „Aufnahme leer“ | `--list-devices`, `audio.input_device` setzen, Datenschutzeinstellung prüfen. |
| „Es wurde keine Sprache erkannt“ | Näher sprechen; Whisper-Modell größer wählen (`stt.model: medium`). |
| Whisper-/Stimmen-Download scheitert | Internetverbindung/Proxy; Dateien landen in `models/`. |
| Plugin nicht geladen | Meldung steht im Startbanner und in `/plugins`; Details in `logs/jarvis.log`. |

Mehr Details: `logging.level: "DEBUG"` und `logging.console: true` in der Config.

## Neue Plugins und Tools

* Plugin erstellen: [docs/adding_plugins.md](docs/adding_plugins.md) (Vorlage: `plugins/example/`).
* Architektur und Erweiterungspunkte (Provider, STT, TTS, Wake Word): [docs/architecture.md](docs/architecture.md).

## Entwicklung und Tests

```powershell
pip install -e ".[dev]"
python -m pytest            # 115 Tests, brauchen weder Modelle noch Audiogeräte
ruff check .                # Linting
ruff format .               # Formatierung
```

## Offene manuelle Tests

Diese Punkte sind implementiert, aber **nicht auf echter Hardware geprüft** und sollten einmal
durchgespielt werden:

1. `jarvis --check` mit laufendem Ollama und installiertem Modell → „LLM: OK“.
2. Textfrage mit echtem `llama3.1:8b`; Frage nach dem System löst das Tool `system_info` aus
   (zeigt, ob das Modell Tool-Calling nutzt).
3. Hands-free: „Jarvis, wie spät ist es?“ – wird das Wake Word zuverlässig erkannt (auch aus
   2–3 m Entfernung)? Reagiert Jarvis nicht auf Fernseher/Gespräche? Funktioniert das Folgefenster?
   Passt die Rausch-Kalibrierung? Whisper `small` auf der CPU: Dauer pro Äußerung prüfen.
   Alternativ `/voice` (Push-to-Talk, Enter beendet die Aufnahme).
4. Erste Sprachausgabe: Piper-Download der Stimme, Hörbarkeit, Klang der Stimme, `speed`.
5. `piper`-Python-API: Der Code nutzt `PiperVoice.load`, `voice.synthesize(..., syn_config=...)`,
   `SynthesisConfig` und `python -m piper.download_voices` gemäß `piper-tts` ≥ 1.3. Ändert sich die
   API in deiner Version, ist nur `PiperTTS` in `src/jarvis/audio/tts.py` anzupassen.
6. Unterbrechen der Sprachausgabe (`/stop`), sauberes Beenden mit Strg+C während Aufnahme/Ausgabe.
7. Verhalten bei abgezogenem Mikrofon (sollte Fehlermeldung, kein Absturz sein).

## Nächste sinnvolle Schritte

Echtes Wake-Word-Modell (z. B. openWakeWord) statt Whisper-Transkript, Unterbrechen per Stimme; Streaming-Sprachausgabe direkt aus
`LLMProvider.stream()`; erster Cloud-Adapter (empfohlen: `openai_compatible`, da einfach per HTTP);
Plugins für Home Assistant, Systemüberwachung, Docker/Proxmox; Langzeitgedächtnis-Nutzung über
`LongTermMemory`.
