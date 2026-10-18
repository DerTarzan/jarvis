# Ein Plugin hinzufügen

> **Sicherheitshinweis:** Plugins sind **keine Sandbox**. Ein Plugin ist beliebiger Python-Code und
> läuft mit allen Rechten des Jarvis-Prozesses (Dateien, Netzwerk, Umgebungsvariablen inkl. API-Schlüssel).
> Aktiviere nur Plugins, deren Code du gelesen hast und denen du vertraust.

## Schritt für Schritt

1. Vorlage kopieren: `plugins/example/` → `plugins/mein_plugin/` (Ordnername = Plugin-Name,
   Buchstaben/Ziffern/Unterstrich; Ordner mit `_` oder `.` am Anfang werden ignoriert).
2. In `plugins/mein_plugin/plugin.py` eine Klasse **`Plugin`** von `BasePlugin` ableiten:

```python
from jarvis.plugins import BasePlugin


class Plugin(BasePlugin):
    description = "Kurze Beschreibung"
    version = "0.1.0"

    def setup(self) -> None:
        wert = self.ctx.config.get("einstellung", "standard")
        self.ctx.register_tool(
            name="mein_tool",
            description="Was das Tool tut – das liest das Sprachmodell.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 200}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=self.mein_tool,
            requires_confirmation=False,  # True bei Aktionen mit Nebenwirkungen!
        )

    def mein_tool(self, text: str) -> dict:
        return {"antwort": text.upper()}

    def teardown(self) -> None:  # optional
        pass
```

3. Aktivieren und konfigurieren in `config/config.yaml`:

```yaml
plugins:
  enabled: ["example", "mein_plugin"]
  settings:
    mein_plugin:
      einstellung: "wert"      # landet in self.ctx.config
```

4. Starten und prüfen: `jarvis --check` bzw. `/plugins` und `/tools` in der Sitzung.

## Was der Kern dem Plugin bereitstellt (`self.ctx`)

| Attribut | Zweck |
|---|---|
| `ctx.config` | Plugin-Konfiguration (`plugins.settings.<name>`) |
| `ctx.log` | Logger `jarvis.plugin.<name>` (nie Geheimnisse loggen) |
| `ctx.events` | `EventBus` (z. B. auf `user_text`, `assistant_text`, `error` hören) |
| `ctx.data_dir` | eigener Datenordner (`data/plugins/<name>`; Ordner selbst anlegen) |
| `ctx.register_tool(...)` | Tool beim zentralen Registry anmelden (wird beim Entladen entfernt) |

## Regeln und Verhalten

* **Fehlerisolierung:** Wirft `import` oder `setup()` eine Ausnahme, wird nur dieses Plugin verworfen
  (bereits registrierte Tools werden entfernt); Jarvis startet trotzdem. Die Meldung steht im Banner
  und in `/plugins`.
* **Tool-Namen** sind global eindeutig; ein doppelter Name lässt das zweite Plugin scheitern.
* **Tool-Handler** dürfen sync oder `async` sein, bekommen validierte Keyword-Argumente und sollten
  JSON-serialisierbare Ergebnisse liefern. Ausnahmen werden als Fehlertext an das Modell gemeldet.
  Standard-Timeout: `tools.timeout_s` (pro Tool mit `timeout_s=` überschreibbar).
* **Parameter** immer streng als JSON-Schema beschreiben (`additionalProperties: false`, Grenzen).
* **Gefährliche Aktionen** (Dateien löschen, Geräte schalten, Nachrichten senden): immer
  `requires_confirmation=True`. Im Hands-free-Modus werden solche Aktionen ohne Bestätigung verweigert.
  Niemals Shell-Befehle oder `eval`/`exec` aus Modellparametern ausführen.
* **Unabhängig testbar:** Plugin-Klasse mit einem `PluginContext` instanziieren, `setup()` aufrufen und
  Handler direkt testen – Beispiel: `tests/test_plugins.py`.

## Home Assistant

Das Plugin `homeassistant` listet Lampen auf und kann eine einzelne Lampe an-/ausschalten oder beim
Einschalten Helligkeit (Prozent) und eine Farbe setzen. Jede Änderung erfordert eine Bestätigung.
Es verwendet die Home-Assistant-REST-API und braucht einen Long-Lived Access Token:

1. Erstelle in Home Assistant unter **Einstellungen → Personen → dein Benutzer → Long-Lived Access Tokens**
   einen Token.
2. Trage ihn in `.env` als `HOME_ASSISTANT_TOKEN=...` ein. Teile den Token nicht im Chat und committe
   die `.env`-Datei nicht.
3. Passe `plugins.settings.homeassistant.base_url` in `config/config.yaml` an deine Home-Assistant-URL
   an, falls `http://homeassistant.local:8123` bei dir nicht stimmt.
4. Starte Jarvis neu. Mit `/plugins` und `/tools` kannst du prüfen, ob das Plugin geladen wurde.

Farben: rot, grün, blau, gelb, orange, lila/violett, rosa/pink, weiß, warmweiß und türkis.
Andere Domains als `light.*` sind absichtlich nicht steuerbar.

Proxmox, Docker/Compose, NAS/Dateien, Kalender, Wetter, Websuche, Musik, E-Mail und Automatisierungen
sind nicht implementiert.
