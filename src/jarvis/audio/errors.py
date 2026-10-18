class AudioError(Exception):
    """Mikrofon-, Aufnahme- oder Wiedergabeproblem (Meldung ist nutzerverständlich)."""


class STTError(Exception):
    """Spracherkennung nicht verfügbar oder fehlgeschlagen."""


class TTSError(Exception):
    """Sprachausgabe nicht verfügbar oder fehlgeschlagen."""


class WakeWordError(Exception):
    """Wake-Word-Erkennung nicht verfügbar."""
