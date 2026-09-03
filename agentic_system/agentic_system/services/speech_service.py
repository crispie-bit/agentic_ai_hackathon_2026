"""Optional local speech input and output helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path


def speech_ready() -> bool:
    try:
        import pyttsx3  # noqa: F401
        import speech_recognition  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Transcribe browser-recorded audio using the configured recognizer backend."""
    import speech_recognition as sr

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = Path(audio_file.name)
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    finally:
        audio_path.unlink(missing_ok=True)


def speak(text: str) -> str:
    import pyttsx3

    output_path = Path(tempfile.mktemp(suffix=".wav"))
    engine = pyttsx3.init()
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()
    return str(output_path)
