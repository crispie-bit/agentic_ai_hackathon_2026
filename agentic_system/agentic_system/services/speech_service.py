"""Voice/speech layer stub.

This remains offline until speech capabilities are intentionally enabled.
"""


def speech_ready() -> bool:
    return False


def speak(text: str) -> str:
    return f"Speech disabled in setup mode. Planned output: {text}"
