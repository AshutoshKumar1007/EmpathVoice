"""Unit tests for pause injection."""

from __future__ import annotations

from app.tts.pause_injection import inject_pauses

LONG = (
    "I was walking home and then suddenly everything changed "
    "because the sky turned dark but I kept going"
)


def test_low_level_leaves_text_untouched():
    assert inject_pauses(LONG, 0.2) == LONG


def test_short_text_untouched():
    assert inject_pauses("just three words", 0.9) == "just three words"


def test_high_level_inserts_commas():
    out = inject_pauses(LONG, 0.9)
    assert out.count(",") >= 1


def test_never_stacks_punctuation():
    text = "Stop. Wait, please, this is urgent and important and critical right now okay"
    out = inject_pauses(text, 0.92)
    assert ",," not in out
    assert ".," not in out
    assert " ," not in out


def test_empty_text():
    assert inject_pauses("   ", 0.9) == ""
