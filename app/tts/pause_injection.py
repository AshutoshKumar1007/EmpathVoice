"""Intra-sentence pause injection.

edge-tts does not expose SSML ``<break>`` control, so pauses are approximated by
inserting commas at clause boundaries / fixed intervals, scaled by the pause
level. Behaviour is unchanged from the original implementation.
"""

from __future__ import annotations

import numpy as np

CLAUSE_WORDS = frozenset({
    "but", "and", "or", "because", "since", "while", "although",
    "however", "therefore", "moreover", "so", "yet", "then",
    "still", "also", "unfortunately", "fortunately", "honestly",
    "actually", "meanwhile", "otherwise", "nevertheless", "though",
    "instead", "perhaps", "certainly", "definitely", "really",
})

_PUNCT = frozenset(".,!?;:-")


def inject_pauses(text: str, pause_level: float) -> str:
    text = text.strip()
    if not text or pause_level < 0.35:
        return text

    words = text.split()
    if len(words) < 4:
        return text

    # Higher pause level inserts pauses more frequently.
    interval = int(np.interp(pause_level, [0.35, 0.92], [14, 4]))
    min_gap = max(3, interval // 2)

    out: list[str] = []
    since_pause = 0

    for i, word in enumerate(words):
        out.append(word)
        since_pause += 1

        if i >= len(words) - 1:
            continue

        # Reset if the current token already ends a natural clause/sentence.
        if word.rstrip()[-1:] in _PUNCT:
            since_pause = 0
            continue

        next_word = words[i + 1].lower().strip(".,!?;:")
        at_boundary = next_word in CLAUSE_WORDS and since_pause >= min_gap
        at_interval = since_pause >= interval and pause_level > 0.65

        if at_boundary or at_interval:
            # Never stack punctuation: only add a comma if the token doesn't
            # already end in some punctuation mark.
            if out[-1][-1:] not in _PUNCT:
                out[-1] = out[-1] + ","
            since_pause = 0

    return " ".join(out)
