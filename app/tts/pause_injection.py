import numpy as np

CLAUSE_WORDS = frozenset({
    'but','and','or','because','since','while','although',
    'however','therefore','moreover','so','yet','then',
    'still','also','unfortunately','fortunately','honestly',
    'actually','meanwhile','otherwise','nevertheless','though',
    'instead','perhaps','certainly','definitely','really'
})


def inject_pauses(text, pause_level):

    text = text.strip()

    if not text or pause_level < 0.35:
        return text

    words = text.split()

    if len(words) < 4:
        return text

    # Higher pause level inserts pauses more frequently.
    interval = int(np.interp(pause_level,[0.35,0.92],[14,4]))

    min_gap = max(3,interval//2)

    out=[]

    since_pause=0

    for i,word in enumerate(words):

        out.append(word)

        since_pause+=1

        if i >= len(words)-1:
            continue

        # Reset if the current token already ends a natural sentence/clause.
        if word.rstrip()[-1:] in set('.,!?;:-'):
            since_pause = 0
            continue

        next_word = words[i+1].lower().strip('.,!?;:')

        at_boundary = next_word in CLAUSE_WORDS and since_pause >= min_gap
        at_interval = since_pause >= interval and pause_level > 0.65

        if at_boundary or at_interval:

            if not out[-1].endswith(","):
                out[-1] = out[-1] + ","

            since_pause=0

    return " ".join(out)