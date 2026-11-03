"""Masked examples: a hidden base with 5-30 known bases on each side.

An example is a row of WIDTH = 2 * MAX_CONTEXT + 1 codes centred on the hidden base. Positions
outside the known context, and the hidden base itself, hold SEP ("unknown").
"""

import numpy as np

from genome import BASES, SEP, Genome

MIN_CONTEXT, MAX_CONTEXT = 5, 30
WIDTH = 2 * MAX_CONTEXT + 1
CENTRE = MAX_CONTEXT


def sample_examples(genome: Genome, n: int, seed: int, min_context: int = MIN_CONTEXT,
                    max_context: int = MAX_CONTEXT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw n masked examples uniformly over the positions that have at least `min_context`
    bases on both sides. The left and right context lengths are drawn independently from
    [min_context, max_context] and cut where the sequence ends.

    Returns windows (n, WIDTH) uint8, targets (n,) uint8 and context lengths (n, 2).
    """
    rng = np.random.default_rng(seed)
    usable = genome.lengths - 2 * min_context
    weights = np.clip(usable, 0, None).astype(float)
    record = rng.choice(len(genome.lengths), size=n, p=weights / weights.sum())
    pos = min_context + (rng.random(n) * usable[record]).astype(np.int64)  # position inside the record

    left = np.minimum(rng.integers(min_context, max_context + 1, n), pos)
    right = np.minimum(rng.integers(min_context, max_context + 1, n), genome.lengths[record] - 1 - pos)

    centre = genome.starts[record] + pos
    offsets = np.arange(-max_context, max_context + 1)
    index = np.clip(centre[:, None] + offsets, 0, len(genome.bases) - 1)
    windows = genome.bases[index]
    known = (offsets >= -left[:, None]) & (offsets <= right[:, None]) & (offsets != 0)
    windows = np.where(known, windows, SEP).astype(np.uint8)
    targets = genome.bases[centre]
    return windows, targets, np.stack([left, right], axis=1)


def parse_query(text: str, gap: str = "_") -> tuple[np.ndarray, tuple[int, int]]:
    """Window for a query such as "CGTAAC_TCATT": the gap is the base to predict.

    Spaces are ignored; context beyond MAX_CONTEXT on either side is cut off.
    """
    text = text.replace(" ", "").upper()
    if text.count(gap) != 1:
        raise ValueError(f"The query must contain exactly one {gap!r}")
    left_text, right_text = text.split(gap)
    if not set(left_text + right_text) <= set(BASES):
        raise ValueError("Context may only contain A, C, G and T")
    left_text, right_text = left_text[-MAX_CONTEXT:], right_text[:MAX_CONTEXT]
    window = np.full(WIDTH, SEP, dtype=np.uint8)
    for i, base in enumerate(reversed(left_text)):
        window[CENTRE - 1 - i] = BASES.index(base)
    for i, base in enumerate(right_text):
        window[CENTRE + 1 + i] = BASES.index(base)
    return window, (len(left_text), len(right_text))
