"""(dst, src) pair sets for PAIR MODE attention signals (experiment 3-1).

Positions are ABSOLUTE, 0 == BOS. A pair (d, s) is the attention weight that the token
at d puts on the token at s, so 0 <= s <= d. (0, 0) is always excluded: BOS can only
attend to itself, its weight is exactly 1.0, and there is nothing to report.

WHY A GRID AND NOT ALL PAIRS. Evaluation reads every listed pair on every test document
(PairListTokenSelector), one introspection prompt per (document, pair). All pairs of a
length-m text is m(m+1)/2 - 1 prompts per document -- 131,327 at m=512 -- so beyond
m=32 a fixed, log-spaced subset is evaluated instead:

    d     in {2^k - 1, 2^k} below m, plus m - 1 (the last token)
    src   d - delta for delta in {0, 1, 2, 4, ...} with delta < d, plus s = 0 (BOS)

Both 2^k - 1 and 2^k are kept because the training lengths are powers of two: a run
trained at n has seen d <= n - 1 and never d = n, so the grid straddles every n boundary.
Since m - 1 = 2^k - 1 for every eval length, grid(m) is a subset of grid(2m): a pair
evaluated at one length is evaluated at every longer one, which is what lets the
length sweep compare the SAME (document, d, s) across m.
"""
from __future__ import annotations

from functools import lru_cache

# Up to this length the grid IS all pairs (135 at 16, 527 at 32).
DENSE_MAX_LEN = 32
# Every eval length of the 3-1 sweep; the injection pool covers the union of their grids.
EVAL_LENGTHS = (16, 32, 64, 128, 256, 512)


def _check(pairs, seq_len: int):
    for d, s in pairs:
        assert 0 <= s <= d < seq_len, f"pair (d={d}, s={s}) outside 0 <= s <= d < {seq_len}"
    assert (0, 0) not in pairs, "(0, 0) is BOS -> BOS (weight 1.0) and must be excluded"
    assert len(set(pairs)) == len(pairs), "duplicate pairs"


@lru_cache(maxsize=None)
def all_pairs(seq_len: int) -> tuple[tuple[int, int], ...]:
    """Every causal pair of a seq_len-token text except (0, 0): seq_len(seq_len+1)/2 - 1.

    Cached: the random-pair sampler draws from this once per document, and rebuilding an
    8K-entry list 300K times would dominate the caching cost at n=128.
    """
    assert seq_len >= 2, f"seq_len {seq_len} has no pair besides (0, 0)"
    pairs = tuple((d, s) for d in range(seq_len) for s in range(d + 1) if (d, s) != (0, 0))
    assert len(pairs) == seq_len * (seq_len + 1) // 2 - 1, len(pairs)
    _check(pairs, seq_len)
    return pairs


def _grid_dsts(seq_len: int) -> list[int]:
    dsts = {seq_len - 1}
    k = 0
    while 2 ** k < seq_len:
        dsts.update({2 ** k - 1, 2 ** k})
        k += 1
    return sorted(d for d in dsts if 1 <= d < seq_len)


@lru_cache(maxsize=None)
def pair_grid(seq_len: int) -> tuple[tuple[int, int], ...]:
    """The evaluated pairs at eval length seq_len (see the module docstring)."""
    if seq_len <= DENSE_MAX_LEN:
        return all_pairs(seq_len)
    pairs = set()
    for d in _grid_dsts(seq_len):
        pairs.add((d, 0))                       # the BOS column (attention sink)
        delta = 0
        while delta < d:                        # delta < d keeps s >= 1, apart from BOS
            pairs.add((d, d - delta))
            delta = 1 if delta == 0 else delta * 2
    pairs = tuple(sorted(pairs))
    _check(pairs, seq_len)
    return pairs


@lru_cache(maxsize=None)
def pool_pairs(lengths: tuple[int, ...] = EVAL_LENGTHS) -> tuple[tuple[int, int], ...]:
    """Union of the eval grids: every pair any eval length can ask the injection pool for."""
    pairs = tuple(sorted({p for m in lengths for p in pair_grid(m)}))
    _check(pairs, max(lengths))
    return pairs
