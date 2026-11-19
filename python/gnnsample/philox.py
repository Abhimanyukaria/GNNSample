"""Philox-4x32-10 counter-based RNG (Salmon et al., SC'11), vectorised in NumPy.

This is the reference for the SYCL implementation in cpp/include/gnnsample/philox.hpp:
both must produce identical words for the same (counter, key), so the GPU sampler can be
checked bit-for-bit against reference.py.
"""

import numpy as np

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
MASK32 = np.uint64(0xFFFFFFFF)


def philox4x32(c0, c1, c2, c3, k0, k1):
    """Returns the four uint32 output words; inputs broadcast against each other."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint32) for c in (c0, c1, c2, c3))
    c0, c1, c2, c3 = np.broadcast_arrays(c0, c1, c2, c3)
    k0, k1 = np.uint32(k0), np.uint32(k1)
    with np.errstate(over="ignore"):
        for r in range(10):
            if r:
                k0, k1 = k0 + W0, k1 + W1
            p0 = M0 * c0.astype(np.uint64)
            p1 = M1 * c2.astype(np.uint64)
            c0, c1, c2, c3 = (
                (p1 >> np.uint64(32)).astype(np.uint32) ^ c1 ^ k0,
                (p1 & MASK32).astype(np.uint32),
                (p0 >> np.uint64(32)).astype(np.uint32) ^ c3 ^ k1,
                (p0 & MASK32).astype(np.uint32),
            )
    return c0, c1, c2, c3


def draw_u32(seed: int, hop: int, node, draw):
    """Random word number `draw` for `node` at `hop`. Counter = (node, hop, draw // 4, 0),
    key = (seed lo, seed hi); word draw % 4 of the output block."""
    node = np.asarray(node, dtype=np.uint32)
    draw = np.asarray(draw, dtype=np.uint32)
    words = philox4x32(node, np.uint32(hop), draw >> np.uint32(2), np.uint32(0),
                       seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)
    words = np.stack(np.broadcast_arrays(*words))
    lane = np.broadcast_to(draw & np.uint32(3), words.shape[1:]).astype(np.int64)
    return np.take_along_axis(words, lane[None], 0)[0]


def bounded(r, n):
    """Map uint32 r to [0, n) by multiply-shift (Lemire). Bias <= n / 2^32 — under 0.1%
    for every degree in our graphs; documented in docs/decisions.md D7."""
    return ((np.asarray(r, dtype=np.uint64) * np.asarray(n, dtype=np.uint64))
            >> np.uint64(32)).astype(np.int64)
