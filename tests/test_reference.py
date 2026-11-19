"""Tests for the reference sampler: RNG known answers, structural invariants and
statistical uniformity. Every fast sampler is later checked against this one."""

import numpy as np
import pytest
from scipy import stats

from gnnsample.philox import bounded, draw_u32, philox4x32
from gnnsample.reference import sample
from gnnsample.synthetic import power_law_csc

FANOUTS = [15, 10, 5]


def test_philox_known_answers():
    # Known-answer vectors from Random123 (kat_vectors, philox4x32_10).
    out = philox4x32(0, 0, 0, 0, 0, 0)
    assert [int(w) for w in out] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    f = 0xFFFFFFFF
    out = philox4x32(f, f, f, f, f, f)
    assert [int(w) for w in out] == [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]


@pytest.fixture(scope="module")
def graph():
    return power_law_csc(num_nodes=20_000, avg_degree=20, seed=1)


def test_structure(graph):
    indptr, indices = graph
    rng = np.random.default_rng(0)
    seeds = rng.choice(len(indptr) - 1, 512, replace=False)
    s = sample(indptr, indices, seeds, FANOUTS, seed=7)

    assert len(np.unique(s.node)) == len(s.node), "node list must be unique"
    assert np.array_equal(s.node[: len(seeds)], seeds), "seeds come first"
    # Every sampled edge exists: CSC position lies inside the target's neighbor range and
    # the stored neighbor matches.
    tgt = s.node[s.col]
    assert np.all((s.edge >= indptr[tgt]) & (s.edge < indptr[tgt + 1]))
    assert np.array_equal(indices[s.edge], s.node[s.row])
    # No duplicate edge per target (sampling without replacement; graph has no multi-edges).
    assert len(np.unique(s.edge)) == len(s.edge)

    # Count per expanded node = min(degree, fanout), hop by hop.
    lo, hi = 0, len(seeds)
    e0 = 0
    for hop, k in enumerate(FANOUTS):
        frontier = s.node[lo:hi]
        expected = np.minimum(indptr[frontier + 1] - indptr[frontier], k).sum()
        assert s.num_sampled_edges[hop] == expected
        cols = s.col[e0:e0 + expected]
        assert cols.min() >= lo and cols.max() < hi, "hop expands only the previous hop's new nodes"
        e0 += expected
        lo, hi = hi, hi + s.num_sampled_nodes[hop + 1]


def test_deterministic(graph):
    indptr, indices = graph
    seeds = np.arange(100)
    a = sample(indptr, indices, seeds, FANOUTS, seed=3)
    b = sample(indptr, indices, seeds, FANOUTS, seed=3)
    c = sample(indptr, indices, seeds, FANOUTS, seed=4)
    assert np.array_equal(a.edge, b.edge)
    assert not np.array_equal(a.edge, c.edge)


def test_uniform_without_replacement():
    """Each neighbor of a degree-d node must be picked with probability k/d.
    Chi-square over 4000 independent draws (different RNG seeds)."""
    d, k, trials = 60, 15, 4000
    indptr = np.array([0, d] + [d] * d, dtype=np.int64)  # node 0 has neighbors 1..d
    indices = np.arange(1, d + 1, dtype=np.int32)
    counts = np.zeros(d, dtype=np.int64)
    for t in range(trials):
        s = sample(indptr, indices, [0], [k], seed=t)
        counts[s.edge] += 1
    assert counts.sum() == trials * k
    _, p = stats.chisquare(counts)
    assert p > 1e-3, f"neighbor selection is not uniform (p={p:.2e})"


def test_bounded_range():
    r = draw_u32(5, 0, np.arange(100_000), 0)
    b = bounded(r, 7)
    assert b.min() == 0 and b.max() == 6
