"""The SYCL sampler must reproduce the reference sampler bit-for-bit (same nodes, local
ids, edges, per-hop counts) on every device it runs on."""

import numpy as np
import pytest

from gnnsample.reference import sample
from gnnsample.synthetic import power_law_csc

_C = pytest.importorskip("gnnsample._C")
FANOUTS = [15, 10, 5]


def devices():
    out = []
    for dev in ("gpu", "cpu"):
        try:
            g = power_law_csc(100, 3, seed=0)
            _C.Sampler(g[0], g[1], FANOUTS, 4, dev)
            out.append(dev)
        except Exception:  # device not present
            pass
    return out


@pytest.fixture(scope="module")
def graph():
    return power_law_csc(num_nodes=50_000, avg_degree=25, seed=2)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("batch,seed", [(1, 0), (64, 1), (1024, 2), (1024, 3)])
def test_matches_reference(graph, device, batch, seed):
    indptr, indices = graph
    seeds = np.random.default_rng(seed).choice(len(indptr) - 1, batch, replace=False)
    s = _C.Sampler(indptr, indices, FANOUTS, 1024, device)
    got = s.sample(seeds.astype(np.int32), seed)
    ref = sample(indptr, indices, seeds, FANOUTS, seed=seed)

    assert list(got["num_sampled_nodes"]) == ref.num_sampled_nodes
    assert list(got["num_sampled_edges"]) == ref.num_sampled_edges
    np.testing.assert_array_equal(got["node"], ref.node)
    np.testing.assert_array_equal(got["edge"], ref.edge)
    np.testing.assert_array_equal(got["row"], ref.row)
    np.testing.assert_array_equal(got["col"], ref.col)


# Every combination of the first four bits, plus the no-sync micro-optimizations.
ALL_OPTIONS = list(range(16)) + [9 | 16, 9 | 32, 9 | 64, 9 | 112, 8 | 112, 1 | 112,
                                 9 | 128, 8 | 128, 105 | 128, 121 | 128, 233 | 256, 8 | 256,
                                 489 | 512, 489 | 1024, 489 | 1536, 9 | 1536, 8 | 512,
                                 2025 | 2048, 2025 | 4096, 2025 | 6144, 8 | 2048, 13 | 4096,
                                 8169 | 8192, 8 | 8192, 9 | 8192]


@pytest.mark.parametrize("options", ALL_OPTIONS)
def test_options_match_reference(graph, options):
    """Every optimization combination is bit-identical to the reference, including when
    minibatches follow each other (CLEAR_USED relies on the previous batch's slots)."""
    indptr, indices = graph
    s = _C.Sampler(indptr, indices, FANOUTS, 1024, "gpu")
    s.set_options(options)
    rng = np.random.default_rng(options)
    for b in range(3):
        seeds = rng.choice(len(indptr) - 1, 1024, replace=False)
        got = s.sample(seeds.astype(np.int32), b)
        ref = sample(indptr, indices, seeds, FANOUTS, seed=b)
        np.testing.assert_array_equal(got["node"], ref.node)
        np.testing.assert_array_equal(got["row"], ref.row)
        np.testing.assert_array_equal(got["edge"], ref.edge)


def test_switching_options_between_batches(graph):
    """Layout and reset strategy may change between minibatches (the A/B benchmark does)."""
    indptr, indices = graph
    s = _C.Sampler(indptr, indices, FANOUTS, 256, "gpu")
    rng = np.random.default_rng(0)
    # includes tagged <-> untagged switches around CLEAR_USED (13), which must not reuse
    # slot indices recorded against the other table
    for b, options in enumerate([0, 7, 5, 4, 1, 6, 7, 0, 8, 9, 13, 15, 1, 8, 121, 9, 89, 73,
                                 13, 233, 13, 13, 137, 233, 249, 5, 237,
                                 2025, 489, 1001, 1545, 2025, 13, 1513,
                                 4073, 2025, 8169, 6121, 13, 4073, 5, 16361, 8169, 8192]):
        s.set_options(options)
        seeds = rng.choice(len(indptr) - 1, 256, replace=False)
        got = s.sample(seeds.astype(np.int32), b)
        ref = sample(indptr, indices, seeds, FANOUTS, seed=b)
        np.testing.assert_array_equal(got["edge"], ref.edge)
        np.testing.assert_array_equal(got["row"], ref.row)


@pytest.mark.parametrize("options", [0, 1, 8, 9, 9 | 112, 233])
def test_empty_and_tiny_batches(graph, options):
    """An empty seed batch used to leave the no-sync path's device state uninitialised
    (GPU fault). Empty and tiny batches must work and leave the sampler usable."""
    indptr, indices = graph
    s = _C.Sampler(indptr, indices, FANOUTS, 1024, "gpu")
    s.set_options(options)
    empty = s.sample(np.array([], dtype=np.int32), 0)
    assert len(empty["node"]) == 0 and list(empty["num_sampled_edges"]) == [0, 0, 0]
    for b, n in enumerate([1, 4, 0, 1024]):
        seeds = np.random.default_rng(b).choice(len(indptr) - 1, n, replace=False)
        got = s.sample(seeds.astype(np.int32), b)
        ref = sample(indptr, indices, seeds, FANOUTS, seed=b)
        np.testing.assert_array_equal(got["edge"], ref.edge)
        np.testing.assert_array_equal(got["row"], ref.row)


@pytest.mark.parametrize("device", devices())
def test_repeated_calls_independent(graph, device):
    """Hash table and buffers are reset between minibatches."""
    indptr, indices = graph
    s = _C.Sampler(indptr, indices, FANOUTS, 256, device)
    seeds = np.arange(256, dtype=np.int32)
    a = s.sample(seeds, 5)
    s.sample(np.arange(256, 512, dtype=np.int32), 6)
    b = s.sample(seeds, 5)
    np.testing.assert_array_equal(a["edge"], b["edge"])
    np.testing.assert_array_equal(a["row"], b["row"])


def test_gen_tags_many_batches(graph):
    """Generation tags must keep working across many minibatches without any reset
    (stale slots from every earlier batch stay in the table)."""
    indptr, indices = graph
    s = _C.Sampler(indptr, indices, FANOUTS, 64, "gpu")
    s.set_options(233)  # packed | nosync | assignscan | slot32 | gentags
    rng = np.random.default_rng(3)
    for b in range(300):  # the 8-bit generation wraps after 255 minibatches
        seeds = rng.choice(len(indptr) - 1, 64, replace=False)
        got = s.sample(seeds.astype(np.int32), b)
        if b in (9, 253, 254, 255, 256, 299):
            ref = sample(indptr, indices, seeds, FANOUTS, seed=b)
            np.testing.assert_array_equal(got["edge"], ref.edge)
            np.testing.assert_array_equal(got["row"], ref.row)
            np.testing.assert_array_equal(got["node"], ref.node)


def test_from_files_matches_arrays(graph, tmp_path):
    """The chunked file -> device loader must give the same sampler as the array path."""
    indptr, indices = graph
    indptr.astype(np.int64).tofile(tmp_path / "indptr.int64.bin")
    indices.astype(np.int32).tofile(tmp_path / "indices.int32.bin")
    a = _C.Sampler(indptr, indices, FANOUTS, 512, "gpu")
    f = _C.Sampler.from_files(str(tmp_path / "indptr.int64.bin"),
                              str(tmp_path / "indices.int32.bin"), FANOUTS, 512, "gpu")
    seeds = np.random.default_rng(9).choice(len(indptr) - 1, 512, replace=False).astype(np.int32)
    for s in (a, f):
        s.set_options(8169)
    ra, rf = a.sample(seeds, 4), f.sample(seeds, 4)
    for k in ("node", "row", "col", "edge"):
        np.testing.assert_array_equal(ra[k], rf[k])


def test_from_files_on_bind_mount():
    """Same check on the project's bind-mounted data directory (Docker Desktop's file system
    behaves differently from the container's own; pinned staging reads failed there)."""
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "results" / ".tmp_from_files"
    d.mkdir(parents=True, exist_ok=True)
    indptr, indices = power_law_csc(num_nodes=8_000, avg_degree=12, seed=3)
    indptr.astype(np.int64).tofile(d / "indptr.int64.bin")
    indices.astype(np.int32).tofile(d / "indices.int32.bin")
    try:
        f = _C.Sampler.from_files(str(d / "indptr.int64.bin"), str(d / "indices.int32.bin"),
                                  FANOUTS, 64, "gpu")
        a = _C.Sampler(indptr, indices, FANOUTS, 64, "gpu")
        seeds = np.arange(64, dtype=np.int32)
        np.testing.assert_array_equal(f.sample(seeds, 1)["edge"], a.sample(seeds, 1)["edge"])
    finally:
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
