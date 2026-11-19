"""Prototype D22: both multi-minibatch modes must return, for every minibatch, exactly what
the single-minibatch sampler (and the NumPy reference) returns for the same seeds and key.
D25 adds: hash vs direct dedup tables (also across a generation wrap), and a MultiSampler
built on another sampler's device graph."""

import gc

import numpy as np
import pytest

from gnnsample.reference import sample
from gnnsample.synthetic import power_law_csc

_C = pytest.importorskip("gnnsample._C")
FANOUTS = [15, 10, 5]
MAX_GEN = 255  # GenHashView::kMaxGen: tables are fully reset when the generation wraps


@pytest.fixture(scope="module")
def graph():
    return power_law_csc(num_nodes=40_000, avg_degree=20, seed=21)


def check(ms, indptr, indices, seeds, keys, counts, what=""):
    for m in range(len(seeds)):
        got = ms.result(m)
        ref = sample(indptr, indices, seeds[m], FANOUTS, seed=keys[m])
        assert list(counts[m][0]) == ref.num_sampled_nodes, f"counts m={m} {what}"
        assert list(counts[m][1]) == ref.num_sampled_edges, f"edge counts m={m} {what}"
        for k in ("node", "row", "col", "edge"):
            np.testing.assert_array_equal(got[k], getattr(ref, k), err_msg=f"{k} m={m} {what}")


@pytest.mark.parametrize("table", ["direct", "hash"])
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("mode,wg", [("batched", 0), ("megakernel", 256), ("megakernel", 1024)])
@pytest.mark.parametrize("K", [1, 3, 8])
def test_each_minibatch_matches_reference(graph, device, mode, wg, K, table):
    indptr, indices = graph
    ms = _C.MultiSampler(indptr, indices, FANOUTS, 256, 8, device)
    ms.set_table(table)
    assert ms.table == table
    rng = np.random.default_rng(K)
    for rep in range(3):  # later calls reuse tables (generation tags)
        seeds = [rng.choice(len(indptr) - 1, int(rng.integers(1, 257)), replace=False).astype(np.int32)
                 for _ in range(K)]
        keys = [int(rng.integers(0, 2**40)) for _ in range(K)]
        counts = ms.run(seeds, keys, mode, wg or 512)
        check(ms, indptr, indices, seeds, keys, counts, f"rep={rep}")


@pytest.mark.parametrize("table", ["direct", "hash"])
@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("mode", ["batched", "megakernel"])
def test_generation_wrap(device, mode, table):
    """D25: more runs than generations (255), so every table is fully reset once mid-test.
    A small graph makes consecutive minibatches share most nodes (and, for the hash table,
    probe through slots that earlier generations claimed), so a stale entry read as current
    would change the output. K cycles 1/3/8, so some tables sit unused for whole runs."""
    indptr, indices = power_law_csc(num_nodes=3_000, avg_degree=10, seed=22)
    ms = _C.MultiSampler(indptr, indices, FANOUTS, 32, 8, device)
    ms.set_table(table)
    rng = np.random.default_rng(23)
    checked = set(range(3)) | {120} | set(range(MAX_GEN - 3, MAX_GEN + 5))
    for run in range(MAX_GEN + 5):
        K = (1, 3, 8)[run % 3]
        seeds = [rng.choice(len(indptr) - 1, int(rng.integers(1, 33)), replace=False).astype(np.int32)
                 for _ in range(K)]
        keys = [int(rng.integers(0, 2**40)) for _ in range(K)]
        counts = ms.run(seeds, keys, mode, 256)
        if run in checked:
            check(ms, indptr, indices, seeds, keys, counts, f"run={run}")


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_table_switch_and_auto(graph, device):
    """Switching table kinds between runs (each switch frees one table and fully resets the
    other) keeps the output; "auto" picks hash only when direct is >= 1.5x larger."""
    indptr, indices = graph
    ms = _C.MultiSampler(indptr, indices, FANOUTS, 256, 3, device)
    rng = np.random.default_rng(24)
    for table in ("direct", "hash", "hash", "direct", "auto"):
        ms.set_table(table)
        seeds = [rng.choice(len(indptr) - 1, 200, replace=False).astype(np.int32) for _ in range(3)]
        keys = [int(rng.integers(0, 2**40)) for _ in range(3)]
        check(ms, indptr, indices, seeds, keys, ms.run(seeds, keys, "batched", 512), table)
        mem = ms.memory()
        per_mb = mem["hash_bytes_per_mb"] if ms.table == "hash" else mem["direct_bytes_per_mb"]
        assert mem["tables"] == 3 * per_mb
        assert mem["total"] == mem["graph"] + mem["tables"] + mem["buffers"] == ms.device_bytes()
    # 40 000 nodes, batch 256: every node fits a minibatch, direct (8 B/node) is smaller.
    assert ms.table == "direct"
    # Batch 4: at most 3 664 nodes -> 4 096 hash slots (64 KB) < 320 KB direct.
    small = _C.MultiSampler(indptr, indices, FANOUTS, 4, 2, device)
    assert small.table == "hash" and small.memory()["tables"] == 2 * 16 * 4096
    seeds = [np.array([5, 7, 11, 13], dtype=np.int32), np.array([2], dtype=np.int32)]
    check(small, indptr, indices, seeds, [3, 4], small.run(seeds, [3, 4], "batched", 512), "auto")


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("table", ["direct", "hash"])
def test_shared_graph_matches_own_copy(graph, device, table):
    """D25: a MultiSampler on a Sampler's device graph (no second copy) returns the same
    minibatches as one with its own copy, and keeps that Sampler alive."""
    indptr, indices = graph
    own = _C.MultiSampler(indptr, indices, FANOUTS, 256, 4, device)
    ps = _C.Sampler(indptr, indices, FANOUTS, 256, device)
    shared = _C.MultiSampler.from_sampler(ps, FANOUTS, 256, 4)
    del ps
    gc.collect()  # keep_alive: the Sampler (and its graph) must survive this
    assert shared.memory()["graph"] == 0 and own.memory()["graph"] > 0
    assert own.device_bytes() - shared.device_bytes() == own.memory()["graph"]
    rng = np.random.default_rng(25)
    for ms in (own, shared):
        ms.set_table(table)
    for rep in range(3):
        seeds = [rng.choice(len(indptr) - 1, 256, replace=False).astype(np.int32) for _ in range(4)]
        keys = [int(rng.integers(0, 2**40)) for _ in range(4)]
        for mode in ("batched", "megakernel"):
            c_own = own.run(seeds, keys, mode, 512)
            want = [own.result(m) for m in range(4)]
            c_sh = shared.run(seeds, keys, mode, 512)
            assert c_sh == c_own
            for m in range(4):
                got = shared.result(m)
                for k in ("node", "row", "col", "edge"):
                    np.testing.assert_array_equal(got[k], want[m][k], err_msg=f"{k} m={m} {mode}")
    check(shared, indptr, indices, seeds, keys, c_sh, "shared")
