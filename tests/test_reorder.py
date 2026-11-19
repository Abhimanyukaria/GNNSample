"""D23: reordering, hub cache hints and the cross-minibatch schedule must not change the
sample. Sampling a renumbered graph (with its mapping) must give exactly the reference
sample of the original graph: same nodes, local ids, edges and CSC positions."""

import numpy as np
import pytest

from gnnsample.reference import sample
from gnnsample.reorder import METHODS, compute_order, hub_count, inverse, permute_csc, pos_shift
from gnnsample.synthetic import power_law_csc

_C = pytest.importorskip("gnnsample._C")
FANOUTS = [15, 10, 5]


@pytest.fixture(scope="module")
def graph():
    # Shuffle the synthetic graph's ids (it is generated hub-first) so that every method
    # really moves nodes.
    indptr, indices = power_law_csc(num_nodes=40_000, avg_degree=20, seed=7)
    shuffle = compute_order(indptr, indices, "random", seed=1)
    return permute_csc(indptr, indices, shuffle)


@pytest.fixture(scope="module", params=METHODS)
def reordered(request, graph):
    indptr, indices = graph
    order = compute_order(indptr, indices, request.param)
    new_indptr, new_indices = permute_csc(indptr, indices, order, chunk_edges=10_000)
    return request.param, order, new_indptr, new_indices


def test_permutation_preserves_rows(graph, reordered):
    indptr, indices = graph
    _, order, ni, nx = reordered
    assert sorted(order.tolist()) == list(range(len(order)))
    perm = inverse(order)
    for v_new in np.random.default_rng(0).choice(len(order), 500, replace=False):
        v = order[v_new]
        np.testing.assert_array_equal(nx[ni[v_new]:ni[v_new + 1]], perm[indices[indptr[v]:indptr[v + 1]]])
    shift = pos_shift(indptr, ni, order)
    np.testing.assert_array_equal(ni[:-1] + shift, indptr[order])


def test_hub_first_methods(graph):
    indptr, indices = graph
    deg = np.diff(indptr)
    for method in ("degree", "hubsort"):
        order = compute_order(indptr, indices, method)
        nd = deg[order]
        h = int((deg > deg.mean()).sum())
        assert (np.diff(nd[:h]) <= 0).all() and nd[:h].min() >= nd[h:].max()
    ni, _ = permute_csc(indptr, indices, compute_order(indptr, indices, "degree"))
    h = hub_count(ni, 64 << 10)
    assert 8 * h + 4 * ni[h] <= 64 << 10 < 8 * (h + 1) + 4 * ni[h + 1]


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("options", [0, 8169, 8169 | 8192, 9 | 112])
def test_sampler_on_reordered_graph_matches_reference(graph, reordered, device, options):
    indptr, indices = graph
    method, order, ni, nx = reordered
    s = _C.Sampler(ni, nx, FANOUTS, 1024, device)
    s.set_options(options)
    s.set_reorder(order, pos_shift(indptr, ni, order))
    s.set_hubs(hub_count(ni, 256 << 10))
    rng = np.random.default_rng(options)
    for b in range(3):
        seeds = rng.choice(len(order), 1024, replace=False)
        got = s.sample(seeds.astype(np.int32), b)
        ref = sample(indptr, indices, seeds, FANOUTS, seed=b)
        assert list(got["num_sampled_nodes"]) == ref.num_sampled_nodes, method
        for k in ("node", "row", "col", "edge"):
            np.testing.assert_array_equal(got[k], getattr(ref, k), err_msg=f"{method} {k}")
        raw = s.sample(seeds.astype(np.int32), b, raw=True)
        np.testing.assert_array_equal(order[raw["node"]], ref.node)


def test_hub_hints_identical_on_original_graph(graph):
    indptr, indices = graph
    s = _C.Sampler(indptr, indices, FANOUTS, 512, "gpu")
    seeds = np.arange(0, 40_000, 80, dtype=np.int32)
    s.set_options(8169)
    a = s.sample(seeds, 3)
    for hubs in (0, 100, 10_000):
        s.set_options(8169 | _C.HUB_HINTS)
        s.set_hubs(hubs)
        b = s.sample(seeds, 3)
        for k in ("node", "row", "edge"):
            np.testing.assert_array_equal(a[k], b[k])


def test_reorder_switch_back(graph, reordered):
    indptr, indices = graph
    _, order, ni, nx = reordered
    s = _C.Sampler(ni, nx, FANOUTS, 256, "gpu")
    s.set_options(8169)
    seeds = np.arange(256, dtype=np.int32)
    s.set_reorder(order)
    got = s.sample(seeds, 1)
    assert (got["edge"] == -1).all()  # no pos_shift given
    np.testing.assert_array_equal(got["node"], sample(indptr, indices, seeds, FANOUTS, seed=1).node)
    s.set_reorder(None)
    np.testing.assert_array_equal(s.sample(seeds, 1)["node"], sample(ni, nx, seeds, FANOUTS, seed=1).node)


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("mode,sched", [("batched", False), ("batched", True), ("megakernel", False)])
def test_multi_on_reordered_graph(graph, reordered, device, mode, sched):
    indptr, indices = graph
    method, order, ni, nx = reordered
    ms = _C.MultiSampler(ni, nx, FANOUTS, 256, 6, device)
    ms.set_reorder(order, pos_shift(indptr, ni, order))
    ms.set_hubs(hub_count(ni, 256 << 10), True)
    ms.set_schedule(sched)
    rng = np.random.default_rng(5)
    for rep in range(2):
        K = 6 if rep else 3
        seeds = [rng.choice(len(order), int(rng.integers(1, 257)), replace=False).astype(np.int32)
                 for _ in range(K)]
        keys = [int(rng.integers(0, 2**40)) for _ in range(K)]
        ms.run(seeds, keys, mode, 512)
        for m in range(K):
            got = ms.result(m)
            ref = sample(indptr, indices, seeds[m], FANOUTS, seed=keys[m])
            for k in ("node", "row", "col", "edge"):
                np.testing.assert_array_equal(got[k], getattr(ref, k), err_msg=f"{method} {k} m={m}")


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("K", [1, 4, 8])
def test_schedule_on_original_graph(graph, device, K):
    """The locality schedule alone (no reordering) leaves every minibatch unchanged."""
    indptr, indices = graph
    ms = _C.MultiSampler(indptr, indices, FANOUTS, 512, 8, device)
    ms.set_schedule(True)
    rng = np.random.default_rng(K)
    seeds = [rng.choice(len(indptr) - 1, 512, replace=False).astype(np.int32) for _ in range(K)]
    keys = list(range(100, 100 + K))
    ms.run(seeds, keys, "batched")
    for m in range(K):
        ref = sample(indptr, indices, seeds[m], FANOUTS, seed=keys[m])
        got = ms.result(m)
        for k in ("node", "row", "col", "edge"):
            np.testing.assert_array_equal(got[k], getattr(ref, k), err_msg=f"{k} m={m}")


def test_gpu_sampler_export_maps_ids(graph, reordered):
    """XPU export path: node ids, gathered features and edge positions are the caller's."""
    torch = pytest.importorskip("torch")
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip("no XPU")
    from gnnsample.loader import GpuSampler

    indptr, indices = graph
    _, order, ni, nx = reordered
    feat = np.random.default_rng(0).standard_normal((len(order), 8)).astype(np.float32)
    gs = GpuSampler(ni, nx, FANOUTS, 512, feat=feat, device="xpu", order=order,
                    pos_shift=pos_shift(indptr, ni, order))
    seeds = np.arange(0, 40_000, 100, dtype=np.int32)
    r = gs._sample_full(seeds, 4, with_features=True, with_edge_pos=True)
    ref = sample(indptr, indices, seeds, FANOUTS, seed=4)
    np.testing.assert_array_equal(r["node"].cpu().numpy(), ref.node)
    np.testing.assert_array_equal(r["edge_pos"].cpu().numpy(), ref.edge)
    np.testing.assert_array_equal(r["x"].cpu().numpy(), feat[ref.node])
