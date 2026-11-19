"""The device-resident pipeline must hand PyTorch exactly what the host path returns, and
the gathered features must equal a CPU index_select."""

import numpy as np
import pytest
import torch

from gnnsample.synthetic import power_law_csc

_C = pytest.importorskip("gnnsample._C")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("needs an XPU device", allow_module_level=True)

from gnnsample.loader import GpuSampler  # noqa: E402

FANOUTS = [15, 10, 5]


@pytest.mark.parametrize("usm", ["host", "device"])
def test_export_matches_host_path(usm):
    indptr, indices = power_law_csc(num_nodes=30_000, avg_degree=20, seed=4)
    feat = np.random.default_rng(0).standard_normal((len(indptr) - 1, 12)).astype(np.float32)
    seeds = np.random.default_rng(1).choice(len(indptr) - 1, 512, replace=False)

    gs = GpuSampler(indptr, indices, FANOUTS, 512, feat=feat, usm=usm)
    node, edge_index, x, n_nodes, n_edges = gs.sample(seeds, seed=9)
    ref = _C.Sampler(indptr, indices, FANOUTS, 512, "gpu").sample(seeds.astype(np.int32), 9)

    assert node.device.type == "xpu" and x.device.type == "xpu"
    np.testing.assert_array_equal(node.cpu().numpy(), ref["node"])
    np.testing.assert_array_equal(edge_index[0].cpu().numpy(), ref["row"])
    np.testing.assert_array_equal(edge_index[1].cpu().numpy(), ref["col"])
    np.testing.assert_array_equal(x.cpu().numpy(), feat[ref["node"]])
    assert list(n_nodes) == list(ref["num_sampled_nodes"])
    assert list(n_edges) == list(ref["num_sampled_edges"])


def test_torch_can_use_exported_tensors():
    """Tensors written by our queue are usable by torch kernels afterwards."""
    indptr, indices = power_law_csc(num_nodes=5_000, avg_degree=10, seed=5)
    feat = np.arange((len(indptr) - 1) * 4, dtype=np.float32).reshape(-1, 4)
    gs = GpuSampler(indptr, indices, FANOUTS, 64, feat=feat)
    node, edge_index, x, _, _ = gs.sample(np.arange(64), seed=1)
    assert torch.equal((x[:, 0] / 4).long(), node)
    assert int(edge_index.max()) < len(node)


def test_prefetching_matches_serial():
    """Minibatch i+1 sampled during step i must equal the serial result, and a slot must
    not be overwritten while the consumer's kernels still read it."""
    from gnnsample.loader import PrefetchingGpuSampler

    indptr, indices = power_law_csc(num_nodes=30_000, avg_degree=20, seed=6)
    feat = np.random.default_rng(2).standard_normal((len(indptr) - 1, 16)).astype(np.float32)
    rng = np.random.default_rng(3)
    batches = [rng.choice(len(indptr) - 1, 256, replace=False) for _ in range(8)]

    serial = GpuSampler(indptr, indices, FANOUTS, 256, feat=feat)
    expected = []
    for i, b in enumerate(batches):
        node, ei, x, _, _ = serial.sample(b, seed=100 + i)
        expected.append((node.cpu(), ei.cpu(), x.cpu()))

    pre = PrefetchingGpuSampler(indptr, indices, FANOUTS, 256, feat=feat)
    w = torch.randn(16, 512, device="xpu")
    acc = torch.zeros((), device="xpu")
    for i, (node, ei, x, _, _) in enumerate(pre.iterate(batches, seed_of=lambda i: 100 + i)):
        # Heavy, asynchronous use of this slot while the next batch is being prepared.
        for _ in range(20):
            acc = acc + (x @ w).relu().sum()
        assert torch.equal(node.cpu(), expected[i][0]), f"node mismatch at batch {i}"
        assert torch.equal(ei.cpu(), expected[i][1]), f"edge mismatch at batch {i}"
        assert torch.equal(x.cpu(), expected[i][2]), f"feature mismatch at batch {i}"
    torch.xpu.synchronize()
    assert torch.isfinite(acc)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("K,n_batches", [(4, 11), (8, 8), (3, 1)])
def test_batched_loader_matches_serial(overlap, K, n_batches):
    """D24: BatchedGpuSampler (K per launch, per-minibatch gather, slot semaphores) must
    hand out exactly the serial path's tensors, also while the consumer keeps the GPU busy
    with asynchronous work on the slots and when the last block is partial."""
    from gnnsample.loader import BatchedGpuSampler

    indptr, indices = power_law_csc(num_nodes=30_000, avg_degree=20, seed=8)
    feat = np.random.default_rng(4).standard_normal((len(indptr) - 1, 16)).astype(np.float32)
    rng = np.random.default_rng(K)
    batches = [rng.choice(len(indptr) - 1, 256, replace=False) for _ in range(n_batches)]

    serial = GpuSampler(indptr, indices, FANOUTS, 256, feat=feat)
    expected = []
    for i, b in enumerate(batches):
        node, ei, x, nn, ne = serial.sample(b, seed=50 + i)
        expected.append((node.cpu(), ei.cpu(), x.cpu(), list(nn), list(ne)))

    # D25: the lean configuration (serial's graph and features, hash tables, grown slots)
    # and the D24 one (own graph copy, direct tables, worst-case slots).
    shared = BatchedGpuSampler(None, None, FANOUTS, 256, K=K, share_features_with=serial,
                               share_graph_with=serial, table="hash", overlap=overlap)
    own = BatchedGpuSampler(indptr, indices, FANOUTS, 256, K=K, feat=feat, overlap=overlap,
                            table="direct", slots="max")
    w = torch.randn(16, 512, device="xpu")
    for bs in (shared, own):
        acc = torch.zeros((), device="xpu")
        seen = 0
        for i, it in enumerate(bs.iterate_full(batches, seed_of=lambda i: 50 + i)):
            for _ in range(20):
                acc = acc + (it["x"] @ w).relu().sum()
            node, ei, x, nn, ne = expected[i]
            assert torch.equal(it["node"].cpu(), node), f"node mismatch at batch {i}"
            assert torch.equal(it["edge_index"].cpu(), ei), f"edge mismatch at batch {i}"
            assert torch.equal(it["x"].cpu(), x), f"feature mismatch at batch {i}"
            assert list(it["num_sampled_nodes"]) == nn and list(it["num_sampled_edges"]) == ne
            seen += 1
        assert seen == n_batches
        torch.xpu.synchronize()
        assert torch.isfinite(acc)


@pytest.mark.parametrize("overlap", [False, True])
def test_batched_loader_cpu_and_edge_pos(overlap):
    """The batched sampler's CPU path and edge positions match the single-minibatch path."""
    from gnnsample.loader import BatchedGpuSampler

    indptr, indices = power_law_csc(num_nodes=20_000, avg_degree=15, seed=12)
    feat = np.random.default_rng(6).standard_normal((len(indptr) - 1, 8)).astype(np.float32)
    rng = np.random.default_rng(7)
    batches = [rng.choice(len(indptr) - 1, 128, replace=False) for _ in range(7)]
    for device in ("cpu", "xpu"):
        single = GpuSampler(indptr, indices, FANOUTS, 128, feat=feat, device=device)
        bs = BatchedGpuSampler(indptr, indices, FANOUTS, 128, K=3, feat=feat, device=device,
                               overlap=overlap, with_edge_pos=True)
        for i, it in enumerate(bs.iterate_full(batches, seed_of=lambda i: 9 + i)):
            ref = single._sample_full(batches[i], 9 + i, with_edge_pos=True)
            for k in ("node", "edge_index", "x", "edge_pos"):
                assert it[k].device.type == device
                assert torch.equal(it[k].cpu(), ref[k].cpu()), f"{device} {k} batch {i}"


def test_batched_loader_survives_abandoned_epochs():
    """Regression (D24): a consumer that stops early (break), or that never resumes the
    generator after its last minibatch (zip with a shorter list), must not leave slots
    held: the next epoch used to wait forever on one of them."""
    from gnnsample.loader import BatchedGpuSampler

    indptr, indices = power_law_csc(num_nodes=20_000, avg_degree=15, seed=13)
    feat = np.random.default_rng(8).standard_normal((len(indptr) - 1, 8)).astype(np.float32)
    rng = np.random.default_rng(9)
    batches = [rng.choice(len(indptr) - 1, 128, replace=False) for _ in range(10)]
    single = GpuSampler(indptr, indices, FANOUTS, 128, feat=feat)
    expected = [single.sample(b, seed=i)[0].cpu() for i, b in enumerate(batches)]

    bs = BatchedGpuSampler(indptr, indices, FANOUTS, 128, K=3, feat=feat, overlap=True)
    for i, it in enumerate(bs.iterate_full(batches)):  # break mid-block
        if i == 4:
            break
    for _, it in zip(range(len(batches)), bs.iterate_full(batches)):  # never resumed at the end
        pass
    for i, it in enumerate(bs.iterate_full(batches)):  # a full epoch must still work
        assert torch.equal(it["node"].cpu(), expected[i])
    assert i == len(batches) - 1


def test_batched_loader_slot_growth():
    """D25: slots sized from the first block (small minibatches here) must be regrown, by
    the producer while it holds them, when later blocks are larger, without changing any
    output, while the consumer keeps the GPU busy with asynchronous work on the slots; also
    after abandoned epochs (as in test_batched_loader_survives_abandoned_epochs), which can
    stop the producer right after it regrew a slot."""
    from gnnsample.loader import BatchedGpuSampler

    # Large enough that minibatches stay well below the worst case (all nodes).
    indptr, indices = power_law_csc(num_nodes=200_000, avg_degree=20, seed=14)
    feat = np.random.default_rng(10).standard_normal((len(indptr) - 1, 16)).astype(np.float32)
    rng = np.random.default_rng(11)
    sizes = [4, 8, 6, 5] + [256] * 5 + [32, 200, 256, 16, 256, 128, 256, 64]
    batches = [rng.choice(len(indptr) - 1, b, replace=False) for b in sizes]
    serial = GpuSampler(indptr, indices, FANOUTS, 256, feat=feat)
    expected = []
    for i, b in enumerate(batches):
        r = serial._sample_full(b, 70 + i, with_edge_pos=True)
        expected.append({k: r[k].cpu() for k in ("node", "edge_index", "x", "edge_pos")})

    bs = BatchedGpuSampler(None, None, FANOUTS, 256, K=4, share_graph_with=serial,
                           share_features_with=serial, overlap=True, with_edge_pos=True)
    worst = BatchedGpuSampler(indptr, indices, FANOUTS, 256, K=4, feat=feat, overlap=True,
                              with_edge_pos=True, table="direct", slots="max")
    w = torch.randn(16, 512, device="xpu")

    def epoch(stop_at=None, zipped=False):
        acc = torch.zeros((), device="xpu")
        src = bs.iterate_full(batches, seed_of=lambda i: 70 + i)
        if zipped:  # never resumed after the last minibatch
            src = (it for _, it in zip(range(len(batches)), src))
        seen = 0
        for i, it in enumerate(src):
            for _ in range(20):
                acc = acc + (it["x"] @ w).relu().sum()
            for k, v in expected[i].items():
                assert torch.equal(it[k].cpu(), v), f"{k} mismatch at batch {i}"
            seen += 1
            if i == stop_at:
                break
        torch.xpu.synchronize()
        assert torch.isfinite(acc)
        return seen

    assert bs.memory()["graph_slots"] == 0  # nothing is allocated before the first block
    assert epoch(stop_at=5) == 6  # abandoned in the second block (fresh, larger slots)
    assert epoch(zipped=True) == len(batches)
    assert epoch() == len(batches)
    assert bs.regrown > 0  # block 2 reuses block 0's small slots
    for s in (s for gs in bs.gslots for s in gs if s["node"] is not None):
        assert s["node"].numel() <= bs.ms.max_nodes and s["edges"].numel() <= 2 * bs.ms.max_edges
    mem, mem_worst = bs.memory(), worst.memory()
    assert mem["graph"] == 0 and mem_worst["graph"] > 0
    assert mem["graph_slots"] < mem_worst["graph_slots"]
    assert mem["feature_slots"] < mem_worst["feature_slots"]
    assert mem["total"] == bs.device_bytes() < worst.device_bytes()
