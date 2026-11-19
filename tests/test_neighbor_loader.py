"""gnnsample.NeighborLoader must be usable wherever PyG's NeighborLoader is: same batch
attributes, correct contents against the source graph, and PyG models (with layer trimming)
must run on its batches."""

import numpy as np
import pytest
import torch

pytest.importorskip("gnnsample._C")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("needs an XPU device", allow_module_level=True)

from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import NeighborLoader as PygLoader  # noqa: E402

from gnnsample import NeighborLoader  # noqa: E402
from gnnsample.synthetic import power_law_csc  # noqa: E402

FANOUTS = [15, 10, 5]


@pytest.fixture(scope="module")
def data():
    indptr, indices = power_law_csc(num_nodes=6_000, avg_degree=15, seed=11)
    dst = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    edge_index = torch.from_numpy(np.stack([indices.astype(np.int64), dst]))
    perm = torch.randperm(edge_index.shape[1], generator=torch.Generator().manual_seed(0))
    n = len(indptr) - 1
    return Data(x=torch.randn(n, 6), y=torch.randint(0, 5, (n,)), edge_index=edge_index[:, perm])


def edge_set(d):
    return set((d.edge_index[0] * d.num_nodes + d.edge_index[1]).tolist())


@pytest.mark.parametrize("sampler", ["batched", "single"])
@pytest.mark.parametrize("device", ["xpu", "cpu"])
@pytest.mark.parametrize("prefetch", [True, False])
def test_batches_match_graph_and_pyg(data, prefetch, device, sampler):
    train = torch.randperm(data.num_nodes, generator=torch.Generator().manual_seed(1))[:1500]
    ours = NeighborLoader(data, num_neighbors=FANOUTS, input_nodes=train, batch_size=256,
                          shuffle=True, prefetch=prefetch, seed=3, device=device,
                          sampler=sampler, K=3)
    pyg = PygLoader(data, num_neighbors=FANOUTS, input_nodes=train, batch_size=256)
    assert len(ours) == len(pyg)
    edges = edge_set(data)
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    pyg_keys = set(next(iter(pyg)).keys()) - {"e_id"}

    for b in ours:
        assert pyg_keys <= set(b.keys()), f"missing {pyg_keys - set(b.keys())}"
        assert b.x.device.type == device and b.edge_index.device.type == device
        n_id = b.n_id.cpu()
        bs = b.batch_size
        assert torch.equal(n_id[:bs], train[b.input_id]), "seeds first, input_id points to them"
        assert torch.equal(b.x.cpu(), data.x[n_id]) and b.x.shape[1] == 6
        assert torch.equal(b.y.cpu(), data.y[n_id])
        src, dst = n_id[b.edge_index[0].cpu()], n_id[b.edge_index[1].cpu()]
        assert set((src * data.num_nodes + dst).tolist()) <= edges, "edge not in graph"
        assert sum(b.num_sampled_nodes) == len(n_id)
        assert sum(b.num_sampled_edges) == b.edge_index.shape[1]
        # Hop 0 is deterministic in size: min(in-degree, 15) per seed, as in PyG.
        assert b.num_sampled_edges[0] == int(deg[n_id[:bs]].clamp(max=15).sum())


@pytest.mark.parametrize("sampler", ["batched", "single"])
@pytest.mark.parametrize("device,prefetch", [("xpu", True), ("xpu", False), ("cpu", True),
                                             ("cpu", False)])
def test_e_id(data, device, prefetch, sampler):
    loader = NeighborLoader(data, num_neighbors=FANOUTS, batch_size=128, return_e_id=True,
                            device=device, prefetch=prefetch, sampler=sampler)
    for i, b in enumerate(loader):
        n_id = b.n_id.cpu()
        e = data.edge_index[:, b.e_id.cpu()]
        assert torch.equal(e[0], n_id[b.edge_index[0].cpu()])
        assert torch.equal(e[1], n_id[b.edge_index[1].cpu()])
        if i == 3:
            break


def test_pyg_model_with_trimming(data):
    from torch_geometric.nn.models import GraphSAGE

    model = GraphSAGE(6, 32, num_layers=3, out_channels=5).to("xpu")
    loader = NeighborLoader(data, num_neighbors=FANOUTS, batch_size=128, shuffle=True)
    b = next(iter(loader))
    out = model(b.x, b.edge_index, num_sampled_nodes_per_hop=b.num_sampled_nodes,
                num_sampled_edges_per_hop=b.num_sampled_edges)
    loss = torch.nn.functional.cross_entropy(out[: b.batch_size], b.y[: b.batch_size])
    loss.backward()
    assert out.shape[1] == 5 and torch.isfinite(loss)


def test_unsupported_arguments_raise(data):
    with pytest.raises(NotImplementedError):
        NeighborLoader(data, num_neighbors=[10, -1], batch_size=8)
    with pytest.raises(NotImplementedError):
        NeighborLoader(data, num_neighbors=[10], batch_size=8, replace=True)
    with pytest.raises(NotImplementedError):
        NeighborLoader(data, num_neighbors=[10], batch_size=8, time_attr="t")
    NeighborLoader(data, num_neighbors=[10], batch_size=8, num_workers=4, pin_memory=True)


@pytest.mark.parametrize("sampler", ["batched", "single"])
def test_cpu_and_xpu_give_identical_batches(data, sampler):
    """The sampler is bit-identical across devices, so the two loaders must agree exactly."""
    kw = dict(num_neighbors=FANOUTS, batch_size=200, shuffle=True, seed=7, sampler=sampler)
    for i, (bx, bc) in enumerate(zip(NeighborLoader(data, device="xpu", **kw),
                                     NeighborLoader(data, device="cpu", **kw))):
        for k in ("n_id", "edge_index", "x", "y", "input_id"):
            assert torch.equal(bx[k].cpu(), bc[k].cpu()), f"{k} differs at batch {i}"
        assert bx.num_sampled_edges == bc.num_sampled_edges
        if i == 5:
            break


def test_auto_device(data):
    loader = NeighborLoader(data, num_neighbors=[5], batch_size=16)
    assert loader.device.type == "xpu"  # PyTorch sees the iGPU in this container
    assert loader.sampler_kind == "batched"  # D24: the default


@pytest.mark.parametrize("device,prefetch", [("xpu", True), ("xpu", False), ("cpu", True)])
def test_batched_and_single_give_identical_batches(data, device, prefetch):
    """D24: switching the default sampler must not change a single minibatch (whole
    epochs, two of them, last block partial)."""
    train = torch.randperm(data.num_nodes, generator=torch.Generator().manual_seed(2))[:2300]
    kw = dict(num_neighbors=FANOUTS, input_nodes=train, batch_size=100, shuffle=True, seed=5,
              device=device, prefetch=prefetch, return_e_id=True)
    a, b = NeighborLoader(data, sampler="batched", K=4, **kw), NeighborLoader(data, sampler="single", **kw)
    for epoch in range(2):
        n = 0
        for i, (ba, bb) in enumerate(zip(a, b)):
            for k in ("n_id", "edge_index", "x", "y", "input_id", "e_id"):
                assert torch.equal(ba[k].cpu(), bb[k].cpu()), f"{k} differs, epoch {epoch} batch {i}"
            assert ba.num_sampled_nodes == bb.num_sampled_nodes
            assert ba.num_sampled_edges == bb.num_sampled_edges
            n += 1
        assert n == len(a) == 23


def test_batched_falls_back_to_single(data):
    with pytest.warns(UserWarning, match="sampler='single'"):
        loader = NeighborLoader(data, num_neighbors=[20, 5], batch_size=16)
    assert loader.sampler_kind == "single"
    next(iter(loader))
    with pytest.warns(UserWarning):
        assert NeighborLoader(data, num_neighbors=[5], batch_size=16, options=8).sampler_kind == "single"
    with pytest.raises(ValueError):
        NeighborLoader(data, num_neighbors=[5], batch_size=16, sampler="mega")
