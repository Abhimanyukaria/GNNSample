"""gnnsample: GPU neighbor sampling for GNN training (SYCL)."""


def __getattr__(name):
    # Imported lazily so `import gnnsample` (reference sampler, tests) does not need
    # torch_geometric or the compiled extension.
    if name == "NeighborLoader":
        from .neighbor_loader import NeighborLoader

        return NeighborLoader
    raise AttributeError(name)
