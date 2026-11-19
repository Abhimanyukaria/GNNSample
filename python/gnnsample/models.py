"""GraphSAGE for sampled minibatches, with optional layer trimming.

Sampled minibatches are layered: hop h's edges point into the nodes added at hop h-1.
Without trimming, every layer runs over all sampled nodes, although 3rd-hop leaves (~70%
of the nodes for [15,10,5]) only ever serve as inputs. Trimming (the idea behind PyG's
trim_to_layer) gives layer l only the edges and output rows that later layers use.
On ogbn-products this cut the iGPU train step from 1.91 s to 0.30 s (results/train_step_products.json).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def orient(row, col, n_edges, batch):
    """Return (src, dst) so that messages flow neighbor -> target: hop-0 edges must land
    on the seeds (local ids < batch)."""
    e0 = n_edges[0]
    if e0 == 0 or col[:e0].max() < batch:
        return row, col
    assert row[:e0].max() < batch, "neither orientation points into the seeds"
    return col, row


def layer_adjs(src, dst, n_nodes, n_edges, trim, layers=3):
    """Per layer: (edge_index [2, E_l], number of output rows)."""
    cum_e = np.cumsum([0] + list(n_edges))
    cum_n = np.cumsum([0] + list(n_nodes))
    out = []
    for l in range(layers):
        if trim:
            keep = layers - l  # layer 0 uses all hops, the last layer only hop 0
            e_hi, n_dst = int(cum_e[keep]), int(cum_n[keep])
        else:
            e_hi, n_dst = len(src), int(cum_n[-1])
        out.append((torch.stack([src[:e_hi], dst[:e_hi]]), n_dst))
    return out


class SAGE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out, layers=3):
        super().__init__()
        dims = [d_in] + [d_hidden] * (layers - 1) + [d_out]
        self.convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x, adjs):
        for l, (edge_index, n_dst) in enumerate(adjs):
            x = self.convs[l]((x, x[:n_dst]), edge_index)
            if l < len(self.convs) - 1:
                x = F.relu(x)
        return x
