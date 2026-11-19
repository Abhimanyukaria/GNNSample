"""Sampling pipeline behind the PyTorch-facing API, on the Intel GPU (XPU) or the CPU.

device="xpu" (decision D11): sample on the GPU, then gather feature rows from a host-USM
(zero-copy) table straight into PyTorch XPU tensors. Nothing goes through the CPU except
the seed ids and the per-hop counts.

device="cpu": the same SYCL sampler runs on the CPU device (bit-identical output; it uses
all CPU threads unless the runtime is told otherwise, see docs/ARCHITECTURE.md); results
come back as CPU tensors and features are gathered with torch.index_select.

device="auto" (default) picks the XPU when PyTorch sees one, else the CPU.
"""

import time

import numpy as np
import torch

from . import _C

# Best configuration from the Phase 3 A/B ablations (docs/decisions.md D13).
DEFAULT_OPTIONS = (_C.PACKED_HASH | _C.NO_SYNC | _C.ASSIGN_FROM_SCAN | _C.SLOT32 |
                   _C.GEN_TAGS | _C.SUBGROUP_SAMPLE | _C.LABEL_EARLY | _C.SMALL_TABLE |
                   _C.DIRECT_MAP | _C.NO_NODE_SLOT)


def resolve_device(device="auto"):
    """'auto' -> 'xpu' if PyTorch has an XPU, else 'cpu'; 'gpu' is accepted for 'xpu'."""
    if device in (None, "auto"):
        return torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                            else "cpu")
    d = torch.device("xpu" if str(device) == "gpu" else device)
    if d.type not in ("xpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'xpu' or 'cpu', not {device!r}")
    return d


class GpuSampler:
    """Multi-hop neighbour sampler returning PyTorch tensors on `device` (XPU or CPU)."""

    def __init__(self, indptr, indices, fanouts, max_batch, feat=None, usm="host",
                 device="auto", options=DEFAULT_OPTIONS, profiling=False, order=None,
                 pos_shift=None, hubs=0):
        """order / pos_shift: the graph is reorder.py's renumbering (Graph.reorder_args());
        seeds, node ids, features and edge positions all stay in the original ids.
        hubs: hub count for options with HUB_HINTS (reorder.hub_count)."""
        self.device = resolve_device(device)
        sycl_device = "gpu" if self.device.type == "xpu" else "cpu"
        self.s = _C.Sampler(indptr, indices, list(fanouts), max_batch, sycl_device, profiling)
        self.s.set_options(options)
        self.reordered = order is not None  # BatchedGpuSampler(share_graph_with=) checks it
        if order is not None:
            self.s.set_reorder(order, pos_shift)
        self.s.set_hubs(hubs)
        self.dim = None
        self.feat_cpu = None
        if feat is not None:
            self.dim = feat.shape[1]
            if self.device.type == "xpu":
                self.s.set_features(np.ascontiguousarray(feat, dtype=np.float32), usm)
            else:
                self.feat_cpu = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32))

    def sample(self, seeds, seed=0, with_features=True):
        """Returns (node, edge_index, x, num_sampled_nodes, num_sampled_edges), tensors on
        self.device. edge_index is (neighbor, target) local ids; x is None without features."""
        r = self._sample_full(seeds, seed, with_features, with_edge_pos=False)
        return r["node"], r["edge_index"], r["x"], r["num_sampled_nodes"], r["num_sampled_edges"]

    def _sample_full(self, seeds, seed, with_features=True, with_edge_pos=False):
        seeds = np.ascontiguousarray(seeds, dtype=np.int32)
        if self.device.type == "cpu":
            r = self.s.sample(seeds, seed)  # host arrays
            node = torch.from_numpy(r["node"]).long()
            edge_index = torch.from_numpy(np.stack([r["row"], r["col"]])).long()
            x = None
            if with_features and self.feat_cpu is not None:
                x = torch.index_select(self.feat_cpu, 0, node)
            return dict(node=node, edge_index=edge_index, x=x,
                        edge_pos=torch.from_numpy(r["edge"]) if with_edge_pos else None,
                        num_sampled_nodes=r["num_sampled_nodes"],
                        num_sampled_edges=r["num_sampled_edges"])
        c = self.s.run(seeds, seed)
        dev = self.device
        node = torch.empty(c["num_nodes"], dtype=torch.int64, device=dev)
        edge_index = torch.empty(2, c["num_edges"], dtype=torch.int64, device=dev)
        x = epos = None
        if with_features and self.dim is not None:
            x = torch.empty(c["num_nodes"], self.dim, device=dev)
        if with_edge_pos:
            epos = torch.empty(c["num_edges"], dtype=torch.int64, device=dev)
        # Our queue writes these buffers directly; make sure torch has no pending work on
        # memory its caching allocator just handed back to us.
        torch.xpu.current_stream().synchronize()
        self.s.export_to(node.data_ptr(), edge_index.data_ptr(),
                         x.data_ptr() if x is not None else 0,
                         epos.data_ptr() if epos is not None else 0)
        return dict(node=node, edge_index=edge_index, x=x, edge_pos=epos,
                    num_sampled_nodes=c["num_sampled_nodes"],
                    num_sampled_edges=c["num_sampled_edges"])

    def iterate_full(self, batches, seed_of=lambda i: i):
        """Serial: yields _sample_full() dicts for `batches` in order, fresh tensors each."""
        for i, b in enumerate(batches):
            yield self._sample_full(b, seed_of(i),
                                    with_edge_pos=getattr(self, "with_edge_pos", False))

    def set_profile(self, mode=2):
        """0 off, 1 host wall time with waits, 2 GPU timestamps (needs profiling=True)."""
        self.s.set_profile(mode)

    def stage_times(self):
        return dict(self.s.stage_times())


class PrefetchingGpuSampler(GpuSampler):
    """Samples minibatch i+1 while PyTorch trains minibatch i (background thread; the C++
    calls release the GIL).

    On the XPU the next minibatch is exported into one of two preallocated device slots
    while the consumer trains on the other. Before overwriting a slot, the thread waits on
    an XPU event recorded after the consumer's last use of it, so no in-flight training
    kernel can see its inputs change. The yielded tensors are views into a slot: they are
    valid until the consumer asks for the minibatch after next.

    On the CPU every minibatch gets fresh tensors, so there is no reuse to guard.
    """

    def __init__(self, *args, with_edge_pos=False, **kwargs):
        super().__init__(*args, **kwargs)
        import concurrent.futures

        self.with_edge_pos = with_edge_pos
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.slots = None
        if self.device.type != "xpu":
            return
        dev = self.device
        mn, me = self.s.max_nodes, self.s.max_edges
        self.slots = [dict(node=torch.empty(mn, dtype=torch.int64, device=dev),
                           edges=torch.empty(2 * me, dtype=torch.int64, device=dev),
                           x=(torch.empty(mn * self.dim, device=dev)
                              if self.dim is not None else None),
                           epos=(torch.empty(me, dtype=torch.int64, device=dev)
                                 if with_edge_pos else None),
                           free=None)  # XPU event: consumer finished with this slot
                      for _ in range(2)]

    def _prepare(self, slot, seeds, seed):
        if slot is None:  # CPU: fresh tensors per minibatch
            return self._sample_full(seeds, seed, with_edge_pos=self.with_edge_pos)
        c = self.s.run(np.ascontiguousarray(seeds, dtype=np.int32), seed)  # internal buffers
        if slot["free"] is not None:
            slot["free"].synchronize()  # last training step that read this slot is done
        n, e = c["num_nodes"], c["num_edges"]
        self.s.export_to(slot["node"].data_ptr(), slot["edges"].data_ptr(),
                         slot["x"].data_ptr() if slot["x"] is not None else 0,
                         slot["epos"].data_ptr() if slot["epos"] is not None else 0)
        x = slot["x"][: n * self.dim].view(n, self.dim) if slot["x"] is not None else None
        epos = slot["epos"][:e] if slot["epos"] is not None else None
        return dict(node=slot["node"][:n], edge_index=slot["edges"][: 2 * e].view(2, e), x=x,
                    edge_pos=epos, num_sampled_nodes=c["num_sampled_nodes"],
                    num_sampled_edges=c["num_sampled_edges"])

    def iterate(self, batches, seed_of=lambda i: i):
        """Yields (node, edge_index, x, num_sampled_nodes, num_sampled_edges) for `batches` in
        order, one minibatch ahead."""
        for r in self.iterate_full(batches, seed_of):
            yield (r["node"], r["edge_index"], r["x"], r["num_sampled_nodes"],
                   r["num_sampled_edges"])

    def iterate_full(self, batches, seed_of=lambda i: i):
        """Like iterate(), as dicts that also carry edge_pos (CSC positions) when enabled."""
        if not len(batches):
            return
        on_xpu = self.slots is not None
        if on_xpu:
            torch.xpu.current_stream().synchronize()
        slot = (lambda i: self.slots[i % 2]) if on_xpu else (lambda i: None)
        fut = self.pool.submit(self._prepare, slot(0), batches[0], seed_of(0))
        for i in range(len(batches)):
            res = fut.result()
            if i + 1 < len(batches):
                nxt = slot(i + 1)
                if on_xpu:
                    # The consumer has finished (submitted) the step that used this slot two
                    # minibatches ago; mark the point after it on torch's queue.
                    ev = torch.xpu.Event()
                    ev.record()
                    nxt["free"] = ev
                fut = self.pool.submit(self._prepare, nxt, batches[i + 1], seed_of(i + 1))
            yield res


class BatchedGpuSampler:
    """K minibatches per sampling launch (MultiSampler batched mode, D22): the primary
    training sampler (D24). Same minibatches, keys and output as GpuSampler.

    One launch samples K minibatches and exports their graphs; feature rows are gathered
    per minibatch, just before that minibatch is handed out, so features for at most 2
    minibatches are resident (products: ~160 MB each) instead of 2K.

    XPU, overlap=True: a background thread runs ahead (the next block's sampling overlaps
    this block's training). Graphs go into 2 sets of K slots and features into 2 slots.
    Each slot has a semaphore: the producer takes it before writing; the consumer gives it
    back, with an XPU event recorded after its step, when it asks for the next minibatch,
    and the producer waits on that event before overwriting. Yielded tensors are views
    into slots, valid until the consumer asks for the minibatch after next.
    XPU, overlap=False: every minibatch gets fresh tensors (like GpuSampler).
    CPU: MultiSampler on the CPU device, fresh tensors, features by torch.index_select;
    overlap=True still prepares the next minibatches in a background thread.

    Every yielded item carries producer-side stage times in item["t"] (seconds): sample
    and export (the K-minibatch launch, attributed to the block's first item), gather and
    slot_wait.

    Memory (D25); each switch is output-preserving, and the defaults are the lean setting:
      share_graph_with  a GpuSampler on the same device: MultiSampler reads its device graph
                        instead of a second copy (indptr / indices may then be None)
      table             dedup tables: "auto" (whichever needs fewer bytes), "direct", "hash"
      slots             "grow": XPU slots sized to the minibatches actually seen (the first
                        block's largest x SLOT_HEADROOM) and regrown when one does not fit;
                        "max": worst-case slots allocated up front (the D24 behaviour)
    memory() / device_bytes() report what is allocated.
    """

    SLOT_HEADROOM = 1.25  # room over the largest minibatch seen when a slot is (re)sized

    def __init__(self, indptr, indices, fanouts, max_batch, K=8, feat=None, usm="host",
                 device="auto", overlap=True, with_edge_pos=False, share_features_with=None,
                 share_graph_with=None, table="auto", slots="grow"):
        import threading

        self.device = resolve_device(device)
        self.K, self.overlap, self.with_edge_pos = K, overlap, with_edge_pos
        on_xpu = self.device.type == "xpu"
        if slots not in ("grow", "max"):
            raise ValueError("slots must be 'grow' or 'max'")
        if share_graph_with is not None:
            if share_graph_with.device.type != self.device.type:
                raise ValueError("share_graph_with must be a GpuSampler on the same device")
            if getattr(share_graph_with, "reordered", False):
                raise ValueError("share_graph_with holds a reordered graph (not supported)")
            self.ms = _C.MultiSampler.from_sampler(share_graph_with.s, list(fanouts), max_batch, K)
        else:
            self.ms = _C.MultiSampler(indptr, indices, list(fanouts), max_batch, K,
                                      "gpu" if on_xpu else "cpu")
        self.ms.set_table(table)
        self.dim = self.feat_cpu = None
        if share_features_with is not None:  # a GpuSampler on the same device
            self.dim = share_features_with.dim
            if on_xpu:
                self.ms.share_features(share_features_with.s)
            else:
                self.feat_cpu = share_features_with.feat_cpu
        elif feat is not None:
            self.dim = feat.shape[1]
            f = np.ascontiguousarray(feat, dtype=np.float32)
            if on_xpu:
                self.ms.set_features(f, usm)
            else:
                self.feat_cpu = torch.from_numpy(f)
        self.gslots = self.fslots = None
        # Slot capacity target (nodes, edges); "grow": set by the first block that uses
        # slots and raised whenever a slot is regrown, so later allocations match the largest.
        self._target = None
        self.regrown = 0  # slot tensors replaced by larger ones (first allocations not counted)
        if on_xpu:
            # Tensors come from _fit_graph / _fit_feat: lazily, so overlap=False allocates none.
            self.gslots = [[dict(node=None, edges=None, epos=None, sem=threading.Semaphore(1),
                                 ev=None) for _ in range(K)] for _ in range(2)]
            if self.dim is not None:
                self.fslots = [dict(x=None, sem=threading.Semaphore(1), ev=None)
                               for _ in range(2)]
            if slots == "max":
                mn, me = self.ms.max_nodes, self.ms.max_edges
                self._target = (mn, me)
                for s in (s for gs in self.gslots for s in gs):
                    self._fit_graph(s, mn, me)
                for s in self.fslots or ():
                    self._fit_feat(s, mn)
                torch.xpu.current_stream().synchronize()

    def memory(self):
        """Device bytes by part (D25): MultiSampler's graph copy (0 when shared), dedup
        tables and per-minibatch buffers, and the graph / feature slots allocated so far.
        The feature table itself is not included."""
        m = self.ms.memory()

        def nbytes(slots, keys):
            return sum(s[k].numel() * s[k].element_size() for s in slots for k in keys
                       if s[k] is not None)

        gs = nbytes([s for st in self.gslots or () for s in st], ("node", "edges", "epos"))
        fs = nbytes(self.fslots or (), ("x",))
        return dict(graph=m["graph"], tables=m["tables"], buffers=m["buffers"], graph_slots=gs,
                    feature_slots=fs, total=m["total"] + gs + fs, table=m["table"])

    def device_bytes(self):
        return self.memory()["total"]

    def _cap(self, need, i):
        """Capacity for `need` nodes (i=0) or edges (i=1): at least the target, with
        headroom over `need`, never above the worst case."""
        worst = (self.ms.max_nodes, self.ms.max_edges)[i]
        return min(worst, max(self._target[i], int(need * self.SLOT_HEADROOM) + 1))

    # _fit_graph / _fit_feat (re)allocate a slot's tensors when a minibatch does not fit.
    # They run only where the producer holds that slot: its semaphore is taken and the XPU
    # event recorded after the consumer's last step on it has been synchronised (_take), so
    # no queued kernel still reads the old storage, and a view the consumer still references
    # keeps that storage alive (it is past its validity anyway). They return True if they
    # allocated; the caller must then synchronise torch's stream before our queue writes the
    # new memory, since torch's caching allocator may hand back blocks with torch work pending.
    def _fit_graph(self, slot, n, e):
        grew = False
        dev = self.device
        if slot["node"] is None or slot["node"].numel() < n:
            self.regrown += slot["node"] is not None
            slot["node"] = None  # drop the old storage first: lower peak
            cn = self._cap(n, 0)
            slot["node"] = torch.empty(cn, dtype=torch.int64, device=dev)
            self._target = (max(self._target[0], cn), self._target[1])
            grew = True
        if slot["edges"] is None or slot["edges"].numel() < 2 * e:
            self.regrown += slot["edges"] is not None
            slot["edges"] = slot["epos"] = None
            ce = self._cap(e, 1)
            slot["edges"] = torch.empty(2 * ce, dtype=torch.int64, device=dev)
            if self.with_edge_pos:
                slot["epos"] = torch.empty(ce, dtype=torch.int64, device=dev)
            self._target = (self._target[0], max(self._target[1], ce))
            grew = True
        return grew

    def _fit_feat(self, slot, n):
        if slot["x"] is not None and slot["x"].numel() >= n * self.dim:
            return False
        self.regrown += slot["x"] is not None
        slot["x"] = None
        cn = self._cap(n, 0)
        slot["x"] = torch.empty(cn * self.dim, device=self.device)
        self._target = (max(self._target[0], cn), self._target[1])
        return True

    @staticmethod
    def _take(slot, stop):
        """Waits for a slot (polling `stop`, so an abandoned iteration can end its producer);
        returns the seconds waited, or None if stopped."""
        t0 = time.perf_counter()
        while not slot["sem"].acquire(timeout=0.05):
            if stop.is_set():
                return None
        if slot["ev"] is not None:
            slot["ev"].synchronize()  # the consumer's last step that read this slot is done
        return time.perf_counter() - t0

    @staticmethod
    def _release(it):
        """The consumer has submitted its step on this minibatch: hand its slots back."""
        if not it["_slots"]:
            return
        ev = torch.xpu.Event()
        ev.record()
        for s in it["_slots"]:
            s["ev"] = ev
            s["sem"].release()

    def _export(self, j, n, e, slot):
        if self.device.type == "cpu":
            r = self.ms.result(j)
            out = dict(node=torch.from_numpy(r["node"]).long(),
                       edge_index=torch.from_numpy(np.stack([r["row"], r["col"]])).long())
            if self.with_edge_pos:
                out["edge_pos"] = torch.from_numpy(r["edge"])
            return out
        if slot is None:  # fresh tensors; torch may still be using recycled memory
            dev = self.device
            node = torch.empty(n, dtype=torch.int64, device=dev)
            edges = torch.empty(2 * e, dtype=torch.int64, device=dev)
            epos = torch.empty(e, dtype=torch.int64, device=dev) if self.with_edge_pos else None
            torch.xpu.current_stream().synchronize()
        else:
            node, edges, epos = slot["node"], slot["edges"], slot["epos"]
        self.ms.export_to(j, node.data_ptr(), edges.data_ptr(),
                          epos.data_ptr() if epos is not None else 0)
        out = dict(node=node[:n], edge_index=edges[: 2 * e].view(2, e))
        if self.with_edge_pos:
            out["edge_pos"] = epos[:e]
        return out

    def _gather(self, node, n, fslot):
        if self.device.type == "cpu":
            return torch.index_select(self.feat_cpu, 0, node)
        if fslot is None:
            x = torch.empty(n, self.dim, device=self.device)
            torch.xpu.current_stream().synchronize()
            self.ms.gather(node.data_ptr(), n, x.data_ptr())
            return x
        self.ms.gather(node.data_ptr(), n, fslot["x"].data_ptr())
        return fslot["x"][: n * self.dim].view(n, self.dim)

    def _items(self, batches, seed_of, stop):
        K = self.K
        slots = self.gslots is not None and self.overlap
        for b, start in enumerate(range(0, len(batches), K)):
            blk = [np.ascontiguousarray(s, dtype=np.int32) for s in batches[start:start + K]]
            gset = self.gslots[b % 2][:len(blk)] if slots else [None] * len(blk)
            t_wait = 0.0
            if slots:
                for n_taken, s in enumerate(gset):
                    w = self._take(s, stop)
                    if w is None:  # abandoned: give back what this block already holds
                        for held in gset[:n_taken]:
                            held["sem"].release()
                        return
                    t_wait += w
            t0 = time.perf_counter()
            counts = self.ms.run(blk, [seed_of(start + j) for j in range(len(blk))], "batched")
            t1 = time.perf_counter()
            sizes = [(int(sum(nn)), int(sum(ne))) for nn, ne in counts]
            if slots:  # every slot of gset is held here, so it may be (re)sized (_fit_graph)
                if self._target is None:  # first block: its largest minibatch x headroom
                    self._target = (0, 0)
                    self._target = tuple(self._cap(max(sz[i] for sz in sizes), i) for i in (0, 1))
                grew = [self._fit_graph(gset[j], n, e) for j, (n, e) in enumerate(sizes)]
                if any(grew):
                    torch.xpu.current_stream().synchronize()
            graphs = [self._export(j, n, e, gset[j]) for j, (n, e) in enumerate(sizes)]
            t2 = time.perf_counter()
            for j, g in enumerate(graphs):
                nn, ne = counts[j]
                t = dict(sample=t1 - t0 if j == 0 else 0.0, export=t2 - t1 if j == 0 else 0.0,
                         slot_wait=t_wait if j == 0 else 0.0, gather=0.0)
                x, fs = None, None
                if self.dim is not None:
                    if slots:
                        fs = self.fslots[(start + j) % 2]
                        w = self._take(fs, stop)
                        if w is None:  # abandoned: this block's unsent graph slots go back
                            for held in gset[j:]:
                                held["sem"].release()
                            return
                        t["slot_wait"] += w
                        if self._fit_feat(fs, sizes[j][0]):  # held here: see _fit_graph
                            torch.xpu.current_stream().synchronize()
                    tg = time.perf_counter()
                    x = self._gather(g["node"], sizes[j][0], fs)
                    t["gather"] = time.perf_counter() - tg
                yield dict(node=g["node"], edge_index=g["edge_index"], x=x,
                           edge_pos=g.get("edge_pos"), num_sampled_nodes=list(nn),
                           num_sampled_edges=list(ne), t=t,
                           _slots=([gset[j]] + ([fs] if fs is not None else [])) if slots else [])

    def iterate(self, batches, seed_of=lambda i: i):
        """Yields (node, edge_index, x, num_sampled_nodes, num_sampled_edges) in order."""
        for r in self.iterate_full(batches, seed_of):
            yield (r["node"], r["edge_index"], r["x"], r["num_sampled_nodes"],
                   r["num_sampled_edges"])

    def iterate_full(self, batches, seed_of=lambda i: i):
        """Yields dicts (node, edge_index, x, edge_pos, num_sampled_nodes, num_sampled_edges,
        t) in order."""
        import queue
        import threading

        if not len(batches):
            return
        if self.device.type == "xpu":
            torch.xpu.current_stream().synchronize()
        stop = threading.Event()
        th, q, err = None, None, []
        if self.overlap:
            q = queue.Queue()

            def run():
                try:
                    for it in self._items(batches, seed_of, stop):
                        q.put(it)
                except BaseException as ex:  # noqa: BLE001 - re-raised in the consumer
                    err.append(ex)
                q.put(None)

            th = threading.Thread(target=run, daemon=True)
            th.start()
            source = iter(q.get, None)
        else:
            source = self._items(batches, seed_of, stop)
        pending = None
        try:
            for it in source:
                pending = it
                yield it
                pending = None
                self._release(it)
        finally:
            # Exhausted, abandoned (break / close) or failed: every slot must be free again,
            # or the next iteration waits forever on one this iteration still holds.
            if pending is not None:
                self._release(pending)
            stop.set()
            if th is not None:
                th.join()
                while not q.empty():  # prepared but never consumed
                    it = q.get_nowait()
                    if it is not None:
                        self._release(it)
            else:
                source.close()
        if err:
            raise err[0]


# Primary sampling design (D24, user decision): batched. "single" = one minibatch per launch.
DEFAULT_SAMPLER = "batched"
BATCHED_MAX_FANOUT = 16  # MultiSampler: one sub-group of 16 lanes per frontier node


def training_sampler(indptr, indices, fanouts, max_batch, feat=None, usm="host", device="auto",
                     sampler=DEFAULT_SAMPLER, prefetch=True, K=8, with_edge_pos=False,
                     options=DEFAULT_OPTIONS):
    """The training-loop sampler: an object with iterate_full(batches, seed_of) yielding
    dicts (node, edge_index, x, edge_pos, num_sampled_nodes, num_sampled_edges).

    sampler="batched" (default): BatchedGpuSampler, K minibatches per launch.
    sampler="single": one minibatch per launch (PrefetchingGpuSampler, or GpuSampler with
    prefetch=False). Both give bit-identical minibatches for the same seeds and keys.
    Batched supports fanouts up to 16 and the default options only."""
    if sampler == "batched":
        if options != DEFAULT_OPTIONS:
            raise ValueError("options apply to sampler='single'; batched runs the default set")
        if max(fanouts) > BATCHED_MAX_FANOUT:
            raise ValueError(f"batched sampler supports fanouts <= {BATCHED_MAX_FANOUT}; "
                             "use sampler='single'")
        return BatchedGpuSampler(indptr, indices, fanouts, max_batch, K=K, feat=feat, usm=usm,
                                 device=device, overlap=prefetch, with_edge_pos=with_edge_pos)
    if sampler != "single":
        raise ValueError("sampler must be 'batched' or 'single'")
    if prefetch:
        return PrefetchingGpuSampler(indptr, indices, fanouts, max_batch, feat=feat, usm=usm,
                                     device=device, options=options, with_edge_pos=with_edge_pos)
    gs = GpuSampler(indptr, indices, fanouts, max_batch, feat=feat, usm=usm, device=device,
                    options=options)
    gs.with_edge_pos = with_edge_pos
    return gs
