"""Download a graph and convert it to the binary CSC layout every sampler reads.

Output, in data/csc/<name>/:
  indptr.int64.bin   n+1 offsets into indices
  indices.int32.bin  in-neighbors of node v are indices[indptr[v]:indptr[v+1]]
  meta.json          node/edge counts, source, checksum of the download
  feat.npy, label.npy, {train,valid,test}_idx.npy   (OGB datasets only)

All samplers (ours, PyG, DGL) are fed this exact CSC, so they sample the same graph.
Conventions:
  - Node IDs are compacted to 0..n-1 (SNAP IDs have gaps).
  - Undirected graphs are symmetrised (both directions stored).
  - Directed graphs keep in-edges (u -> v stored under v).
  - Self loops and duplicate edges are removed.

Memory: the Docker VM has ~8 GB, so edges are held as int32 and the (dst, src) keys are
sorted in place. Peak is about 16 bytes per stored (directed) edge.

Usage: python data/prepare.py <name> [--raw-dir data/raw] [--out-dir data/csc]
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SNAP = "https://snap.stanford.edu/data"
DATASETS = {
    "livejournal": dict(url=f"{SNAP}/soc-LiveJournal1.txt.gz", directed=True),
    "orkut": dict(url=f"{SNAP}/bigdata/communities/com-orkut.ungraph.txt.gz", directed=False),
    "friendster": dict(url=f"{SNAP}/bigdata/communities/com-friendster.ungraph.txt.gz",
                       directed=False),
    "products": dict(url="http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
                     ogb_dir="products", directed=False),
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"using cached {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, t0, last = 0, time.time(), 0.0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if total and time.time() - last > 10:
                last = time.time()
                print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB "
                      f"({done / 1e6 / max(last - t0, 1e-9):.1f} MB/s)", flush=True)
    tmp.rename(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def read_edges(path: Path, sep: str, comment: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Stream an edge list into int32 (src, dst) arrays."""
    src, dst = [], []
    reader = pd.read_csv(path, sep=sep, comment=comment, header=None, names=["u", "v"],
                         dtype=np.int64, engine="c", chunksize=20_000_000)
    for chunk in reader:
        u, v = chunk["u"].to_numpy(), chunk["v"].to_numpy()
        assert u.max() < 2**31 and v.max() < 2**31
        src.append(u.astype(np.int32))
        dst.append(v.astype(np.int32))
        print(f"  parsed {sum(map(len, src)):,} edges", flush=True)
    return np.concatenate(src), np.concatenate(dst)


def compact_ids(src: np.ndarray, dst: np.ndarray) -> int:
    """Relabel node IDs to 0..n-1 in place (dense table over the ID range)."""
    present = np.zeros(int(max(src.max(), dst.max())) + 1, dtype=bool)
    present[src] = True
    present[dst] = True
    remap = (np.cumsum(present, dtype=np.int64) - 1).astype(np.int32)
    src[:] = remap[src]
    dst[:] = remap[dst]
    return int(present.sum())


def build_csc(src: np.ndarray, dst: np.ndarray, n: int, directed: bool):
    keep = src != dst
    if directed:
        key = dst[keep].astype(np.int64) * n + src[keep]
    else:
        m = int(keep.sum())
        key = np.empty(2 * m, dtype=np.int64)
        key[:m] = dst[keep].astype(np.int64) * n + src[keep]
        key[m:] = src[keep].astype(np.int64) * n + dst[keep]
    del keep
    key.sort()
    if len(key):
        key = key[np.concatenate(([True], key[1:] != key[:-1]))]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(key // n, minlength=n), out=indptr[1:])
    indices = (key % n).astype(np.int32)
    return indptr, indices


def read_ogb(spec: dict, raw_dir: Path, out: Path):
    root = raw_dir / spec["ogb_dir"]
    if not root.exists() and (raw_dir / f"ogbn_{spec['ogb_dir']}").exists():
        root = raw_dir / f"ogbn_{spec['ogb_dir']}"  # layout left by the ogb package
    if not root.exists():
        zpath = raw_dir / spec["url"].rsplit("/", 1)[1]
        download(spec["url"], zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw_dir)
    n = int(pd.read_csv(root / "raw/num-node-list.csv.gz", header=None).iloc[0, 0])
    print("  node features", flush=True)
    feat = pd.read_csv(root / "raw/node-feat.csv.gz", header=None, dtype=np.float32,
                       engine="c").to_numpy()
    # pandas hands back a column-major (Fortran) array; a row gather on that touches one
    # cache line per feature and was 15x slower (docs/decisions.md D9 measurement 3).
    np.save(out / "feat.npy", np.ascontiguousarray(feat))
    del feat
    label = pd.read_csv(root / "raw/node-label.csv.gz", header=None).to_numpy().reshape(-1)
    np.save(out / "label.npy", label.astype(np.int64))
    split = next((root / "split").iterdir())
    for part in ("train", "valid", "test"):
        idx = pd.read_csv(split / f"{part}.csv.gz", header=None).to_numpy().reshape(-1)
        np.save(out / f"{part}_idx.npy", idx.astype(np.int64))
    src, dst = read_edges(root / "raw/edge.csv.gz", sep=",", comment=None)
    return src, dst, n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=sorted(DATASETS))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/csc"))
    args = p.parse_args()

    spec = DATASETS[args.name]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if "ogb_dir" in spec:
        src, dst, n = read_ogb(spec, args.raw_dir, out)
        checksum = None
    else:
        raw = args.raw_dir / spec["url"].rsplit("/", 1)[1]
        download(spec["url"], raw)
        checksum = sha256(raw)
        src, dst = read_edges(raw, sep=r"\s+", comment="#")
        n = compact_ids(src, dst)

    indptr, indices = build_csc(src, dst, n, spec["directed"])
    del src, dst
    indptr.tofile(out / "indptr.int64.bin")
    indices.tofile(out / "indices.int32.bin")
    deg = np.diff(indptr)
    meta = dict(name=args.name, num_nodes=n, num_edges=int(len(indices)),
                directed=spec["directed"], source=spec["url"], sha256=checksum,
                max_in_degree=int(deg.max()), mean_in_degree=float(deg.mean()),
                zero_in_degree=int((deg == 0).sum()))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
