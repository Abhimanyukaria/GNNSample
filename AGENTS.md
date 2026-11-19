# gnnsample — agent instructions

See [`CLAUDE.md`](CLAUDE.md) for commands and the full rule list; this file repeats the
rules for agents that read `AGENTS.md`.

- Read `docs/ARCHITECTURE.md` first and `docs/decisions.md` before changing a design choice.
- Reproduce in the order of `docs/PROCEDURE.md`: build, datasets, tests, benchmarks.
- Correctness bar: every sampler path and option combination is bit-identical to
  `python/gnnsample/reference.py`. Never change the reference to make GPU code pass.
- Every optimisation is a new run-time `SamplerOption` bit; keep `options = 0` working and
  register the bit in `cpp/src/module.cpp`, `bench/ablate.py` and the switching test.
- Back claims with paired, interleaved measurements (`bench/ablate.py`), median ratio on
  products, LiveJournal and Orkut; record each as a `D<n>` entry in `docs/decisions.md`.
- Long jobs go through `tools/run_logged.sh` with a timeout, watched by `tools/watch_runs.sh`.
- One GPU benchmark at a time; AC power for any timing you quote.
- Never hand-edit `results/`, `data/csc/` or `.gpu_cache/`; never commit datasets or results.
