# Reproducibility

## What's seeded

`kkbox.determinism.seed_everything(seed)` seeds Python's `random`, `numpy`, `torch` (CPU and all CUDA
devices if present), `PYTHONHASHSEED`, and calls `torch.use_deterministic_algorithms(True, warn_only=True)`.
DataLoaders that shuffle should be built via `kkbox.data.make_loader(..., seed=...)`, which wires
`worker_init_fn` and a seeded `torch.Generator` so multi-worker shuffling is also reproducible.

Call `seed_everything(seed)` once at the start of any training script/notebook cell, before constructing the
model or any DataLoader.

## Known remaining nondeterminism

- **`warn_only=True`**: some PyTorch ops used in this project (e.g. certain `scatter`/embedding-backward
  paths) have no deterministic implementation on all backends. `warn_only=True` means these emit a warning
  and proceed nondeterministically rather than raising - if you see `UserWarning: ... does not have a
  deterministic implementation`, that op is a source of run-to-run variance even with a fixed seed.
- **BatchNorm on CUDA**: `MultiTaskFMNet`'s backbone uses `nn.BatchNorm1d`. cuDNN's BatchNorm backward pass
  is not guaranteed bit-exact across runs even with deterministic algorithms enabled - this is a known
  PyTorch/cuDNN limitation, not something `seed_everything` can fix. Effect is small (differences appear
  many significant figures deep) but not exactly zero. Not applicable when running on CPU (this project's
  dev machine, an M2 MacBook Air, has no CUDA device - see the GPU-vs-CPU benchmark earlier in this
  project's history, where MPS was actually *slower* than CPU for this model size, so CPU is the default
  execution path here regardless).
- **DuckDB thread scheduling**: label/feature construction in `kkbox.labels` uses DuckDB's default
  multi-threaded query execution. Aggregate results (SUM, MAX, COUNT, GROUP BY) are deterministic regardless
  of thread scheduling, so this does not affect correctness of labels/features - but it does mean wall-clock
  timing varies run to run, and DuckDB's `connect()` doesn't expose a query-level determinism knob because
  none is needed for the aggregate-only queries this project runs.
- **Cross-machine floating point**: exact bit-reproducibility across different CPU architectures (e.g. this
  repo's results were produced on Apple Silicon/ARM64) is not guaranteed even with identical seeds and code -
  only same-machine, same-backend reproducibility is claimed here.

## What "reproducible" means for this repo

Given a fixed seed, same machine, and CPU execution: **identical DataLoader batch order and identical model
initialization**, and losses/metrics that agree to many decimal places across reruns. Bit-for-bit identical
final checkpoints are not guaranteed (see BatchNorm/scatter caveats above) and should not be expected on
mixed CPU/GPU environments.

## Reproducing this repo's results

See `make reproduce` in the `Makefile`, and `make smoke` for a fast (2% subsample) correctness check before
committing to a full run.
