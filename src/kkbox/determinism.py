"""Seeds every source of randomness used in this project.

Known remaining nondeterminism (see project_report.md section 10): a few ops
have no deterministic implementation on all backends (warn_only=True lets
these proceed with a warning rather than raising), BatchNorm backward passes
aren't guaranteed bit-exact across runs, and DuckDB's multi-threaded query
execution means wall-clock timing (not correctness) varies run to run. Only
same-machine, same-backend reproducibility is claimed - not bit-identical
reruns across machines/backends.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed):
    """Seeds python's random, numpy, and torch (CPU + all CUDA devices if present)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # warn_only=True: some ops (e.g. certain scatter/index ops) have no
    # deterministic CUDA implementation. We want the ones that exist to be
    # deterministic and a visible warning for the ones that don't, not a
    # hard crash that would block CPU-only runs on this project's dev machine.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Pass as DataLoader(worker_init_fn=seed_worker) to seed each worker process."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    """Pass as DataLoader(generator=...) for deterministic shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
