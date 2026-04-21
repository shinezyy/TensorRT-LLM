# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-10 divergent-flags driver (srun -n 2, bypasses mp.spawn PMIx issue).

Usage::

    srun -n 2 --ntasks-per-node=2 --mpi=pmix ... \\
        python -m tests.unittest._torch.visual_gen.multi_gpu._u8_drivers.ac10_driver
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TLLM_DISABLE_MPI", "1")

import torch
import torch.distributed as dist

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[6]
_MULTI_GPU_DIR = _THIS.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_MULTI_GPU_DIR))


def main() -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_SRUN_COMM_HOST", "localhost"))
    os.environ.setdefault("MASTER_PORT", "29702")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = int(os.environ.get("SLURM_LOCALID", str(rank % torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from test_ltx2_ulysses_cross_attn_parity import _logic_ac10_divergent_flags_forward

    try:
        _logic_ac10_divergent_flags_forward(rank, world_size)
    except Exception as exc:
        print(f"[rank {rank}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    print(f"[rank {rank}] PASSED: ac10_divergent_flags", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
