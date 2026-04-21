# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-5 / AC-5.2 U=8 driver for srun-native multi-task launch.

Usage::

    srun -n 8 --ntasks-per-node=8 --mpi=pmix ... \\
        python -m tests.unittest._torch.visual_gen.multi_gpu._u8_drivers.ac5_u8_driver <test>

Each SLURM task reads ``SLURM_PROCID`` / ``SLURM_NTASKS`` and becomes a
first-class PMIx process. This bypasses the OMPI/PMIx abort that fires
when ``torch.multiprocessing.spawn(nprocs=8)`` inherits the parent task's
PMIx context (see BitLesson ``BL-20260421-mp-spawn-pmix-at-u8``).
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# TRT-LLM import hits MPI_Init; we disable its MPI to stay under --mpi=pmix.
os.environ.setdefault("TLLM_DISABLE_MPI", "1")

import torch
import torch.distributed as dist

# Repo root needs to precede any site-packages editable install so that
# ``import tensorrt_llm`` picks up the checked-in source under this workspace
# rather than a sibling editable install on the same node. The driver lives
# at ``<repo>/tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/``, so
# the repo root is 6 parents up.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[6]
_MULTI_GPU_DIR = _THIS.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_MULTI_GPU_DIR))


def _init_distributed() -> tuple[int, int]:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_SRUN_COMM_HOST", "localhost"))
    os.environ.setdefault("MASTER_PORT", "29700")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = int(os.environ.get("SLURM_LOCALID", str(rank % torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size


def main(test_name: str) -> None:
    rank, world_size = _init_distributed()
    print(f"[rank {rank}/{world_size}] initialized", flush=True)

    # Lazy import after distributed is up.
    from test_ltx2_ulysses_cross_attn_parity import (
        _logic_ac5_negative_eager_strip,
        _logic_ac5_negative_missing_a2v_mask,
        _logic_audio_attn1_pad_parity_u8,
        _logic_ltx2_pad_mask_parity_u8,
    )

    workers = {
        "ac5_full": _logic_ltx2_pad_mask_parity_u8,
        "ac52_audio_attn1": _logic_audio_attn1_pad_parity_u8,
        "ac5_neg_eager_strip": _logic_ac5_negative_eager_strip,
        "ac5_neg_missing_a2v_mask": _logic_ac5_negative_missing_a2v_mask,
    }
    if test_name not in workers:
        raise ValueError(f"unknown test: {test_name}; options={sorted(workers)}")

    try:
        workers[test_name](rank, world_size)
    except Exception as exc:
        print(f"[rank {rank}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    print(f"[rank {rank}] PASSED: {test_name}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: python {_THIS.name} <test_name>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
