# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-9 minimal nsys driver: exercises the Ulysses AV cross-attention NVTX
ranges on 1x8 so nsys can count the kernels launched inside each range.

Usage::

    srun -n 8 --ntasks-per-node=8 --mpi=pmix ... \\
        nsys profile -o <out>.nsys-rep -t cuda,nvtx -f true \\
            python .../_u8_drivers/ac9_nsys_driver.py [num_iters]

This is NOT a full LTX2-T2V pipeline run — it constructs a reduced-config
``LTXModel`` (1 layer, 8 heads, ``audio_frames=125``) and runs N forward
passes with Ulysses=8 so nsys captures the full Python NVTX ranges
``ltx2.a2v_cross_attn`` / ``ltx2.v2a_cross_attn`` plus the audio/video
self-attention ranges. The kernel-count and byte audit in
``tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_audit.py``
is dimension-independent: 3 AllToAll + 0 AllGather per AV cross-attn
range is the HARD contract regardless of scale.
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


def main(num_iters: int = 3) -> None:
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_SRUN_COMM_HOST", "localhost"))
    os.environ.setdefault("MASTER_PORT", "29920")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    local_rank = int(os.environ.get("SLURM_LOCALID", str(rank % torch.cuda.device_count())))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    from test_ltx2_ulysses_cross_attn_parity import (
        _AV_CONFIG,
        _build_av_modalities,
        _build_ltx2_model,
        _init_weights_deterministic,
        _make_model_config_with_pg,
    )

    dtype = torch.bfloat16
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    cfg = _make_model_config_with_pg(
        ulysses_size=world_size, group=dist.group.WORLD, rank=rank
    )
    torch.manual_seed(9001)
    model = _build_ltx2_model(cfg, dtype=dtype, device=device).eval()
    _init_weights_deterministic(model, seed=7777, zero_biases=False)
    a_frames = 125
    model.configure_audio_ulysses(a_frames)

    torch.manual_seed(9002)
    video_mod, audio_mod = _build_av_modalities(
        1, 2, 4, 4, a_frames, 4, device=device, dtype=dtype
    )

    # Warmup outside the NVTX capture window.
    with torch.no_grad():
        model(video=video_mod, audio=audio_mod)
    torch.cuda.synchronize()
    dist.barrier()

    # Main profiled iterations.
    for i in range(num_iters):
        with torch.no_grad():
            model(video=video_mod, audio=audio_mod)
        torch.cuda.synchronize()

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"AC-9 driver completed {num_iters} iterations at world_size={world_size}.", flush=True)


if __name__ == "__main__":
    try:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    except Exception as exc:
        print(f"[rank {os.environ.get('SLURM_PROCID','?')}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)
