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
self-attention ranges.

When ``AC9_BYTES_SIDECAR`` is set, the driver also monkey-patches
``all_to_all_4d`` / ``all_to_all_5d`` to record per-call payload byte
counts and writes them to ``AC9_BYTES_SIDECAR``-per-rank JSON at exit,
so ``ac9_audit.py --measured-bytes-sidecar`` can verify the bytes are
within ±10%% of the plan's theoretical value.
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


_BYTES_COUNTERS = {
    # ``_local`` holds raw ``elem_size * numel`` (the full local tensor
    # size on the calling rank). Informational only.
    "a2a_q_bytes_local_per_call": [],
    "a2a_kv_bytes_local_per_call": [],
    "a2a_out_bytes_local_per_call": [],
    # ``_communicated`` holds the portion of the local tensor that is
    # actually sent (and received) by the rank in one all-to-all op:
    # ``elem_size * numel * (U - 1) / U``. This is the quantity that
    # matches the plan's theoretical per-rank-per-tensor byte count of
    # ``((U - 1) / U^2) * B * S_kv * H_kv * D_h * elem_size`` after
    # dividing the fused K|V figure by 2 (K + V stacked on dim=2 of the
    # 5D input tensor).
    "a2a_q_bytes_communicated_per_call": [],
    "a2a_kv_bytes_communicated_per_call": [],
    "a2a_out_bytes_communicated_per_call": [],
}


def _install_bytes_counters(ulysses_size: int):
    """Patch ``all_to_all_4d`` / ``all_to_all_5d`` to record payload bytes.

    ``UlyssesCrossAttention.forward`` calls them with explicit tensor
    shapes known at call time; we compute both the raw local tensor size
    (``elem_size * numel``, informational) and the communicated portion
    (``elem_size * numel * (U - 1) / U``, comparable to the plan's
    theoretical value) before the collective and accumulate per-op.
    Three NVTX ranges in ``UlyssesCrossAttention.forward``
    (``ulysses.cross.a2a.{q,kv,out}``) tell us which op this call is
    for via a small tracker.
    """
    from tensorrt_llm._torch import distributed as _dist_mod

    _orig_4d = _dist_mod.all_to_all_4d
    _orig_5d = _dist_mod.all_to_all_5d

    # Use the stack of NVTX ranges active on this thread via a small
    # domain tracker so we can dispatch counters to the right bucket.
    import threading
    _active = threading.local()
    _active.stack = []

    _ORIG_NVTX_PUSH = torch.cuda.nvtx.range_push

    def _push(label):
        try:
            if not hasattr(_active, "stack"):
                _active.stack = []
            _active.stack.append(label)
        finally:
            return _ORIG_NVTX_PUSH(label)

    _ORIG_NVTX_POP = torch.cuda.nvtx.range_pop

    def _pop():
        try:
            if hasattr(_active, "stack") and _active.stack:
                _active.stack.pop()
        finally:
            return _ORIG_NVTX_POP()

    torch.cuda.nvtx.range_push = _push
    torch.cuda.nvtx.range_pop = _pop

    def _record(kind: str, t):
        local = t.element_size() * t.numel()
        communicated = (local * (ulysses_size - 1)) // ulysses_size
        _BYTES_COUNTERS[f"a2a_{kind}_bytes_local_per_call"].append(local)
        _BYTES_COUNTERS[f"a2a_{kind}_bytes_communicated_per_call"].append(communicated)

    def _wrap_4d(t, *a, **kw):
        label = _active.stack[-1] if getattr(_active, "stack", None) else ""
        if label == "ulysses.cross.a2a.q":
            _record("q", t)
        elif label == "ulysses.cross.a2a.out":
            _record("out", t)
        return _orig_4d(t, *a, **kw)

    def _wrap_5d(t, *a, **kw):
        label = _active.stack[-1] if getattr(_active, "stack", None) else ""
        if label == "ulysses.cross.a2a.kv":
            _record("kv", t)
        return _orig_5d(t, *a, **kw)

    _dist_mod.all_to_all_4d = _wrap_4d
    _dist_mod.all_to_all_5d = _wrap_5d

    # Also patch the attribute where ``parallel.py`` imported it:
    import tensorrt_llm._torch.visual_gen.attention_backend.parallel as _par
    _par.all_to_all_4d = _wrap_4d
    _par.all_to_all_5d = _wrap_5d


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

    if os.environ.get("AC9_BYTES_SIDECAR"):
        _install_bytes_counters(ulysses_size=world_size)

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

    sidecar = os.environ.get("AC9_BYTES_SIDECAR")
    if sidecar:
        import json as _json
        sidecar_path = Path(sidecar).parent / f"{Path(sidecar).stem}_rank{rank}.json"
        out = {
            f"{k}_avg": (sum(v) / len(v)) if v else 0
            for k, v in _BYTES_COUNTERS.items()
        }
        out["num_iters_measured"] = num_iters
        out["ulysses_size"] = world_size
        out["per_call_counts"] = {k: len(v) for k, v in _BYTES_COUNTERS.items()}
        out["a2a_kv_bytes_communicated_per_call_samples"] = \
            _BYTES_COUNTERS["a2a_kv_bytes_communicated_per_call"]
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(_json.dumps(out, indent=2))

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
