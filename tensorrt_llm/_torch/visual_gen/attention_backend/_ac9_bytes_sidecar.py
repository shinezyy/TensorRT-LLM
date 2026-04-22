# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-9 per-call bytes sidecar instrumentation (env-gated).

When the ``AC9_BYTES_SIDECAR`` env var is set, this module patches
``all_to_all_4d`` / ``all_to_all_5d`` as imported by
:mod:`tensorrt_llm._torch.visual_gen.attention_backend.parallel` so that
each call made from inside ``UlyssesCrossAttention.forward`` records its
payload byte count into a per-rank counter, keyed by the surrounding
``ulysses.cross.a2a.{q,kv,out}`` NVTX label. At process exit the
counters are written to a JSON sidecar named
``<stem>_rank<rank>.json`` where ``<stem>`` is the basename (without
extension) of the ``AC9_BYTES_SIDECAR`` env var and ``<rank>`` is the
per-process rank read from ``RANK`` / ``SLURM_PROCID`` / ``OMPI_COMM_WORLD_RANK``.

When the env var is unset, importing this module is a noop. This
keeps the plan-scale ``trtllm-serve`` path free of instrumentation
overhead unless explicitly asked for, while still providing a single
installation site shared between the reduced ``ac9_nsys_driver.py``
path and the full plan-scale ``serve-nsys-job.sh`` path.

Rationale
---------
AC-9.b requires the measured per-rank per-K/V byte count to stay
within ±10% of the plan's theoretical ``((U-1)/U^2) * B * S_kv * H_kv *
D_h * elem_size``. The only code that previously emitted the required
sidecar was the reduced ``ac9_nsys_driver.py`` (LTX2 model,
``audio_frames=125``). Without this hook, a plan-scale
``trtllm-serve`` nsys capture cannot prove AC-9.b per rank.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import threading
from pathlib import Path
from typing import Dict, List, Optional

import torch


# Module-level state. Guarded by ``_INSTALL_LOCK`` to make
# double-installation (e.g., from a test driver that also imports
# this module) a noop.
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False

# Write the sidecar every ``_WRITE_EVERY_N_CALLS`` a2a invocations so a
# worker killed by SIGTERM / SIGKILL before the atexit handler fires
# still persists its counters. ``trtllm-serve`` ends workers via
# asynchronous shutdown signals and Python does NOT run atexit on
# SIGKILL, so atexit alone is insufficient.
_WRITE_EVERY_N_CALLS = 50
_CALLS_SINCE_LAST_WRITE = 0
_WRITE_LOCK = threading.Lock()

# Per-op accumulated bytes. ``_local`` is ``elem_size * numel`` on the
# calling rank (informational); ``_communicated`` is ``local * (U-1) /
# U`` (the portion the rank actually sends+receives), which maps to
# the plan's theoretical per-rank-per-tensor byte count after dividing
# the fused K|V entry by 2.
_BYTES_COUNTERS: Dict[str, List[int]] = {
    "a2a_q_bytes_local_per_call": [],
    "a2a_kv_bytes_local_per_call": [],
    "a2a_out_bytes_local_per_call": [],
    "a2a_q_bytes_communicated_per_call": [],
    "a2a_kv_bytes_communicated_per_call": [],
    "a2a_out_bytes_communicated_per_call": [],
}


# NVTX label stack, tracked per-thread. ``UlyssesCrossAttention.forward``
# wraps each a2a in a labeled NVTX range (``ulysses.cross.a2a.{q,kv,out}``);
# the patched ``range_push`` / ``range_pop`` maintain a matching Python
# stack so the wrapped ``all_to_all_*`` can dispatch to the right bucket.
_ACTIVE_LABELS = threading.local()


def _current_label() -> str:
    stack = getattr(_ACTIVE_LABELS, "stack", None)
    if not stack:
        return ""
    return stack[-1]


def _resolve_rank() -> int:
    """Resolve the per-process rank from the standard MPI/SLURM env vars.

    Order is ``RANK`` (torchrun / generic), then ``SLURM_PROCID``
    (SLURM), then ``OMPI_COMM_WORLD_RANK`` (Open MPI). Falls back to 0
    if none are set, which is the single-rank smoke-test case.
    """
    for var in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.environ.get(var)
        if val is not None and val != "":
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def _resolve_ulysses_size() -> int:
    """Resolve ``U`` (ulysses_size) from env at write time.

    The ``(U-1)/U`` factor is not known at call time in the plan-scale
    path (``trtllm-serve`` initializes process groups inside the
    worker, after this module is imported). We read ``AC9_ULYSSES_SIZE``
    first (explicit override), then ``WORLD_SIZE`` / ``SLURM_NTASKS``
    (implicit). If none resolve to a positive integer, ``_communicated``
    columns are omitted from the sidecar so the audit can fall back to
    computing them itself from ``_local`` and a passed-in ``U``.
    """
    for var in ("AC9_ULYSSES_SIZE", "WORLD_SIZE", "SLURM_NTASKS"):
        val = os.environ.get(var)
        if val is not None and val != "":
            try:
                U = int(val)
                if U > 0:
                    return U
            except ValueError:
                pass
    return 0


def _record(kind: str, t: torch.Tensor, ulysses_size: int) -> None:
    local = t.element_size() * t.numel()
    _BYTES_COUNTERS[f"a2a_{kind}_bytes_local_per_call"].append(local)
    if ulysses_size > 0:
        communicated = (local * (ulysses_size - 1)) // ulysses_size
        _BYTES_COUNTERS[f"a2a_{kind}_bytes_communicated_per_call"].append(communicated)


def _sidecar_path_for_rank(sidecar_env: str, rank: int) -> Path:
    """Derive per-rank sidecar path from ``AC9_BYTES_SIDECAR``.

    Example: ``AC9_BYTES_SIDECAR=/run/profile/bytes.json`` with rank 3
    yields ``/run/profile/bytes_rank3.json``.
    """
    base = Path(sidecar_env)
    return base.with_name(f"{base.stem}_rank{rank}.json")


def _write_sidecar(sidecar_env: str) -> None:
    rank = _resolve_rank()
    U = _resolve_ulysses_size()
    path = _sidecar_path_for_rank(sidecar_env, rank)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Best-effort: if the parent directory can't be created we let
        # the write fail loudly below.
        pass

    counters = {k: list(v) for k, v in _BYTES_COUNTERS.items()}
    payload = {
        f"{k}_avg": (sum(v) / len(v)) if v else 0
        for k, v in counters.items()
    }
    payload["num_calls_recorded_per_op"] = {k: len(v) for k, v in counters.items()}
    payload["ulysses_size"] = U
    payload["rank"] = rank
    payload["a2a_kv_bytes_communicated_per_call_samples"] = (
        counters["a2a_kv_bytes_communicated_per_call"]
    )
    # The reduced driver wrote ``a2a_kv_bytes_per_call_avg`` as the
    # fused K|V figure. Aliased here so the existing
    # ``ac9_audit.py`` sidecar reader (``_key_for`` with the
    # ``a2a_kv_bytes_per_call_avg`` fallback) keeps working.
    payload["a2a_kv_bytes_per_call_avg"] = (
        sum(counters["a2a_kv_bytes_communicated_per_call"])
        / max(1, len(counters["a2a_kv_bytes_communicated_per_call"]))
        if counters["a2a_kv_bytes_communicated_per_call"]
        else 0
    )
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        # A failing sidecar write is a soft error: we don't want the
        # process exit path to abort just because the audit couldn't
        # persist its counters. Log so it shows up in server.log.
        print(
            f"[ac9_bytes_sidecar] WARNING: failed to write {path}: {exc}",
            flush=True,
        )


def install_if_enabled() -> bool:
    """Install the counter patch iff ``AC9_BYTES_SIDECAR`` is set.

    Returns ``True`` if the patch was installed this call, ``False``
    otherwise (env unset, already installed, or install failed).
    """
    global _INSTALLED

    sidecar_env = os.environ.get("AC9_BYTES_SIDECAR")
    if not sidecar_env:
        return False

    with _INSTALL_LOCK:
        if _INSTALLED:
            return False

        # Patch the ``all_to_all_4d`` / ``all_to_all_5d`` bindings in
        # ``parallel`` (the UlyssesCrossAttention module imports them
        # at module level, so this is the one site that matters for
        # dispatch). We do not patch
        # ``tensorrt_llm._torch.distributed`` globally -- other call
        # sites (e.g., UlyssesAttention's fused QKV path) don't carry
        # the ``ulysses.cross.a2a.{q,kv,out}`` NVTX labels and would
        # only add noise.
        from . import parallel as _par

        _orig_4d = _par.all_to_all_4d
        _orig_5d = _par.all_to_all_5d

        # Grab the live NVTX hooks so a later monkey-patch (e.g., by a
        # debugger) stacks on top of ours rather than underneath.
        _orig_push = torch.cuda.nvtx.range_push
        _orig_pop = torch.cuda.nvtx.range_pop

        def _push(label):
            if not hasattr(_ACTIVE_LABELS, "stack"):
                _ACTIVE_LABELS.stack = []
            _ACTIVE_LABELS.stack.append(label)
            return _orig_push(label)

        def _pop():
            if getattr(_ACTIVE_LABELS, "stack", None):
                _ACTIVE_LABELS.stack.pop()
            return _orig_pop()

        torch.cuda.nvtx.range_push = _push
        torch.cuda.nvtx.range_pop = _pop

        # U is resolved at write time (sidecar finalization), but we
        # also compute ``_communicated`` per-call if U is already
        # knowable at call time (rare in the plan-scale path, common
        # in the reduced driver path). Falling back to per-write
        # computation lets the audit still verify the bound even when
        # ``WORLD_SIZE`` is set only after the module imports.
        #
        # ``torch._dynamo.disable(recursive=True)`` prevents torch.compile
        # from tracing the wrapper. Without it, the serve path (which
        # compiles every transformer block) traces the a2a call, hits
        # the list-append side effect on ``_BYTES_COUNTERS``, and
        # graph-breaks into a recompile on every call site -- the
        # `` hit config.recompile_limit (128)`` error observed in the
        # first Round 12 nsys attempt.
        def _wrap_4d_eager(t, *a, **kw):
            label = _current_label()
            if label == "ulysses.cross.a2a.q":
                _record("q", t, _resolve_ulysses_size())
                _maybe_write_sidecar(sidecar_env)
            elif label == "ulysses.cross.a2a.out":
                _record("out", t, _resolve_ulysses_size())
                _maybe_write_sidecar(sidecar_env)
            return _orig_4d(t, *a, **kw)

        def _wrap_5d_eager(t, *a, **kw):
            label = _current_label()
            if label == "ulysses.cross.a2a.kv":
                _record("kv", t, _resolve_ulysses_size())
                _maybe_write_sidecar(sidecar_env)
            return _orig_5d(t, *a, **kw)

        try:
            _wrap_4d = torch._dynamo.disable(_wrap_4d_eager, recursive=True)
            _wrap_5d = torch._dynamo.disable(_wrap_5d_eager, recursive=True)
        except Exception:
            # torch._dynamo unavailable (e.g., old torch) -- fall back.
            _wrap_4d = _wrap_4d_eager
            _wrap_5d = _wrap_5d_eager

        _par.all_to_all_4d = _wrap_4d
        _par.all_to_all_5d = _wrap_5d

        # Atexit covers the clean-shutdown case. Signal handlers cover
        # the common trtllm-serve worker-lifecycle case where the
        # master sends SIGTERM for a graceful shutdown. SIGKILL still
        # loses data, which is why ``_maybe_write_sidecar`` also
        # flushes every ``_WRITE_EVERY_N_CALLS`` calls.
        atexit.register(_write_sidecar, sidecar_env)
        _install_signal_handlers(sidecar_env)

        _INSTALLED = True
        return True


def _maybe_write_sidecar(sidecar_env: str) -> None:
    """Periodic checkpoint write. Called from the wrapper on every a2a.

    Writes the sidecar once every ``_WRITE_EVERY_N_CALLS`` recorded
    calls, giving us a complete snapshot even if the worker is killed
    via SIGKILL before ``atexit`` can fire.
    """
    global _CALLS_SINCE_LAST_WRITE
    with _WRITE_LOCK:
        _CALLS_SINCE_LAST_WRITE += 1
        if _CALLS_SINCE_LAST_WRITE >= _WRITE_EVERY_N_CALLS:
            _CALLS_SINCE_LAST_WRITE = 0
            _write_sidecar(sidecar_env)


def _install_signal_handlers(sidecar_env: str) -> None:
    """Write the sidecar on SIGTERM / SIGINT then re-raise.

    Python's default atexit does NOT run on signal-delivery-triggered
    exits. ``trtllm-serve`` sends SIGTERM to workers for graceful
    shutdown, so without this handler workers lose their recorded
    counters. The handler chains to the previous handler after
    writing, so existing shutdown logic continues to run.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
        except (OSError, ValueError):
            continue

        def _handler(signum, frame, _prev=prev, _env=sidecar_env, _sig=sig):
            try:
                _write_sidecar(_env)
            except Exception:
                pass
            # Chain to the prior handler so the normal shutdown path
            # runs. SIG_DFL / SIG_IGN get special-cased.
            if _prev is signal.SIG_DFL:
                signal.signal(_sig, signal.SIG_DFL)
                os.kill(os.getpid(), _sig)
            elif _prev is signal.SIG_IGN:
                return
            elif callable(_prev):
                _prev(signum, frame)

        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            # Non-main thread or signal not installable -- skip.
            pass


def reset_for_tests() -> None:
    """Clear accumulated counters (test helper only).

    This does NOT uninstall the monkey-patches, because atexit is
    already registered and undoing that cleanly is more hassle than
    it's worth for a test. Tests that want a clean slate should
    call this at setUp.
    """
    for key in _BYTES_COUNTERS:
        _BYTES_COUNTERS[key].clear()


# Installed automatically at import time. The gate inside
# ``install_if_enabled`` ensures zero effect when the env var is
# unset -- this is intentionally a cheap noop in production runs.
install_if_enabled()
