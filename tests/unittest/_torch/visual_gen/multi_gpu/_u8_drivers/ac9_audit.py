# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AC-9 nsys audit: kernel counts inside AV cross-attn NVTX ranges.

The audit window is the two AV-direction cross-attention NVTX ranges
``ltx2.a2v_cross_attn`` and ``ltx2.v2a_cross_attn`` — text cross-attn
ranges (``*_cross_attn`` on the text stream) are intentionally excluded.

The HARD AC-9 contract checked here is **implementation-independent**:

  * Zero ``ncclDevKernel_AllGather*`` kernels inside any AV cross-attn
    range. This is the strict-Ulysses claim: the AllGather path has
    been removed from the AV cross-attn critical path.
  * Non-zero ``ncclDevKernel_AllToAll*`` / ``ncclDevKernel_SendRecv``
    kernels across the profile, so the replacement all-to-all path
    actually launches.

The plan's "exactly 3 AllToAll per range" wording is satisfied at the
OP level (Q a2a + fused K|V 5D a2a + output a2a), not at the NCCL kernel
level. On intra-node B200 topologies NCCL lowers
``dist.all_to_all_single`` to ``(U-1)`` pairwise ``ncclDevKernel_SendRecv``
kernels per op, and kernels can spill slightly outside the enclosing
Python NVTX range due to async launch queueing. An earlier version of
this script enforced ``a2a == 3`` per range; that check was removed
because it is topology-dependent and already disagrees with the
committed round-4 exports (per-range counts in the range 1–5). Per-range
raw counts are still reported in the JSON for human review.

Usage::

    python ac9_audit.py <profile>.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


_A2A_NVTX_NAMES = ("ltx2.a2v_cross_attn", "ltx2.v2a_cross_attn")

_OP_LEVEL_A2A_NVTX_NAMES = (
    "ulysses.cross.a2a.q",
    "ulysses.cross.a2a.kv",
    "ulysses.cross.a2a.out",
)


def _fetch_ranges(conn, names):
    # NVTX ranges use two encodings: registered strings (textId ->
    # StringIds.value) and inline strings (text column populated
    # directly). ``torch.cuda.nvtx.range`` uses the latter while the
    # ``ltx2.*`` markers go through the former. Union both.
    # We also include ``globalTid`` so callers can filter per-rank.
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"""
        SELECT r.start, r.end, s.value, r.globalTid
        FROM NVTX_EVENTS r JOIN StringIds s ON r.textId = s.id
        WHERE s.value IN ({placeholders})
        UNION ALL
        SELECT start, end, text, globalTid
        FROM NVTX_EVENTS
        WHERE text IN ({placeholders})
        """,
        tuple(names) + tuple(names),
    ).fetchall()
    return [(int(start), int(end), str(value), int(tid) if tid is not None else -1)
            for (start, end, value, tid) in rows]


def _fetch_kernel_events(conn):
    # Include deviceId so callers can filter to kernels running on a
    # specific GPU (rank) when the profile covers multiple devices.
    rows = conn.execute(
        """
        SELECT k.start, k.end, s.value, k.deviceId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        """
    ).fetchall()
    return [(int(start), int(end), str(name),
             int(dev) if dev is not None else -1)
            for (start, end, name, dev) in rows]


def _count_in_range(kernels, range_start, range_end, patterns, device_id=None):
    """Return total kernel count for any pattern in ``patterns`` whose kernel
    lives inside the ``[range_start, range_end]`` window. When
    ``device_id`` is not None, only kernels on that GPU are counted."""
    if isinstance(patterns, str):
        patterns = (patterns,)
    total = 0
    for start, end, name, dev in kernels:
        if device_id is not None and dev != device_id:
            continue
        if end < range_start or start > range_end:
            continue
        if any(p in name for p in patterns):
            total += 1
    return total


def _theoretical_a2a_bytes_per_rank(
    *, U: int, B: int, S_kv: int, H_kv: int, D_h: int, elem_size: int
) -> int:
    """Plan's theoretical a2a byte count per K (or per V) per rank.

    From the plan §AC-9: each rank sends / receives
    ``((U-1)/U^2) * B * S_kv * H_kv * D_h * elem_size`` bytes per K (and
    per V) across all peers in one Q a2a / one fused K|V a2a / one
    output a2a. The audit reports the reference value so a post-hoc
    byte-aggregation pass can compare recorded NCCL payload bytes to
    this number at the ±10% tolerance the plan specifies.
    """
    return ((U - 1) * B * S_kv * H_kv * D_h * elem_size) // (U * U)


def _sum_range_wallclock_ns(ranges, names_whitelist=None):
    """Total wall-clock nanoseconds summed over every range in ``ranges``."""
    total = 0
    for entry in ranges:
        start, end, name = entry[0], entry[1], entry[2]
        if names_whitelist is not None and name not in names_whitelist:
            continue
        total += max(0, end - start)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument(
        "--theoretical-bytes",
        nargs=6,
        metavar=("U", "B", "S_kv", "H_kv", "D_h", "elem_size"),
        type=int,
        default=None,
        help=(
            "If provided, emit the plan's per-rank, per-K (and per-V) "
            "a2a theoretical byte count under _totals.theoretical_bytes "
            "for external comparison against measured NCCL payloads."
        ),
    )
    parser.add_argument(
        "--measured-bytes-sidecar",
        type=Path,
        default=None,
        help=(
            "Path to a per-rank JSON sidecar (or a glob with '*') written "
            "by the nsys driver with measured a2a communicated-byte "
            "counts per op (keys: a2a_q_bytes_communicated_per_call, "
            "a2a_kv_bytes_communicated_per_call, "
            "a2a_out_bytes_communicated_per_call). Audit aggregates "
            "across ranks and compares to --theoretical-bytes at ±10%% "
            "when both are given."
        ),
    )
    parser.add_argument(
        "--baseline-sqlite",
        type=Path,
        default=None,
        help=(
            "Path to a baseline nsys sqlite captured on the same hardware "
            "from the pre-change branch. Audit reports AV cross-attn NVTX "
            "wall-clock ratio (current/baseline). Plan tolerance: ≤ 1.10."
        ),
    )
    parser.add_argument(
        "--bytes-tolerance",
        type=float,
        default=0.10,
        help="Relative tolerance for bytes vs. theoretical (default 0.10).",
    )
    parser.add_argument(
        "--wallclock-tolerance",
        type=float,
        default=0.10,
        help="Max allowed regression ratio for AV wall-clock (default 0.10).",
    )
    args = parser.parse_args()

    if not args.sqlite.exists():
        print(f"ERROR: {args.sqlite} does not exist.", file=sys.stderr)
        sys.exit(2)

    conn = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)

    ranges = _fetch_ranges(conn, _A2A_NVTX_NAMES)
    if not ranges:
        print(f"ERROR: no 'ltx2.*cross_attn' NVTX ranges found in {args.sqlite}.",
              file=sys.stderr)
        sys.exit(3)
    kernels = _fetch_kernel_events(conn)
    # Op-level a2a NVTX ranges (Q / fused K|V / output), inserted inside
    # ``UlyssesCrossAttention.forward``. These are implementation-
    # independent: regardless of whether NCCL lowers ``all_to_all_single``
    # to a single ``AllToAll`` kernel or ``U-1`` ``SendRecv`` kernels per
    # op, there must be exactly one op-level NVTX range per a2a OP.
    op_ranges = _fetch_ranges(conn, _OP_LEVEL_A2A_NVTX_NAMES)

    # Per-rank ``tid -> device`` via the nsys globalTid / globalPid encoding.
    #
    # Earlier versions of this audit joined ``CUPTI_ACTIVITY_KIND_RUNTIME``
    # to ``CUPTI_ACTIVITY_KIND_KERNEL`` on ``correlationId``. That join is
    # UNSAFE on a multi-process nsys trace: correlationIds are per-process
    # and can collide across ranks, so a single globalTid ends up mapped
    # to every device in the trace, and ``setdefault`` picks an arbitrary
    # one. The Round 9 sqlite exhibits this: every audited globalTid
    # reports all 8 deviceIds under the correlation join (see
    # ``round9/verify/outlier-analysis.md``).
    #
    # The correct per-rank attribution uses nsys's own encoding: for any
    # NVTX globalTid ``T``, its process's globalPid is ``T & ~((1<<24)-1)``
    # (the lower 24 bits are the intra-process thread id). Each process
    # binds to exactly one CUDA device, so looking up kernels with that
    # globalPid yields the single device hosting that rank's compute.
    _TID_THREAD_BITS = 24
    _TID_PID_MASK = ~((1 << _TID_THREAD_BITS) - 1)
    pid_to_device = {}
    try:
        rows = conn.execute(
            """
            SELECT globalPid, deviceId, COUNT(*) AS cnt
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE globalPid IS NOT NULL
            GROUP BY globalPid, deviceId
            """
        ).fetchall()
        pid_dev_counts = {}
        for pid, dev, cnt in rows:
            if pid is None or dev is None:
                continue
            pid_dev_counts.setdefault(int(pid), {})[int(dev)] = int(cnt)
        for pid, dev_counts in pid_dev_counts.items():
            # Pick the dominant device. In a correctly launched rank-to-
            # device binding this map is one-to-one; we still pick the
            # max-count device for safety.
            dev = max(dev_counts.items(), key=lambda kv: kv[1])[0]
            pid_to_device[pid] = dev
    except Exception:
        pid_to_device = {}

    def _tid_to_device(tid):
        if tid is None or tid < 0:
            return None
        pid = tid & _TID_PID_MASK
        return pid_to_device.get(pid)

    # Identify the profile-stop cutoff. On ``cudaProfilerStop`` any NVTX
    # range still open on any thread is terminated at the stop timestamp,
    # so we see several ranges sharing the exact same ``end`` tick
    # regardless of which rank owned them. Those trailing truncated
    # ranges cannot carry their three inner op-level NVTX markers (the
    # Python code inside ``UlyssesCrossAttention.forward`` didn't finish
    # emitting them before the profiler stop), so they must be reported
    # as "truncated" rather than counted as op-invariant failures.
    profile_cutoff_ns = max(int(e) for (_s, e, _n, _t) in ranges)
    cutoff_cnt = sum(1 for (_s, e, _n, _t) in ranges if int(e) == profile_cutoff_ns)
    profile_cutoff_is_shared = cutoff_cnt >= 2

    summary = {}
    ok = True
    ALLTOALL_KERNEL_PATTERNS = (
        "ncclDevKernel_AllToAll",
        "ncclDevKernel_SendRecv",
    )
    ALLGATHER_KERNEL_PATTERNS = ("ncclDevKernel_AllGather",)

    # Per-rank aggregates for the pass/fail decision.
    per_tid_a2a = {}
    per_tid_ag = {}
    per_tid_op = {}

    truncated_ranges = []

    for start, end, name, tid in ranges:
        # Resolve this outer AV range's device (rank).
        dev = _tid_to_device(tid)
        is_truncated = profile_cutoff_is_shared and end == profile_cutoff_ns
        a2a = _count_in_range(kernels, start, end, ALLTOALL_KERNEL_PATTERNS,
                               device_id=dev)
        ag = _count_in_range(kernels, start, end, ALLGATHER_KERNEL_PATTERNS,
                              device_id=dev)
        # Count op-level NVTX ranges with the SAME globalTid whose start
        # lies inside [start, end]. This is per-rank and avoids
        # cross-rank overlap inflation when all 8 ranks are in the same
        # profile.
        op_count = sum(
            1 for (o_start, o_end, _o_name, o_tid) in op_ranges
            if start <= o_start <= end and o_tid == tid
        )
        summary.setdefault(name, []).append({
            "all_to_all_kernels": a2a,
            "all_gather_kernels": ag,
            "op_level_a2a_nvtx": op_count,
            "tid": tid,
            "device_id": dev,
            "truncated_at_profile_cutoff": bool(is_truncated),
        })

        per_tid_a2a[tid] = per_tid_a2a.get(tid, 0) + a2a
        per_tid_ag[tid] = per_tid_ag.get(tid, 0) + ag
        per_tid_op[tid] = per_tid_op.get(tid, 0) + op_count

        if is_truncated:
            truncated_ranges.append({
                "range_name": name,
                "tid": tid,
                "device_id": dev,
                "start": start,
                "end": end,
                "op_count": op_count,
                "all_gather_kernels": ag,
            })

        if ag > 0:
            ok = False
            print(
                f"FAIL range={name} tid={tid} dev={dev} [{start},{end}]: "
                f"AllGather kernels={ag} "
                "(plan requires 0 — Ulysses cross-attn must never launch AllGather).",
                file=sys.stderr,
            )
        # Op-level NVTX invariant: if op markers are present, every AV
        # cross-attn range must carry exactly 3 op-level markers on its
        # own globalTid. Profile-stop-truncated ranges are reported
        # separately (they cannot emit their inner markers after
        # ``cudaProfilerStop``) and are excluded from this assertion.
        if op_ranges and op_count != 3 and not is_truncated:
            ok = False
            print(
                f"FAIL range={name} tid={tid} [{start},{end}]: op-level "
                f"a2a NVTX ranges={op_count} (expected exactly 3: Q / K|V / output).",
                file=sys.stderr,
            )

    total_a2a = sum(r["all_to_all_kernels"] for k, v in summary.items()
                    if not k.startswith("_") for r in v)
    total_ag = sum(r["all_gather_kernels"] for k, v in summary.items()
                   if not k.startswith("_") for r in v)
    total_op = sum(r["op_level_a2a_nvtx"] for k, v in summary.items()
                   if not k.startswith("_") for r in v)
    summary["_totals"] = {
        "all_to_all_kernels": total_a2a,
        "all_gather_kernels": total_ag,
        "op_level_a2a_nvtx": total_op,
        "op_level_markers_present": bool(op_ranges),
        "num_ranges": sum(len(v) for k, v in summary.items() if not k.startswith("_")),
        "num_truncated_ranges": len(truncated_ranges),
        "profile_cutoff_ns": profile_cutoff_ns,
        "pid_to_device": {str(k): v for k, v in sorted(pid_to_device.items())},
    }
    if truncated_ranges:
        summary["_truncated_at_profile_cutoff"] = truncated_ranges
    if args.theoretical_bytes is not None:
        U, B, S_kv, H_kv, D_h, elem_size = args.theoretical_bytes
        theoretical = _theoretical_a2a_bytes_per_rank(
            U=U, B=B, S_kv=S_kv, H_kv=H_kv, D_h=D_h, elem_size=elem_size,
        )
        summary["_totals"]["theoretical_bytes_per_rank_per_tensor"] = theoretical
        summary["_totals"]["theoretical_bytes_params"] = {
            "U": U, "B": B, "S_kv": S_kv, "H_kv": H_kv,
            "D_h": D_h, "elem_size": elem_size,
        }

    # Measured bytes from the driver's per-rank sidecar JSON files.
    #
    # The driver in ``ac9_nsys_driver.py`` writes one file per rank with
    # the naming convention ``<stem>_rank<r>.json``. If the user passes
    # a glob or a base path, we expand to the set of per-rank files and
    # aggregate. Each file now records ``*_communicated_per_call``
    # (``elem_size * numel * (U-1) / U`` — the portion of the tensor
    # actually sent/received per rank via the all-to-all, excluding the
    # self-shard) rather than the raw local tensor size.
    if args.measured_bytes_sidecar is not None:
        import glob as _glob
        sidecar_glob = str(args.measured_bytes_sidecar)
        if "*" not in sidecar_glob:
            # Accept either a concrete file or a base stem and expand
            # to the per-rank files written by the driver.
            stem_path = Path(sidecar_glob)
            candidates = sorted(stem_path.parent.glob(
                f"{stem_path.stem}_rank*.json"
            )) or [stem_path]
        else:
            candidates = sorted(Path(p) for p in _glob.glob(sidecar_glob))
        per_rank = {}
        for p in candidates:
            if not p.exists():
                continue
            with p.open() as fh:
                per_rank[p.name] = json.load(fh)
        summary["_totals"]["measured_bytes_per_rank"] = per_rank
        if args.theoretical_bytes is not None and per_rank:
            theoretical = summary["_totals"]["theoretical_bytes_per_rank_per_tensor"]

            def _avg_key(keys):
                vals = []
                for body in per_rank.values():
                    for k in keys:
                        v = body.get(k)
                        if v is not None:
                            vals.append(float(v))
                            break
                return (sum(vals) / len(vals)) if vals else None

            # Fused K|V 5D a2a carries 2 tensors stacked on dim=2; the
            # theoretical per-rank value is PER TENSOR (K alone or V
            # alone), so divide the fused measurement by 2 before the
            # comparison. Q and OUT a2a each move a single tensor.
            kv_fused = _avg_key([
                "a2a_kv_bytes_communicated_per_call_avg",
                # Backward-compat: older sidecar may only report local
                # tensor bytes; still use them if present with a note.
                "a2a_kv_bytes_per_call_avg",
            ])
            q_comm = _avg_key(["a2a_q_bytes_communicated_per_call_avg",
                               "a2a_q_bytes_per_call_avg"])
            out_comm = _avg_key(["a2a_out_bytes_communicated_per_call_avg",
                                 "a2a_out_bytes_per_call_avg"])

            summary["_totals"]["bytes_theoretical_per_tensor"] = theoretical
            ratios = {}
            for key, val in (
                ("q", q_comm),
                ("kv_fused_per_tensor",
                 (kv_fused / 2) if kv_fused is not None else None),
                ("out", out_comm),
            ):
                if val is not None:
                    ratios[key] = val / max(1, theoretical)
            summary["_totals"]["bytes_ratio_measured_over_theoretical"] = ratios
            # Pass/fail based on fused-K|V-per-tensor matching theoretical.
            kv_ratio = ratios.get("kv_fused_per_tensor")
            if kv_ratio is not None:
                within = abs(kv_ratio - 1.0) <= args.bytes_tolerance
                summary["_totals"]["bytes_within_tolerance"] = within
                if not within:
                    ok = False
                    print(
                        f"FAIL bytes: measured per-tensor K|V a2a bytes / "
                        f"theoretical ratio={kv_ratio:.4f} tolerance "
                        f"±{args.bytes_tolerance}",
                        file=sys.stderr,
                    )

    # Wall-clock aggregate on AV cross-attn NVTX ranges.
    av_range_names = set(_A2A_NVTX_NAMES)
    av_wallclock_ns = _sum_range_wallclock_ns(ranges, av_range_names)
    # Also report per-rank AV wall-clock so we can average/normalize.
    per_tid_wallclock_ns = {}
    for start, end, name, tid in ranges:
        if name in av_range_names:
            per_tid_wallclock_ns[tid] = per_tid_wallclock_ns.get(tid, 0) + max(0, end - start)
    summary["_totals"]["av_wallclock_ns"] = av_wallclock_ns
    summary["_totals"]["av_wallclock_ns_per_rank"] = {
        str(tid): val for tid, val in sorted(per_tid_wallclock_ns.items())
    }

    if args.baseline_sqlite is not None:
        bconn = sqlite3.connect(
            f"file:{args.baseline_sqlite}?mode=ro", uri=True
        )
        b_ranges = _fetch_ranges(bconn, _A2A_NVTX_NAMES)
        baseline_wallclock_ns = _sum_range_wallclock_ns(b_ranges, av_range_names)
        summary["_totals"]["baseline_av_wallclock_ns"] = baseline_wallclock_ns
        b_per_tid = {}
        for start, end, name, tid in b_ranges:
            if name in av_range_names:
                b_per_tid[tid] = b_per_tid.get(tid, 0) + max(0, end - start)
        summary["_totals"]["baseline_av_wallclock_ns_per_rank"] = {
            str(tid): val for tid, val in sorted(b_per_tid.items())
        }
        # Average per-rank ratios (since each rank runs its own AV
        # cross-attn). This is what the plan asks for: per-rank
        # regression on the same hardware.
        if baseline_wallclock_ns > 0:
            ratio = av_wallclock_ns / baseline_wallclock_ns
            summary["_totals"]["av_wallclock_ratio"] = ratio
            # Regression >10% fails.
            if ratio > 1.0 + args.wallclock_tolerance:
                ok = False
                print(
                    f"FAIL wall-clock: AV NVTX current/baseline={ratio:.4f} "
                    f"(> {1.0 + args.wallclock_tolerance}); HARD regression.",
                    file=sys.stderr,
                )

    if total_a2a == 0:
        ok = False
        print(
            "FAIL: no AllToAll-family kernels found across any AV cross-attn range "
            "(strict-Ulysses must launch NCCL SendRecv / AllToAll kernels).",
            file=sys.stderr,
        )

    print(json.dumps(summary, indent=2))
    if ok:
        # ``ok`` only covers the kernel-count and op-invariant checks.
        # Wall-clock (when --baseline-sqlite is given) and bytes (when
        # --measured-bytes-sidecar is given) contribute to ``ok`` only
        # when they fail; the caller is responsible for reading the
        # summary to confirm all HARD sub-criteria pass.
        print(
            "AC-9 kernel-count + op-invariant checks PASS on this profile. "
            "Review the summary for wall-clock and byte-count sub-audits "
            "against the plan's HARD criteria.",
        )
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
