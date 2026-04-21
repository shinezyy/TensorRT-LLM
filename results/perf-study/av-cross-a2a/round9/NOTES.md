# AC-9 Bring-up Notes — Round 9

## Scope

End-to-end AC-9 verification per the plan: fresh container build from
`perf-study-patches` HEAD, 4-row `a2a-tests.csv` sweep, fresh 1x8 nsys
capture on the new container, and kernel-count + byte + wall-clock
audit against a pre-change baseline.

## Artifacts

```
results/perf-study/av-cross-a2a/round9/
  NOTES.md                 (this file)
  nsys/
    perf-study/
      profile.nsys-rep     (1x8 nsys from perf-study-patches HEAD)
      profile.sqlite       (nsys export for audit)
    baseline/
      profile.nsys-rep     (1x8 nsys from cross-attn-ulysses-base)
      profile.sqlite       (baseline export)
  verify/
    summary.json           (machine-readable AC-9 verdict)
    audit-perf-vs-baseline.json   (per-range audit output)
    audit-perf-vs-baseline.err    (per-range FAIL messages, if any)
```

## task22 — Fresh container (PASS)

- Built from `perf-study-patches` HEAD = `e11db0baf`.
- Output: `/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/perf-study-patches-r9.sqfs`
  (14 GB).
- Updated `plugins/aigv-bench-dev1/workspace.config` `image_visual_gen_ltx2`
  to point at the new image.

## task23 — 4-row sweep (PASS, 4/4 OK)

Sweep ran via `visual-bench:bench-visual-gen --csv a2a-tests.csv`
against the new container (tag `round9-a2a-sweep_20260422_002811`).

| Row | GPUs (cfg × ulysses) | Denoising | Pipeline total | Status |
|-----|----------------------|-----------|----------------|--------|
| 1x2 | 2 | 21.87s | 22.89s | OK |
| 1x8 | 8 | 14.61s | 15.62s | OK |
| 2x2 | 4 | 13.85s | 14.87s | OK |
| 2x4 | 8 | 13.97s | 14.99s | OK |

All 4 rows produced `.avi` output and reported `Successful requests: 1,
Failed requests: 0`.

## task24 — 1x8 nsys captures

- **perf-study**: captured with `perf-study-patches-r9.sqfs` and
  workspace on `perf-study-patches` branch (commit `e11db0baf`). Tag
  `round9-nsys-perf_20260422_002932`. `profile.nsys-rep` 94 MB.
- **baseline**: captured with the SAME `perf-study-patches-r9.sqfs`
  container image but with the lustre workspace switched to
  `cross-attn-ulysses-base` branch (commit `036c5c13d`). Tag
  `round9-nsys-baseline-v2_20260422_012719`. `profile.nsys-rep` 57 MB.
- Both used `--profile-start-step 5 --profile-num-steps 6` (6 denoise
  steps captured at steady-state).
- Using the same container for both avoids a second 30+ min build; the
  editable install on the workspace means import `tensorrt_llm` resolves
  to whichever source tree is checked out, giving a fair apples-to-apples
  comparison on the same hardware, same compiled C++ side.

## task25 — Audit

Run:
```
python tests/unittest/_torch/visual_gen/multi_gpu/_u8_drivers/ac9_audit.py \
    <perf-study>.sqlite --baseline-sqlite <baseline>.sqlite
```

### Kernel-count half — PASS

- `AllGather` kernels **in every AV cross-attn range, on its owning rank**: **0**.
- `AllToAll` / `SendRecv` kernels globally inside AV ranges: 13972 (> 0).
- Initial audit incorrectly reported 7 AllGather kernels inside AV ranges
  because the cross-rank multi-GPU trace mixes all 8 ranks' NVTX in one
  file and my first counter didn't filter by owning GPU. After adding a
  tid→device map via `CUPTI_ACTIVITY_KIND_RUNTIME` correlation, the per-
  rank AllGather count is exactly 0 in every AV range. The 7 aggregate
  matches were from audio/video self-attn AllGathers on OTHER ranks
  happening concurrently.

### Op-level per-range invariant — PASS

Using the 3 op-level NVTX markers inserted in Round 6 inside
`UlyssesCrossAttention.forward`
(`ulysses.cross.a2a.{q,kv,out}`):

- `ltx2.a2v_cross_attn`: **2326/2326 ranges** have exactly 3 op-level markers.
- `ltx2.v2a_cross_attn`: **2324/2326 ranges** have exactly 3; 2 outlier
  ranges (both sub-100 μs) have 0 markers — likely warmup/startup
  artifacts. 99.96% hit rate.

This confirms the plan's "exactly 3 a2a OPs per AV range" invariant at
the implementation-independent op level.

### Wall-clock regression — **FAIL**

Per-rank AV NVTX wall-clock (sum of `ltx2.a2v_cross_attn` +
`ltx2.v2a_cross_attn` range durations on each rank, averaged over 8
ranks):

| Metric | perf-study (Ulysses a2a) | baseline (AllGather) | Ratio |
|--------|--------------------------|----------------------|-------|
| Avg per-rank AV wall-clock | 1115.6 ms | 688.7 ms | **1.62x** |
| Aggregate across 8 ranks | 8924.7 ms | 5509.9 ms | **1.62x** |

**Observed regression: 62.0%** (plan tolerance: ≤10%).

This is a HARD FAIL against AC-9 as written. The Ulysses cross-attn
refactor, while it correctly reduces per-rank K/V bytes moved from
`((U-1)/U) · T_kv` (AllGather) to `((U-1)/U²) · T_kv` per tensor
(AllToAll), introduces TWO additional collectives per AV cross-attn
per layer: a Q all-to-all and an output all-to-all that the AllGather
baseline did not have. On intra-node 1x8 B200 NVLink topology, these
extra collectives dominate the savings from the smaller K/V payload.

### Byte audit — Not run at plan scale

The theoretical byte formula (`((U-1)/U²) · B · S_kv · H_kv · D_h ·
elem_size`) is recorded in `ac9_audit.py` as a reference. A per-rank
measured-byte path exists in `ac9_nsys_driver.py` (gated by
`AC9_BYTES_SIDECAR` env), but running it through the full
`trtllm-serve` pipeline at plan scale requires extra wiring the serve
path does not currently expose. The op-level NVTX invariant (exactly 3
a2a OPs per AV range) combined with the deterministic tensor shapes
(fully specified by B, S_kv, H_kv, D_h, U in the serve config) makes
the byte count redundant with the op-level check — each op processes a
known-shape tensor, so bytes follow from shape × elem_size. This is
documented rather than empirically verified for this round.

## AC-9 overall verdict

- Kernel-count half: **PASS** (per-rank AllGather=0; AllToAll/SendRecv
  present).
- Op-level invariant: **PASS** (3 ops per AV range, 99.96%).
- Wall-clock regression: **FAIL** (1.62x, plan requires ≤1.10).
- Byte audit: **Deferred** (topology follows from ops + shapes; not
  empirically recorded this round).

**Overall: AC-9 does NOT pass the plan's HARD wall-clock criterion.**

This is an honest test outcome. The implementation correctly enforces
the strict-Ulysses a2a topology, but does not deliver the expected
improvement on 1x8 B200 intra-node NVLink. At smaller Ulysses sizes
(2, 4) the extra Q + output collectives may amortize better, but that
is not what the plan's HARD 1x8 criterion tests. Two possible paths
forward:

1. **Revise the implementation** — investigate whether the Q a2a and
   output a2a can be overlapped with other work, or whether the per-rank
   K/V payload reduction alone justifies keeping just the fused K|V a2a
   and leaving Q/output gathered as before. This is a plan-level
   discussion, not an audit issue.
2. **Revise the AC** — acknowledge that 1x8 B200 NVLink topology
   inverts the trade-off, and scope AC-9's ≤10% regression bound to
   configurations where the savings actually net out.

Neither path is a valid mid-round RLCR action without user/Codex
approval; this round's deliverable is the **evidence** that the plan's
criterion fails on this hardware.
