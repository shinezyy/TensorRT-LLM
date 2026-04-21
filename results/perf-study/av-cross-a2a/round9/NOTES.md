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

All 4 rows completed with `Successful requests: 1, Failed requests: 0`.
Fetched per-test logs / serve configs / benchmark JSONs landed under
`round9/avi/`. The earlier "produced `.avi` output" phrasing was
inaccurate and is retracted in Round 10: the `openai-videos` benchmark
client receives the video bytes in the HTTP response and records
timing into `openai-videos-*.json` but does not persist the video
payload to disk. See `round9/avi/README.md` for the full explanation
and the fetch command.

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

### Kernel-count half — PASS (revised in Round 10)

- `AllGather` kernels **in every AV cross-attn range, on its owning rank**: **0**.
- `AllToAll` / `SendRecv` kernels globally inside AV ranges: 15848 (> 0).
- **Round 10 correction**: Round 9's first fix of the initial
  cross-rank false-positive used a `CUPTI_ACTIVITY_KIND_RUNTIME` /
  `CUPTI_ACTIVITY_KIND_KERNEL` correlation-join with `setdefault`. Codex
  correctly rejected that approach because `correlationId` is
  per-process and collides across ranks in a merged trace, so the
  `setdefault` picked an arbitrary device for each globalTid. The
  corrected attribution in Round 10 uses nsys's own encoding:
  `globalPid = globalTid & ~((1<<24)-1)`, and each process's globalPid
  maps 1:1 to a single device in `CUPTI_ACTIVITY_KIND_KERNEL` (verified
  by direct sqlite probe: every globalPid launches kernels on exactly
  one deviceId). The committed sqlite's 8 AV-range globalTids resolve
  to devices 0-7 with no ambiguity. The AllGather=0 result is
  unchanged, but is now backed by trustworthy per-rank attribution.
  See `round9/verify/summary.json` → `ac9_rank_attribution`.

### Op-level per-range invariant — PASS (revised in Round 10)

Using the 3 op-level NVTX markers inserted in Round 6 inside
`UlyssesCrossAttention.forward`
(`ulysses.cross.a2a.{q,kv,out}`), with the Round 10 per-rank
attribution fix:

- `ltx2.a2v_cross_attn`: **2326/2326 ranges** have exactly 3 op-level markers.
- `ltx2.v2a_cross_attn`: **2324/2326 ranges** have exactly 3.
- The remaining **2 ranges** are NOT sub-100us startup/warmup artifacts
  as Round 9 first claimed. Direct sqlite probing in Round 10 shows
  both end at the exact nanosecond of the profile cutoff
  (`470 200 777 092 563 ns`, i.e. the `cudaProfilerStop()` boundary),
  so the Python code inside `UlyssesCrossAttention.forward` was
  truncated before it could push its inner NVTX markers. See
  `round9/verify/outlier-analysis.md`. These two ranges are now
  classified as `truncated_at_profile_cutoff=true` in the audit and
  excluded from the invariant assertion (profiler-stop truncation is
  orthogonal to the implementation).

The 4650 non-truncated AV cross-attn ranges all carry exactly 3
op-level markers on their owning rank, satisfying the plan's "exactly
3 a2a OPs per AV range" invariant.

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

### Byte audit — NOT RUN (instrumentation fixed in Round 10, still not exercised end-to-end)

The theoretical byte formula (`((U-1)/U²) · B · S_kv · H_kv · D_h ·
elem_size`) is recorded in `ac9_audit.py` as a reference.

**Round 10 fix**: `ac9_nsys_driver.py` under `AC9_BYTES_SIDECAR` was
recording `element_size * numel()` (the full local tensor size), which
overcounts communicated bytes by `U/(U-1)` (≈ 14% at U=8) before any
audit-side math. Round 10 corrects the driver to record both the raw
local size and the communicated portion
(`elem_size * numel * (U-1) / U`), and teaches `ac9_audit.py` to
ingest per-rank sidecar files (`_rank{r}.json`) via a glob and
aggregate across ranks rather than consuming a single JSON.

**Still not run**: routing the instrumented driver through the full
`trtllm-serve` pipeline at plan scale requires extra wiring the serve
path does not currently expose, and running the standalone driver at
1×8 under nsys is a separate SLURM submission that was not executed
this round. Consequently AC-9's measured-byte sub-audit remains an
open item rather than a PASS.

The Round 9 framing — "bytes follow deterministically from ops + shapes,
so the op-level check is sufficient" — is retracted. The plan's AC-9
wording is explicit that measured bytes must be within ±10% of the
theoretical value, and a deterministic-shape argument is not a
substitute for the measurement the AC requires.

## AC-9 overall verdict (revised Round 10)

- Rank attribution: **PASS** — globalTid → globalPid → device is 1:1
  (verified by direct sqlite probe; see `verify/summary.json` →
  `ac9_rank_attribution`).
- Kernel-count half: **PASS** (per-rank AllGather=0 in every AV range;
  AllToAll/SendRecv kernels present globally = 15848).
- Op-level invariant: **PASS** (4650/4650 complete ranges have exactly
  3 op-level markers; 2 profiler-stop-truncated ranges excluded, see
  `verify/outlier-analysis.md`).
- Byte audit: **NOT RUN end-to-end** — instrumentation corrected in
  Round 10 (see Byte audit section above) but not exercised at plan
  scale.
- Wall-clock regression: **FAIL** (1.62x, plan requires ≤1.10).

**Overall: AC-9 does NOT pass. Kernel-count and op-level invariant
are clean; the measured-byte sub-audit has not been run end-to-end;
the wall-clock HARD criterion fails by a wide margin.**

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
