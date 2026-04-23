# V1 Bisection Verdict (AC-1)

## Verdict: **PASS** — H1 confirmed.

V1 one-line bisection patch (`overlap_ok = False`) at the frozen pre-fix base SHA produces non-dark 1x2 video. Unpatched frozen base produces pure-black. H1 (overlap path in `UlyssesCrossAttention.forward` is the root cause of pure-black 1x2 output) is confirmed. AC-1 closed.

## Frozen Pre-fix Base SHA

`bb414983cc6fc16090f17482c4305f4e0006ac8c` — captured in `base.sha` (Round 0, task1).

## Patch Applied (V1 one-liner, task2 positive leg)

```python
# tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py — UlyssesCrossAttention.forward
# V1 bisect (task2, AC-1): force serial path to isolate the
# overlap-branch as root cause of pure-black 1x2 output.
overlap_ok = False  # V1-bisect-patch: was (q.is_cuda and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing())
```

No other code changes. The patch forces the `if overlap_ok:` guard to skip the Q/KV side-stream overlap block and fall through to the `else` serial fallback.

## Cluster Execution Summary

- **Cluster**: prenyx
- **Account**: `coreai_comparch_inferencex` (FairShare 0.785, top of {inferencex, trtllm, infbench} for yaoyangz)
- **Image**: `/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/perf-study-patches-r12.sqfs` (14 GB, Apr 21 13:49; baked at Round 12 pre-fix state)
- **Workspace rotation** (R3 + R4): `git bundle` → `scp` → `git fetch <bundle>` into the existing remote `/lustre/.../visualgen-ltx2` workspace (linked-worktree-safe, avoids `git worktree add` which fails on this repo). Branches created and deleted within the remote workspace for each job; state reverted at end of round.

### Positive (V1-patched frozen base) — PASS — JOBID 2116795 (R3)

Direct `visual_gen_ltx2.py` sbatch (per `BL-20260423-visualgen-bench-wrapper-interrupts-encoder`: bypass bench-visual-gen wrapper, `.avi` output):
- Config: `ltx2-t2v-sfp4-vanilla-1x2-cache0-tcompile1-cg0` — LTX2 T2V, 720x1280, 121 frames, 40 steps, VANILLA attention, static-nvfp4 linear, torch_compile on, cuda_graph off, cfg=1, ulysses=2.
- Output: `results/perf-study/av-cross-ulysses-dark-fix/bisect/media/v1-patched-bb414983c.avi` (53 MB).
- Denoising 40/40 in 27.86s (0.70s/step). Pipeline 29.43s. Generation 31.65s.
- Brightness verification: `python scripts/verify_video_not_dark.py ...avi` →
  ```
  PASS .../v1-patched-bb414983c.avi size=1280x704 mean=151.11 std=61.943 max=255.0 thresholds(dark_mean<=5.0, flat_std<=1.0)
  exit: 0
  ```

### Negative (UNPATCHED frozen base) — FAIL — JOBID 2119590 (R4, canonical)

**This is the canonical AC-1 negative artifact**, replacing the Round-12 equivalence-proof reuse used in R3. Same direct `visual_gen_ltx2.py` sbatch pattern on the SAME frozen base `bb414983c` with NO `overlap_ok=False` patch applied (overlap branch active):
- Config: identical to the positive run.
- Output: `results/perf-study/av-cross-ulysses-dark-fix/bisect/media/task6-unpatched-bb414983c.avi` (1.78 MB — much smaller than the patched 53 MB because a uniformly-zero video compresses trivially).
- Brightness verification: `python scripts/verify_video_not_dark.py ...avi` →
  ```
  FAIL .../task6-unpatched-bb414983c.avi size=1280x704 mean=0.00 std=0.000 max=0.0 thresholds(dark_mean<=5.0, flat_std<=1.0)
    reason: mean luminance 0.00 <= dark_mean_max 5.0
    reason: pixel std 0.000 <= flat_std_max 1.0
  exit: 1
  ```

This same run also serves as task6 T-SWEEP-PRE evidence toward AC-3 (see tracker).

### Supplementary (Round 12 reuse) — deprecated by task6

R3 originally cited `results/perf-study/av-cross-a2a/round12/avi/1x2.avi` as the negative artifact, justified by the byte-equivalence of `parallel.py` between `16c51e2d3` (where round12 was captured) and the frozen base `bb414983c` — `git log 16c51e2d3..bb414983c -- tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py` returns only `e6367acab` (the `_ac9_bytes_sidecar` module-level import at file bottom, not on the overlap path). R4's JOBID 2119590 makes this reuse no longer required; the round12 artifact remains in the repository as supplementary evidence but is NOT the canonical AC-1 negative.

## Verdict Summary Table

| State | parallel.py overlap branch | JOBID | avi mean | avi std | verifier exit |
|-------|----------------------------|-------|----------|---------|---------------|
| **Pre-fix (UNPATCHED frozen base)** | active | 2119590 (R4) | **0.00** | **0.000** | **1 (FAIL as required)** |
| **V1 patched (`overlap_ok = False`)** | bypassed (serial fallback) | 2116795 (R3) | **151.11** | **61.943** | **0 (PASS)** |

## Conclusion

H1 confirmed. The Q/KV stream-overlap branch in `UlyssesCrossAttention.forward` is the proximate cause of the pure-black 1x2 multi-GPU output. Forcing serial execution restores correct video. Commit A (full revert of the overlap branch + `_kv_side_stream` field + gated `VG_DEBUG_FINITE_CHECK` asserts) is the sanctioned fix. AC-1 closed with both positive and negative legs anchored on direct executions of `bb414983c`.
