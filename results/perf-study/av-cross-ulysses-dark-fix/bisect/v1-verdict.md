# V1 Bisection Verdict (AC-1)

## Verdict: **PASS** — H1 confirmed.

V1 one-line bisection patch (`overlap_ok = False`) at the frozen pre-fix base SHA produces non-dark 1x2 video. H1 (overlap path in `UlyssesCrossAttention.forward` is the root cause of pure-black 1x2 output) is confirmed. Plan advances to Milestone 2 push (task5a / task5b).

## Frozen Pre-fix Base SHA

`bb414983cc6fc16090f17482c4305f4e0006ac8c` — captured in `base.sha` (Round 0, task1).

## Patch Applied (V1 one-liner)

```python
# tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py — UlyssesCrossAttention.forward
# V1 bisect (task2, AC-1): force serial path to isolate the
# overlap-branch as root cause of pure-black 1x2 output.
overlap_ok = False  # V1-bisect-patch: was (q.is_cuda and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing())
```

No other code changes. The patch forces the `if overlap_ok:` guard to skip the Q/KV side-stream overlap block and fall through to the `else` serial fallback.

## Cluster Execution

- **Cluster**: prenyx
- **Account**: `coreai_comparch_inferencex` (FairShare 0.785, top of {inferencex, trtllm, infbench} for yaoyangz)
- **Image**: `/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/perf-study-patches-r12.sqfs` (14 GB, Apr 21 13:49; baked at Round 12 pre-fix state)
- **Workspace**: `/lustre/fsw/coreai_comparch_infbench/yaoyangz/visualgen-ltx2` on branch `v1-bisect-task2` (pushed from this repo's `bb414983c` via `git bundle`; linked-worktree blocker noted in tracker worked around using `git fetch <bundle>`).
- **JOBID**: 2116795 (direct `visual_gen_ltx2.py` invocation; earlier JOBIDs 2116724 / 2116735 went through `bench-visual-gen` but the wrapper's interrupted shutdown truncated the `PurePythonEncoder` AVI save; JOBID 2116756 attempted `.mp4` which `PurePythonEncoder` rejects without `ffmpeg`; JOBID 2116795 switched to `.avi` and completed cleanly).
- **Config**: `ltx2-t2v-sfp4-vanilla-1x2-cache0-tcompile1-cg0` — LTX2 T2V, 720x1280, 121 frames, 40 steps, VANILLA attention, static-nvfp4 linear, torch_compile on, cuda_graph off, cfg=1, ulysses=2.
- **Output**: `/lustre/fsw/coreai_comparch_infbench/yaoyangz/aigv-results/results/bench-visual-gen/v1bisect-patched-bb414983c-direct_2116795/output.avi` (53 MB).
- **Fetched to**: `results/perf-study/av-cross-ulysses-dark-fix/bisect/media/v1-patched-bb414983c.avi`.
- **Bench timing**: Denoising 40/40 steps in 27.86s (0.70s/step). Pipeline 29.43s. Generation completed 31.65s.

## Brightness Verification

Run: `python scripts/verify_video_not_dark.py results/perf-study/av-cross-ulysses-dark-fix/bisect/media/v1-patched-bb414983c.avi`

### Positive (patched base) — PASS

```
PASS .../v1-patched-bb414983c.avi size=1280x704 mean=151.11 std=61.943 max=255.0 thresholds(dark_mean<=5.0, flat_std<=1.0)
exit: 0
```

`mean=151.11` >> `dark_mean_max=5.0`. `std=61.943` >> `flat_std_max=1.0`. Verifier exits 0.

### Negative (unpatched base) — FAIL (reused Round 12 1x2 artifact)

Reused `results/perf-study/av-cross-a2a/round12/avi/1x2.avi`. Equivalence justification: between commit `16c51e2d3` (where the round12 .avi was produced) and the frozen base `bb414983c`, the only change to `parallel.py` is the `_ac9_bytes_sidecar` module-level side-effect import at the bottom of the file (`git log 16c51e2d3..bb414983c -- tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py` returns only `e6367acab`, a non-overlap-path change). The overlap branch logic is byte-identical between these SHAs, so the round12 1x2.avi is a valid negative artifact for the frozen base.

```
FAIL results/perf-study/av-cross-a2a/round12/avi/1x2.avi size=1280x704 mean=0.00 std=0.000 max=0.0 thresholds(dark_mean<=5.0, flat_std<=1.0)
  reason: mean luminance 0.00 <= dark_mean_max 5.0
  reason: pixel std 0.000 <= flat_std_max 1.0
exit: 1
```

## Verdict Summary Table

| State | SHA on parallel.py | mean | std | verifier exit |
|-------|---------------------|------|-----|---------------|
| **Pre-fix (negative)** | `bb414983c` (overlap branch active) | 0.00 | 0.000 | **1 (FAIL as required)** |
| **V1 patched (positive)** | `bb414983c` + `overlap_ok = False` | **151.11** | **61.943** | **0 (PASS)** |

## Conclusion

H1 confirmed. The Q/KV stream-overlap branch in `UlyssesCrossAttention.forward` is the proximate cause of the pure-black 1x2 multi-GPU output. Forcing serial execution restores correct video. Commit A (full revert of the overlap branch + `_kv_side_stream` field + gated `VG_DEBUG_FINITE_CHECK` asserts) is the sanctioned fix. AC-1 closed.
