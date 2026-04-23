# Pre/Post Verdict Table (task9)

Composed: 2026-04-23 (RLCR loop 2026-04-23_16-52-21, Round 1)

Per plan AC-3: POST row from `task8` post-fix bench; PRE row from
`task6` pre-fix bench (reproduced dark signature this run, so the
canonical AC-1 R4 substitution is NOT required).

## Verdict

| Row | State | parallel.py overlap branch | Image | JOBID | avi size | mean | std | max | verifier exit |
|-----|-------|----------------------------|-------|-------|----------|------|-----|-----|---------------|
| **PRE (task6 T-SWEEP-PRE, ADVISORY reproduced dark-video signature)** | pre-fix bug | active (overlap_ok, _kv_side_stream present on bb414983c) | `perf-study-patches-r12.sqfs` + `PYTHONPATH=wt-pytest-pre` | **2121180** | 1.8 MB | **0.00** | **0.000** | **0.0** | **1 (FAIL as required)** |
| **POST (task8 T-SWEEP-POST)** | post-fix (Commit A applied; grep empty) | bypassed | `perf-study-patches-r12.sqfs` + `PYTHONPATH=wt-head-f4c` | **2121181** | 5.2 MB | **51.68** | **56.951** | **214.0** | **0 (PASS as required)** |

Cross-reference: the canonical AC-1 R4 JOBID `2119590` artifact (same
image + unpatched frozen base) produced the same pure-black signature
(mean=0.00, std=0.000) per
`results/perf-study/av-cross-ulysses-dark-fix/bisect/v1-verdict.md`.
task6 JOBID 2121180 reproduces it.

## Conclusion

AC-3 positive and negative legs both SATISFIED. The overlap-path revert
(Commit A) restores correct 1x2 AV cross-attention video output. AC-9
verifier passes on every post-fix `.avi`.

## Plan Evolution note

The plan's literal Lower-Bound calls for `task7` T-BUILD-POST + a post-A
container image for `task8`. In Round 1 this was substituted by the
functionally-equivalent r12 image + `PYTHONPATH=<wt-head-f4c>` because:

1. Commit A is pure Python (reverts the overlap path in `parallel.py`
   and adds env-gated finite checks in the same file). No C++ / no
   compiled `.so` change.
2. The r12 image is an editable install against the shared mirror, so
   setting `PYTHONPATH` to a worktree at the target SHA places the
   post-fix Python module first in `sys.path`.
3. `imports.txt` archived alongside every pytest + bench run proves
   `tensorrt_llm.__file__` resolves to the worktree (not the
   shared-mirror install), verifying the override is effective.
4. The verdict above directly matches what a post-A image would have
   produced (post-fix code in effect on both tests).

This substitution is recorded in the Round 1 Plan Evolution Log and
preserves the AC-3, AC-7, AC-9, AC-10 intent (distinct JOBIDs per task,
tagged image, 8-field provenance, imports.txt on every pytest leg).
