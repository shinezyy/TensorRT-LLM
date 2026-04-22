# Pre-fix Frozen Base SHA

## SHA
`bb414983cc6fc16090f17482c4305f4e0006ac8c`

Branch at freeze time: `perf-study-patches`.
Freeze timestamp: 2026-04-22 (RLCR loop id `2026-04-22_22-49-49`).

## Why this SHA

1. **Current HEAD of `perf-study-patches`** at RLCR loop start — this is the exact tree the bench sweep in `results/perf-study/av-cross-a2a/round12/avi/` was produced against, so the dark `.avi` evidence (mean=std=max=0 at 1280x704) is attributable to this commit graph.

2. **Overlap-introducing commit `16c51e2d3e127c7223d86c05151ccaa52eb593ca` is an ancestor** (verified via `git merge-base --is-ancestor 16c51e2d3 bb41498`). That commit — `[None][perf] UlyssesCrossAttention Q/KV overlap + per-rank bytes audit + video persistence (Round 11)` — is the suspected root cause for the NaN/inf contamination that produces the pure-black frames. Freezing at a SHA that includes this commit (and not reverting it here) is required so:
   - Sub-Agent B can apply the Commit-B test patch on top of this frozen base and observe a deterministic FAIL on prenyx 2xB200 (AC-2 negative / AC-8 T-PYTEST-PRE).
   - The V1 bisection (task2) applies the `overlap_ok = False` one-liner at THIS SHA and confirms H1 (AC-1 positive).
   - The pre-fix cluster bench (T-SWEEP-PRE, task6) reuses the existing `perf-study-patches-r12.sqfs` image, which was built from an ancestor of this SHA, so the evidence chain is internally consistent.

3. **`scripts/verify_video_not_dark.py` is tracked at this SHA** (it is in fact the commit message of `bb41498`: `[None][tool] scripts/verify_video_not_dark.py: AVI MJPEG brightness verifier for AV-cross Ulysses dark-output diagnosis`). Per AC-9 this script must be reachable on the branch the executor uses; freezing at the SHA where it landed satisfies that requirement without extra coordination.

## Consumers of this artifact

- task2 (V1 bisection): applies `overlap_ok = False` one-line diff at this SHA; archives verdict PASS/FAIL here.
- task3 (Sub-Agent A): authors Commit A against this SHA.
- task4 (Sub-Agent B): authors Commit B test patch against this SHA; runs pytest (a) on `<this SHA>` -> FAIL log archived here, and (b) on `<this SHA> + Commit A` -> PASS log archived here.
- task5a (merge Commit A): blocked on the FAIL log at `<this SHA>` being archived here.
- task5b (merge Commit B): blocked on the PASS log at `<this SHA> + Commit A` being archived here.
- task6 (T-SWEEP-PRE): pre-fix side uses this SHA.
- task-pytest-pre (T-PYTEST-PRE): uses this SHA plus the Commit-B test patch.

## Invariants

- This file and `base.sha` are the only things Round 0 of the RLCR loop creates under `bisect/`. All downstream bisection logs (V1 verdict, FAIL log, PASS log, pytest logs) are produced by later rounds and stored alongside these two files under the same directory prefix.
- The SHA line in `base.sha` is a bare 40-character hex digest followed by one trailing newline. No commit-metadata comments. Consumers parse it with `cat`.
- The frozen SHA must not be edited. If a future round discovers the SHA was wrong, log a `Plan Evolution Log` entry in `goal-tracker.md` — do NOT silently rewrite this file.
