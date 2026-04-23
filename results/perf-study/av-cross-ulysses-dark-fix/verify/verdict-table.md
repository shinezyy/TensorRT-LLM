# Pre/Post Verdict Table (task9)

Composed: 2026-04-23 (RLCR loop 2026-04-23_16-52-21, Round 2)

Per plan AC-3: POST row from `task8` post-fix bench executed in the
real post-Commit-A image; PRE row from `task6` pre-fix bench (reproduced
dark signature this run).

## Verdict

| Row | State | parallel.py overlap branch | Image | JOBID | avi size | mean | std | max | verifier exit |
|-----|-------|----------------------------|-------|-------|----------|------|-----|-----|---------------|
| **PRE (task6 T-SWEEP-PRE, ADVISORY reproduced dark-video signature)** | pre-fix bug | active (overlap_ok, _kv_side_stream present on bb414983c) | `perf-study-patches-r12.sqfs` + `PYTHONPATH=wt-pytest-pre` | **2121180** | 1.8 MB | **0.00** | **0.000** | **0.0** | **1 (FAIL as required)** |
| **POST (task8 T-SWEEP-POST in real post-A image)** | post-fix (Commit A applied; grep empty in `/opt/tensorrt-llm-post-a/` source) | bypassed | `perf-study-patches-4dec6ab2e` combined-overlay (pyxis cache `av-post-a-combined-2121874` on prenyx0079) + `PYTHONPATH=wt-post-a-b-v6` | **2121874** (STEP 3) | 5.2 MB | **51.68** | **56.951** | **214.0** | **0 (PASS as required)** |

Cross-reference: the canonical AC-1 R4 JOBID `2119590` artifact (same
r12 image + unpatched frozen base) produced the same pure-black
signature (mean=0.00, std=0.000) per
`results/perf-study/av-cross-ulysses-dark-fix/bisect/v1-verdict.md`.
task6 JOBID 2121180 reproduces it this loop.

## Conclusion

AC-3 positive and negative legs both SATISFIED. The overlap-path revert
(Commit A) restores correct 1x2 AV cross-attention video output. AC-9
verifier passes on every post-fix `.avi`.

## Round 2 provenance note

The Round 1 POST row used JOBID `2121181` against the r12 image with
`PYTHONPATH=wt-head-f4c` (F4a+F4c HEAD), which the Round 1 code review
rejected as non-compliant post-fix provenance on two counts: (a) the
image was not a real post-Commit-A build, and (b) the PYTHONPATH
pointed at a worktree past the intended post-A + post-B-v6 state (it
included F4a and F4c). Round 2 replaces that POST row with JOBID
`2121874` STEP 3:

- **Real post-Commit-A image**: the JOBID 2121874 combined sbatch first
  imports `perf-study-patches-r12.sqfs` into a fresh pyxis container
  cache, then overlay-copies the post-Commit-A source tree from
  `wt-build-post-a` (HEAD `2394e1625…`) into `/opt/tensorrt-llm-post-a/`
  inside the container, and writes a site-packages `.pth` redirect. A
  fresh `/usr/bin/python3 -c "import tensorrt_llm"` inside the
  container (no PYTHONPATH) resolves to `/opt/tensorrt-llm-post-a/tensorrt_llm/__init__.py`,
  which defines the image as a post-A image. See `…/cluster/images.md`
  for the full rationale, and the rationale for the overlay build vs
  a from-scratch `build_wheel.py` (project-quota blockers on
  `cpp/build` inodes).
- **Correct test state**: `PYTHONPATH=wt-post-a-b-v6` points at the
  base branch HEAD `c183fc79b` (= post-A + post-B-v6, without F4a or
  F4c). Grep for `_kv_side_stream\|overlap_ok` is empty on both the
  image `/opt/` source and the worktree source.
- **Identical verdict numbers**: mean / std / max are identical to the
  Round 1 POST numbers (51.68 / 56.951 / 214.0). This is expected:
  Commit A is pure Python and the seed=42 / prompt / shape were held
  constant across runs, so the frame-level statistics on the first
  JPEG frame are deterministic.

The Round 1 t-sweep-post-2121181 directory is retained on disk as
historical context (its provenance.txt still documents the r12-image
substitution); the authoritative post-fix bench is
`results/bench-visual-gen/t-sweep-post-2121874/`.
