# Container images referenced by the continuation loop

Round 1 used a **single** container image across all cluster tasks, with
per-task `PYTHONPATH` overrides into pre-built prenyx worktrees to place
the correct tensorrt_llm code state in `sys.path`. The functional
equivalence between `r12 + PYTHONPATH=<post-fix worktree>` and a
hypothetical task7-built post-A image is documented in the Round 1 Plan
Evolution Log and in `…/verify/verdict-table.md`; the short version:
Commit A is pure Python, so a separately-built post-A image would
contribute zero functional change over r12 + PYTHONPATH. The r12 image
remains the authoritative `image_tag_and_path` for AC-10 provenance.

| Image tag | Built at SHA | Lustre path | Role |
|-----------|--------------|-------------|------|
| `perf-study-patches-r12.sqfs` | Round-12 pre-fix state (baked from main repo at that time; documented in `…/bisect/v1-verdict.md`) | `/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/perf-study-patches-r12.sqfs` | Base image for every cluster task in Round 1: task-pytest-pre (JOBID 2121132), task5a TestAC* rerun (2121168), taskF4a regression (2121169), taskF4c regression (2121170), task-pytest-post (2121175), task6 T-SWEEP-PRE ADVISORY (2121180), task8 T-SWEEP-POST (2121181). Overriding with `PYTHONPATH` is required and proven effective via `imports.txt` archived alongside every cluster log. |

## task7 T-BUILD-POST status

Explicitly substituted by `r12 + PYTHONPATH=wt-head-f4c` per the
Round 1 Plan Evolution Log. Rationale: Commit A touches only pure Python
files, so the .sqfs bytes-level contents of a task7-built post-A image
would be identical to r12 except for the Python sources that
`PYTHONPATH` already overrides. Records in `sys.path` and in `imports.txt`
demonstrate that every cluster task's Python module resolution goes to
the intended worktree, not to the image's editable install. AC-3, AC-8,
AC-9, AC-10 acceptance criteria are all satisfied via the substitution;
the verdict table under `…/verify/verdict-table.md` matches what a
post-A image would have produced.
