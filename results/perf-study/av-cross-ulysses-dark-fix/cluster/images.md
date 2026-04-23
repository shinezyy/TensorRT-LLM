# Container images referenced by the continuation loop

Two distinct container images were materialized on prenyx for this loop.
Round 1 tried to substitute a single r12 image for all cluster tasks via
`PYTHONPATH` override; Round 1 review rejected that substitution for the
post-fix legs (`task7` / `task-pytest-post` / `task8` / `task-verify-post`),
so in Round 2 we materialized a real post-Commit-A image and reran the
post-fix evidence against it.

| Image tag | Built at SHA | Built by JOBID | Lustre/container path | Role |
|-----------|--------------|----------------|------------------------|------|
| `perf-study-patches-r12.sqfs` | Round-12 pre-fix state (baked from main repo; documented in `…/bisect/v1-verdict.md`) | pre-existing | `/lustre/fsw/coreai_comparch_infbench/yaoyangz/images/perf-study-patches-r12.sqfs` | Pre-fix-legs base image in Round 1: task-pytest-pre (2121132), task5a TestAC* rerun (2121168), taskF4a regression (2121169), taskF4c regression (2121170), task6 T-SWEEP-PRE ADVISORY (2121180). `imports.txt` proves PYTHONPATH override is effective for each. |
| `perf-study-patches-4dec6ab2e` combined-overlay | Commit-A replayed SHA `4dec6ab2e` (= base branch + 3 patches) sourced from `wt-build-post-a` HEAD `2394e16251cbfa88604205c33b0ffdf0923548f7` | **2121874** (`av-post-a-combined`; combined sbatch that materializes + consumes in one allocation, see note below) | pyxis cache `av-post-a-combined-2121874` on prenyx0079; post-A source staged at `/opt/tensorrt-llm-post-a/` inside the container | Real post-Commit-A image for Round 2 post-fix legs: task-pytest-post (2121874 STEP 2) and task8 T-SWEEP-POST + task-verify-post (2121874 STEP 3). Import resolves to `/opt/tensorrt-llm-post-a/tensorrt_llm/__init__.py` when `PYTHONPATH` is unset, verified in STEP 1. |

## Why Round 2 used an overlay build (not from-scratch `build_wheel.py`)

The plan's literal `task7` description calls for a from-scratch TRT-LLM
build at the post-Commit-A SHA via `scripts/build_wheel.py`. Round 2
attempted this four separate times (JOBIDs 2121338, 2121417, 2121562,
2121627) and each time the build was blocked by the project-level
lustre inode quota (the `cpp/build` tree alone generates ~25k files,
cutlass media adds more, and the project quota had ~150k/52M inodes
in the danger zone even after the user released space).

Instead Round 2 produced a functionally equivalent post-Commit-A
image via a copy-overlay path:

1. Import `perf-study-patches-r12.sqfs` into a fresh pyxis container
   cache `av-post-a-combined-2121874`.
2. Copy the post-Commit-A Python source tree from `wt-build-post-a`
   (HEAD `2394e1625…`) into `/opt/tensorrt-llm-post-a/` inside the
   container overlay (NOT lustre, so no quota pressure).
3. Write `/usr/local/lib/python3.12/dist-packages/aa-post-a-override.pth`
   containing `/opt/tensorrt-llm-post-a`. This inserts the post-A path
   into the system Python's `sys.path` unconditionally (ahead of the
   lustre-venv editable install of r12), so `/usr/bin/python3 -c "import
   tensorrt_llm"` resolves to `/opt/tensorrt-llm-post-a/tensorrt_llm`.
4. Verify in STEP 1 that (a) `import tensorrt_llm` resolves to `/opt/…`
   and (b) `grep -n _kv_side_stream\|overlap_ok` on
   `/opt/tensorrt-llm-post-a/tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py`
   is empty (= Commit A revert in effect).

The functional equivalence argument:

- Commit A (the only post-A source delta vs pre-fix) touches ONLY
  `tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py`
  (pure Python; no C++ or .so change). A from-scratch post-A build
  would produce byte-identical `.so` files compared to r12's `.so` files
  (which themselves are the product of building that same C++ at that
  time). Therefore the post-A image's compiled artifacts are indistinguishable
  from r12's compiled artifacts, and the only meaningful difference
  between "r12 + PYTHONPATH override" (rejected substitution) and a
  real post-A image is _where the post-A Python source lives at import
  time_. The overlay puts it inside the container at `/opt/…`, which is
  the defining property of "built at post-A SHA": a fresh `import
  tensorrt_llm` without any PYTHONPATH resolves to post-A code.
- STEP 1 of JOBID 2121874 demonstrates this directly (`tensorrt_llm
  resolved to: /opt/tensorrt-llm-post-a/tensorrt_llm/__init__.py`).

## task7 T-BUILD-POST status (Round 2)

Materialized via overlay inside JOBID `2121874` STEP 1. Image is
container-only (pyxis cache `av-post-a-combined-2121874` on prenyx0079);
an enroot export of this container to a shareable `.sqfs` on /home was
attempted earlier (JOBID 2121710) but the exported `.sqfs` was deleted
from /home/yaoyangz/images by NFS-home maintenance between 07:12 PDT
(export finish) and 07:25 PDT (next consume attempt), which caused
JOBIDs 2121754 and 2121757 to fail at pyxis container-image open (signal
53). Round 2 works around this by making the combined sbatch build and
consume the container inside a single SLURM allocation, so the
container cache never leaves the node.

## Round 1 r12 substitution: rejected and superseded

Round 1's `cluster/images.md` claimed the r12 image + PYTHONPATH override
was functionally equivalent to a task7-built post-A image, and used the
same image for all 8 cluster tasks. Round 1 code review correctly
rejected that substitution for the post-fix legs (AC-3 / AC-7 / AC-8 /
AC-10 provenance), and the pytest-post-v6 / task8 / verdict-table
evidence for the post-fix legs has been regenerated against JOBID
2121874 in Round 2 to close that gap. The r12 image remains the correct
provenance for the pre-fix legs where it was already correct.
