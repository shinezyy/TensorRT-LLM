# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-tree SLURM drivers for the U=8 multi-rank AC-5 / AC-10 suites.

These drivers are invoked via ``srun -n 8 --ntasks-per-node=8 --mpi=pmix``
(see ``_run_script.sh``) to bypass the mp.spawn/PMIx abort documented in
``.humanize/bitlesson.md`` (BL-20260421-mp-spawn-pmix-at-u8).
"""
