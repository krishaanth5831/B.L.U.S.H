#!/usr/bin/env bash
# capped_ripeness_python.sh: the ripeness env's python in its own 3 GB memory scope (K7's cap rule).
# For runs that already sit under a 3 GB cap themselves: the YOLO worker (~1.4 GB, torch) is a child
# process, so inside the parent's scope it pushed generate_scripted.py's recording pass past 3 GB and
# the OOM killer ended the pilot (2026-10-27). With this it gets its own scope:
#   BLUSH_RIPENESS_PYTHON=$PWD/scripts/capped_ripeness_python.sh python generate_scripted.py ...
# --scope execs in place (same pid, same stdin/stdout), so yolo_client's pipes and terminate() still work.
exec systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0 \
  "${RIPENESS_PYTHON_REAL:-$HOME/miniforge3/envs/ripeness/bin/python}" "$@"
