#!/usr/bin/env bash
# M1 registered RoboGCG targeted-redirect attack (box step 3).
# Faithful to configs/m1_viability.yaml step 3 EXCEPT --exclusive-gpu is OMITTED:
# the box is shared (another user + the GUI live on GPU 1); this run is pinned to
# cuda:0 (uncontended) and is NOT a timing measurement, so per the runbook a
# production attack run does not need — and should not over-claim — an exclusive
# window. GPU 0 is dedicated to this process; GPU 1 is a different card (L4: one
# card per claim, no cross-card mixing).
#
# Auto-restart until-loop: a transient OOM self-recovers (--resume skips the
# per-unit write-once checkpoints already on disk). Capped to avoid a crash-loop.
#
# Arg 1 = n_attacked (default 10). Run `... 1` for a 1-unit registered pilot, then
# `... 10` to --resume the remaining units into the SAME write-once run dir
# (results/m1-robogcg-redirect/). git_commit must be identical across the two calls
# for the BUG3 resume-header guard to accept the reuse — do not commit between them.
set -u
cd "$HOME/vla-injection" || exit 1
N_ATTACKED="${1:-10}"

export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$HOME/LIBERO"

MAX_RESTARTS=50
n=0
until uv run --no-sync python scripts/run_attack.py \
        --config configs/m1_viability.yaml \
        --schema-from results/m1-benign-baseline/schema_repinned.json \
        --n-attacked "$N_ATTACKED" --n-steps 500 --search-width 512 --eval-batch 32 \
        --openvla-root "$HOME/openvla" --device cuda:0 \
        --resume ; do
  code=$?
  n=$((n + 1))
  echo "[launcher] run_attack exited $code (restart $n/$MAX_RESTARTS) $(date -u)"
  if [ "$n" -ge "$MAX_RESTARTS" ]; then
    echo "[launcher] hit MAX_RESTARTS=$MAX_RESTARTS -> giving up $(date -u)"
    exit 1
  fi
  sleep 10
done
echo "[launcher] run_attack completed cleanly (exit 0) $(date -u)"
