#!/usr/bin/env bash
#
# Data-parallel judge evaluation across 2 GPUs.
#
# Splits dataset_tts.jsonl into two halves, runs single_pipeline.py on each half
# on its own GPU in parallel, then concatenates the shard outputs back into the
# canonical dataset_eval.jsonl / dataset_eval_votes.jsonl (records are keyed by
# annotation_id + pipeline, so plain concatenation reconstructs them in order).
#
# Any extra flags are forwarded to BOTH workers, e.g.:
#     ./run_eval_2gpu.sh --votes 10 --temperature 0.7
# Note: --limit is applied per shard (so --limit 100 -> ~200 pipelines total).
#
# Overridable via env: PYTHON, INPUT, EVAL_OUT, VOTES_OUT, GPUS ("0 1").
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="$REPO/src/data"
TEMPERATURE=1.0

# How to invoke python: `uv run` against the repo venv (the box has no
# pyproject.toml, so pin the interpreter explicitly). Override with
# PYTHON="/path/to/python" or PYTHON="uv run ... python".
if [[ -n "${PYTHON:-}" ]]; then
    read -r -a RUN <<< "$PYTHON"
else
    RUN=(uv run --no-project --python "$REPO/.venv/bin/python" python)
fi

INPUT="${INPUT:-$DATA/dataset_tts.jsonl}"
EVAL_OUT="${EVAL_OUT:-$DATA/dataset_eval.jsonl}"
VOTES_OUT="${VOTES_OUT:-$DATA/dataset_eval_votes.jsonl}"

GPUS="${GPUS:-0 1}"
read -r -a GPU_ARR <<< "$GPUS"
[[ ${#GPU_ARR[@]} -eq 2 ]] || { echo "GPUS must name exactly 2 devices (got '$GPUS')" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "input not found: $INPUT" >&2; exit 1; }

# vLLM workarounds for this box (see repo run.sh / reference_broken_gpu0_a6000_box).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

WORK="$DATA/.eval_2gpu"
rm -rf "$WORK"; mkdir -p "$WORK"

# --- split in half (exact line balance, first half gets the extra line) -------
n=$(wc -l < "$INPUT")
h=$(( (n + 1) / 2 ))
head -n "$h"            "$INPUT" > "$WORK/shard00.jsonl"
tail -n +"$((h + 1))"  "$INPUT" > "$WORK/shard01.jsonl"
echo ">> $INPUT : $n lines  ->  shard00=$(wc -l < "$WORK/shard00.jsonl")  shard01=$(wc -l < "$WORK/shard01.jsonl")"

# --- launch one worker per GPU ----------------------------------------------
pids=()
for i in 0 1; do
    g="${GPU_ARR[$i]}"
    echo ">> GPU $g  <-  $WORK/shard0$i.jsonl   (log: $WORK/log0$i.txt)"
    CUDA_VISIBLE_DEVICES="$g" "${RUN[@]}" "$HERE/single_pipeline.py" \
        --input        "$WORK/shard0$i.jsonl" \
        --output       "$WORK/eval0$i.jsonl" \
        --temperature  "$TEMPERATURE" \
        --votes-output "$WORK/votes0$i.jsonl" \
        "$@" > "$WORK/log0$i.txt" 2>&1 &
    pids+=("$!")
done

tail -n +1 -F "$WORK/log00.txt" "$WORK/log01.txt" & tail_pid=$!

status=0
for i in 0 1; do
    rc=0; wait "${pids[$i]}" || rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!! worker $i (GPU ${GPU_ARR[$i]}) exited $rc  --  see $WORK/log0$i.txt" >&2
        status=$rc
    fi
done
kill "$tail_pid" 2>/dev/null || true
wait "$tail_pid" 2>/dev/null || true

[[ $status -eq 0 ]] || { echo "aborting: a worker failed (intermediate files kept in $WORK)" >&2; exit "$status"; }

# --- reconstruct the full outputs ------------------------------------------
mkdir -p "$(dirname "$EVAL_OUT")" "$(dirname "$VOTES_OUT")"
cat "$WORK/eval00.jsonl"  "$WORK/eval01.jsonl"  > "$EVAL_OUT"
cat "$WORK/votes00.jsonl" "$WORK/votes01.jsonl" > "$VOTES_OUT"

echo ">> done"
echo "   $EVAL_OUT   : $(wc -l < "$EVAL_OUT") records"
echo "   $VOTES_OUT  : $(wc -l < "$VOTES_OUT") vote records"
echo "   intermediate files in $WORK (safe to delete)"
echo ">> agreement stats:  ${RUN[*]} $HERE/agreement.py --votes $VOTES_OUT"
