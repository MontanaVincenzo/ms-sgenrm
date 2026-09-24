#!/usr/bin/env bash
# Wrapper so the vLLM runtime workarounds for this box survive edits to the .py.
# See: reference_broken_gpu0_a6000_box in Claude memory.
set -euo pipefail
cd "$(dirname "$0")"

export VLLM_WORKER_MULTIPROC_METHOD=spawn      # engine-core child must not fork a live CUDA ctx
export VLLM_USE_FLASHINFER_SAMPLER=0           # no nvcc -> JIT sampler can't build
exec .venv/bin/python src/asr_stage.py
