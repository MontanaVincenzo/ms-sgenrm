#!/usr/bin/env bash
# Wrapper so the vLLM runtime workarounds for this box survive edits to the .py.
# Extra flags are forwarded to asr_stage.py. Override the config with CONFIG=/path/to/config.yaml.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

export VLLM_WORKER_MULTIPROC_METHOD=spawn      # engine-core child must not fork a live CUDA ctx
export VLLM_USE_FLASHINFER_SAMPLER=0           # no nvcc -> JIT sampler can't build
exec "$REPO/data_generation/.venv/bin/python" "$HERE/asr_stage.py" --config_path "${CONFIG:-$REPO/config.yaml}" "$@"
