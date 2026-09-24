#!/usr/bin/env bash

set -euo pipefail
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 reward_model/.venv/bin/python sft.py
