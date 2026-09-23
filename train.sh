#!/usr/bin/env bash

set -euo pipefail
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python sft.py
