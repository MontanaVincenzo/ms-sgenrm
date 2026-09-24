#!/usr/bin/env bash
# Build the project's three uv virtual environments from their frozen requirements.
#
#   data_generation   data_generation/.venv                               (py3.12, torch cu132 + vllm-omni)
#   orpheus           data_generation/01_input_request/02_tts_synthesis/.venv   (py3.11, vllm 0.7.3 for Orpheus-TTS)
#   training          reward_model/.venv                                  (py3.12, reward-model SFT/eval)
#
# Usage:
#   ./setup_envs.sh                     # all three
#   ./setup_envs.sh orpheus training    # a subset
#   FORCE=1 ./setup_envs.sh             # recreate venvs that already exist
#   VLLM_OMNI_SRC=/path/to/vllm-omni ./setup_envs.sh data_generation
#
# vllm-omni is installed editable from a local checkout (it carries local patches), not from PyPI.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_SRC="${VLLM_OMNI_SRC:-/home/vmontana/synthetic_data_generation/vllm-omni}"
FORCE="${FORCE:-0}"

# name -> "python_version|requirements file|venv dir|extra uv pip args"
declare -A ENVS=(
  # --no-deps: this freeze is complete but not self-consistent (numpy 2.5.2 vs mistral-common's <2.4), so it is
  # installed verbatim instead of resolved.
  [data_generation]="3.12|data_generation/data_generation_requirements.txt|data_generation/.venv|--no-deps --extra-index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match"
  [orpheus]="3.11|data_generation/orpheus_requirements.txt|data_generation/01_input_request/02_tts_synthesis/.venv|"
  [training]="3.12|reward_model/training_requirements.txt|reward_model/.venv|"
)
ORDER=(data_generation orpheus training)

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }

build_env() {
  local name="$1"
  IFS='|' read -r py req venv extra <<<"${ENVS[$name]}"
  req="$ROOT/$req"
  venv="$ROOT/$venv"

  echo "=== [$name] python $py -> ${venv#"$ROOT"/}"
  if [[ -d "$venv" && "$FORCE" != 1 ]]; then
    echo "    already exists, skipping (FORCE=1 to recreate)"
    return
  fi
  uv venv --clear --python "$py" --prompt "$name" "$venv"

  # Local editable installs (-e file:///...) are machine-specific: drop them here, handle below.
  local tmp_req
  tmp_req="$(mktemp)"
  grep -vE '^-e (file://)?/' "$req" >"$tmp_req"
  # shellcheck disable=SC2086  # $extra is intentionally word-split into separate flags
  uv pip install --python "$venv/bin/python" $extra -r "$tmp_req"
  rm -f "$tmp_req"

  if [[ "$name" == data_generation ]]; then
    if [[ -d "$VLLM_OMNI_SRC" ]]; then
      uv pip install --python "$venv/bin/python" --no-deps -e "$VLLM_OMNI_SRC"
    else
      echo "    WARNING: vllm-omni checkout not found at $VLLM_OMNI_SRC; set VLLM_OMNI_SRC and rerun with FORCE=1" >&2
    fi
  fi
  if [[ "$name" == orpheus ]]; then
    # Local copy of the orpheus_tts package (importable as `orpheus_tts`)
    uv pip install --python "$venv/bin/python" --no-deps -e "$ROOT/data_generation/01_input_request/02_tts_synthesis/Orpheus-TTS/orpheus_tts_pypi"
  fi
  echo "    done"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=("${ORDER[@]}")
for name in "${targets[@]}"; do
  [[ -n "${ENVS[$name]+x}" ]] || { echo "unknown env '$name' (choose from: ${ORDER[*]})" >&2; exit 1; }
  build_env "$name"
done
