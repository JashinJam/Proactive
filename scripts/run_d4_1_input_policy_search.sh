#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/quewenjun/miniconda3/envs/wearable_ai/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/d4_1_internvl35_1b_input_policy_search_v1.json}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/20260720_internvl35_1b_d4_1_input_policy_search_v1}"
NUM_GPUS="${NUM_GPUS:-4}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "D4.1 Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

args=(
  -m proactive_d4_1.run_search
  --config "${CONFIG}"
  --experiment-dir "${EXPERIMENT_DIR}"
  --num-gpus "${NUM_GPUS}"
)

if [[ -n "${GPU_IDS:-}" ]]; then
  args+=(--gpu-ids "${GPU_IDS}")
fi
if [[ -n "${NUM_SHARDS:-}" ]]; then
  args+=(--num-shards "${NUM_SHARDS}")
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  args+=(--model-path "${MODEL_PATH}")
fi
if [[ -n "${INPUT_JSONL:-}" ]]; then
  args+=(--input-jsonl "${INPUT_JSONL}")
fi
if [[ -n "${VIDEO_DIR:-}" ]]; then
  args+=(--video-dir "${VIDEO_DIR}")
fi
if [[ -n "${STARTER_KIT_DIR:-}" ]]; then
  args+=(--starter-kit-dir "${STARTER_KIT_DIR}")
fi
if [[ -n "${HEAD_PATH:-}" ]]; then
  args+=(--head-path "${HEAD_PATH}")
fi
if [[ -n "${MAX_TASK_ATTEMPTS:-}" ]]; then
  args+=(--max-task-attempts "${MAX_TASK_ATTEMPTS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi
if [[ "${SMOKE_ONLY}" == "1" ]]; then
  args+=(--smoke-only)
fi

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src"
exec "${PYTHON_BIN}" "${args[@]}"
