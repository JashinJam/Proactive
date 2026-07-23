#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/quewenjun/miniconda3/envs/wearable_ai/bin/python}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/d6_internvl35_1b_query_memory_lora_oof_v1.json}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/output/experiments/20260722_internvl35_1b_d6_query_memory_lora_oof_v1}"
MAXIMUM_GPUS="${MAXIMUM_GPUS:-5}"
ZERO_INIT_SUMMARY="${EXPERIMENT_DIR}/smokes/zero_init/summary.json"
TRAINABILITY_SUMMARY="${EXPERIMENT_DIR}/smokes/rotation_0_trainability/summary.json"
LAUNCHER_DIR="${EXPERIMENT_DIR}/launcher"
LOG_PATH="${LAUNCHER_DIR}/supervisor.log"
LOCK_PATH="${LAUNCHER_DIR}/run.lock"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "D6 Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
for required_path in "${CONFIG}" "${ZERO_INIT_SUMMARY}" "${TRAINABILITY_SUMMARY}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "D6 required input is unavailable: ${required_path}" >&2
    exit 2
  fi
done
if ! [[ "${MAXIMUM_GPUS}" =~ ^[1-5]$ ]]; then
  echo "MAXIMUM_GPUS must be an integer from 1 through 5: ${MAXIMUM_GPUS}" >&2
  exit 2
fi

mkdir -p "${LAUNCHER_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "Another D6 launcher/evaluator script holds ${LOCK_PATH}" >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src"

{
  echo "[$(date -Is)] Starting D6 five-fold OOF launcher"
  "${PYTHON_BIN}" -m proactive_d6.launch_folds \
    --config "${CONFIG}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --zero-init-summary "${ZERO_INIT_SUMMARY}" \
    --trainability-summary "${TRAINABILITY_SUMMARY}" \
    --maximum-gpus "${MAXIMUM_GPUS}" \
    --allow-shared-gpus

  echo "[$(date -Is)] Five folds completed; starting frozen D6 evaluation"
  "${PYTHON_BIN}" -m proactive_d6.evaluate \
    --config "${CONFIG}" \
    --experiment-dir "${EXPERIMENT_DIR}"
  echo "[$(date -Is)] D6 evaluation completed"
} 2>&1 | tee -a "${LOG_PATH}"
