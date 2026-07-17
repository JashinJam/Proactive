#!/usr/bin/env bash
set -euo pipefail

cd /home/lanjinxin/workspace/wearable_ai_challenge

python_bin=/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python
config=configs/d1_internvl35_1b_neural_features.json
base=output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1

pids=()
for shard in 0 1 2 3; do
  output_dir="${base}_shard${shard}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing shard directory: ${output_dir}" >&2
    exit 1
  fi
  env CUDA_VISIBLE_DEVICES="${shard}" PYTHONNOUSERSITE=1 PYTHONPATH=src \
    "${python_bin}" -m proactive_d1.extract_neural \
    --config "${config}" \
    --device cuda:0 \
    --num-shards 4 \
    --shard-index "${shard}" \
    --output-dir "${output_dir}" \
    >"/tmp/d1_neural_shard${shard}.launcher.log" 2>&1 &
  pids+=("$!")
  echo "started shard=${shard} physical_gpu=${shard} pid=${pids[-1]}"
done

status=0
for shard in 0 1 2 3; do
  if wait "${pids[$shard]}"; then
    echo "completed shard=${shard}"
  else
    echo "failed shard=${shard}; see /tmp/d1_neural_shard${shard}.launcher.log" >&2
    status=1
  fi
done
exit "${status}"
