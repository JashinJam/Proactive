# D6 Query Memory + Late Attention LoRA

本包实现 `annotations/d6_query_memory_lora_oof_v1/PROTOCOL.md` 中冻结的唯一 D6
候选。正式执行只能使用原 D4.2 session manifest，不得改结构或训练参数。

## 入口

所有命令应从项目根目录运行，并使用包含 PyTorch/Transformers 的 challenge conda
环境。默认 GPU 入口会在模型加载前拒绝有外部 compute process 或空闲显存不足
75 GiB 的卡。用户授权的共享执行必须显式传入 `--allow-shared-gpu`；详见
`annotations/d6_query_memory_lora_oof_v1/RESOURCE_AMENDMENT_20260722.md`。

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
PY=/home/quewenjun/miniconda3/envs/wearable_ai/bin/python
EXP=output/experiments/20260722_internvl35_1b_d6_query_memory_lora_oof_v1

CUDA_VISIBLE_DEVICES=6 "$PY" -m proactive_d6.run_zero_init_smoke \
  --config configs/d6_internvl35_1b_query_memory_lora_oof_v1.json \
  --output-dir "$EXP/smokes/zero_init" --device cuda:0

CUDA_VISIBLE_DEVICES=1 "$PY" -m proactive_d6.run_fold \
  --config configs/d6_internvl35_1b_query_memory_lora_oof_v1.json \
  --output-dir "$EXP/smokes/rotation_0_trainability" \
  --device cuda:0 --fold 0 --trainability-smoke --allow-shared-gpu
```

两个 smoke 的 `summary.json` 全部门通过后，每折默认使用一张独占卡。共享执行在
下列命令末尾增加 `--allow-shared-gpu`：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID "$PY" -m proactive_d6.run_fold \
  --config configs/d6_internvl35_1b_query_memory_lora_oof_v1.json \
  --output-dir "$EXP/folds/fold_FOLD" --device cuda:0 --fold FOLD --formal \
  --zero-init-summary "$EXP/smokes/zero_init/summary.json" \
  --trainability-summary "$EXP/smokes/rotation_0_trainability/summary.json"
```

也可让冻结 launcher 自动选择最多五张合格 A800 并阻塞到五折结束：

```bash
"$PY" -m proactive_d6.launch_folds \
  --config configs/d6_internvl35_1b_query_memory_lora_oof_v1.json \
  --experiment-dir "$EXP" \
  --zero-init-summary "$EXP/smokes/zero_init/summary.json" \
  --trainability-summary "$EXP/smokes/rotation_0_trainability/summary.json" \
  --maximum-gpus 5 --allow-shared-gpus
```

单折训练 checkpoint 只含 adapter/optimizer/PRNG/session-boundary 状态；特征记录按
session 原子追加。相同命令重启会校验配置、模型与数据 hash，并从已落盘边界恢复。

五折完成后运行：

```bash
"$PY" -m proactive_d6.evaluate \
  --config configs/d6_internvl35_1b_query_memory_lora_oof_v1.json \
  --experiment-dir "$EXP"
```

汇总器只合并带 `SENTINEL_UNSEALED` 的冻结 test 预测，随后才附加 gold 并运行官方
scorer、paired-session bootstrap 和全部预注册门。任何门失败均终止该结构族；本包
不会上传 leaderboard 或外部 registry。

只有当 `evaluation/summary.json` 的全部门通过时，以下唯一 refit 才会运行；否则
入口直接拒绝。它固定使用五折中位 best epoch、L2 和 threshold，随后执行独立
102-chunk 在线重放审计：

```bash
CUDA_VISIBLE_DEVICES=GPU_ID "$PY" -m proactive_d6.refit \
  --config configs/d6_internvl35_1b_query_memory_lora_oof_v1.json \
  --experiment-dir "$EXP" --output-dir "$EXP/final_refit" --device cuda:0 \
  --allow-shared-gpu
```
