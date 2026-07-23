# C1 Small D6 可训练性与资源 GPU 审计

## 结论

D6 rotation-0 完整一轮 trainability/resource smoke 通过正式五折前置硬门。
该 smoke 只验证冻结架构能够更新、checkpoint 能在 session 边界确定性恢复，
以及正式单折资源预算可行；它不形成、比较或选择 efficacy 结论。

在用户授权的 shared-GPU 资源修订下，训练从 session-boundary checkpoint
跨 GPU 恢复。原协议和 canonical config 均保持字节不变，架构、优化器、损失、
数据划分和评估门槛没有调整。

## 可训练性结果

- epoch-0 calibration BCE：`3.8371416557`。
- epoch-1 calibration BCE：`0.6612109072`；best epoch 为 `1`。
- epoch-1 fit：418 sessions、6,003 chunks；calibration 为 141 sessions、
  1,922 chunks。
- 48/48 adapter tensors 发生变化，合计 L2 delta 为 `4.08935`，最大绝对
  delta 为 `0.022626`。
- 48/48 optimizer moment tensors 非零，最大 optimizer step 为 `85`。
- calibration memory residual norm 均值为 `16.9996`，归一化 attention
  entropy 均值为 `0.48122`。

## 资源与恢复审计

- peak allocated memory：`7.1092 GiB`，低于 `70 GiB` 门。
- maximum session model time：`26.8671s`，低于 `240s` smoke 门。
- 包含已完成恢复段的累计训练审计墙钟：`23,846.88s`。
- 按冻结完整单折工作量线性估算：`37.3365h`，低于 `48h` 门。
- 恢复只发生在已保存的 session 边界；参数不会在同一 session 内因迁移而改变。
- 两次迁移只终止本实验自己的进程，未终止、迁移或修改外部进程。

五项机器门均为 `true`：显存、正式单折时间、最长 session 时间、adapter-only
checkpoint，以及非零 optimizer 更新。最佳 adapter checkpoint 不包含基础模型
权重。

## 冻结指纹

- protocol SHA256：
  `eda95ccca756966d57b59a2ce4b21bc71699dd51acd63b096e0b007c5d4ae4d7`。
- canonical config SHA256：
  `ecbe9b3cd187cfdf4793f0001e9b22a6905f1f1c44d1a981ebf5d5835c39d2e1`。
- config file SHA256：
  `e381a9e825e0878072a0c56b4eea0741a76e998b4eae4341673ac7710aef7633`。
- shared-GPU resource amendment SHA256：
  `7b43ffcfd8400accd5a423c9e14d743694e37f0bef5c808930616b8f1f4c6883`。
- best adapter SHA256：
  `10b658014253a9cbec41a747d8d82851144b7990f3df0c581ada75b53ac3f43c`。
- training checkpoint SHA256：
  `56a792f01a34c7f737570c73fceef014ada0eeecc9397f0a07b93ad04e796b03`。
- summary SHA256：
  `2aca7d6a88df9a207adfb1f258dedd49b28261fb00b349f9d8bb0479e122a00b`。

机器可读产物位于：

```text
output/experiments/20260722_internvl35_1b_d6_query_memory_lora_oof_v1/
  smokes/rotation_0_trainability/
```

## 证据边界与后续状态

本审计没有读取 test-fold labels 来形成候选选择，也不报告 Macro F1。正式
五折 OOF 已按冻结 D4.2 manifest 启动；在五折合并和官方评分完成前，D6 没有
efficacy 结果，D4.2 `history8` 仍是活动 leaderboard-engineering 候选。
所有未来 D6 指标仍只能称为 post-selection、val-supervised OOF，不是
hidden-test 或独立泛化证据。外部上传未授权。
