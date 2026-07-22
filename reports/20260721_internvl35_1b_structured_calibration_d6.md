# D6 低维结构化门槛校准 OOF 报告

> 实验 ID：`20260721_internvl35_1b_structured_calibration_d6_oof_v1`  
> 日期：2026-07-21  
> 状态：冻结五折 OOF、三组新稳定性 split 和独立复现全部完成；主候选明确失败  
> 性质：public-validation-supervised leaderboard diagnostic，不是隐藏测试证据

## 1. 结论

D6 回答了 D5 留下的一个具体问题：近期 action-history 是否不需要进入 2,100 维融合
head，只需按少量历史状态调整 D4 的 interrupt threshold 就能稳定获益？答案是否定的。

- D4 再次逐字节复现为 Macro F1 `0.6846`；
- 唯一轻微正向的控制是按 chunk 位置收缩门槛，达到 `0.6855`，仅 `+0.0009`，bootstrap
  区间 `[-0.00221,+0.00398]` 跨 0；
- 只按上一 action 收缩门槛降到 `0.6778`；
- 按最近两次 action 独立校准、不收缩时降到 `0.6633`；
- 预注册主候选 `last2_shrunk` 虽用固定规则向全局门槛收缩，仍降到 `0.6747`，相对 D4
  为 `-0.0099`；
- 主候选五个 fixed folds、四个 domains 和三组新 stability splits 全部为负。

因此 D6 不晋级，不做门槛 transport、部署集成或 GPU smoke。D4 继续是唯一冻结的
leaderboard-engineering submission candidate；D5/D6 的 action-history 搜索在当前公共数据上
停止，不根据结果改分组、64 条下限、256 收缩常数或局部门槛目标后重跑。

## 2. 为什么做 D6

D5 的完整融合达到 `0.6912`，但 bootstrap 和额外 split 不稳定；同时 2,100 维矩阵中有
三对完全重复列，无法把全部增益解释成新历史信息。D6 因而采用更窄的检验：

1. 每折仍训练和选择完全相同的 D4 head；
2. 所有 D6 变体共用同一个 D4 model、L2 和 logits；
3. 候选只改变 calibration fold 上得到的少量 threshold；
4. 每行候选输入只有一个 D4 logit 和一个互斥阶段名，不追加任何数值特征列。

这样，D6 与 D4 的差异不可能来自重复列改变线性 head 的有效正则强度。

## 3. 冻结分组与收缩规则

协议在任何 D6 拟合或指标读取前冻结：

```text
annotations/d6_structured_calibration_v1/PROTOCOL.md
SHA256 f8d9a1e91fdb2c1a669b9983f0471b6ca8e70721400607bd781493238960389d
```

五个固定变体为：

| 变体 | 分组 | 是否收缩 | 角色 |
|---|---|---|---|
| `d4_global_replay` | 全局单门槛 | - | 精确基线 |
| `position_shrunk` | first/second/2-4/5-9/10+ | 是 | 位置控制 |
| `last_action_shrunk` | previous interrupt/silent | 是 | 一阶 action 控制 |
| `last2_unshrunk` | second/II/IS/SI/SS | 否 | 过拟合控制 |
| `last2_shrunk` | second/II/IS/SI/SS | 是 | 唯一主候选 |

first chunk 永远使用 D4 全局门槛。其他组至少需要 calibration fold 中 64 条且两类标签都
存在。局部门槛按该组较少类别的有效样本量向全局门槛收缩：

```text
effective_n = 2 * min(interrupt_count, silent_count)
weight = effective_n / (effective_n + 256)
applied = global + weight * (local - global)
```

所有常数、分组和 tie break 都在运行前冻结；test-fold labels 不参与 threshold 或权重。

## 4. 信息边界与工程检查

- 700 sessions / 9,935 chunks 完整覆盖；
- stage builder 默认拒绝含 `answers` 的输入；
- stage 只来自当前可见的累计 dialog prefix，不读取 future dialog/chunk；
- 9,235 个 non-first chunks 的 visible previous action 在事后解释性核查中与 previous gold
  action 全部一致，但 gold 不参与构造；
- action stage 数量为 first/second/II/IS/SI/SS =
  `700/700/1919/2635/2499/1482`；
- malformed、empty、multi-add assistant turn 均为 0；
- D6 6 项测试与 R0/D1/D3/D4 48 项冻结回归共 `54/54` 通过；
- D4 prediction 和 metric hashes 均精确复现；
- 全程 CPU-only，未重跑模型 inference，未使用 GPU、人评、外部数据或外部上传。

## 5. 五折 OOF 结果

| 变体 | Macro F1 | vs D4 | Interrupt F1 | Silent F1 | Non-first Macro |
|---|---:|---:|---:|---:|---:|
| D4 global replay | 0.6846 | 0 | 0.6893 | 0.6799 | 0.6596 |
| Position shrunk | **0.6855** | **+0.0009** | 0.6948 | 0.6761 | 0.6614 |
| Last action shrunk | 0.6778 | -0.0068 | 0.6867 | 0.6688 | 0.6530 |
| Last two unshrunk | 0.6633 | -0.0213 | 0.6819 | 0.6447 | 0.6389 |
| Last two shrunk | 0.6747 | -0.0099 | 0.6857 | 0.6637 | 0.6500 |

收缩把 last-two 的损失从 `-0.0213` 缩小到 `-0.0099`，证明它确实抑制了部分过拟合，
但不足以恢复 D4。主候选 bootstrap 95% 区间为
`[-0.01482,-0.00492]`，全部位于 0 以下。

Position control 修复 127 个 D4 错误，同时新增 116 个错误，净增加 11 个正确 decision；
但它只有 3/5 folds、2/4 domains 为正，增益远低于 `+0.005` 门槛，不能在看到结果后改成
正式主候选。

## 6. 主候选为何失败

`last2_shrunk` 修复 246 个 D4 错误，却新增 341 个错误，净减少 95 个正确 decision。
相对 D4，它增加 47 个 TP，但也增加 142 个 FP；即更积极地 interrupt，却损失了更多
silent 判断。135 个 session 变好、191 个变差、374 个持平。

五折增益全部为负：

```text
fold 0  -0.01595
fold 1  -0.00983
fold 2  -0.00941
fold 3  -0.00455
fold 4  -0.01113
```

四域同样全部为负：Arts and Crafts `-0.02334`、Chef `-0.00206`、Handyman
`-0.01036`、Tutorial `-0.00589`。除 second `+0.00574` 外，2--4、5--9 和 10+
分别下降 `-0.00795/-0.00902/-0.01709`。

一个关键诊断是：在 II/IS/SI/SS 各组内部单独计算 Macro F1 时，每组都显示正增益，
但合并后的全局 Macro F1 明显下降。Macro F1 不能按分组相加；分别优化每组的局部 Macro
会改变全局 TP/FP/TN/FN 的平衡，尤其会把更多 silent 推成 interrupt。这说明当前
“各组独立选门槛再拼接”的目标与最终排行榜指标不一致，而不是继续微调收缩强度就能可靠
修复的问题。

## 7. 三组新稳定性 Split

| Split | D4 | D6 primary | D6-D4 | Bootstrap 95% 区间 |
|---|---:|---:|---:|---:|
| A | 0.6939 | 0.6859 | -0.0080 | [-0.01272,-0.00310] |
| B | 0.6908 | 0.6889 | -0.0019 | [-0.00629,+0.00258] |
| C | 0.6827 | 0.6761 | -0.0066 | [-0.01199,-0.00096] |

三组均为负，A/C 的完整区间也低于 0。与 D5 的 split-sensitive 小正信号相比，D6 是跨
split 一致的负结果，因此无需进行任何部署阶段工作。

## 8. 晋级门槛

| 条件 | 结果 |
|---|---|
| Macro 增益至少 +0.005 | 失败，-0.0099 |
| Bootstrap lower bound > 0 | 失败，-0.01482 |
| 至少 4/5 positive folds | 失败，0/5 |
| 至少 3/4 positive domains | 失败，0/4 |
| Non-first Macro 提升 | 失败 |
| Interrupt/Silent F1 均至少 0.67 | 失败，Silent 0.6637 |
| 三组 stability split 全为正 | 失败，0/3 |

最终：`promotion_passed = false`。

## 9. 复现与工件

正式运行 wall time `444.427s`。独立第二次运行使用不同输出目录；主比较、stage audit、
五个预测文件、主候选 diagnostics 和三组 stability comparison 均逐字节一致。

| Artifact | SHA256 |
|---|---|
| Protocol | `f8d9a1e91fdb2c1a669b9983f0471b6ca8e70721400607bd781493238960389d` |
| Config | `fd0165db3d4e50deea6a38b0ce74b80100142ad85d9912d6f71c63e45275a285` |
| `comparison.json` | `572ce2d1659fbe519fd7aed678bea32f116cf300101d34929a406d7f47a29a27` |
| `stage_audit.json` | `a8462b5cabe09112387ac381cb9528618d458f029a9db6bdef55949ab3ef1a01` |
| D4 replay predictions | `467bd7c567d66b3041425b336343290c0b3d742f10d25d2279de987770ae43f5` |
| Position predictions | `8bcb2d2f7d5a902ff777dbe863bc514e3c1a3bfa1d1ff1ee211fe0073d3954ea` |
| Primary predictions | `4d18da82156b387839037df116433aa97fc918b8bbab9dd0b9bf20645e254955` |

完整产物：

```text
output/experiments/20260721_internvl35_1b_structured_calibration_d6_oof_v1/
```

## 10. 路线决定

1. D6 v1 冻结，不做 threshold transport、all-public 调整、adapter 集成或 GPU smoke。
2. D4 bundle、head、manifest、threshold 和 submission adapter 全部保持不变。
3. 不在相同公共数据上继续搜索 action window、stage bins、收缩常数或局部门槛目标；
   position `+0.0009` 也不事后晋级。
4. D5 表明 action history 作为联合表示可能有信号，D6 表明按组独立校准不适合全局 Macro；
   若未来有独立证据恢复这一方向，必须直接针对全局指标并重新预注册，不能沿用本轮结果调参。
5. 当前可执行主线转为：接收并独立验证同学的 frame/history 输入策略；官方 Docker 模板
   发布后完成 D4 容器适配、资源/许可证审计和授权后的外部提交。
6. 新的人评、utterance/state 训练、S1、granularity 和 GRPO 继续暂停。
