# D5 决策融合与 Causal Action-History OOF 报告

> 实验 ID：`20260721_internvl35_1b_decision_fusion_d5_oof_v1`
>
> 状态：冻结五折 OOF 与三组稳定性 split 全部完成；主候选未通过预注册晋级门槛
>
> 运行范围：700 sessions / 9,935 chunks；CPU-only；未运行模型推理；未使用人工评测
>
> 2026-07-22 口径更正：本实验运行时采用“隐藏输入沿用公开 chunk-aligned dialog”这一工作假设；公开数据和 starter runner 支持该契约，但当前正式规则没有明确保证隐藏字段或 assistant 历史来源。以下因果结论对公开输入成立，隐藏部署仍需组织方或 Docker 模板确认。

## 1. 结论

D5 得到一个真实但不够稳定的正信号：

- D4 被精确复现为官方 OOF Macro F1 `0.6846`；
- action-history 单独加入 D4 后达到 `0.6889`，提升 `+0.0043`；
- D3 full dynamics 单独加入 D4 后只有 `0.6853`，提升 `+0.0007`；
- 预注册主候选 `D4 + full dynamics + action history` 达到本轮最高的 `0.6912`，
  相对 D4 提升 `+0.0066`；
- 主候选 5/5 folds 为正、3/4 domains 为正，non-first Macro 从 `0.6596`
  提升到 `0.6676`；
- 但是 session-bootstrap 95% 区间为 `[-0.00025, +0.01356]`，下界仍略低于 0；
- 三组额外 session split 的同 split D5-D4 差值为 `+0.0072/-0.0007/+0.0016`，
  没有全部为正。

因此 D5 **不通过**预注册门槛，不进行 all-public full refit，不接入 D4 submission，
也不启动 GPU smoke。当前排行榜提交候选仍是 D4；D5 作为“有信号但 split-sensitive”
的自动实验保留。

## 2. 为什么做 D5

D3 与 D4 使用的不是同一类历史信息：

- D3 显式计算当前 hidden/tag margin 相对前一 chunk 和历史均值的变化；
- D4 使用当前 D1 fused 表征加 8 个官方 dialog-stage 标量；
- D4 没有包含 D3 的完整 dynamics；
- D3 与 D4 在冻结 OOF 上只有 `82.49%` 的逐 chunk 决策一致率。

在 9,935 个 chunk 中，D3/D4 同时正确 5,858 个，同时错误 2,337 个，只有 D4
正确 944 个，只有 D3 正确 796 个。这个互补性使“合并 D3 dynamics 与 D4 dialog”
成为一个合理但此前未执行的决策实验。

同时，本实验按隐藏输入沿用公开 chunk-aligned cumulative dialog 的工作假设设计。对当前 chunk `i`，比较
`dialog[i]` 与前一个可见 prefix 的 assistant 数量，可以恢复组织方在 `i-1` chunk
之后是否插入 assistant turn。该信号严格位于当前 chunk 之前，因此在官方输入契约内
因果可用。D5 在 D4 的粗粒度 stage 信息之外，加入了短期 prior-action pattern。

## 3. 预注册与信息边界

协议在任何 D5 拟合或指标读取前冻结：

```text
annotations/d5_decision_fusion_v1/PROTOCOL.md
SHA256 66d1c6344b9996ef4041654eac51f724c515d6195d68cbb4025aa405966a40f3
```

配置：`configs/d5_internvl35_1b_decision_fusion_oof_v1.json`。

特征构造顺序固定为：

```text
source rows
  -> strip_answers()
  -> D1 label-free scalar rows
  -> frozen D1 neural cache
  -> D4 dialog-stage
  -> D3 causal dynamics
  -> D5 causal prior-action history
  -> 最后才 attach public labels 供 fit/calibration/test scorer 使用
```

特征构造器拒绝含 `answers` 的输入，不读取任何 D1/D3/D4 prediction，也不读取未来
dialog 或未来 chunk。所有模型拟合仍属于 public-validation-supervised，不能描述成隐藏
测试泛化结果。

## 4. 冻结特征与变体

### 4.1 D4 基础矩阵

D4 基础矩阵保持原样，共 1,051 维：D1 fused 的 18 个 causal scalar、1 个 tag
margin、1,024 维 current hidden，以及 8 个 answer-stripped dialog-stage scalar。

### 4.2 D3 dynamics

D3 dynamics 原有 8 个 scalar 和 1,024 维 hidden delta。合入 D4 时，
`has_previous_chunk` 与 D4 同名同值，因此按预注册只保留 D4 版本，追加其余 7 个
dynamic scalar；full 变体再追加 1,024 维 `current_hidden - previous_hidden`。

### 4.3 D5 action-history

新增 18 个特征：

- lag-2/3/4 action 与 availability mask；
- 最近 2/4/8 actions 的 interrupt rate；
- 连续 interrupt/silent 长度；
- 距离最近 silent 的长度；
- 最近两个 actions 的 `II/IS/SI/SS` 模式；
- 最近 4/8 actions 的 transition rate。

离线构造与在线状态定义相同：处理 chunk `i` 的 dialog prefix 时，先把组织方刚刚暴露
的 `i-1` action 加入历史，再构造当前特征；从不把当前模型 prediction 写回历史。

### 4.4 五个冻结变体

| 变体 | 特征数 | 每 fold head 参数 |
|---|---:|---:|
| `d4_replay` | 1,051 | 1,052 |
| `d4_plus_dynamic_scalar` | 1,058 | 1,059 |
| `d4_plus_full_dynamics` | 2,082 | 2,083 |
| `d4_plus_action_history` | 1,069 | 1,070 |
| `d4_plus_full_dynamics_history` | 2,100 | 2,101 |

最后一个是唯一可晋级的预注册主候选，其余均为解释性控制。

## 5. 自动因果与复现检查

- D5 核心单元测试 `6/6` 通过；
- R0/D1/D3/D4 冻结回归 `48/48` 通过，总计 `54/54`；
- D4 replay predictions SHA256 精确复现为
  `467bd7c567d66b3041425b336343290c0b3d742f10d25d2279de987770ae43f5`；
- D4 replay metrics SHA256 精确复现为
  `3c1f8b81eeac385597e202763c17f0a6fc2ec61b875b19243059fcd19c1d21d1`；
- action-history lag-1 与 D4 `assistant_added_since_previous` 在全部 9,935 chunks
  逐值一致；
- 所有 OOF predictions 完成后再做 gold cross-check：9,235/9,235 non-first chunks
  的可见 previous action 与 previous gold interrupt 一致；
- 700 个 first chunks 的 action-history 全为 0；
- dialog/action-history 中无 malformed、empty 或 multi-add assistant turn；
- OOF 和 stability 全程未使用 GPU、人工评测或外部数据。

previous-gold 的 100% 一致只用于解释官方 dialog 构造策略，不参与特征生成。

## 6. 五折 OOF 结果

| 变体 | Macro F1 | vs D4 | Interrupt F1 | Silent F1 | Non-first Macro |
|---|---:|---:|---:|---:|---:|
| D4 replay | 0.6846 | 0.0000 | 0.6893 | 0.6799 | 0.6596 |
| + dynamic scalar | 0.6847 | +0.0001 | 0.6914 | 0.6780 | 0.6601 |
| + full dynamics | 0.6853 | +0.0007 | 0.6954 | 0.6753 | 0.6614 |
| + action history | 0.6889 | +0.0043 | 0.6955 | 0.6824 | 0.6647 |
| + full dynamics + action history | **0.6912** | **+0.0066** | **0.7007** | **0.6818** | **0.6676** |

主候选 prediction SHA256：

```text
db4bec8e538945669525246dd517043ded32618efa17e788dca7a7bf9d11a498
```

新增 action-history 是主要增益来源。完整 D3 dynamics 在 D4 之上单独只增加 `0.0007`；
在 action-history 基础上加入 full dynamics，使 `0.6889` 进一步到 `0.6912`，但这个
增量没有表现出足够的跨 split 稳定性。

## 7. 主候选错误变化

主候选相对 D4 改变了 1,456/9,935 个 decisions：

| 变化 | 数量 |
|---|---:|
| 修复 D4 FN | 402 |
| 修复 D4 FP | 360 |
| 新增 FN | 290 |
| 新增 FP | 404 |
| 两者均正确且不变 | 6,108 |
| 两者均错误且不变 | 2,371 |

总计修复 762 个错误，同时引入 694 个错误，净增加 68 个正确 decision。按 session
统计，244 个 session 变好，218 个变差，238 个完全持平。正负 session 数量接近，
与 bootstrap 下界贴近 0 的结果一致。

## 8. Fold、Domain 与位置分层

五个固定 test folds 的 Macro 增益：

```text
fold 0  +0.00021
fold 1  +0.00364
fold 2  +0.01414
fold 3  +0.00717
fold 4  +0.00840
```

虽然 5/5 都为正，但 fold 0 几乎没有增益，fold 2 明显更高，提示 split 敏感性。

Domain 增益：

```text
Arts and Crafts  +0.00600
Chef             -0.00205
Handyman         +0.01019
Tutorial         +0.01503
```

Chunk 位置增益：

```text
first    +0.00000
second   +0.01415
2-4      +0.01334
5-9      +0.01824
10+      -0.00768
```

D5 的收益集中在 second 到 5--9 chunks，10+ chunks 反而下降。它没有依赖 first-chunk
约定，但也没有形成全 session 阶段一致的改进。

## 9. Session Bootstrap

主候选相对 D4 的 5,000 次 paired session bootstrap：

```text
median            +0.00672
95% lower bound   -0.00025
95% upper bound   +0.01356
positive fraction  0.9692
```

绝大多数 bootstrap 样本为正，但预注册要求 lower bound 严格大于 0。这里不能因为只差
`0.00025` 就事后放宽门槛。action-history-only 的区间同样跨 0：
`[-0.00061,+0.00947]`，因此不能事后将较小模型改成正式主候选。

## 10. 三组额外稳定性 Split

每组都重新进行 label-independent domain-stratified session 分配，并在同一 split 内分别
重训 D4 和 D5：

| Split | D4 | D5 | D5-D4 | Bootstrap 95% 区间 |
|---|---:|---:|---:|---:|
| A | 0.6915 | 0.6987 | +0.0072 | [-0.00039,+0.01472] |
| B | 0.6888 | 0.6881 | **-0.0007** | [-0.00824,+0.00690] |
| C | 0.6964 | 0.6980 | +0.0016 | [-0.00592,+0.00900] |

绝对分数随 split 变化，所以这里只解释同 split 差值。B 为负已经直接使“全部 split
为正”门槛失败；三组区间也都跨 0。

## 11. 结构冗余审计

自动逐列比较发现三对完全相同的结构列：

```text
assistant_added_since_previous
  == assistant_add_count_since_previous

log1p_chunks_since_assistant_addition
  == action_log1p_consecutive_silents

action_log1p_consecutive_interrupts
  == action_log1p_chunks_since_silent
```

第一对来自原 D4，因为本数据没有 multi-add chunk；后两对来自 D5 action-history 定义。
这些重复列不会造成 target leakage，但在线性 L2 模型中会改变对应信号的有效正则强度。
因此不能把 D5 的 `+0.0066` 全部解释为新历史模式，也不能在结果之后删除重复列再重跑
并挑选更高者。该问题与 split 不稳定一起支持“不晋级”的决定。

## 12. 晋级门槛判定

| 条件 | 结果 |
|---|---|
| Macro 增益至少 +0.005 | 通过，+0.0066 |
| Bootstrap lower bound > 0 | **失败，-0.00025** |
| 至少 4/5 positive folds | 通过，5/5 |
| 至少 3/4 positive domains | 通过，3/4 |
| Non-first Macro 提升 | 通过 |
| Interrupt/Silent F1 均至少 0.67 | 通过，0.7007/0.6818 |
| 三组 stability split 全为正 | **失败，B=-0.0007** |

最终：`promotion_passed = false`。

## 13. 运行资源与产物

| 项目 | 结果 |
|---|---:|
| Sessions / chunks | 700 / 9,935 |
| 固定变体 | 5 |
| Stability splits | 3 |
| 最大 head 参数 | 2,101 |
| Wall time | 880.678 秒 |
| GPU / 模型 inference | 未使用 / 未重跑 |
| 人工评测 / 外部上传 | 未使用 / 未执行 |

完整产物：

```text
output/experiments/20260721_internvl35_1b_decision_fusion_d5_oof_v1/
```

关键入口包括 `comparison.json`、`feature_audit.json`、`runtime.json`、
`variants/*/{predictions,metrics,diagnostics}` 和 `stability/*/comparison.json`。

## 14. 后续路线

1. D4 继续保持当前可部署排行榜候选，bundle、head、manifest 和 submission adapter
   均不修改。
2. 冻结 D5 v1，不做 all-public refit，不做 GPU smoke，不在同一协议下删列、改窗口或
   调 L2/threshold。
3. action-history 有方向性，但主要问题是 split/threshold 稳定性。下一阶段应独立预注册
   结构化 calibration 或低维 stage policy，而不是继续扩张 2,100 维线性特征。
4. 下一实验必须将完全重复列预先合并，并把“表示增益”和“有效正则变化”分开控制。
5. 可优先研究按 dialog stage 收缩的 threshold 或只在低维 causal scalar 上使用浅层
   结构模型；仍使用自动 official scorer、session OOF 和多 split 稳定性，不恢复人工评测。
6. 同学的 frame/history 输入策略到达后，应先在 D4 上独立确认，再决定是否重新生成
   neural cache；不要与新的 calibration 方法同时改变。

D5 最重要的结论不是“0.6912 已经替代 D4”，而是：官方 prior-action history 确实包含
额外决策信号，但当前融合方式的收益尚未稳定到足以成为单一隐藏测试模型。
