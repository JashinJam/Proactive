# D4 Dialog-Stage 排行榜候选：全量 Refit 与在线部署报告

## 1. 结论

D4 已完成从冻结候选到在线 GPU 推理的完整工程闭环：

- 唯一特征集：D1 fused 的 1,043 维 + 8 个 answer-stripped dialog-stage 标量；
- 公共 validation 五折 OOF 参考：Macro F1 `0.6846`；
- 最终 head：1,051 features / 1,052 parameters；
- 总系统参数：1,060,898,844，约 `1.060899B`，满足 C1 Small `<2B`；
- 全量 700 sessions / 9,935 chunks refit 已序列化；
- 离线/在线 dialog 特征逐值完全一致，9,935/9,935 decisions 一致；
- shared-vision GPU smoke 的 10/10 raw response、prompt、tag margin、hidden、dialog
  features、decision 和 answer 全部与冻结工件一致。

D4 现在是当前最强的**排行榜工程候选**。它不是新的科学晋级结果：来源 D3-D 协议明确
规定所有变体不可晋级，`0.6846` 也只是在公共 validation 上的诊断 OOF 证据。

## 2. 为什么建立 D4

D3-D 发现，官方 dialog prefix 会暴露 previous gold action：非首 chunks 中，是否新增
assistant turn 与上一 gold interrupt 在 9,235/9,235 条上完全一致。只用 8 个
dialog-stage 标量就达到 `0.6618`；与 D1 fused 合并后达到 `0.6846`，超过 D3
`0.6690`。

由于本项目优先追排行榜，但不能事后修改实验身份，因此采取两层结论：

1. D3 `0.6690` 保持正式晋级的科学/部署 baseline；
2. 把唯一固定的 8 特征方案作为 D4 leaderboard-engineering candidate 做一次完整 refit
   和在线验证，不继续搜索其他 dialog 特征。

## 3. 冻结模型与特征

Backbone 为 `OpenGVLab/InternVL3_5-1B-HF`：

| 项目 | 数值 |
|---|---:|
| Vision parameters | 304,012,288 |
| Language parameters | 751,632,384 |
| Projector parameters | 5,253,120 |
| Backbone total | 1,060,897,792 |
| D4 head | 1,052 |
| System total/active | 1,060,898,844 |

基础 D1 fused 特征：

- 18 个 strictly causal `response_temporal` 标量；
- 1 个 tag margin；
- 1,024 维当前 causal hidden。

冻结追加的 8 个 dialog-stage 标量：

```text
has_previous_chunk
assistant_added_since_previous
assistant_add_count_since_previous
log1p_visible_assistant_turns
assistant_turns_per_elapsed_chunk
log1p_chunks_since_assistant_addition
log1p_last_assistant_text_length
last_assistant_has_interrupt_tag
```

这些特征只从当前可见的官方 `dialog[i]` 和此前 prefix 构造；构造前删除 `answers`，
不读取 D1/D3/D4 prediction，也不读取未来 dialog。

## 4. Full-Development Refit

五个 OOF folds 都选择 `L2=0.01`。最终规则在运行前冻结为：

```text
L2        = median(0.01, 0.01, 0.01, 0.01, 0.01) = 0.01
threshold = median(
  -0.0785817419690357,
   0.21264921000590017,
   0.11188985852786787,
   0.15641524265203316,
   0.1263874797442615
) = 0.1263874797442615
```

然后在全部 700 public-development sessions / 9,935 chunks 上拟合一次 class-balanced
float64 LBFGS linear head。运行不使用 GPU，耗时 8.557 秒。

| 项目 | 结果 |
|---|---:|
| Head features / params | 1,051 / 1,052 |
| Head SHA256 | `531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13` |
| Predicted interrupt rate | 43.96% |
| Full-fit Macro F1 | 0.7393 |
| Full-fit interrupt/silent F1 | 0.7336 / 0.7450 |

`0.7393` 是在全部训练标签上的 train-fit sanity，不是开发泛化成绩，也不能用于与 OOF
`0.6846` 或未来 leaderboard 分数直接比较。Full-fit predictions SHA256 为：

```text
c2be105287e495212f192733266e0b750a1647e01d33b13860c5db03533eb0ad
```

## 5. 在线 Dialog 状态验证

在线状态每个 session 独立初始化，逐 chunk 消费官方 `dialog[i]`，维护：

- 上一个 prefix 的 assistant count；
- 距最近一次 assistant 新增经过的 chunks；
- 预期下一个 chunk index。

在冻结 cache 上完整回放 9,935 chunks：

| 检查 | 结果 |
|---|---:|
| Dialog feature 最大绝对差 | 0.0 |
| Decision exact | 9,935 / 9,935 |
| Logit 最大绝对差 | `2.5469e-7` |
| 冻结容差 | `1e-6` |

这证明离线训练特征与在线 prefix state 的定义一致，没有使用 future session 信息。

## 6. Shared-Vision GPU Smoke

运行前 GPU 状态：物理 GPU 0 有约 22 GB 已有进程，GPU 1--7 空闲。Smoke 选择物理
GPU 1，并启用 exclusive 检查；没有停止或修改任何已有进程。

范围为原始 session 0 的 10 chunks，沿用：

- 每 interval 16 帧，累计最多 32 帧；
- 最近 4 turns；
- R0 greedy generation；
- shared-vision current hidden/tag feature extraction；
- BF16 + SDPA；
- D4 online dialog state。

| 项目 | 结果 |
|---|---:|
| Session compute / total wall | 32.16s / 41.314s |
| Peak GPU memory | 3,466,037,248 bytes |
| Raw response | 10/10 exact |
| Prompt tokens | 10/10 exact |
| Tag margin | 10/10 exact |
| Current hidden | 10/10 exact |
| 8 dialog features | 10/10 exact |
| Decision / answer | 10/10 exact / 10/10 exact |
| 最大 logit 差 | `6.3171e-8` |

子集 Macro `0.8990` 只描述这 10 个训练集内 chunks，不能当性能估计。Smoke 的意义是
整条 GPU 推理管线与冻结 R0/cache/final 工件一致。

## 7. 风险与适用范围

D4 使用的是官方 benchmark 明确提供、时间上严格因果的 dialog prefix，因此没有未来
泄漏；但这个信号由官方对话构造策略产生。风险包括：

- hidden test 若 dialog 构造策略变化，收益可能下降；
- 真实 self-fed deployment 中，模型自己的错误会改变后续 history；
- `0.6846` 来自同一 public validation 的多变体机制诊断，不能称为独立确认；
- D4 提升不能表述成视觉进展或程序状态理解。

排行榜可利用性和研究解释必须分开报告。

## 8. 当前完成度与下一步

已经完成：

1. 冻结协议和配置；
2. 全量 refit 与 head 序列化；
3. 9,935-chunk 离线/在线等价；
4. D1 通用 deployment CLI 接入；
5. 10-chunk shared-vision GPU 等价 smoke；
6. 参数、环境、输入和输出哈希记录。

尚未执行：

1. 对全部 700 sessions 重新跑 GPU 在线 inference；现有 train-fit predictions 已由同一
   冻结 R0/cache 特征生成，但完整 GPU 重跑耗时较高，主要价值是额外工程验证；
2. 官方 submission 目录/容器打包与入口验收；
3. 任何外部上传或 leaderboard 提交。

下一步应优先做 submission packaging 审计，确认平台要求的是单独
`predictions.jsonl` 还是可运行模型包。外部上传仍需用户明确授权。

## 9. 复现工件

| 产物 | 路径 / SHA256 |
|---|---|
| 协议 | `annotations/d4_dialog_stage_candidate_v1/PROTOCOL.md` |
| Final config | `configs/d4_internvl35_1b_dialog_stage_final_v1.json` / `387f1a7c64691ea9204df669250caf8cd18821e9a8e3e0751c8e4aa76777d898` |
| Final artifact | `output/experiments/20260719_internvl35_1b_d4_dialog_stage_final_v1/` |
| Online audit | `c233961af275402dd381be196125bc5b9cbb7e3eeb181695283716f719d2e1b9` |
| Deploy config | `configs/d4_internvl35_1b_dialog_stage_deploy_shared_vision_v1.json` / `ba4a4fff81d0ce0f6851cf1502336457074c51ff9003e5d6f4a4589034e6bb89` |
| GPU smoke | `output/experiments/20260719_internvl35_1b_d4_dialog_stage_deploy_shared_vision_v1_smoke1/` |
| Equivalence audit | `b873e740518eb61a73f7b6cb14f30dcc5eb9a3f6306445d3077f7f209acb8932` |

测试：D4 3/3、D4+D3-D+D1 相关集合 22/22 通过；`compileall` 通过。
