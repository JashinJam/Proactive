# U1-V 视觉依赖审计与 D3-D 对话策略控制报告

## 1. 核心结论

本轮两个实验共同回答了一个关键问题：当前系统的增益究竟主要来自视觉/程序状态，
还是来自官方对话历史。

结论很明确：**当前系统首先依赖对话历史，视觉只提供不稳定的局部修正。D3 的大部分
提升也可以由官方 dialog 中显式暴露的上一动作和干预节奏重建。**

- U1-V 去掉 assistant history 后，80/80 全部生成空 continuation 并回退固定句；
- 去掉当前 chunk 视觉后，fallback 没有增加，预注册的 current-visual gate 未触发；
- 全灰视觉会明显改写文本，但 fallback 不变，且相似度仅以 `0.0021` 的边界触发阈值，
  这只能说明视觉内容会影响措辞，不能证明可靠 grounding；
- 只用 8 个 dialog-stage 标量、不用视觉或 hidden，Macro F1 已达 `0.6618`；
- D1 fused 加 8 个 dialog-stage 标量达到 `0.6846`，比 D3 `0.6690` 高 `0.0156`；
- 官方 dialog 的 assistant 新增信号与 previous gold interrupt 在 9,235/9,235 个非首
  chunks 完全一致。

因此，S1 大规模状态标注继续暂停，当前不进入 granularity 或 GRPO。排行榜路线转向
冻结一个 D4 dialog-stage candidate 并完成全量 refit/在线验证；研究路线则需要更严格的
视觉 grounding 或独立数据证据。

## 2. 实验边界

两项实验都使用公共 validation，因此不是 hidden-test generalization。

- U1-V：20 sessions / 80 个已经用于 U1 人工评测的 chunks；只做生成机制诊断；
- D3-D：700 sessions / 9,935 chunks；沿用冻结的 5-fold session-level OOF；
- 没有读取 reviewer B 文件；
- 没有使用 reviewer A 分数选择 U1-V 阈值或样本；
- dialog 特征在删除 `answers` 后构造；previous-gold 一致率只在全部 OOF 完成后核对；
- 两项实验均不训练 backbone，不使用外部数据。

## 3. U1-V：视觉与历史依赖

### 3.1 设计

冻结 U1 的 InternVL3.5-1B、prompt、`$interrupt$` prefill、最多 32 帧、最近 4 turns、
greedy decoding 和 64-token 上限。比较四个视图：

| 视图 | 对话 | 视觉 |
|---|---|---|
| `full` | 冻结的最近 4 turns | 截至当前 chunk 的真实累计帧 |
| `no_assistant_history` | 移除 assistant 历史 | 与 full 相同 |
| `no_current_interval_video` | 与 full 相同 | 只保留过去 chunks 的真实帧 |
| `masked_video` | 与 full 相同 | 同帧数/尺寸的中灰帧 |

`full` 直接复用原始 80 条输出，另外三个视图在一次模型加载中生成 240 条配对输出。
模型共 1,060,897,792 参数，峰值显存约 3.48 GB，运行 572.7 秒。

### 3.2 总体结果

| 视图 | fallback | 相对 full 的 answer exact | 平均文本相似度 | 预注册判断 |
|---|---:|---:|---:|---|
| full | 30.00% | 100% | 1.0000 | 参照 |
| no assistant history | 100.00% | 30.00% | 0.3000 | `history_necessary=true` |
| no current interval video | 26.25% | 46.25% | 0.7734 | `current_visual_material=false` |
| masked video | 30.00% | 31.25% | 0.6479 | `any_visual_material=true` |

去掉 assistant history 后，full 中原本 56 个非 fallback 样本全部变成 fallback。这不是
轻微质量下降，而是模型不再愿意继续 `$interrupt$` 前缀，直接生成 EOS。按位置看，full
的 fallback 为 second `80%`、2--4 `35%`、5--9 `0%`、10+ `5%`；这与评测员 A
观察到的“越晚越能说”完全一致。

去掉当前 chunk 视觉并没有增加 fallback，反而从 24 条降到 21 条。80 条中仍有 27 条
相似度低于 `0.7`、19 条低于 `0.5`，说明当前视觉会改变部分具体内容，但不负责解决
“能不能生成”的主要问题。典型差异包括：

- full 要求把烤芝士放入锅中，去掉当前画面后退回到“先把一片面包放到锅里”；
- full 要求倒入剩余液体并混合，去掉当前画面后改成“加入半杯水”；
- full 已进入写最后一句，masked 后却说“列表已准备好，进入下一个任务”。

这些例子表明视觉确实影响 step wording，但影响不稳定，历史对话仍能独立生成一条
看似合理但可能过时的指导。

masked 的平均相似度 `0.6479` 仅略低于冻结阈值 `0.65`，fallback 状态只有 2/80
发生互换。它说明真实像素对文本有影响，却不能证明这个影响方向正确。固定的完成断言
词法规则在 full/masked 中只命中 3/1 条，而且漏掉单独的 `Done!` 等表达，只能视为
下界诊断。

### 3.3 U1-V 结论

当前 forced utterance 管线的实际行为更接近：

```text
assistant history 决定是否有话可说并提供步骤骨架
                 +
视觉对部分名词、动作和步骤位置做不稳定修正
```

它不是“先可靠理解当前画面，再基于 state 生成”。因此不能根据中后段高人工分直接
扩展状态标注；更直接的问题是 early-chunk language cold start、历史捷径和 grounding。

## 4. D3-D：官方 dialog policy 控制

### 4.1 动机与设计

D3 原结果是 Macro `0.6690`，相对 D1 `0.6341` 提升 `+0.0349`。此前只知道 hidden/
margin dynamics 中混有 dialog 信号，本实验把 dialog 中的信号显式写成 8 个因果标量：

```text
是否非首 chunk
上一 chunk 后是否新增 assistant turn
新增 assistant turn 数
累计 assistant turn 数
assistant turn / 已经过 chunk 比例
距离上次新增 assistant 已经过多少 chunks
最后一条 assistant 文本长度
最后一条 assistant 是否带 $interrupt$ tag
```

所有特征从 answer-stripped 官方输入构造。实际数据没有 malformed/空 turn、没有一次
新增多条 assistant，也没有 assistant count 下降。

### 4.2 OOF 结果

| 变体 | 特征数 | Macro F1 | 相对 D1 | 重建 D3 增益 | 95% session bootstrap vs D1 |
|---|---:|---:|---:|---:|---:|
| D1 fused replay | 1,043 | 0.6341 | 0 | 0% | `[0, 0]` |
| dialog increment only | 2 | 0.6475 | +0.0134 | 38.4% | `[+0.0006,+0.0265]` |
| dialog stage only | 8 | 0.6618 | +0.0277 | 79.4% | `[+0.0169,+0.0391]` |
| D1 + dialog increment | 1,045 | 0.6749 | +0.0408 | 116.9% | `[+0.0324,+0.0489]` |
| D1 + dialog stage | 1,051 | **0.6846** | **+0.0505** | **144.7%** | `[+0.0418,+0.0591]` |

`D1 + dialog stage` 的 interrupt/silent F1 为 `0.6893/0.6799`，非首 chunk Macro
从 D1 `0.6045` 提升到 `0.6596`。相对 D1：

- 5/5 folds 全为正：`+0.0447/+0.0448/+0.0584/+0.0438/+0.0624`；
- 4/4 domains 全为正：`+0.0620/+0.0387/+0.0515/+0.0537`；
- second、2--4、5--9、10+ 均为正：`+0.0157/+0.0248/+0.0550/+0.0971`。

因此结果不是某一折、某个领域或单一长 session 的偶然性。

### 4.3 学到的是什么

在 9,235 个非首 chunks 上：

```text
dialog[i] 是否新增 assistant turn
            ==
answers[i-1] 是否为 $interrupt$

一致：9,235 / 9,235
```

线性头 5 个 folds 中最稳定、绝对值最大的 dialog 特征都是
`log1p_chunks_since_assistant_addition`，标准化权重约 `+0.61` 到 `+0.67`。模型实际上在
学习：“已经连续沉默多久、最近说话有多频繁、当前处于对话流程的什么阶段”。

这对官方 benchmark 是严格因果且可见的，因而可以合法用于排行榜模型；但它依赖组织方
如何构造 dialog prefix。在真实 self-fed 部署中，如果历史由模型自己的预测产生，错误会
改变后续 dialog，数据分布会漂移。

`D1 + dialog stage` 与 D3 的逐 chunk 决策一致率只有 `82.49%`，所以 `0.6846` 不是把
D3 决策简单复制一遍。`captured_d3_gain > 100%` 也不是“负视觉贡献”的可加分解，只表示
这一显式特征集在同一公共 OOF 上比 D3 dynamics 更容易被线性头利用。

## 5. 路线调整

### 5.1 立即停止或继续暂停

- S1 继续停在 2/32 sessions、23/444 states；不继续剩余 421 个状态标注；
- 不进入 granularity 专门建模；
- 不启动 GRPO；
- 不继续在同一 5 folds 搜索更多 dialog、history window、L2 或 threshold。

U1 state package 和 reviewer B 结果到达后仍按冻结 gate 读取，但它们不再阻塞排行榜
decision 路线。

### 5.2 排行榜路线

D3 `0.6690` 保持为已经正式晋级、已经序列化和通过 GPU smoke 的科学/部署基线。
本次 D3-D 协议预先规定所有变体“机制诊断、不可晋级”，因此不能事后把最高的
`0.6846` 宣称为新的正式 baseline。

下一步应建立唯一的 **D4 dialog-stage leaderboard candidate**：

1. 固定且只使用本次 8 个 dialog-stage 特征，不再增删；
2. 按既有 D1/D3 full-development finalization 规则做一次全量 refit；
3. 序列化 D1 fused + dialog-stage head；
4. 在在线 session runner 中严格复现 8 个 prefix 特征；
5. 对 9,935 个缓存 chunks 做特征/决策精确验证；
6. 做共享视觉 GPU CLI smoke；
7. 通过后生成 validation `predictions.jsonl` 和后续提交/容器工件。

D4 的 `0.6846` 必须继续标为 public-validation OOF evidence；full refit 的 train-fit 分数
只能作 sanity check，不能替代 OOF。

执行更新（2026-07-19）：上述 D4 已按固定 8 特征完成 full refit、9,935-chunk 在线
回放和 10-chunk shared-vision GPU smoke；下一步已推进为 submission packaging/
entry-point audit。详见 [D4 报告](20260719_internvl35_1b_d4_dialog_stage_candidate.md)。

### 5.3 Utterance 路线

官方 C1 Macro 只看 interrupt/silent tag，当前排行榜优先级应先完成 D4。内容侧后续最有
针对性的实验不是扩大 state，而是：

- 对 early chunks 做专门的 forced-generation cold-start 训练/提示控制；
- 加入当前画面 grounding 的反事实一致性或监督；
- 保持 gate 与正文生成解耦；
- 以人工 hallucination/groundedness 作为内容门，不用文本相似度替代质量。

## 6. 复现工件

U1-V：

```text
协议: annotations/u1_visual_reliance_v1/PROTOCOL.md
配置: configs/u1_internvl35_1b_visual_reliance_v1.json
实验: output/experiments/20260718_internvl35_1b_u1_visual_reliance_v1/
content_records.jsonl SHA256:
dbc6538d1af4da6fa818c39c44e1c5397c03fcac04e2acd6945800d42ce8fbc6
analysis.json SHA256:
981cfcde311876c03d7dd505bff9369454b28fb3e5b311e33c7af7c9a3e13531
```

D3-D：

```text
协议: annotations/d3_dialog_policy_control_v1/PROTOCOL.md
配置: configs/d3_internvl35_1b_dialog_policy_control_v1.json
实验: output/experiments/20260718_internvl35_1b_d3_dialog_policy_control_v1/
comparison.json SHA256:
cc5eee7d1f8b0d561cc99b49f735af5d84f3e66b1a1cd4c8c07b4ca5d1083088
D1 + dialog-stage predictions SHA256:
467bd7c567d66b3041425b336343290c0b3d742f10d25d2279de987770ae43f5
```
