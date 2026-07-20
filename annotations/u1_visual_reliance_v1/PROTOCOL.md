# U1-V：强制 utterance 的视觉/对话依赖审计协议

## 1. 目的与边界

本实验诊断冻结的 U1 `forced_no_state` utterance 生成到底主要依赖：

1. 当前 chunk 的视觉；
2. 过去 chunk 的视觉；
3. 当前可见的历史对话。

实验不训练模型、不改变 D1 interrupt 决策、不读取当前答案或未来信息，也不评价
utterance 的真实质量。80 个样本已经用于 U1 人工评测，因此本实验属于开发集上的
机制诊断，不是独立 held-out 证据，也不是模型晋级实验。

## 2. 冻结样本和基准输出

- 样本：`annotations/u1_forced_generation_v1/sample_items.jsonl`，80 chunks / 20 sessions。
- `full` 基准：复用
  `output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/content_records.jsonl`
  中的 80 条 `forced_no_state` 记录，禁止重新生成或替换。
- 输入容器在送入生成管线前必须删除所有当前/未来 `answers`。
- 模型、prompt、`$interrupt$` assistant prefill、greedy decoding、帧采样和 token 上限
  与冻结 U1 完全相同。

## 3. 四个视图

每个样本只改变下列一个输入因素：

| 视图 | 对话 | 视觉 | 解释目标 |
|---|---|---|---|
| `full` | 最多最近 4 个可见 turn | 截至当前 chunk 的真实累计帧 | 冻结参照 |
| `no_assistant_history` | 仅 system + query | 与 `full` 完全相同 | 历史对话是否必要 |
| `no_current_interval_video` | 与 `full` 完全相同 | 只保留当前 chunk 之前的真实帧 | 当前视觉是否提供增量信息 |
| `masked_video` | 与 `full` 完全相同 | 与 `full` 相同帧数和尺寸，但所有像素替换为中灰色 | 任意真实视觉内容是否必要 |

`no_current_interval_video` 仍保留过去真实帧。冻结样本全部来自第二个或更晚的
chunk，因此该视图不会产生空视频。`masked_video` 保留帧数、尺寸和图像 token 结构，
以减少“有无图像模态”本身造成的格式混杂。

## 4. 固定推理设置

- backbone：`OpenGVLab/InternVL3_5-1B-HF`，本地冻结 snapshot；
- dtype：BF16；attention：SDPA；
- 每 interval 抽 16 帧，累计后均匀截到最多 32 帧；
- 输入视频帧尺寸：448；
- `full` 历史上限：4 turns；
- 最大新生成 token：64；
- 解码：greedy，`do_sample=false`；
- assistant prefill：`$interrupt$`；
- seed：20260713。

三个反事实视图必须在一次模型加载中完成。运行前检查 GPU 占用；允许与轻量进程
共享，但不得启动第二个大模型副本或中断已有进程。

## 5. 自动诊断指标

所有指标按 80 个配对样本计算，并报告 overall、position bin 和 domain：

- fallback rate、非空率、平均生成词数；
- 固定 completion-claim 词法规则的命中率；规则只匹配 `you are/you're done`、
  `you have/you've finished/completed`、`it/this/that is done/finished/complete`、
  `all set` 和 `great/good job` 等显式完成断言；
- 与 `full` 的 answer/content exact-match；
- fallback decision agreement；
- `difflib.SequenceMatcher` 字符串相似度均值；
- `full` 非 fallback 但消融后 fallback 的数量；
- 消融非 fallback 但 `full` fallback 的数量。

相似度和 exact-match 只能表示敏感性，不能表示 grounding 或质量。
completion-claim 命中也不能自动判定 hallucination；它只用于筛选需要人工查看的
“可能过早宣告完成”样例。

## 6. 预注册诊断阈值

在查看反事实输出前冻结下列阈值：

- `no_assistant_history` 相对 `full` 的 fallback rate 增加至少 `0.20`：记为
  `history_necessary=true`；
- `no_current_interval_video` 的 fallback rate 增加至少 `0.10`，或平均文本相似度
  低于 `0.70`：记为 `current_visual_material=true`；
- `masked_video` 的 fallback rate 增加至少 `0.15`，或平均文本相似度低于 `0.65`：
  记为 `any_visual_material=true`。

这里的 `*_material=true` 只表示该输入足以改变输出，不表示变化方向更好。阈值未触发
也不证明模型完全没有使用相应输入，因为生成可能在表面文本上保持不变。

## 7. 结果解释与后续门控

1. 若去掉 assistant history 造成最大退化，而两个视觉消融变化很小，优先研究
   early-chunk language cold start、对话捷径和视觉 grounding；不能直接归因于 state 缺失。
2. 若去掉当前视觉显著改变输出，人工抽查 discordant cases，判断是正确 grounding、
   错误敏感还是随机措辞变化。
3. 若 masked 视觉显著而 no-current 不显著，说明生成可能主要利用历史视觉或视觉模态
   结构，不能声称理解当前进展。
4. 仅当 U1 人工评分、U1-V 的定性抽查以及后续状态/残差证据共同指向
   step/progress 时，才恢复 S1 或进一步做 granularity 建模。
5. 本实验不得接触 reviewer B 文件，也不得用 reviewer A 分数选择阈值或样本。

## 8. 完成条件

- 80 个样本的四视图记录齐全，`full` 记录 SHA256 与配置一致；
- 三个生成视图均严格因果，且未读取任何 `answers`；
- `masked_video` 对每个样本保留 `full` 的模型输入帧数；
- 配对键、样本顺序和冻结设置通过程序校验；
- 自动分析、配置、命令、环境、代码状态、运行时间和输入哈希完整落盘；
- 中文报告明确区分“输出敏感性”和“生成质量”。
