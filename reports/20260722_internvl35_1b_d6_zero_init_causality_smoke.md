# C1 Small D6 零初始化等价与因果 GPU 审计

## 结论

D6 query-conditioned causal visual memory 与 late-attention LoRA 的冻结
zero-init 路径通过正式训练前 GPU 硬门。审计覆盖 D4.3 固定的四领域 source
indices `143,356,472,609`，共 4 sessions / 102 chunks。该结果只证明实现等价、
因果和资源可行，不是 efficacy 估计；D4.2 `history8` 仍是活动
leaderboard-engineering 候选。

## 等价与因果结果

- raw response、prompt token count 和 model input frame count 均为 102/102
  精确匹配。
- hidden state、tag margin、silent log-probability 和 interrupt
  log-probability 的最大绝对差异均为 `0.0`。
- zero-init memory residual norm 最大值为 `0.0`。
- silent/interrupt 两个 batch-1 language forward 独立核对的 memory update
  最大差异为 `0.0`，session state 只提交一次。
- 在 source 143 上只修改首 chunk 之后的 dialog snapshots 和 intervals，重新
  推理首 chunk 的 frame provenance、prompt、hidden 和全部 tag 数值精确不变。
- D6 的 10 项单元测试、真实单 chunk backward 审计，以及复用层的 D1 18 项和
  D4.2 20 项回归测试均通过。

## 资源与参数

- peak allocated memory：`2.9096 GiB`，低于 `70 GiB` 门。
- maximum session model time：`22.9376s`，低于 smoke `240s` 门。
- 102 chunks 累计模型时间：`80.5457s`。
- 四 session 本次完整恢复段墙钟：`237.6926s`。
- 真实 backward 检查中 48/48 adapter tensors 获得有限梯度，18 个 tensor 在
  zero-init 首步非零；base gradient tensor 数为 0，峰值 `4.5556 GiB`。
- 参数仍为 base `1,060,897,792` + memory `627,072` + LoRA `327,680` +
  head `1,052` = `1,061,853,596`，低于 Small 2B。

## 冻结指纹

- 协议 SHA256：
  `eda95ccca756966d57b59a2ce4b21bc71699dd51acd63b096e0b007c5d4ae4d7`。
- canonical config SHA256：
  `ecbe9b3cd187cfdf4793f0001e9b22a6905f1f1c44d1a981ebf5d5835c39d2e1`。
- config file SHA256：
  `e381a9e825e0878072a0c56b4eea0741a76e998b4eae4341673ac7710aef7633`。
- D4.3 reference records SHA256：
  `e12310419db40eb1957ded6ae513872e151bf47fd62eb2c0ff2c05bbc337970e`。
- D6 smoke records SHA256：
  `6baa9cc6509f3df2a7a5457746fbdb8490e71642640ccd8293ea0beb28d16108`。
- Summary SHA256：
  `920d9b35cca629477b0d61ff9b786b8a63f6ff2c97cecafb999d5bae64e111fa`。

完整机器可读产物位于：

```text
output/experiments/20260722_internvl35_1b_d6_query_memory_lora_oof_v1/
  smokes/zero_init/
```

## 证据边界与下一门

本审计没有读取 test/validation labels 来形成预测选择，也不报告 Macro F1。
D6 使用 public validation 进行后续 adapter/head 监督，因此未来五折结果仍只能称为
post-selection、val-supervised OOF，不是 hidden-test 或独立泛化证据。

正式五折仍被 rotation-0 完整一轮 trainability/resource smoke 阻塞。该 smoke
必须同时满足峰值显存、最长 session model time 和线性估算单折 `<=48h`，否则按
冻结停止规则终止 D6，不缩模型或修改结构。外部上传未授权。
