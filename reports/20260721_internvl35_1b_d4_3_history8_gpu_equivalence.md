# C1 Small D4.3 history8 GPU 等价与候选晋升报告

## 结论

D4.2 `history8=(32,16,8,64)` 通过单卡 shared-vision 在线等价验证。
固定的四领域长 session 共包含 102 chunks；raw response、prompt tokens、
model frame count、八项 dialog features、decision 和 answer 均为 102/102
一致，hidden、tag probabilities 和 tag margin 的最大差异为 0，head logit
最大绝对差异为 `1.2215805633708499e-7`，低于 `1e-6` 门槛。

因此 `history8` 晋升为活动 leaderboard-engineering 基线，并生成独立的
`submission/d4_2_history8_small` 候选。原 `submission/d4_small`、D4 配置和
D4 head 均未覆盖。D3 `0.6690` 仍是正式科学基线。

## 协议和资源

- 协议在 GPU 前冻结，SHA256 为
  `d0e54823587cc9f736a4ea99fcac4dd791f180dc5a497fa6b2ade1da17e81fa3`。
- session indices 为 `143,356,472,609`，覆盖 Handyman、Tutorial、Arts and
  Crafts、Chef，分别为 22、26、24、30 chunks。
- 单 session 墙钟为 `63.77/76.22/77.09/112.70s`，均低于 300 秒；总墙钟
  `344.536s`，峰值分配显存 `3,481,809,920` bytes。
- 总参数 `1,060,898,844`，符合 Small `<=2B` 限制。
- 在线 predictions SHA256 为
  `c9b0b2da7a0e9083e308326a698b5f419fe1ac8eee20757b0fe7efe5c3ac6dca`。

## 证据边界

该 smoke 只证明在线实现复现冻结缓存，不提供新的性能估计。D4.2 OOF
Macro `0.6988` 仍是 post-selection、`val-supervised` public-validation 证据，
不能表述为 hidden-test 或独立泛化提升。外部上传、官方 Docker 模板适配
和项目源码许可证仍是独立门槛。
