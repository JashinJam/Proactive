# C1 Small D5 视觉时序残差 D4-session-fold 报告

## 结论

冻结的 39,073 参数 causal visual-temporal residual 在 D4.2 五折 session OOF
上取得 Macro/G-mean `0.6983/0.6983`，相对 `history8` 基线下降 `-0.0005`。
5,000 次 paired-session bootstrap 95% 区间为
`[-0.002131,+0.000987]`，只有 1/5 folds 严格提升，因此候选不晋升并终止
这一视觉时序残差模型族。

该负结果只拒绝当前冻结的视觉池化与小型 GRU 残差，不能外推为视觉信息对
任务无用。

## 冻结设计与结果

基模型、官方 prompt/dialog、BF16 greedy decoding、`history8` 输入策略、
1,051 维基线特征、D4.2 session folds 和 OOF 轮转全部冻结。视觉分支使用
`causal_multiscale_16_8_8_v1` 的已观测帧、冻结 vision tower 的 1,024 维
pooler 输出，以及 `Linear(1024,32) -> GRU(32,32) -> Linear(32,1)` 残差。

fold delta 为 `[-0.0014,-0.0021,+0.0011,-0.0005,+0.0000]`；domain delta
为 Arts `+0.0015`、Chef `-0.0025`、Handyman `+0.0005`、Tutorial
`-0.0011`；previous-interrupt / previous-silent delta 为
`-0.0026/+0.0017`。64 个决策变化，其中修正 29 个错误、新增 35 个错误。
所有晋升门槛除 no-class-collapse 外均失败。

协议 SHA256 为
`5193e652b98882e130e4866932f5197aa57ccd5e7078a08f56a24075e7c69dcc`。
训练在物理 GPU 1 上完成，进程内设备为 `cuda:0`，墙钟约 811.48 秒；复用
已冻结的视觉特征 cache，没有重新提取 backbone 特征。

## 产物与边界

完整实验位于
`output/experiments/20260722_internvl35_1b_d5_visual_temporal_session_oof_v1`。
predictions SHA256 为
`9eaf6b2ee9032e6cc7380f25e8301c7588b9db910f2c14ab5223cac01ada6896`，
summary SHA256 为
`968f6ee9aff3599110aa696f8a4599dfe02a2380ab3b3c62bc4d3b954f4a556f`。

本实验仍是 post-selection、val-supervised public-validation 证据，不是
hidden-test 或独立泛化证据，不授权外部上传。
