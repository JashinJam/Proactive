# C1 Small D5 因果多尺度采样 D4-session-fold 报告

## 结论

冻结的 `causal_multiscale_16_8_8_v1` 在 D4.2 五折 session OOF 上取得
Macro/G-mean `0.6988/0.6988`，与 D4.2 `history8` 基线四舍五入后相同，
Macro delta 为 `+0.0000`。5,000 次 paired-session bootstrap 95% 区间为
`[-0.006055,+0.006116]`，只有 2/5 folds 严格提升，因此不晋升并终止该
采样族。

## 冻结改动与结果

模型、官方 prompt/dialog、BF16 greedy decoding、`history8` 文本窗口、
1,051 维特征 schema、D4.2 session folds、线性学习器、L2 网格和 calibration
threshold 规则全部保持不变。唯一变化是已观测帧的因果选择：当前 interval
最多 16 帧，紧邻上一 interval 8 帧，更早历史按绝对时间取 8 个锚点。

候选 interrupt/silent F1 为 `0.6987/0.6990`，TP/FP/TN/FN 为
`3469/1109/3474/1883`。相对基线：

- fold delta 为 `[+0.0073,-0.0051,-0.0050,+0.0055,-0.0030]`；
- domain delta 为 Arts `+0.0016`、Chef `+0.0015`、Handyman `+0.0030`、
  Tutorial `-0.0063`；
- previous-interrupt / previous-silent delta 为 `-0.0045/+0.0068`；
- 887 个决策变化，修正 443 个错误，同时新增 444 个错误。

最小增益、正 bootstrap 下界、4/5 正 folds 和两个 previous-response strata
不下降均失败。协议 SHA256 为
`bac68b2a50d506f7bc541872c611be107b0657f220d34de1868b81b39be91095`。

## 产物与边界

完整实验位于
`output/experiments/20260722_internvl35_1b_d5_causal_multiscale_session_oof_v1`。
predictions SHA256 为
`9507a4828ba7c3d6230453d5768b46921f649b8a7a9fe8ee4d9ec976ed2e80c1`，
summary SHA256 为
`01f5edaa8d584f3429805856ec485ca841db22d624f9e6f6e0dc9423f5e9120f`。

本结果是 post-selection、val-supervised public-validation 证据，不改变
D4.2 活动候选，也不授权 hidden-test 声明或外部上传。
