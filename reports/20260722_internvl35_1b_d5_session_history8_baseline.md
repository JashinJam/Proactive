# C1 Small D5 D4-session-fold history8 基线报告

## 结论

D5 基线在 D4.2 的原始五折 session manifest 上重新训练后，精确复现 D4.2
`history8=(32,16,8,64)` 的完整 OOF 结果：Macro/G-mean 为
`0.6988/0.6988`，interrupt/silent F1 为 `0.7049/0.6928`，TP/FP/TN/FN
为 `3572/1211/3372/1780`。predictions 与 metrics 文件的 SHA256 均和
D4.2 冻结参考逐字节一致，因此本实验是后续 D5 候选唯一有效的比较基线。

## 评价协议

本轮撤销 exact-query-grouped folds，直接复用 D4.2 fold manifest：算法
`domain_stratified_sha256_round_robin`、seed `d1-session-oof-v1`、五折 session
OOF。每个 rotation 使用三折 fit、一折 calibration、一折 test；线性头为
1,052 参数的标准化、class-balanced float64 logistic regression，L2 网格为
`{1e-5,1e-4,1e-3,1e-2}`，阈值只在 calibration fold 上按 Macro-F1 选择。

fold manifest SHA256 为
`bd537e9e155586cf3af9f26052fda277fa3e1930e378538346cc197432ff86c0`，
协议 SHA256 为
`cc026bf7f68292cc6b5d1f7da1c476b153c527597bebf0655c321507f2e23327`。
完整覆盖 700 sessions / 9,935 chunks。

## 产物与边界

完整实验位于
`output/experiments/20260722_internvl35_1b_d5_session_history8_baseline_v1`。
predictions SHA256 为
`d154789b8f41583558878e93b9bb618643a5f64d1ad5b397d84cfd592e31c121`，
summary SHA256 为
`0e606683279ce809b23362f4c2510104384a32dd3b0ca7899d5af58b37ac85b1`。

这是同一 public validation 上的 post-selection、val-supervised 精确重放，
不是 hidden-test 或独立泛化证据。旧 grouped-fold 输出只保留为历史产物，
不再作为活动协议、基线或晋升依据。
