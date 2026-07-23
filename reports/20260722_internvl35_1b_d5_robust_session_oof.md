# C1 Small D5 鲁棒多视图线性头 D4-session-fold 报告

## 结论

冻结的四视图等权线性头在 clean `history8` 上取得 Macro `0.6918`，相对
精确复现的 standard-head 基线 `0.6988` 下降 `-0.0070`，paired-session
95% 区间 `[-0.012605,-0.001188]` 完全为负。静态 gate 失败，实验按协议
停止在 self-fed 推理之前，不晋升且不继续搜索视图权重、阈值、L2 或扰动强度。

## 冻结设计

四个 answer-free 视图为 clean history8、history4、assistant-drop 和确定性
half-stride frame-jitter。standard 对照只在各自 clean 训练协议下拟合；唯一
鲁棒候选在四个视图上等权拼接 fit/calibration 样本。二者都使用 D4.2 session
folds、三折 fit / 一折 calibration / 一折 test、class-balanced float64
logistic regression 和冻结的阈值选择规则。

协议 SHA256 为
`bcf8199d51c5d5e1718173bd8a628256b362dbff797213128f361d44a1972415`。
standard clean predictions 精确复现 D4.2 参考 SHA256
`d154789b8f41583558878e93b9bb618643a5f64d1ad5b397d84cfd592e31c121`。

## 静态结果

| 视图 | Standard Macro | Robust Macro | Delta | Paired-session 95% 区间 |
|---|---:|---:|---:|---:|
| clean history8 | 0.6988 | 0.6918 | -0.0070 | [-0.012605,-0.001188] |
| history4 | 0.6905 | 0.6917 | +0.0012 | [-0.004476,+0.007082] |
| assistant-drop | 0.3500 | 0.5715 | +0.2215 | [+0.213912,+0.229175] |
| frame-jitter | 0.6958 | 0.6934 | -0.0024 | [-0.008053,+0.003148] |

clean retention 失败；history4、assistant-drop 和 frame-jitter 也没有全部达到
冻结的逐视图增益门槛。`self_fed_eligible=false`，因此没有生成 self-fed
结果或全开发集鲁棒 head。

## 产物与边界

完整实验位于
`output/experiments/20260722_internvl35_1b_d5_robust_session_oof_v1`。
鲁棒 clean predictions SHA256 为
`50e06fe78da8036d4b2f3b14c21634ab97848265cdd99106981ce11ccf1bfbc1`，
summary SHA256 为
`82193791fd73de521881fadfd5756f7f6b2fc662bd7aa16a45a54a9fa4c4fda8`。

这是 post-selection、val-supervised public-validation 鲁棒性审计，不是
hidden-test 或独立泛化证据；它不改写 D4/D4.2 submission，也不授权外部上传。
