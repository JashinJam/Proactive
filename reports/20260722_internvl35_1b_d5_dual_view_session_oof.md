# C1 Small D5 双视图融合 D4-session-fold 报告

## 结论

两个冻结的双视图线性候选均显著低于 D4.2 `history8` 基线 `0.6988`。
`shared_delta` 的 Macro 为 `0.6846`，delta `-0.0142`，paired-session 95%
区间 `[-0.022314,-0.006260]`；`dialog_gated_delta` 为 `0.6793`，delta
`-0.0195`，区间 `[-0.027861,-0.011013]`。整个融合族终止，不继续搜索
门控、交互项或正则强度。

## 候选与门槛

两个候选都使用冻结的 uniform-history8 1,051 维基特征，并加入 multiscale
相对 uniform 的 tag-margin 与 1,024 维 hidden difference：

- `shared_delta`：2,076 features，2,077 个 head parameters；
- `dialog_gated_delta`：按 `assistant_added_since_previous` 门控 difference，
  共 3,101 features、3,102 个 head parameters。

所有候选均使用 D4.2 session folds、三折 fit / 一折 calibration / 一折 test
及同一冻结线性头规则。gated 相对 shared 的内部选择 margin 为 `-0.0053`，
未达到 `+0.002` 门槛，因此 `shared_delta` 仅按规则机械选中，
`selected_promoted=false`。两个候选的 5/5 folds 与 4/4 domains 均低于基线。
协议 SHA256 为
`b8f272c8681c6b0f3d3d01782ea91496dcc575e70d46e57f41bae38c7947643d`。

## 产物与边界

完整实验位于
`output/experiments/20260722_internvl35_1b_d5_dual_view_session_oof_v1`。
选中候选 predictions SHA256 为
`c3c04a61d09887bc3873668390668148fde7e662faa4570c6a6d01584f1ca799`，
summary SHA256 为
`7920b3d0120770b3311d5b79efbb5ad1040f2f3fa1b58f9bdc9df03ebed7cc16`。

这是同一 public validation 上的 post-selection、val-supervised 负结果；
它不构成 hidden-test 证据，不改变 D4/D4.2 submission。
