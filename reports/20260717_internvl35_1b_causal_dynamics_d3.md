# InternVL3.5-1B 冻结缓存因果动态 D3 正式报告

> 日期：2026-07-17  
> 任务：ECCV Wearable AI Challenge 1 EgoProactive，Small 组  
> 结论：预注册主变体 `dynamics_fused` 通过全部推广门槛，成为新的科学 OOF 基线；shared-vision GPU CLI 等价 smoke 已通过，完整 submission/container packaging 尚未执行  
> 实验性质：public-validation-supervised、session-level OOF，不是 hidden-test 泛化证据

## 1. 实验要回答的问题

D1 `fused_linear` 只使用当前 chunk 的 18 个因果标量、一个固定标签 margin 和 1,024 维因果 hidden state，官方 OOF Macro F1 为 `0.6341`。它没有显式表示“当前状态相对上一时刻发生了什么变化”。

D3 检验一个单一假设：

> 在不重新运行 backbone、不读取 U1 人工评分、不引入未来信息的条件下，当前与历史 frozen D1 表征之间的严格因果变化，能否稳定改善 interrupt/silent 决策？

本轮不训练 InternVL、不修改模型权重、不改变 utterance generation。它只复用已冻结的 label-free D1 缓存，在 CPU 上训练新的线性决策头。

## 2. 模型、输入与模块设计

### 2.1 Backbone 与参数预算

| 项目 | 配置 |
|---|---|
| Backbone | `OpenGVLab/InternVL3_5-1B-HF` |
| Revision | `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| License | Apache-2.0 |
| Backbone params | 1,060,897,792 |
| D3 head params | 2,076 |
| 总参数 / active params | 1,060,899,868，约 1.060900B |
| Precision | Backbone BF16；缓存与动态特征 float32；线性拟合 float64 |

D3 头替换 D1 的 1,044 参数头，不是叠加两个决策头。总参数仍远低于 Small 组 2B 上限。

### 2.2 完整输入输出链路

```text
当前及历史 causal frames + query + 当前可见 prior dialog
                         |
                         v
          冻结 InternVL / D1 shared representation
          current tag margin + current 1024-d hidden
                         |
             +-----------+-----------+
             |                       |
             v                       v
       D1 current features      session-local history
       18 scalar + margin       previous / prefix mean
       + current hidden                  |
             |                           v
             |                  8 dynamics scalars
             |                  + 1024-d hidden delta
             +-------------+-------------+
                           v
                 2075-d standardized linear head
                           |
                     interrupt / silent
```

输出仍由既有 `decision_answer` 组装。D3 只改变二元 gate，不解决 D1 fallback utterance 的内容质量问题。

### 2.3 冻结动态特征

八个标量为：

```text
has_previous_chunk
tag_margin_delta_previous
tag_margin_abs_delta_previous
tag_margin_delta_history_mean
hidden_cosine_previous
hidden_delta_rms_previous
hidden_cosine_history_mean
hidden_delta_rms_history_mean
```

高维动态特征为：

```text
hidden_delta = current_hidden - previous_hidden
```

总特征数：

```text
D1 fused             1043
dynamics scalars        8
hidden delta          1024
--------------------------
D3 fused             2075
```

因果约束如下：

- 每个 session 的首 chunk 动态特征全部为 0；
- history mean 只包含当前 chunk 之前的状态；
- session 切换时状态完全重置；
- chunk 必须从 0 开始并严格连续；
- 动态构造不读取 label、未来 row、未来 dialog 或未来 frame；
- prefix-invariance 单元测试验证修改未来 row 不会改变前缀特征。

## 3. 预注册实验协议

正式结果出现前，配置和推广门槛已写入 `configs/d3_internvl35_1b_causal_dynamics_oof.json` 与 `CURRENT_ROUTE.md`。

四个冻结变体：

| 变体 | 输入 | 地位 |
|---|---|---|
| `d1_fused_replay` | 原 D1 1,043 维 | 精确重放控制 |
| `dynamics_scalar` | D1 + 8 个动态标量 | 诊断，不可推广 |
| `dynamics_hidden` | D1 + 1,024 维 hidden delta | 诊断，不可推广 |
| `dynamics_fused` | D1 + 两类 dynamics | 唯一可推广主变体 |

数据和训练协议保持 D1 不变：700 sessions / 9,935 chunks，五折 session-level split；每轮三折 fit、一折 calibration、一折 test。L2 网格固定为 `[1e-5, 1e-4, 1e-3, 1e-2]`，使用 sum reduction；L2 和 threshold 只由 calibration fold 选择。

推广门槛：

1. 官方 Macro F1 相对 D1 至少 `+0.005`；
2. paired session bootstrap 95% 下界大于 0；
3. 至少 4/5 folds 和 3/4 domains 正增益；
4. non-first Macro 正增益；
5. interrupt/silent F1 均不低于 `0.60`；
6. 结果出现后不得搜索新 feature、rolling window、L2 或 threshold。

## 4. 执行与审计

### 4.1 训练前审计

`--audit-only` 在训练和读取监督指标前验证：

- 700 sessions / 9,935 chunks 完整对齐；
- 恰有 700 个首 chunk，全部 dynamics 为 0；
- 四个矩阵形状分别为 `1043/1051/2067/2075` 维；
- dynamics 不读取 label 或未来 row；
- D1 cache SHA256 为 `cc5a5b3c6184987edc5f041eb2cb01a51ccfd88d6328d8308fcce3a4bd9122bf`。

### 4.2 正式运行

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d3.run \
  --config configs/d3_internvl35_1b_causal_dynamics_oof.json \
  --output-dir output/experiments/20260717_internvl35_1b_causal_dynamics_d3_oof_v1
```

正式 OOF wall time 为 `306.657s`，只使用 CPU，没有重新运行模型推理，也没有访问正在进行的 U1 双人评分。

正式 runner 首先精确复现 D1：Macro `0.6341`、9,935 个 decisions、prediction SHA256 `04183a4083d160662d5f91bff5432a7ca96595dd66b2b0b64f3b430799143ad9` 全部一致，之后才训练 dynamics 变体。

## 5. 正式结果

### 5.1 四个变体

| 变体 | Macro F1 | Interrupt F1 | Silent F1 | 相对 D1 | Positive folds/domains |
|---|---:|---:|---:|---:|---:|
| D1 exact replay | 0.6341 | 0.6352 | 0.6330 | 0 | 0/5，0/4 |
| dynamics scalar | 0.6551 | 0.6633 | 0.6469 | +0.0210 | 5/5，4/4 |
| dynamics hidden | 0.6594 | 0.6734 | 0.6454 | +0.0253 | 4/5，4/4 |
| **dynamics fused** | **0.6690** | **0.6845** | **0.6535** | **+0.0349** | **5/5，4/4** |

两个诊断变体虽有正增益，但按冻结协议不能替代主变体。主变体通过全部推广门槛。

### 5.2 官方主指标

| 指标 | D3 dynamics fused |
|---|---:|
| Macro F1 | **0.6690** |
| G-mean F1 | 0.6688 |
| Interrupt precision / recall / F1 | 0.7051 / 0.6652 / 0.6845 |
| Silent precision / recall / F1 | 0.6332 / 0.6751 / 0.6535 |
| TP / FP / TN / FN | 3560 / 1489 / 3094 / 1792 |
| Predicted interrupt | 5,049 / 9,935，50.82% |

Paired session bootstrap 5,000 次的 Macro 增益中位数为 `+0.03506`，95% 区间 `[+0.02654, +0.04372]`，正增益比例为 1.0。

### 5.3 稳定性

五折增益依次为：

```text
fold 0  +0.02804
fold 1  +0.02272
fold 2  +0.04856
fold 3  +0.02623
fold 4  +0.05057
```

四个 domain 均提升：Arts and Crafts `+0.04753`、Chef `+0.00664`、Handyman `+0.04479`、Tutorial `+0.04948`。Chef 的收益明显较小，是后续需要重点观察的迁移风险。

按位置，首 chunk 仅 `+0.00036`；second、2--4、5--9、10+ 分别为 `+0.02344/+0.04174/+0.04638/+0.04730`。Non-first Macro 从 `0.60454` 提升到 `0.64460`，说明总收益不是首 chunk convention 造成的。

D3 相对 D1 共改变 2,142 个判定，其中 2,141 个发生在非首 chunk：修复 755 个 FN 和 493 个 FP，同时新引入 360 个 FN 和 534 个 FP，净修复 354 个 chunk。

## 6. 解释性审计与重要边界

官方输入的 `dialog[i]` 只包含当前 chunk 之前可见的 turns，因此从因果定义上合法。但在这批 public validation 数据中，对全部 9,235 个非首 chunk：

```text
dialog[i] 相对 dialog[i-1] 是否新增 assistant turn
    ==
上一 chunk 的 gold interrupt/silent
```

一致率为 `9235/9235 = 100%`。当前 chunk 的 gold interrupt rate 在两组中差异很大：

| 上一时刻 dialog 新增 assistant turn | Chunks | 当前 gold interrupt rate | D1 Macro | D3 Macro | Delta |
|---|---:|---:|---:|---:|---:|
| 否 | 4,118 | 64.01% | 0.5949 | 0.6355 | +0.0406 |
| 是 | 5,117 | 39.42% | 0.5901 | 0.5915 | +0.00145 |

两组的动态幅度也显著不同：未新增 assistant turn 时 `hidden_delta_rms_previous` 均值为 `0.2680`，新增时为 `1.3392`；对应 tag-margin 绝对变化均值为 `0.3924` 与 `2.0466`。这说明 D3 能通过表征变化识别官方历史对话状态。

正确解释是：

- D3 在官方 benchmark 的当前前缀输入下严格因果，没有读取当前或未来 gold；
- 它稳定利用了跨 chunk 历史状态，排行榜价值真实存在；
- 但收益包含“官方 dialog 序列策略 / 上一 gold action 痕迹”，不能全部归因于纯视觉程序进展理解；
- 若真实部署改成模型自回灌自己的历史输出，该信号分布可能变化，必须另做 self-fed robustness 审计。

该事后分析只解释冻结预测，不参与推广、调参或特征选择。产物为 `analysis.json`，SHA256 `f0ed176408d2599ed57e5725f8315fbf0e48f3750e6bc7b4b82921d9885bfd97`。

## 7. 最终头与在线状态闭环

五折均选择 L2 `0.01`。最终 full-development 头冻结为五折 L2 中位数 `0.01`，threshold 使用五个 calibration threshold 的中位数 `0.14439966662436324`，没有在 full-fit predictions 上重选。

最终头在全部 700 public-development sessions 上重拟合：

| 项目 | 结果 |
|---|---|
| Head features / params | 2,075 / 2,076 |
| Head SHA256 | `c1f3445d0fd5205983174c7d60b6d0930c97ec2dcbd115e0b67c7ca03996b420` |
| Full-fit prediction SHA256 | `cc0e9716383d7422253fbcf47c861cc26fdc4c42a6606fdebfc91bb2ecc03ea7` |
| Full-fit Macro | 0.7544，仅作训练闭环检查 |
| Refit wall time / GPU | 12.11s / 未使用 |

独立 full refit 的 head、records、predictions、metrics 与正式产物逐字节一致。

在线状态机按 session 顺序逐 chunk 更新 previous state 和 prefix sum。全量 9,935-chunk 回放结果：

- 8 个动态标量最大绝对差 `0.0`；
- 1,024 维 hidden delta 最大绝对差 `0.0`；
- 9,935/9,935 decisions 完全一致；
- 最大 logit 差 `2.95e-7`，小于冻结容差 `1e-6`。差异来自离线标量矩阵为 float32、在线因果标量保留 Python float 精度；
- online audit SHA256 为 `6cf48767e682c575ce69f21358627757d1edc394aafc83dcd1748c8ae0ee31bb`。

随后将 `dynamics_fused` 接入现有 `proactive_d1.run_deploy`，继续使用已推广的 `shared_vision` 特征路径。启动前检查显示物理 GPU 2--7 为空闲；smoke 选择 GPU 2，未访问 GPU 0/1 上已有的约 22/21 GB 进程。范围为原始 session 0 的 10 个 chunk，开启 hidden/delta 记录：

| 项目 | 结果 |
|---|---:|
| Wall / session compute | 42.989s / 33.73s |
| Peak GPU memory | 3,466,037,248 bytes，约 3.466 GB |
| Raw response / prompt tokens | 10/10 exact / 10/10 exact |
| Tag margin / current hidden | 10/10 exact / 10/10 exact |
| Dynamics scalars / hidden delta | 10/10 exact / 10/10 exact |
| Decision / answer | 10/10 exact / 10/10 exact |
| Max logit difference | `7.2178e-8` |

该子集官方 Macro `0.8990` 只描述 10 个训练集内 chunk，不是性能估计。关键结论是在线 GPU 路径与冻结 R0、D1 cache、D3 final head 的八类中间量/输出完全对齐。Equivalence audit SHA256 为 `3dfab7cc98d9e01b25e4dceda327bbb97ed4460c39c417ef136fc5aea2ae6191`。

至此可以称为“D3 serialized head、session-local state、shared-vision GPU CLI 已闭环”。仍不能称为“提交容器已验收”：完整 prediction packaging、容器启动命令和官方平台上传均未执行。

## 8. 复现与产物

| 产物 | 路径 / SHA256 |
|---|---|
| OOF config | `configs/d3_internvl35_1b_causal_dynamics_oof.json` / `0373ccf9971229d4515d79c027731f97b8f71e1f16328bdae2ec839ab8e54d07` |
| OOF artifact | `output/experiments/20260717_internvl35_1b_causal_dynamics_d3_oof_v1/` |
| Primary predictions | `c8b836d90768ef609747b40d968d1551c5b90aa91c72fd6d017514eed3b3535a` |
| Official metrics | `4f661845fb6c7adc92c22ca6aab7d3e641efc7fee56bbd25b1125a9dbdfaeae0` |
| Final config | `configs/d3_internvl35_1b_causal_dynamics_final.json` / `f0e828002c8372a891382cbcb5ac52ba1614186805fd1c6cb2a80d98b733da9d` |
| Final artifact | `output/experiments/20260717_internvl35_1b_causal_dynamics_d3_final_v1/` |
| Deploy config | `configs/d3_internvl35_1b_causal_dynamics_deploy_shared_vision.json` / `bc81144dc9904a01376c828024e1a5011bb55e51f32c99b3926071b0c690abd1` |
| GPU smoke | `output/experiments/20260717_internvl35_1b_causal_dynamics_d3_deploy_shared_vision_v1_smoke1/` |

第二个独立 OOF 运行与正式产物的 comparison、feature audit、四个变体 predictions/metrics 和主 diagnostics 全部逐字节一致。`src/proactive_d3` 的 5 个单元测试与 `compileall` 均通过。

## 9. 结论与下一步

D3 回答了原问题：严格因果的跨 chunk dynamics 在 public validation session-level OOF 上有稳定、显著且非首 chunk 的增益，主结果 `0.6690` 相对 D1 `+0.0349`，应推广为新的科学 OOF 基线和提交候选决策头。

下一步按优先级为：

1. 不再对同一五折搜索 dynamics 特征、窗口、L2 或 threshold；
2. 在不改动冻结模型的前提下做 dialog-policy / self-fed robustness 审计，量化 gold-history 痕迹消失后的退化；
3. U1 双人评分继续独立完成，D3 不替代 utterance/state 质量结论；
4. 真正提交前完成 prediction/container packaging 审计，并重新核对官方实时规则；
5. 任何外部上传或 leaderboard 提交仍需用户授权。
