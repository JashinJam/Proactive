# U0 双评测员聚合、一致性与 Early-Chunk 选择报告

> 日期：2026-07-20  
> 实验 ID：`20260720_u0_dual_reviewer_analysis_v1`  
> 状态：A/B 原始评分冻结、完整聚合和分歧清单已完成  
> 性质：public-validation 困难分层诊断，不替代官方 scorer，不估计总体发生率

## 1. 结论

U0 的双人结果确认了两个稳定结论，也暴露了一个不能静默平均的问题：

1. 固定 fallback 的内容质量显著低于 nonfallback，且主要错误稳定地表现为 generic；
2. second chunk 仍是最弱的内容位置，pair-average composite 只有 `1.7857`；
3. 两名评测员对 correctness/specificity 有较好一致性，但对 groundedness 使用了明显不同
   的评分尺度。groundedness 的 A/B 均值为 `2.4375/4.0125`，quadratic kappa 仅
   `0.0508`，不能把两人平均值直接当作可靠的 grounding promotion 指标。

因此，U0 支持继续做 early-chunk utterance 对照，但新方案必须重新盲评
groundedness/hallucination，并保留原始分歧；当前结果不支持恢复大规模 state 标注。

## 2. 输入与完整性

| 输入 | 数量 | SHA256 |
|---|---:|---|
| Reviewer A | 200 | `4e1476f6576150d1e135b9ad8ef047029c04829d78a234114e93592d6d0c9feb` |
| Reviewer B | 200 | `7d9eb9fbea15bdfd5b89d4c4165268b05442f65e42ce8ef8c085ae1a7b93d750` |
| Blind items | 200 | `35fef3e15a4fbe73efdfc61a50e9ca550556434316ac7ff0cc15784d9c298337` |
| Review key | 200 | `84e8508bf90bb1c76398402dcff051d82bbbd89bbc804640c65fbef33d7f2afb` |

程序验证了 200 个 `review_id` 均恰有 A/B 各一条，无重复、缺失或 slot 错位。160 个
`model_action=spoke` 项内容字段完整；40 个 silent 项内容字段全部为空，没有把缺失值当 0。

## 3. 总体双人结果

| 指标 | Reviewer A | Reviewer B | Pair average | B-A / 95% session CI |
|---|---:|---:|---:|---:|
| Timeliness | 3.0550 | 3.4200 | 3.2375 | `+0.3650` / `[+0.1384,+0.5950]` |
| Content composite | 2.0125 | 2.6000 | 2.3063 | `+0.5875` / `[+0.4564,+0.7190]` |
| Correctness | 1.8063 | 1.9688 | 1.8875 | `+0.1625` / `[0,+0.3313]` |
| Specificity | 1.9313 | 2.2563 | 2.0938 | `+0.3250` / `[+0.1706,+0.4817]` |
| Actionability | 1.7625 | 2.2875 | 2.0250 | `+0.5250` / `[+0.3273,+0.7250]` |
| Groundedness | 2.4375 | 4.0125 | 3.2250 | `+1.5750` / `[+1.3750,+1.7750]` |
| Plan consistency | 2.1250 | 2.4750 | 2.3000 | `+0.3500` / `[+0.1543,+0.5419]` |

两名评测员共 400 个 timing 判断中，`should_interrupt=yes/no/uncertain` 分别为
216/166/18，即 `54.0%/41.5%/4.5%`。A/B exact agreement 为 `68.0%`，Cohen's kappa
为 `0.4080`；有 46 条出现直接 yes/no 对立。官方标签仍是排行榜依据，人评只描述实际
交互判断。

## 4. 一致性与分歧

| 维度 | Exact | Within one | Quadratic kappa | Spearman |
|---|---:|---:|---:|---:|
| Correctness | 63.13% | 86.25% | 0.6655 | 0.6972 |
| Specificity | 60.00% | 84.38% | 0.7050 | 0.7827 |
| Actionability | 52.50% | 73.75% | 0.5365 | 0.6011 |
| Groundedness | **19.38%** | **35.63%** | **0.0508** | **0.2438** |
| Plan consistency | 27.50% | 81.88% | 0.4672 | 0.3704 |
| Timeliness | 24.00% | 67.00% | 0.3384 | 0.3830 |

Generic flag 一致性很高：exact `98.13%`、kappa `0.9625`。Hallucination flag exact
虽为 `90%`，但 A/B rate 为 `10.0%/3.75%`，kappa 只有 `0.2308`；unsafe flag 为
`5.0%/0%`，8 条不一致。低发生率下不能仅凭 exact agreement 或 kappa 选一个结论。

按冻结规则共有 162/200 条进入宽口径仲裁清单，其中 103 条由 groundedness 相差至少
2 分触发，66 条由 timeliness 触发，76 条 primary error type 不同。这说明仲裁清单适合
定位 rubric 分歧，不应被误读为“81% 样本都完全不可用”。

## 5. Fallback 与位置

| 分组 | Items | Should yes | Timeliness | Content composite | Hallucination rate |
|---|---:|---:|---:|---:|---:|
| Fallback | 80 | 55.00% | 3.1625 | **1.5413** | 0.63% |
| Nonfallback | 80 | 59.38% | 3.2250 | **3.0713** | 13.13% |
| Silent FN | 40 | 41.25% | 3.4125 | - | - |

Nonfallback 明显更有内容，但伴随更高的 hallucination 风险；fallback 不是安全的高质量
方案，只是内容为空泛、因而较少触发 hallucination。

| 位置 | Items / spoken | Should yes | Timeliness | Content composite | Hallucination rate |
|---|---:|---:|---:|---:|---:|
| First | 20 / 19 | 95.00% | 4.3000 | 2.7421 | 5.26% |
| Second | 30 / 21 | 60.00% | 3.2000 | **1.7857** | 4.76% |
| 2--4 | 49 / 39 | 58.16% | 3.3265 | 2.3256 | **11.54%** |
| 5--9 | 50 / 40 | 50.00% | 3.0100 | 2.3725 | 7.50% |
| 10+ | 51 / 41 | 34.31% | 2.9804 | 2.2878 | 3.66% |

这支持把下一轮重点放在 second 与 2--4：second 是内容冷启动最明显的位置，2--4 则
已经开始生成具体内容但 hallucination 风险最高。

## 6. 后续 U2 冻结样本池

为了避免人工挑案例，下一轮使用满足以下全部条件的完整交集：

1. U0 位置为 second 或 2--4；
2. Reviewer A/B 均判断 `should_interrupt=yes`；
3. D4 五折 OOF 和 D4 全公共 refit 都预测 interrupt。

交集共 21 条，不再二次按分数选择：second 9、2--4 12；Arts and Crafts/Chef/Tutorial/
Handyman 为 7/5/6/3；原始 fallback/nonfallback 为 14/7。该池已被人工评分且来自公共
validation，因此只能作为 review-informed development diagnostic，不能作为独立验证。

## 7. 路线结论

1. 固定句 fallback 的失败由双人结果确认，继续维持它不能解决 early utterance。
2. 新实验固定 D4 gate，只改变正文生成，不修改 threshold、decision features 或输入策略搜索。
3. 自动结果先比较 coverage、fallback、文本敏感性和事实块利用；grounding/hallucination
   的质量结论必须来自新的盲评包。
4. 由于 groundedness rubric 尚未校准，新盲评应增加“候选中每个可验证事实是否被当前
   interval 支持”的显式检查，不能只给一个整体主观分。
5. U1-B 和独立 state package 仍未完成，state/granularity/GRPO 继续冻结。

## 8. 复现工件

```text
协议: annotations/u0_dual_reviewer_analysis_v1/PROTOCOL.md
配置: configs/u0_dual_reviewer_analysis_v1.json
实验: output/experiments/20260720_u0_dual_reviewer_analysis_v1/
analysis.json SHA256:
a1bef9ebd2215fa079bfc41d3d5577f7e7c22c82880d6916076ee75b809616f8
item_records.jsonl SHA256:
fa8d6ea6869d7ec55df46182a1cae1ca49dcbfffbb9c4cf3739029427b1caaca
disagreement_cases.jsonl SHA256:
5ad1003c57981caa6451381daa211a6e563580376655f41256850f54b43a2a61
```

分析代码：`src/proactive_u0/ratings.py` 与 `src/proactive_u0/analyze_ratings.py`。
