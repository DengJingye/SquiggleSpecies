# 旧项目实验资产盘点

## 已经尝试了什么

| 路线 | 核心方法 | 结果/结论 |
|---|---|---|
| v2 | frozen Bonito read 768D + MLP/ensemble | macro-F1 约 0.8799 |
| v4 | chunk cache + 手工统计 pooling | 0.8763，负结果 |
| v5 | gated Attention MIL | 0.8869，有小幅提升 |
| v6 | LB01/LB12 hierarchical resolver | gated 0.8871；oracle 0.8972；路由未兑现上限 |
| v8 | chunk Transformer MIL | 约 0.8869，未优于 v5 |
| v9 | hard-pair curriculum/margin | 0.8943，有限提升 |
| v10 | C2LR/MI-Mix inspired pretrain + CE fine-tune | 0.8957，旧内部最佳 |
| v11 | Bonito layer ablation | 未形成比 layer-9 更强的稳定结果；耗时大 |
| v12 | 从零 raw-signal tokenizer | 失败，不扩大 |
| v13 | 复刻旧六分类 metric/Atlas 思路 | 未解决 Zymo9 上限 |
| v14 | DNABERT-S-inspired adapter | 增益极小，不扩大 |
| v15-v19 | 小组电信号基座/PoreGPT corrected、多层 probe | 小规模约 0.58-0.67，弱于 Bonito |
| v20 | sequence-hard stress test | Zymo9 未复现强 signal-hard advantage |
| v21 | LB01/LB12 合并为 8-group | Signal v10 macro-F1 0.9318，作为 coarse 补充 |
| v24-v26 | selective/coarse/abstain 与阈值审计 | 内部 accepted set 较高，但不能代替全覆盖 9 分类 |
| v28 | D6306 独立 10 菌混样 blind evaluation | known-9 macro-F1 0.7051；明显域偏移；unknown false accept 0.4298 |

## 工作量不只是“调了几个参数”

已经覆盖：

- raw CCF5 抽样、标准化、chunk 化与 Bonito embedding cache；
- read-level 与 chunk-level 两套输入；
- MLP、gated attention MIL、Transformer MIL；
- supervised contrastive、C2LR、MI-Mix、mixup、hard curriculum、pair margin；
- embedding atlas/KNN、动态阈值、OOD/Unknown、hierarchical resolver；
- Bonito 多层表征与小组电信号基座诊断；
- k-mer MLP、contrastive、sequence MIL、minimap2；
- sequence-hard、coarse 8-group、selective prediction；
- 约 199 万 eligible signal reads 的真实混样盲推理与 10-reference minimap2 外部评估。

## 新结论

旧实验不是“白做”，它已经排除了大量低收益方向。new0715 不再问“还能加什么 loss”，而是问：

1. 在 CCF-file 隔离后，真实 signal 表征上限是多少；
2. species-aware sequence teacher 能否把语义转给 signal student；
3. 只有蒸馏有效时，梯度是否值得进入 Bonito 后部层；
4. 若仍失败，是否应该从闭集分类转向 raw-signal reference matching。

## new0715 v1：file-held-out 公平对照

| 方法 | val macro-F1 | test macro-F1 | 结论 |
|---|---:|---:|---|
| Sequence k-mer teacher | 0.9863 | 0.9822 | 数据划分可学，sequence species signal 强 |
| Frozen Bonito Transformer MIL + CE | 0.8352 | 0.8080 | 明显低于旧 random-read split |
| Frozen Bonito + cross-modal KD | 0.8375 | 0.8166 | val 仅 `+0.0024`，不扩大 |

补充诊断：

- CE 的 1035 个 test 错误中，369 个为 LB01/LB12 互错，占 `35.65%`。
- KD 将该互错降至 311 个，但整体 8-group macro-F1 没有提升：CE `0.8715`，KD `0.8705`。
- 单个 held-out CCF 文件准确率差异大，CE `0.585-0.925`，KD `0.625-0.930`。
- 结论：瓶颈包含 LB01/LB12，但不止这个 pair；frozen signal representation 对文件/批次变化不够稳健。
- Atlas/centroid 不再作为主分类器。旧 v17 的 classifier/centroid 结果几乎相同，未证明 Atlas 能带来稳定准确率增益。
