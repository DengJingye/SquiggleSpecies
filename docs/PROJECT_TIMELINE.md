# 项目实验时间线

本文使用两套前缀避免版本混淆：`H-v*`表示历史random-read探索，`S-v*`表示严格CCF-file-held-out结果。

## 起点：6分类

历史6分类采用`frozen Bonito 768D -> MetricProjector 256D -> contrastive/OOD -> Atlas KNN`。Normal macro-F1约`0.941`，sequence-hard top50/species约`0.909`，提出了“sequence-hard和signal-hard可能不是同一难度轴”的假设。该结果是可行性起点，不是Zymo9推广证据。

## 数据工程

- 从9类CCF5固定抽取每类10万reads，共90万reads、1296个CCF文件。
- 恢复物理电流后执行Stone或Apple标准化，discard 5000，以6000点窗口和3000 overlap切块。
- 生成约902万chunks、202 GB raw intermediate、2.9 GB read-level 768D和约13.9 GB chunk-level 768D cache。
- 9类basecall、canonical k=6、minimap2和learned sequence baseline使用相同read体系进行对照。

## H-v1至H-v10：Bonito冻结后端探索

| 版本 | 方法 | test macro-F1 | 结论 |
|---|---|---:|---|
| H-v1 | Raw 768D Atlas/KNN | `0.6181` | 原始空间较弱 |
| H-v1 | Contrastive 256D Atlas/KNN | `0.8539` | metric learning有效 |
| H-v2 | MLP ensemble | `0.8799` | 强监督闭集baseline |
| H-v4 | mean/std/max pooling | `0.8763` | 简单pooling负结果 |
| H-v5 | Attention MIL | `0.8869` | chunk权重小幅有效 |
| H-v6 | gated pair resolver | `0.8871` | 可部署增益极小 |
| H-v8 | Transformer MIL | `0.8953` | 历史期主要架构增益 |
| H-v9 | hard curriculum | `0.8943` | 未超过Transformer |
| H-v10 | C2LR/MI-Mix | `0.8957` | 旧协议最高，但只增`0.0004` |

H-v7构建动态confusability graph，确认LB01/LB12为最难pair，但它是诊断工具而非新分类器。

## H-v11至H-v19：表征源和基座替换

- Bonito layer5 test macro-F1 `0.7871`，layer7 val `0.8824`，浅层未超过front-9。
- 从零训练raw tokenizer CE/MI-Mix只有`0.2154/0.2045`，不扩大。
- 旧6分类Metric/Atlas同口径复刻从raw `0.6014`提高到`0.8137`，仍未解决9分类。
- DNABERT-S-style adapter CE/C2LR-MI-Mix为`0.8565/0.8560`，未超过H-v10。
- PoreGPT旧协议约`0.36`；修正token和Apple标准化后hlast为`0.6619`。
- 同read公平对照中Bonito/PoreGPT MLP为`0.8233/0.5848`。
- DNAOLMO_V600小规模为`0.6658`，未过`0.70`停止线。

## 序列与科学审计

| 方法 | test macro-F1 |
|---|---:|
| k-mer centroid | `0.8433` |
| k-mer MLP CE | `0.9891` |
| k-mer MLP contrastive | `0.9894` |
| sequence window MIL | `0.9931` |
| minimap2 known-9 | `0.9437` |

H-v20的sequence-hard top50/species中，Signal H-v10为`0.7383`，低于Seq MLP `0.8236`和Seq MIL `0.9413`，因此9分类没有复现6分类的强signal-hard优势。

H-v21合并LB01/LB12后旧协议macro-F1为`0.9318`。H-v22发现top80%和top50%置信子集accuracy为`0.9651/0.9842`。H-v24将fine/coarse/abstain正式化，accepted rate `0.9183`、accepted hierarchical accuracy `0.9602`。H-v25显示局部attention特征AUC最高仅`0.6475`，因此停止local-selector训练。

## H-v28：真实10菌混样

D6306包含25个CCF5、约147.3 GB signal和6429273条FASTQ。冻结H-v10先盲预测1989919条eligible reads，再用官方10物种参考生成高置信参考比对标签。

- known-9 forced macro-F1 `0.7051`；
- accepted hierarchical accuracy `0.8894`，accepted rate `0.7731`；
- unknown L. fermentum abstain `0.5702`，false accept `0.4298`。

结果揭示跨批次domain shift和eligible selection bias。它不等于strict S-v3外部验证。

## S-v0至S-v7：严格重审

S-v0发现旧manifest中`1296/1296`个CCF文件跨split。新协议改为CCF group隔离，每类train/val/test=`1800/600/600`。

| 版本 | 方法 | test macro-F1 | 决策 |
|---|---|---:|---|
| S-v1 | Frozen Bonito CE | `0.8080` | 严格baseline |
| S-v1 | Sequence KD | `0.8166` | 增益不足，不扩大 |
| S-v1 | Sequence k=6 teacher | `0.9822` | 证明严格split可学习 |
| S-v2 | Apple frozen | `0.6882` | 与当前Bonito不匹配 |
| S-v2 | Apple PFT last-1 | `0.7165` | 相对提升但绝对不足 |
| S-v3 | Stone frozen Transformer | `0.8179` | fresh严格控制 |
| S-v3 | Stone PFT-3 C2LR/MI-Mix | `0.8452` | 正式9分类结果 |
| S-v6 | LB01/LB12 binary | `0.6965` | 停止pair-only优化 |
| S-v7 | Reference-diverse 5-class | `0.9280 +/- 0.0033` | 辅助软件演示 |

S-v4三种子严格9分类macro-F1为`0.8457 +/- 0.0009`。Cross-file GroupDRO validation低于baseline，因此没有运行test。项目随后正式停止闭集模型追分，进入工具包、固定数据、结果归档和写作阶段。

## 当前定位

项目证明了CCF5原始电信号能够完成微生物分类，并形成了防泄漏、预处理绑定、流式CCF推理、模型包验证、选择性分类和外部审计工具链。它没有证明signal全面优于sequence，也没有解决近缘物种的严格跨文件细粒度分类。

正式数字和边界见`docs/RESULTS_AND_LIMITATIONS.md`及`results/`。
