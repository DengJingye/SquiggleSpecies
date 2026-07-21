# 文献依据与新路线

## 文献给出的直接启发

### DNABERT-S

DNABERT-S 的核心不是“在任意 frozen embedding 后加一次 mixup”，而是把 MI-Mix 和课程对比学习用于可训练的 genome foundation model，使 species-aware objective 能改变中间层和最终 embedding。项目地址：<https://github.com/MAGICS-LAB/DNABERT_S>；论文：<https://arxiv.org/abs/2402.08777>。

对本项目的启发：

- species-aware 目标需要进入表征学习过程，而不应只停留在最终分类头。
- hard-negative curriculum 应由训练数据和验证集定义，不能根据 test 结果手工指定。
- MI-Mix 只有在中间表示可学习时，才真正接近论文中的 manifold instance mixup。

官方论文与源码核对后的实现细节：

- 第1阶段训练 Weighted SimCLR，第2阶段才启用 MI-Mix；不是把所有损失从第1个 epoch 同时相加。
- MI-Mix 在随机选取的 encoder layer 混合 hidden states，而不是只混合最终 embedding。
- 官方默认 temperature `0.05`、mix alpha `1.0`，mix 比例来自 `Beta(alpha, alpha)` 并取较大侧。
- 官方 ablation 显示两阶段 Weighted SimCLR + MI-Mix 优于单独 Weighted SimCLR、单独 MI-Mix、i-Mix 和 SupCon。因此项目同时保留单独 SupCon/Mixup作为消融，但不预设它们一定有效。
- 本项目没有同一 genome 的非重叠序列对，使用同物种 reads/同 read 增强视图构造正对；这是向 raw signal 迁移后的任务差异，报告中必须注明。

### SquiggleNet

SquiggleNet 直接用 1D CNN 处理短 raw signal，并特别规避 barcode/adapter confounding；论文使用的数据规模很大，任务也主要是 human-vs-bacteria 等粗粒度分类。论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC8548853/>。

对本项目的启发：

- raw signal 确实可分类，但粗粒度二分类超过 0.90 不等于近缘 9 物种分类也能达到同样水平。
- adapter、barcode、run/file 信息可能形成捷径；分割和划分协议本身与模型同等重要。
- 应增加直接 raw-signal CNN 小基线，但必须先通过 file/run-held-out 协议。

### DeepSelectNet

DeepSelectNet 证明深度模型可直接分类物种 raw current，但论文也报告 intra-species/更细粒度任务更困难，并指出复杂多物种场景仍需扩展。论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9883605/>。

对本项目的启发：不能用二分类或相距很远的物种结果推断近缘多分类上限；Zymo9 必须单独做跨文件、跨批次验证。

### RawHash / RawAlign

RawHash、RawHash2 和 RawAlign 不把任务完全交给闭集分类头，而是把 signal 量化、哈希、播种、链式匹配或精细对齐到参考基因组。论文：<https://arxiv.org/abs/2301.09200>、<https://arxiv.org/abs/2309.05771>、<https://arxiv.org/abs/2310.05037>。

对本项目的启发：如果 species-aware neural embedding 在独立批次仍不稳定，raw-signal reference matching 是更符合信号原生优势的备选主线，不应继续无限叠加分类 loss。

### Bonito 增量适配

已有工作在 nanopore basecaller 上用增量学习和知识蒸馏适配新任务，同时约束遗忘。论文：<https://www.nature.com/articles/s41467-024-51639-5>。

对本项目的启发：相比完全解冻，优先尝试后部少量层的低学习率适配，并保留 basecalling teacher/旧输出约束，降低灾难性遗忘。

## 当前最合理的研究顺序

### Phase 0：协议审计

1. 旧 split 的 read_id 虽无重复，但同一个 CCF 文件同时出现在 train/atlas/val/test。
2. 新 split 必须按 `ccf_file` 分组，整个文件只能属于一个 split。
3. 若能获得独立 run/生物重复，最终必须升级为 run-held-out；file-held-out 只是一道最低门槛。

### Phase 1：冻结特征的小规模可信基线

- 复用现有 layer-9 Bonito chunk cache。
- 每类 3000 reads：train 1800、val 600、test 600。
- 每个 CCF 文件最多取 200 reads，避免单文件占据一个 split。
- 同一模型分别跑 CE 和 sequence-teacher KD。
- 只有 `val macro-F1 >= 0.90` 且复杂方法相对 CE `>= +0.02`，才扩数据。

### Phase 2：跨模态知识蒸馏

训练期：

```text
同一 read 的 basecalled sequence
-> k=6 sequence teacher
-> teacher logits + 256D teacher embedding

同一 read 的 raw signal
-> frozen Bonito 768D chunks
-> Transformer MIL student
-> student logits + 256D student embedding
```

损失：

```text
L = L_species_CE
  + lambda_kd * T^2 * KL(student_logits/T, teacher_logits/T)
  + lambda_align * cosine_distance(student_embedding, teacher_embedding)
```

推理期完全不需要 FASTQ：只保留 signal student。

为避免旧 sequence teacher 的随机 read split 向新评估间接泄漏，v1 会在同一个 CCF-group split 上从头训练一版小型 k-mer teacher。旧 checkpoint 仅作为历史结果和架构参考，不用于生成新 test 的 teacher 目标。

### Phase 3：条件式 partial fine-tuning

仅当 Phase 2 有明确增益才执行：

- 重建小规模 raw chunk cache，不恢复 90 万 reads 全量中间数据。
- 冻结 Bonito 早期卷积和大部分 recurrent blocks。
- 只解冻最后 1-2 个 encoder blocks，学习率设为 head 的 1/10 到 1/50。
- 使用 CE + KD + consistency，验证集早停。

### Phase 4：失败后的备选

若 file/run-held-out 仍明显低于 0.90：

1. 获取每物种至少 2-3 个独立测序域，解除 species 与 batch 的混杂。
2. 评估 RawHash/RawAlign-style reference matching。
3. 将模型定位为 coarse/selective classifier，而不是宣称 9 物种闭集替代序列。

## 明确停止规则

- 小规模 group-held-out CE `<0.80`：先解决数据/域问题，不尝试新 loss。
- KD 的 val 增益 `<0.02`：不扩大 KD。
- partial fine-tuning 的 val 增益 `<0.02`：停止 Bonito 分类头路线。
- 新方法只提高 random-read split、却不提高 file/run-held-out：判定为无效。
- D6306 若用于无监督域适配，就不能再作为最终 blind test；必须另留独立混样。
