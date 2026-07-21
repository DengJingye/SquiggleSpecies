# 方法术语对照

## 我们以前的做法专业上叫什么

| 实际操作 | 推荐术语 | 不建议的叫法 |
|---|---|---|
| 冻结 Bonito，读取中间 768D 表征 | frozen-backbone feature extraction；冻结骨干特征提取 | 重新训练 Bonito |
| 在 768D 后训练任务模块 | feature-based transfer learning；post-backbone task adaptation；表征级下游适配 | 大模型全量微调 |
| 768D 直接接线性层 | linear probing；线性探针 | 端到端训练 |
| 768D 接多层 MLP 分类器 | nonlinear probing / MLP probing；head-only fine-tuning | 只做了一个 KNN |
| 768D -> projector -> 256D + SupCon | supervised metric learning；contrastive representation adaptation | 训练了新的基座模型 |
| 每条 read 是 bag、chunk 是 instance | multiple-instance learning, MIL；多实例学习 | 普通逐 chunk 分类 |
| Transformer 聚合 chunk 序列 | task-specific hierarchical aggregation head；层级聚合任务头 | Bonito Transformer |
| Mixup/MI-Mix | latent-space interpolation；manifold mixup；潜空间插值增强 | 数据简单拼接 |
| Atlas + cosine KNN | embedding atlas / reference embedding bank / non-parametric metric classifier | 知识图谱、图神经网络 |
| 物种混淆关系图 | class-confusability graph / error graph | Atlas |
| pair resolver | hierarchical specialist；gated specialist；类 MoE 路由 | 完整 mixture-of-experts，除非路由和专家均联合学习 |
| 只解冻 Bonito 后部少量层 | partial fine-tuning；discriminative fine-tuning | frozen backbone |
| 在层内插 adapter 或 LoRA | parameter-efficient fine-tuning, PEFT | linear probing |
| 序列 teacher 指导 signal student | cross-modal knowledge distillation；跨模态知识蒸馏 | 推理时使用序列 |

## 对当前系统最准确的一句话

旧主线是：

> 以 basecalling 预训练的 Bonito encoder 作为冻结骨干，提取 chunk-level 768D 表征，再使用任务特定的 Transformer-MIL 聚合头、监督分类目标和对比/潜空间混合目标进行表征级下游适配。

Atlas 版本应描述为：

> 在学习后的 embedding space 中构建 reference embedding bank，并通过 cosine nearest-neighbor retrieval 完成非参数分类和阈值拒识。

它不是知识图谱，也不是图神经网络。

## 为什么旧方法不等于 DNABERT-S

DNABERT-S 的 MI-Mix 会在可训练 backbone 的随机中间层混合隐藏状态，并把 species-aware objective 反传进 backbone。我们的 v10 主要在已经冻结的 Bonito 768D 之后训练 Transformer-MIL，再对 read-level 表征做 mixup/contrastive；梯度不会改变 Bonito 的信号编码。因此二者共享损失思想，但优化深度不同。

