# Zymo 微生物电信号分类开发规范

## 1. 当前执行版本

项目代号：`new0715`。

当前阶段不是继续复跑 v8-v19，而是重新建立可信的数据协议和可打包的软件边界。

当前主线：

```text
Phase 0  legacy protocol audit
Phase 1  file-held-out frozen Bonito CE baseline [complete]
Phase 2  sequence-teacher -> signal-student cross-modal distillation [complete, no scale-up]
Phase 3  latest-apple partial fine-tuning [complete, no scale-up]
Phase 3b legacy-stone bounded ablation [complete]
Phase 4  cross-file robustness and reproducibility closure [smoke complete, full pending]
Phase 5  package stable inference pipeline [raw-cache inference and fixed fixture complete]
Phase 6  pre-registered 5-9 class scaling [manifests/runner ready; wait for v4 GPU closure]
```

## 2. 已知事实

- 旧内部最强 Signal v10：test macro-F1 `0.8957`。
- 旧 sequence k-mer MIL：test macro-F1 `0.9931`。
- 独立 D6306 known-9 forced classification：accuracy `0.7420`、macro-F1 `0.7051`。
- D6306 v24 accepted hierarchical accuracy `0.8894`，但 accepted rate 仅 `0.7731`。
- 未见 `Lactobacillus fermentum` abstain rate `0.5702`，false-accept rate `0.4298`。
- 旧 split 中 read_id 不泄漏，但全部 1296 个 species-CCF 文件都横跨多个 split；这意味着原评估不是 file-held-out。
- 当前 14 GB layer-9 Bonito chunk cache 可复用；旧 raw intermediate chunks 已删除，因此不能直接做 Bonito partial fine-tuning。
- 新 file-held-out sequence teacher test macro-F1 `0.9822`。
- 新 file-held-out frozen Bonito CE test macro-F1 `0.8080`。
- 新 file-held-out cross-modal KD test macro-F1 `0.8166`；val 增益仅 `+0.0024`，不扩大。
- 新 file-held-out 8-group CE/KD test macro-F1 `0.8715/0.8705`。
- test CCF 文件级 accuracy 最低约 `0.59-0.63`，说明问题不只来自 LB01/LB12，还包括文件/批次稳健性。

## 3. 对根因的判断

“Bonito 后面的策略没训练好”只解释了一部分现象。更完整的假设是：

1. Bonito 预训练目标是 basecalling，不是 species discrimination。
2. frozen backbone 阻止 species-aware loss 修改底层表征。
3. read-level 随机 split 允许同一 CCF 文件出现在 train/val/test，可能放大文件/批次捷径。
4. D6306 是独立混样域，外部结果暴露明显 domain shift。
5. 固定 `signal_len >= 11000` 产生物种相关 eligible bias，不能把 eligible-read abundance 当作全 FASTQ abundance。

所以新路线首先修协议，其次才修模型。

## 4. 小规模协议

默认每类 3000 reads：

| split | reads/species | CCF 文件约束 |
|---|---:|---|
| train | 1800 | 与 val/test 文件完全不同 |
| val | 600 | 只做早停和超参决策 |
| test | 600 | 最终一次评估 |

每个 CCF 文件最多抽 200 reads，以增加每个 split 覆盖的文件数。任何 CCF 文件不得跨 split。

v1 CE/KD 公平对照还要求 read_id 存在于已有 sequence k-mer cache。CE 与 KD 使用完全相同的 matched reads；这是一项训练期配对约束，不代表最终 CCF5-only 推理需要 FASTQ。

主指标：macro-F1。辅助指标：accuracy、balanced accuracy、per-species recall、worst-species recall、confusion matrix、ECE/置信度分布。

## 5. 模型消融

### A. Frozen Bonito + Transformer MIL + CE

```text
CCF5
-> latest /mnt/zzbnew/poregpt/shared/signal.py
-> apple standardization with current shared post-processing
-> trim first 5000 points
-> 6000-point chunks, overlap 3000
-> frozen Bonito encoder prefix [0:9]
-> chunk 768D
-> 768->384->256 chunk adapter
-> 2-layer Transformer
-> gated attention pooling
-> 9-class head
```

历史 v1 frozen baseline 曾复用旧 cache；正式 v2 不复用该 cache，而是从 CCF5 按固定 manifest 重新生成 latest-apple raw chunks 和 fresh Bonito 768D。

### B. Frozen Bonito + Sequence Teacher KD

结构与 A 相同，增加训练期 teacher logits 和 256D embedding 对齐。teacher 也必须按新 CCF-group split 从头训练，禁止直接复用旧随机 split checkpoint。推理仍为 CCF5-only。

### C. Partial Bonito Fine-tuning

虽然 B 未达到扩大门槛，但它证明后端 KD 无法跨越 frozen representation 上限。下一项仅允许做一次小规模 backbone-level 诊断：只对当前 27000 条 matched reads 重建 raw chunks，只解冻最后 1-2 个 Bonito blocks。

```text
PFT-A: last 1 block trainable + CE
PFT-B: last 2 blocks trainable + signal augmentation consistency
```

Bonito 后部层使用低学习率，分类/MIL head 使用独立较高学习率。禁止全量解冻，禁止先扩大到 10000/100000 reads/species。

具体实现：

- `PFT-A`: children `[8]` 可训练，即第 5 个 LSTM。
- `PFT-B`: children `[7:9]` 可训练，即最后 2 个 LSTM。
- children `[0:train_start]` 在 `no_grad` 下保持冻结。
- 可训练 Bonito 权重保留 FP32 master copy，前向使用 AMP。
- 先从 apple raw chunks 生成 fresh Bonito 768D cache，并训练 fresh apple frozen CE。
- PFT 从该 fresh apple frozen CE 的 Transformer MIL head 初始化；禁止加载旧 stone CE checkpoint。
- `max_chunks=8`；正式比较前必须以同样 8 chunks 重评 frozen CE。
- raw cache 为 float16 packed memmap，预计 266991 chunks、约 3.2 GB，可按物种和 CCF 文件断点续跑。
- cache 审计只检查 shape、read/chunk 对齐和 finite values，不在此阶段选择分类阈值。

v2 已完成：PFT-A test macro-F1 `0.7165`，未过扩量门槛。64-chunk frozen apple 控制为 `0.6809`，排除了8 chunks是主要根因。

### D. v3 legacy-stone 有界消融

输入标准化固定为旧项目实际使用的：

```text
/mnt/zzbnew/rnamodel/shenhaojie/poregpt/poregpt/utils/signal.py
nanopore_process_signal(signal, "stone")
```

不得用新版 shared stone 冒充该实现。v3 执行顺序：

```text
fresh stone raw chunks
-> fresh stone Bonito 768D
-> frozen mean / attention / Transformer (64 chunks)
-> val 选择聚合头
-> PFT 1 / 2 / 3 / 5 LSTM blocks (16 chunks, CE)
-> val 选择深度
-> CE / SupCon / embedding Mixup / C2LR+MI-Mix
-> val 选择最终模型
-> final test once
```

`embedding Mixup` 只混合最终 read embedding；`MI-Mix` 在随机可训练 Bonito LSTM 中间层混合 hidden states。二者必须在结果表中分开命名。

v3 最终结果：Transformer 聚合、解冻最后3个LSTM、C2LR+random-layer MI-Mix；test accuracy `0.8459`、macro-F1 `0.8452`。该模型是当前严格 file-held-out signal 主模型，但未达到0.90目标。C2LR/MI-Mix 相对同深度CE的 validation 增益仅 `+0.0053`，作为消融保留，不继续扩展参数组合。

### E. v4 有界结题实验

v4 不再搜索层数、聚合头或loss权重，只执行：

1. 固定v3配置，补 `seed=3407/2026` 两次重复，和seed42共同报告mean/std；三者使用同一split和同一frozen head初始化，因此估计的是PFT训练随机性，不冒充完整数据重采样方差。
2. 单一跨文件候选：同物种正样本必须来自不同CCF文件，训练目标为cross-file SupCon与CCF-file GroupDRO。
3. 候选test门槛：validation macro-F1相对v3至少 `+0.015`，同时CCF-file 10%分位accuracy至少 `+0.03`。
4. 任一门槛未通过，不评估候选test，并停止Bonito闭集模型优化。

v4复用 `artifacts/cache/v3_stone_raw_v1_3000`，不重建stone raw cache或768D cache。smoke已经通过，分数不用于结论。

## 6. Atlas 的位置

Atlas 不再作为闭集 9 分类主结果。它保留用于：

- few-shot retrieval；
- prototype distance；
- OOD/Unknown 阈值校准；
- embedding 可解释性。

正式称呼为 `embedding atlas` 或 `reference embedding bank`，不是知识图谱。

定量依据：旧 v17 小规模 Bonito MLP 中，classifier 与 centroid test macro-F1 为 `0.8233` 与 `0.8214`；SupCon 后为 `0.8303` 与 `0.8305`。Atlas/centroid 没有形成稳定增益。它只能改变决策规则，不能补回 frozen embedding 中缺失的物种信息。

## 7. 工程与缓存规范

- 旧缓存通过 `config/resources.json` 只读引用，不复制。
- 所有耗时中间结果必须登记在 `artifacts/cache_registry.csv`。
- 新运行写入 `artifacts/runs/<run_id>/`。
- teacher logits/embedding 可复用，写入 `artifacts/teacher/`。
- 临时数据只能写 `/tmp/new0715_*`，完整性校验后删除。
- 不允许脚本自动删除旧项目文件。
- checkpoint 必须同时保存 config、label map、split manifest hash、best val epoch。

## 8. 执行命令

协议审计：

```bash
cd /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
bash scripts/run_v0_audit.sh
```

CE 与 KD 小规模双卡实验：

```bash
cd /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
nohup bash scripts/run_v1_small_kd.sh \
  > logs/run_v1_small_kd.master.log 2>&1 &
```

查看：

```bash
tail -f logs/run_v1_small_kd.master.log
tail -f logs/v1_ce_gpu0.log
tail -f logs/v1_kd_gpu1.log
```

Bonito partial fine-tuning：

```bash
cd /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715

nohup bash scripts/run_v2_pft_small_2gpu.sh \
  > logs/run_v2_pft_small.master.log 2>&1 &
```

v2 runner 的环境分工固定为：

- `/usr/bin/python3`: `pyccf5` 读取、latest apple 标准化和 raw cache。
- `bonito090_py39`: fresh apple Bonito 768D、fresh frozen baseline、PFT-A/PFT-B。

v3 legacy-stone 单卡周末消融：

```bash
cd /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
nohup bash scripts/run_v3_stone_weekend_cuda0.sh \
  > logs/run_v3_stone_weekend.master.log 2>&1 &
```

该 runner 只使用 `cuda:0`，支持从 raw cache、768D cache、每个解冻深度和每个 objective 断点续跑。

v4 有界结题实验：

```bash
cd /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715
nohup bash scripts/run_v4_closure_cuda0.sh \
  > logs/run_v4_closure.master.log 2>&1 &
```

该runner只使用 `cuda:0`，复用v3 stone raw cache；smoke、两个repeat seed和cross-file候选均按 `summary.json` 断点跳过。

## 9. 决策标准

| 结果 | 决策 |
|---|---|
| CE val macro-F1 < 0.80 | 停止堆模型，优先补独立批次/重复数据 |
| CE 0.80-0.90 | KD 可以小试，不扩大 |
| KD 相对 CE val 增益 < +0.02 | KD 记为负结果；v1 实测 `+0.0024` |
| KD val >= 0.90 且增益 >= +0.02 | 扩到 10000 reads/species 或进入 partial fine-tuning |
| PFT val >= 0.88 且相对 CE 增益 >= +0.03 | 才允许扩大 partial fine-tuning |
| PFT val < 0.86 或增益 < +0.02 | 停止 Bonito 闭集调优，转独立重复数据/reference matching |
| file-held-out 好、外部混样仍差 | 增加 run/domain-held-out 与域适配 |

目标 `>0.90` 必须是在 file/run-held-out 上达到，而不是回到 read-level 随机 split 追分。

这里的模型选择和未来 OOD/Unknown threshold 均在 val 上动态校准。选择性分类不再使用`macro-F1 * coverage`乘积目标；应预先声明accepted accuracy目标、总体coverage下限、每物种coverage下限和最小增益，再选择满足约束的最大coverage。无可行解时必须关闭拒识并报告`threshold_enabled=false`。raw cache 的数值完整性检查不是分类阈值，不能与 Atlas cosine threshold 混为一谈。

## 10. 最终工具包设计

当前已实现：

```text
squiggle-species inventory-ccf INPUT
squiggle-species validate-manifest MANIFEST
squiggle-species predict EMBEDDING_BAGS
squiggle-species predict-raw-cache RAW_BAGS
squiggle-species calibrate VAL_PREDICTIONS
squiggle-species report TEST_PREDICTIONS
```

Backbone扩展采用`SignalBackboneAdapter`：模型提供方自行声明`feature_dim`、兼容预处理profile和有序`trainable_units`，可用`module:factory`动态加载。PFT的“最后N层”是adapter-specific参数；当前Bonito消融选择3个LSTM blocks，不代表其他模型也应解冻3层。

当前V1.0完成的是固定数据、Bonito embedding/raw-cache推理、validation动态校准和报告闭环。任意backbone的接口已实现，但其通用训练runner、直接CCF5统一cache CLI和模型bundle签名仍属于V1.1，文档不得把接口存在写成所有模型都已端到端验证。

## 11. 5–9类扩展实验

v5只回答“最佳冻结路线在不同类别粒度下如何变化”，不再搜索新网络。物种选择不看validation/test分数：所有子集固定包含LB01/LB12困难对，再按参考基因组5-mer组成距离做greedy max-min扩展。

```text
k5: LB01, LB12, LB07, LB06, LB02
k6: k5 + LB08
k7: k6 + LB11
k8: k7 + LB09
k9: k8 + LB18
```

每类`train/val/test=1800/600/600`，CCF-file-held-out。k=5/6/7/8分别重新训练类别专属frozen head与Stone + Bonito PFT-3 + Transformer + C2LR/MI-Mix；k=9复用v3冻结结果。hard-pair权重沿用v3 validation已冻结pair，并只启用当前子集中存在的pair。不得把9类预测事后过滤后称为k类正式结果。

v4完成、`cuda:0`释放后执行：

```bash
GPU=1 nohup bash scripts/run_v5_class_count_scaling.sh \
  > logs/run_v5_class_count_scaling.master.log 2>&1 &
```

输出为`artifacts/summaries/v5_class_count_scaling/class_count_metrics.csv`和`class_count_scaling_curve.png`。即使低类别达到0.90，也必须保留9类主结果、完整物种名单和预注册选择规则。

待v4最终冻结后补`cache-ccf`与`classify-ccf`编排层；底层raw-cache PFT推理已经可用，不需要再设计第二套模型加载逻辑。

最终 GitHub 包必须包含：

- CCF5 reader adapter；
- 标准化、trim、chunk 配置；
- Bonito 权重路径检查；
- 可断点 cache；
- frozen/partial student 推理；
- read-level prediction 与 sample abundance；
- validation risk-coverage、correctness AUROC/AURC、每物种acceptance rate；
- Unknown/abstain及无可行阈值状态；
- 运行时间和完整性审计；
- smoke fixtures、单元测试、示例 config、模型卡。

软著和工具类文章在 CLI 对独立样本稳定后再做，不提前包装未验证模型。

固定数据采用fixture、benchmark-mini、benchmark-full三层。当前checkpoint与`legacy-stone-v1`绑定，Apple是独立profile；Atlas不进入闭集默认链路；D6306 10菌混样只做冻结外部known-9泛化和unknown10拒识。完整规范见`docs/TOOLKIT_BENCHMARK_PROTOCOL.md`。
