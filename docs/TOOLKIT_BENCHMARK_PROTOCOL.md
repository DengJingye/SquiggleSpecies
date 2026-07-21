# 工具包固定数据与测评协议

## 1. 设计目标

工具包必须同时满足三个要求：别人能快速跑通、研究结果可复现、外部混样不污染模型选择。固定对象是 `read_id + CCF文件分组 + split + 预处理profile`，而不是把所有模型强行绑定到一份通用标准化数据。

## 2. 三层固定数据

| 层级 | 每类 train/val/test | 用途 | 是否用于正式结论 |
|---|---:|---|---|
| fixture | 2/2/2 | 安装、路径、推理和出图smoke | 否 |
| benchmark-mini | 300/100/100 | 快速比较、教程和开发回归 | 仅开发结论 |
| benchmark-full | 1800/600/600 | 严格CCF文件留出内部评估 | 是 |

三层均来自同一 `zymo9-file-heldout-v1` 母manifest。read_id不重复，同一个CCF文件不得跨train/val/test。当前固定产物：

```text
artifacts/benchmarks/zymo9_fixture_v1/
artifacts/benchmarks/zymo9_benchmark_mini_v1/
```

fixture已经包含可移植的`legacy-stone-v1` raw chunk bundle，共54条reads、337个chunks、约4 MB。benchmark-mini manifest共4500条reads，其compact raw bundle也已生成：24611个chunks、约283 MB。二者均不默认提交Git中的`*.npy`，发布时进入独立Release/数据归档。

## 3. 预处理不是通用常量

源CCF5和固定read_id是数据层；标准化、trim和chunk参数是模型接口的一部分，必须随checkpoint一起版本化。

### 当前正式profile

`legacy-stone-v1`：

```text
physical signal
-> global median/MAD, scale floor=1.0
-> no smooth clamp
-> discard first 5000 points
-> 6000-point chunks, overlap=3000
```

它与当前严格file-held-out最强Bonito PFT checkpoint绑定。新版shared Stone额外加入sclamp，不能用来替换该profile。

### 未来基座profile

`apple-sclamp-v1`包括物理范围修复、6000点局部中值去尖峰、central-98% MAD、kernel=5中值滤波和`5->6`平滑压缩。它只用于声明兼容的foundation-model/adapter checkpoint。现有实验中Apple+Bonito明显弱于legacy Stone，因此不是当前默认路线。

## 4. 模块化路线与当前默认配置

工具包不是只保留三条硬编码流水线。完整实验由五个可配置层组成：

```text
preprocessing profile
-> backbone
-> frozen / partial fine-tuning
-> read aggregation
-> training objective and decision head
```

配置入口为`config/toolkit_routes_v1.json`。任何checkpoint都必须声明兼容profile，不能在推理时临时把Stone换成Apple。

### PFT是什么

PFT是`partial fine-tuning`，即保留Bonito卷积前端和较早层冻结，只解冻末端若干LSTM，同时训练MIL和分类头。v3的validation结果为：

| 解冻末端LSTM数 | val macro-F1 |
|---:|---:|
| 1 | 0.8431 |
| 2 | 0.8444 |
| 3 | 0.8480 |
| 5 | 0.8391 |

3层最好，5层退化。因此当前默认是3层，不代表工具包只能支持3层。相对fresh Stone frozen Transformer，最终PFT模型test macro-F1从`0.8179`提高到`0.8452`，提升`+0.0273`。

### Transformer是否必须

不必须。聚合头是`mean / gated-attention / transformer-mil`参数：

| 聚合方式 | val macro-F1 |
|---|---:|
| Mean | 0.8318 |
| Gated attention | 0.8255 |
| Transformer MIL | 0.8360 |

Transformer按validation最好，但仅比Mean高`+0.0042`。发布默认使用Transformer；低资源环境可以使用Mean，不能声称两者等效。

### 对比学习和MI-Mix是否保留

保留，而且必须把训练目标和推理结构分开理解。SupCon、C2LR和MI-Mix在训练期塑造embedding；推理时不再计算contrastive loss，只加载训练后的权重执行分类。

| 末端3层PFT objective | val macro-F1 | 相对CE |
|---|---:|---:|
| CE | 0.8480 | - |
| SupCon | 0.8490 | +0.0011 |
| embedding Mixup | 0.8482 | +0.0003 |
| C2LR + random-layer MI-Mix | 0.8533 | +0.0053 |

C2LR+MI-Mix是当前最优objective，尤其适合作为近缘难分物种的研究路线；但严格协议下增益只有约0.5个百分点，所以工具包保留它，不夸大为决定性突破。

### 当前默认：Bonito Stone best

```text
legacy-stone-v1 raw chunks
-> Bonito encoder prefix [0:9]
-> 解冻末端3个LSTM blocks
-> chunk 768D adapter
-> Transformer MIL + gated attention
-> 9-class classifier
```

严格CCF文件留出test accuracy/macro-F1为`0.8459/0.8452`，8-group macro-F1为`0.8932`。C2LR+MI-Mix相对同深度CE的validation增益只有`+0.0053`，因此保留为消融事实，不把复杂loss当作主要卖点。

### 轻量配置：Bonito Stone frozen

```text
precomputed frozen Bonito 768D
-> Transformer MIL
-> 9-class classifier
```

它适合快速开发和不具备Bonito训练环境的演示，但不是当前最高模型。v3 fresh Stone frozen test macro-F1约`0.8179`。

### 实验配置：Foundation Apple

后续电信号基座模型默认使用`apple-sclamp-v1`，但必须通过独立adapter实现并绑定自己的checkpoint。已有基座实验未超过Bonito，因此它保留为实验接口，不是当前发布默认。

### 序列准确率参照

同一file-held-out协议的k=6 sequence teacher test macro-F1为`0.9822`。它用于确认划分和标签可学，并作为准确率上界参照；不能与raw-signal路线混称为同一种输入成本。

### Atlas的位置

闭集9分类默认不需要Atlas。Atlas/reference embedding bank只保留在以下可选模块：

- prototype/KNN retrieval；
- few-shot诊断；
- embedding可视化；
- OOD距离特征研究。

旧实验中Atlas相对classifier没有稳定增益，因此不进入默认推理命令，也不允许使用真实混样更新Atlas。

## 5. 真实10菌混样

D6306必须保留为独立外部benchmark，而不是第四个训练split：

```text
known-9: 跨批次泛化
Lactobacillus fermentum: 未见物种拒识
```

禁止将D6306用于训练、早停、阈值校准或Atlas构建。完整数据约147 GB signals和642万FASTQ reads，不放入GitHub。后续发布一个按高置信参考比对标签分层抽样的`D6306-external-mini`，同时保留known-9与unknown10；minimap2标签应称为“高置信参考比对标签”，不是绝对ground truth。

旧D6306结果来自旧随机read split的Signal v10，known-9 macro-F1为`0.7051`，不能冒充当前file-held-out v3模型的外部结果。当前PFT模型需要在外部mini上重新冻结评估。

## 6. 固定数据生成

生成benchmark-mini manifest：

```bash
python3 scripts/16_build_fixed_benchmark_manifest.py \
  --source-manifest artifacts/manifests/v1_groupheldout_3000_seed42/group_split_manifest.csv \
  --output-dir artifacts/benchmarks/zymo9_benchmark_mini_v1 \
  --tier-name benchmark-mini \
  --train-per-species 300 \
  --val-per-species 100 \
  --test-per-species 100 \
  --seed 42
```

从已经验证的Stone raw cache导出compact bundle：

```bash
python3 scripts/17_export_compact_raw_benchmark.py \
  --benchmark-manifest artifacts/benchmarks/zymo9_benchmark_mini_v1/benchmark_manifest.csv \
  --source-raw-manifest artifacts/cache/v3_stone_raw_v1_3000/raw_chunk_manifest.csv \
  --preprocessing-profile config/preprocessing_legacy_stone_v1.json \
  --output-dir artifacts/benchmarks/zymo9_benchmark_mini_v1/raw_bundle \
  --max-chunks 16
```

导出器按正式评估规则均匀选择最多16个chunks，保存float16数组、相对路径manifest、profile副本和SHA256审计。

## 7. 固定模型推理与出图

当前PFT checkpoint必须走raw-cache入口：

```bash
squiggle-species predict-raw-cache \
  --manifest artifacts/benchmarks/zymo9_fixture_v1/raw_bundle/raw_benchmark_manifest.csv \
  --checkpoint artifacts/runs/v3_stone_weekend/pft_c2lr_mimix_blocks3/model.pth \
  --bonito-model-dir /mnt/zzbnew/rnamodel/dengjingye/bacteria/data/models \
  --split test \
  --device cuda:0 \
  --output output/fixture_test_predictions.csv
```

随后只用validation校准，再冻结到test：

```bash
squiggle-species calibrate --predictions output/val_predictions.csv --output-dir output/calibration
squiggle-species report --predictions output/test_predictions.csv \
  --threshold-json output/calibration/calibration.json --output-dir output/test_report
```

fixture只验证软件链路，不用于宣称模型效果。正式数字来自benchmark-full；benchmark-mini用于快速回归和教程图。

一键fixture demo：

```bash
bash scripts/run_toolkit_fixture_demo.sh
```

它自动执行val/test推理、validation-only校准和test报告，并写入`FIXTURE_ONLY.txt`防止误用。

v4结束后运行benchmark-mini并生成开发测评图：

```bash
MANIFEST=artifacts/benchmarks/zymo9_benchmark_mini_v1/raw_bundle/raw_benchmark_manifest.csv \
DATASET_ROLE=benchmark-mini \
OUTPUT_DIR=output/zymo9_benchmark_mini_v1 \
nohup bash scripts/run_toolkit_benchmark_eval.sh \
  > logs/run_toolkit_benchmark_mini.master.log 2>&1 &
```

benchmark-mini用于开发回归和演示，不替代benchmark-full的正式数字。

## 8. 发布边界

- Git仓库：代码、配置、fixture manifest、文档和测试。
- GitHub Release或Zenodo：约4 MB fixture raw bundle、版本化checkpoint和checksum。
- benchmark-mini/full：受数据授权约束，使用下载脚本或独立数据归档，不直接塞入源码历史。
- 发布前必须确认Zymo标准品数据和模型权重的再分发许可。

## 9. 代码与数据分离交付

建议采用“GitHub代码仓库 + 独立数据目录/Release”的方式：

```text
SquiggleSpecies/                    # GitHub代码、配置、测试、文档
SquiggleSpecies_Benchmark_Data_v1/ # 单独移交或数据仓库
  fixture/
  benchmark-mini/
  checksums.sha256
  README_DATA.md
```

约4 MB fixture可在再分发许可允许时放GitHub Release；约283 MB benchmark-mini适合单独目录、Release或Zenodo，不放进Git历史。它每类共有500条reads，其中train/val/test为`300/100/100`，足以验证训练、校准、推理和出图，又不会像3.1 GB benchmark-full raw cache那样笨重。正式科学结果仍使用full协议，不能用mini替换。

生成独立移交目录：

```bash
python3 scripts/18_build_data_handoff.py \
  --benchmark-root artifacts/benchmarks \
  --output-dir /path/to/SquiggleSpecies_Benchmark_Data_v1 \
  --mode hardlink
```

同一文件系统优先使用hardlink，形成可独立移交的目录但不重复占用磁盘；跨文件系统会自动回退到复制。
