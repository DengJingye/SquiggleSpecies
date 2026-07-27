# 用户操作手册

## 1. 软件用途

Squiggle Species 用于纳米孔 CCF5 原始电信号的微生物分类流程管理和结果评估。软件将信号文件盘点、数据划分审计、Bonito chunk embedding 分类、验证集阈值校准以及结果报告组织为统一命令行接口。

## 2. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

CCF5读取和Bonito编码需要额外安装与数据格式匹配的 `pyccf5` 和 `ont-bonito`。

## 3. 输入检查

```bash
squiggle-species inventory-ccf /data/sample/signals -o output/inventory.json
squiggle-species validate-manifest split_manifest.csv -o output/manifest_audit.json
```

`validate-manifest` 会检查 read_id 重复、read跨split泄漏和CCF文件跨split泄漏。正式训练前必须保证审计状态为 `pass`。

## 4. 模型推理

### 4.1 CCF5端到端流式推理

```bash
PYTHONPATH=src /usr/bin/python3 -m squiggle_species classify-ccf \
  /data/sample/signals \
  --model-bundle /path/to/model_bundle.json \
  --bonito-model-dir /path/to/bonito/models \
  --device cuda:0 \
  --output-dir output/sample_result
```

`model_bundle.json`固定类别顺序、checkpoint与Bonito权重哈希、标准化profile、chunk参数、MIL最大chunk数和validation校准阈值。程序在推理前强制校验这些接口，避免Stone模型误用Apple数据或类别顺序错位。

该命令直接顺序读取CCF5，不依赖`.idx`。raw chunks只在内存batch中存在，推理后立即释放；程序按CCF文件原子写入预测并支持断点续跑。主要输出为：

- `read_predictions.csv`：read级类别、置信度和各类别概率；
- `report/species_abundance.csv/png`：accepted reads预测组成；
- `report/confidence_distribution.png`：冻结阈值与置信度分布；
- `classification_summary.json`：输入、模型、profile、覆盖和输出审计。

单一运行环境必须同时包含`pyccf5`、`bonito`和CUDA PyTorch。当前服务器使用`/usr/bin/python3`已经完成真实CCF smoke。

### 4.2 建立可复用raw cache

```bash
PYTHONPATH=src /usr/bin/python3 -m squiggle_species cache-ccf \
  /data/sample/signals \
  --profile legacy-stone-v1 \
  --output-dir output/raw_cache
```

缓存以float16保存标准化后的6000点raw chunks，而不是把一个read的chunks先求均值。时间维均值发生在Bonito对单个chunk编码之后；MIL再聚合一个read包含的多个768D chunk向量。大型混样优先使用流式`classify-ccf`，避免raw cache占满磁盘。

### 4.3 Frozen 768D开发基线

```bash
squiggle-species predict \
  --manifest bag_manifest.csv \
  --checkpoint model.pth \
  --config config/experiment_apple_frozen_small.json \
  --split test \
  --device cuda:0 \
  --output output/test_predictions.csv
```

输入 manifest 的每一行代表一个 read bag，记录 chunk embedding 路径、chunk起点、chunk数量、物种标签和read_id。推理结果包含 predicted_label、confidence 和每个类别的概率。

### 4.4 Bonito部分微调正式模型

```bash
squiggle-species predict-raw-cache \
  --manifest raw_benchmark_manifest.csv \
  --checkpoint model.pth \
  --bonito-model-dir /path/to/bonito/models \
  --split test \
  --device cuda:0 \
  --output output/test_predictions.csv
```

该命令输入已经按checkpoint兼容profile处理的raw chunk bag，并同时恢复checkpoint中的Bonito末端可训练层和MIL分类器。当前模型要求`legacy-stone-v1`；Apple cache不能混用。

## 5. 动态阈值校准

```bash
squiggle-species calibrate \
  --predictions output/val_predictions.csv \
  --target-accuracy 0.90 \
  --min-coverage 0.5 \
  --min-per-class-coverage 0.5 \
  --min-accuracy-gain 0.01 \
  --output-dir output/calibration
```

阈值只根据验证集选择。软件按置信度排序计算risk-coverage曲线、accepted accuracy、accepted macro-F1、每物种coverage、正确性AUROC和AURC，并选择满足目标accuracy、总体coverage、每物种coverage及最小增益要求的最大覆盖工作点。若约束不可同时满足，`calibration.json`写入`threshold_enabled=false`，不伪造可用阈值。可行阈值冻结后才能应用到test或外部样本。

## 6. 报告生成

```bash
squiggle-species report \
  --predictions output/test_predictions.csv \
  --threshold-json output/calibration/calibration.json \
  --output-dir output/test_report
```

主要输出包括：

- `report_summary.json`：总体分类与接受率摘要；
- `species_abundance.csv`：accepted reads中的预测物种组成；
- `per_species_metrics.csv`：各物种precision、recall和F1；
- `confusion_matrix.csv/png`：混淆矩阵底层数据和图片。

报告同时保存全覆盖指标、accepted子集指标和每物种acceptance rate。accepted accuracy不能替代全覆盖accuracy；若困难物种被大量拒绝，必须从逐物种coverage图中明确呈现。

## 7. 结果边界

- test和真实混样不能用于调阈值。
- 外部混样丰度首先解释为eligible reads上的组成。
- minimap2标签应称为高置信参考比对标签，而不是绝对ground truth。
- Atlas不是闭集主分类器，只用于retrieval、few-shot和OOD诊断。
- fixture只用于跑通软件；benchmark-mini用于快速回归；正式结论使用benchmark-full。
- D6306真实10菌混样是冻结外部评估，不能进入训练、阈值校准或Atlas。

固定数据、标准化绑定和混样边界详见`docs/TOOLKIT_BENCHMARK_PROTOCOL.md`。
