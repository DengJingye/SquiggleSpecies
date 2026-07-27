# Squiggle Species

Squiggle Species（纳米孔微生物原始电信号智能分类与评估软件）用于管理 CCF5 原始信号数据、审计训练/验证/测试划分、运行 read-level 微生物分类、基于验证集动态校准拒识阈值，并生成分类与物种组成报告。

软件面向 Linux 服务器和批处理场景。研究数据、模型权重及大规模缓存不进入代码仓库。

当前发布口径基于严格CCF文件级留出，而不是同一CCF文件内的随机read留出。正式9分类test accuracy/macro-F1为`0.8459/0.8452`，三种子macro-F1为`0.8457 +/- 0.0009`。完整结果、负结果和适用边界见[结果与限制](docs/RESULTS_AND_LIMITATIONS.md)，从6分类前史到严格重审的路线见[项目实验时间线](docs/PROJECT_TIMELINE.md)。

## 功能

- 盘点 CCF5、BLOW5 和 SLOW5 文件数量、大小及分组。
- 从CCF5恢复物理电流，按模型绑定的Stone/Apple profile切分raw chunks。
- 检查 read_id 重复、跨 split 泄漏和 CCF 文件级泄漏。
- 从 Bonito chunk embedding cache执行轻量推理，或从标准化raw chunk cache执行Bonito部分微调模型推理。
- 仅使用 validation predictions 动态选择置信度阈值。
- 对 test 或外部样本应用冻结阈值，输出分类、拒识和物种组成结果。
- 保存 JSON、CSV 和 PNG 底层结果，便于重新绘图和复核。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

开发模式：

```bash
pip install -e .
```

## 快速使用

检查输入信号目录：

```bash
squiggle-species inventory-ccf /data/sample/signals -o output/ccf_inventory.json
```

直接从CCF5进行流式分类（推荐，不持久保存大型raw cache）：

```bash
PYTHONPATH=src /usr/bin/python3 -m squiggle_species classify-ccf \
  /data/sample/signals \
  --model-bundle /path/to/model_bundle.json \
  --bonito-model-dir /path/to/bonito/models \
  --device cuda:0 \
  --output-dir output/sample_result
```

该命令执行`pA恢复 -> 模型绑定标准化 -> chunk -> Bonito/PFT -> MIL -> 冻结阈值 -> 报告`。按CCF文件保存中间预测并支持断点续跑，但不会保存全量raw chunks，因此适合大型混合样本。

需要复用raw chunks时显式建立缓存：

```bash
PYTHONPATH=src /usr/bin/python3 -m squiggle_species cache-ccf \
  /data/sample/signals \
  --profile legacy-stone-v1 \
  --discard-first 5000 \
  --chunk-size 6000 \
  --overlap 3000 \
  --output-dir output/raw_cache
```

检查模型包及权重兼容性：

```bash
squiggle-species inspect-model-bundle \
  /path/to/model_bundle.json \
  --bonito-model-dir /path/to/bonito/models
```

检查数据划分：

```bash
squiggle-species validate-manifest split_manifest.csv -o output/manifest_audit.json
```

对已经完成 Bonito 编码的 read bag 进行分类：

```bash
squiggle-species predict \
  --manifest bag_manifest.csv \
  --checkpoint model.pth \
  --config config/experiment_apple_frozen_small.json \
  --split val \
  --device cuda:0 \
  --output output/val_predictions.csv
```

对当前正式的Bonito部分微调模型进行分类：

```bash
squiggle-species predict-raw-cache \
  --manifest raw_benchmark_manifest.csv \
  --checkpoint model.pth \
  --bonito-model-dir /path/to/bonito/models \
  --split test \
  --device cuda:0 \
  --output output/test_predictions.csv
```

在验证集上动态校准阈值：

```bash
squiggle-species calibrate \
  --predictions output/val_predictions.csv \
  --target-accuracy 0.90 \
  --min-coverage 0.50 \
  --min-per-class-coverage 0.50 \
  --min-accuracy-gain 0.01 \
  --output-dir output/calibration
```

软件选择满足可靠性与覆盖约束的最大 validation coverage，而不是最大化一个偏向全覆盖的乘积得分。若不存在可行工作点，`threshold_enabled=false`，程序关闭拒识并明确报告不可行；不会用 test 或真实混样重新找阈值。

将冻结阈值应用到 test：

```bash
squiggle-species report \
  --predictions output/test_predictions.csv \
  --threshold-json output/calibration/calibration.json \
  --output-dir output/test_report
```

## 方法概述

当前严格CCF文件留出实验的signal流程绑定`config/preprocessing_legacy_stone_v1.json`：

```text
CCF5 raw signal
-> traceable legacy-stone standardization
-> discard first 5000 points
-> 6000-point chunks with 3000 overlap
-> Bonito encoder prefix [0:9]
-> partial fine-tuning of the last 3 Bonito LSTM blocks
-> Transformer MIL and gated attention pooling
-> read-level classification
-> validation-calibrated selective prediction
```

该配置的严格文件留出test accuracy/macro-F1为`0.8459/0.8452`。以validation accepted accuracy `0.90`为目标动态得到阈值约`0.712`，冻结到test后coverage `0.8767`、accepted accuracy `0.8954`、accepted macro-F1 `0.8895`。这说明置信度排序有一定作用，但不能把模型包装成全覆盖`0.90`；全覆盖指标始终是主结果。V1.1同时支持已有raw cache推理和CCF5流式端到端推理。

固定数据采用fixture、benchmark-mini和benchmark-full三层协议。预处理是checkpoint接口的一部分：当前模型只能使用无sclamp的`legacy-stone-v1`；`apple-sclamp-v1`仅供明确兼容的未来foundation-model路线。完整设计见[工具包固定数据与测评协议](docs/TOOLKIT_BENCHMARK_PROTOCOL.md)。

工具包不把Transformer、PFT或对比学习写死。`config/toolkit_routes_v1.json`分别配置预处理、backbone、解冻深度、read聚合头、训练objective和决策头；当前默认组合只是现有严格validation中表现最好的已验证profile。外部模型通过`SignalBackboneAdapter`与`package.module:factory`接入，并自行声明输出维度、兼容预处理和可解冻单元；Bonito的PFT-3不是其他backbone的默认结论，详见[Backbone接入规范](docs/BACKBONE_ADAPTER_GUIDE.md)。

5–9类历史扩展采用预注册的“LB01/LB12困难对锚定 + 参考基因组组成多样性”嵌套方案。最终另设一个严格标注的reference-diverse 5-class软件演示，固定排除LB01/LB12并只依据参考基因组距离选择物种。三个seed的test macro-F1为`0.9280 ± 0.0033`；该数字只用于粗粒度软件演示，不替代严格9分类`0.8452`主结果。

v4结束、`cuda:0`空闲后可用一条命令跑fixture推理与出图：

```bash
bash scripts/run_toolkit_fixture_demo.sh
```

该fixture只有每类2条test read，结果只验证软件链路，不是模型效果数字。

`scripts/run_toolkit_benchmark_eval.sh`是同一流程的通用runner，可通过`MANIFEST`、`CHECKPOINT`、`DATASET_ROLE`和`OUTPUT_DIR`运行benchmark-mini或后续冻结数据。

闭集分类不依赖 Atlas。Atlas/reference embedding bank 仅用于检索、few-shot、原型距离和 OOD 诊断。任何分类、OOD 或拒识阈值都必须在 validation split 上选择，test 与真实混合样本只用于冻结后的最终评估。

## 仓库结构

```text
src/squiggle_species/   可安装Python包
scripts/                可复现实验与缓存脚本
config/                 参数配置
tests/                  小型自动化测试
examples/               演示输入
docs/                   设计、术语和实验记录
results/                小型正式结果表，不包含模型权重或大型缓存
artifacts/              本地产物目录，不提交大模型和大缓存
```

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 第三方依赖

核心报告和推理模块依赖 NumPy、scikit-learn、PyTorch 和 Matplotlib。CCF5流式入口要求同一Python同时能够导入`pyccf5`、`bonito`和CUDA版PyTorch；当前服务器验证环境为`/usr/bin/python3`。若环境分离，可先用`cache-ccf`提取，再在Bonito环境运行`predict-raw-cache`。第三方库按各自许可证使用，不属于本项目自研源代码。
