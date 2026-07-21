# 软件架构设计

## 分层结构

```text
Command line interface
  |-- input inventory and preflight
  |-- manifest leakage audit
  |-- model inference
  |-- validation calibration
  `-- report generation

Signal pipeline
  CCF5 -> checkpoint-bound preprocessing profile -> raw chunks
       -> registered backbone adapter (frozen or partial fine-tune) -> chunk embeddings
       -> Transformer MIL -> gated attention -> read prediction

Output pipeline
  predictions -> validation risk/coverage calibration
              -> feasible frozen threshold or explicit disabled state
              -> accept/abstain -> full + accepted metrics, CSV, JSON and PNG
```

## 模块职责

- `inventory.py`：信号文件清点与分组统计。
- `manifest.py`：read和CCF文件级划分审计。
- `data.py`：mmap chunk bag数据读取和批处理。
- `preprocessing.py`：版本化legacy Stone和Apple信号profile。
- `backbones/`：任意signal backbone的注册、预处理兼容性和adapter-specific解冻契约。
- `bonito_pft.py`：标准化raw chunk bag读取和当前已验证Bonito末端部分微调实现。
- `models.py`：Transformer MIL、gated attention和分类头。
- `inference.py`：checkpoint加载、概率计算和read-level结果写出。
- `calibration.py`：验证集可靠性/总体coverage/每物种coverage约束、AUROC/AURC与动态工作点冻结。
- `reporting.py`：全覆盖与accepted分类指标、每物种acceptance rate、物种组成和混淆矩阵。
- `cli.py`：统一命令分发和参数校验。

## 可复现性

所有训练、验证和测试数据通过manifest追踪。大型chunk cache采用mmap按需读取。模型选择使用validation macro-F1，test只用于冻结后的最终评估。输出同时保存机器可读JSON/CSV和展示用图片。

闭集9分类主链路不依赖Atlas。真实10菌混样属于独立外部层，不参与模型选择或校准。

当前完整度边界：固定数据、Bonito推理、校准、报告和benchmark runner已经闭环；任意backbone的接口和registry已经建立，但外部adapter的通用训练runner、直接CCF5到cache的统一CLI、模型bundle签名仍待V1.1完成。
