# Changelog

## 1.1.0 - 2026-07-23

- 新增模型包schema与checkpoint、Bonito权重、类别顺序和预处理兼容校验。
- 新增`cache-ccf`持久raw chunk缓存入口。
- 新增`classify-ccf`流式端到端推理，避免大型混样产生全量raw中间缓存。
- 无标签外部样本新增物种组成图与置信度分布图。
- 当前严格Zymo9模型固定使用validation选择的`0.7118`阈值，不再使用历史`0.245`近似no-op阈值。
- 新增严格Zymo9正式结果CSV/JSON、三种子重复性和任务粒度结果表。
- 新增完整项目时间线与结果限制文档，明确历史random-read协议和严格CCF-file-held-out协议不可混用。
- 新增`zymo9_strict_v3`模型包示例，记录类别顺序、预处理、权重哈希和validation校准阈值。

## 1.0.0

- Added installable `squiggle-species` command line interface.
- Added CCF5/BLOW5/SLOW5 input inventory.
- Added read and CCF-file split leakage audit.
- Added Bonito chunk-bag Transformer MIL inference.
- Added validation-only risk/coverage calibration with reliability, overall coverage, per-class coverage and gain constraints.
- Added explicit disabled-threshold status when no feasible selective operating point exists.
- Added full-versus-accepted metrics, per-species acceptance rates, correctness AUROC and AURC diagnostics.
- Added automated smoke tests and example inputs.
- Added checkpoint-bound `legacy-stone-v1` and `apple-sclamp-v1` preprocessing profiles.
- Added deterministic fixture/benchmark-mini manifests and portable compact raw-cache export.
- Added `predict-raw-cache` for end-to-end Bonito partial-finetune checkpoint inference.
- Added one-command fixture and generic benchmark evaluation runners.
