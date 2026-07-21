# 5–9 类分类扩展协议

## 目的

评估同一条最佳 signal 路线随类别数从 5 增加到 9 时的性能变化。每个点必须单独训练对应类别数的模型，不能只在 9 类 test predictions 中事后删除类别。

## 物种选择

预注册规则为 `hard-anchor-plus-reference-diversity-v1`：

1. 每个子集固定保留近缘困难对 `LB01/LB12`；
2. 其余物种只依据参考基因组 canonical 5-mer cosine distance，以 greedy max-min 方式逐步加入；
3. 不读取 validation/test 预测或分数来选物种。

嵌套顺序：

```text
LB01, LB12, LB07, LB06, LB02, LB08, LB11, LB09, LB18
```

对应任务：

| 类别数 | 物种条码 |
|---:|---|
| 5 | LB01, LB12, LB07, LB06, LB02 |
| 6 | 上述 + LB08 |
| 7 | 上述 + LB11 |
| 8 | 上述 + LB09 |
| 9 | 上述 + LB18 |

进化或参考基因组距离远通常意味着任务更容易。保留 LB01/LB12 是为了避免低类别任务退化为“只挑容易菌”的结果筛选。这里使用的是参考基因组 5-mer 组成距离，不应称为严格的系统发育分支长度。

## 固定训练口径

每类均使用：

```text
train=1800, val=600, test=600
CCF-file-held-out
legacy-stone-v1
Bonito prefix [0:9]
PFT last 3 LSTM blocks
Transformer MIL
C2LR + random-layer MI-Mix
```

Hard-pair weights沿用v3已冻结的validation结论，并只保留当前子集中存在的pair：`LB01/LB12`始终启用，k7起加入`LB11/LB02`，k8起加入`LB07/LB09`。这不是用v5 test重新找pair。

所有子集 read 和 CCF group 跨 split 泄漏均为 0。9 类使用已冻结 v3 结果；5–8 类分别重新训练类别专属 head 和 PFT 模型。

## 执行

当前 v4 占用 `cuda:0`。若`cuda:1`确认空闲，可以并行运行：

```bash
cd /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715

GPU=1 nohup bash scripts/run_v5_class_count_scaling.sh \
  > logs/run_v5_class_count_scaling.master.log 2>&1 &
```

监控：

```bash
tail -f logs/run_v5_class_count_scaling.master.log
tail -f logs/v5_class_count_scaling/03_k5_pft_best.log
```

Runner 会复用现有 raw chunk 与 768D cache，不重新做 CCF5 标准化或 Bonito frozen embedding。v4与v5使用不同GPU和输出目录，但共享CPU与磁盘带宽，因此并行时可能略慢。单张A40需要完成4组正式训练，保守估计约2–3天；支持按summary断点跳过。

## 结果解释

- 5–8 类若达到或超过 `0.90`，可以报告“在预注册的较低类别粒度下达到较高准确率”，但不能替代 9 类主结果。
- 曲线下降说明类别粒度和近缘类增加带来难度；不能写成 9 类模型删类后变好。
- 即使 5 类很好，也必须同时展示其包含 LB01/LB12 以及完整物种名单。
