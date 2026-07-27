# 结果与适用边界

## 正式评价协议

正式结果采用CCF文件级留出：每个CCF文件只能属于train、validation或test之一。旧流程虽然没有重复read_id，但`1296/1296`个CCF文件跨越多个split，因此旧指标只代表同文件随机read留出，不代表新文件泛化。

严格Zymo9每类使用`train/val/test=1800/600/600` reads，每个CCF文件最多200 reads。模型选择、PFT深度、objective和阈值只使用validation，test只用于最终冻结评估。

## 物种

| ID | 物种 |
|---|---|
| LB01 | Escherichia coli |
| LB06 | Pseudomonas aeruginosa |
| LB07 | Staphylococcus aureus |
| LB08 | Bacillus subtilis |
| LB09 | Listeria monocytogenes |
| LB12 | Salmonella enterica |
| LB18 | Enterococcus faecalis |
| LB11 | Saccharomyces cerevisiae |
| LB02 | Cryptococcus neoformans |

## 正式结果

严格signal主路线：

```text
CCF5 physical-current restoration
-> legacy-stone-v1
-> discard 5000
-> 6000-point chunks, overlap 3000
-> Bonito prefix [0:9]
-> partial fine-tuning of the last 3 LSTM blocks
-> Transformer MIL
-> C2LR + random-layer MI-Mix
```

| 结果 | test accuracy | test macro-F1 |
|---|---:|---:|
| Strict Zymo9 signal | `0.8459` | `0.8452` |
| Strict Zymo8 group, LB01/LB12 merged | `0.8976` | `0.8932` |
| Sequence canonical k=6 teacher | `0.9822` | `0.9822` |

严格Zymo9三种子test macro-F1为`0.84521/0.84696/0.84486`，均值和总体标准差为`0.84568 +/- 0.00092`。

reference-diverse 5-class只依据参考基因组canonical 5-mer距离预先选择`LB06/LB07/LB02/LB08/LB11`。三种子test macro-F1为`0.92830/0.92378/0.93191`，均值`0.92800 +/- 0.00333`。这是辅助软件演示，不替代9分类。

## 选择性分类

旧`0.245`阈值接近全覆盖，不再使用。当前模型通过validation约束选择阈值`0.71179676`：

- validation coverage `0.8894`，accepted accuracy `0.9001`；
- test coverage `0.8767`，accepted accuracy `0.8954`，accepted macro-F1 `0.8895`。

accepted指标只能描述接受子集，不能替代全覆盖9分类结果。

## 已确认的限制

1. LB01/LB12严格二分类test macro-F1只有`0.6965`，是当前主要细粒度瓶颈。
2. C2LR/MI-Mix相对同深度CE的validation增益只有`+0.0053`，复杂loss不是主要性能来源。
3. sequence learned baseline显著高于signal，因此不声称signal全面优于sequence。
4. 历史D6306结果使用旧random-read H-v10模型，known-9 macro-F1为`0.7051`，不能称为当前strict-v3外部验证。
5. D6306还存在跨批次domain shift和signal-length eligible selection bias。

机器可读结果位于`results/`。所有大型权重、CCF5、FASTQ和chunk cache均在仓库外单独移交。
