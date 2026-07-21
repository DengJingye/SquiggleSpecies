# Backbone 接入规范

## 设计原则

工具包不能把模型写死为 Bonito 或某个电信号基座。一个 backbone 只负责把标准化后的 signal chunks 编码成 chunk embeddings；Mean/Attention/Transformer 聚合、分类头、训练 objective 和决策策略属于独立模块。

外部模型实现 `squiggle_species.backbones.SignalBackboneAdapter`，并声明：

- `backbone_id`：模型与权重版本；
- `feature_dim`：chunk embedding 维度，不要求必须是 768；
- `preprocessing_profiles`：允许的标准化版本；
- `trainable_units`：该模型可部分解冻的有序单元；
- `encode_chunks()`：`[n_chunks, signal_length] -> [n_chunks, feature_dim]`。

配置使用 Python locator，不需要修改工具包源码：

```json
{
  "adapter": "your_package.your_adapter:create_backbone",
  "kwargs": {"model_path": "/path/to/model"},
  "preprocessing_profile": "your-profile-v1",
  "adaptation": {
    "mode": "partial_finetune",
    "selection": {"unfreeze_last_n": 2}
  }
}
```

也可以用具名单元：

```json
{
  "adaptation": {
    "mode": "partial_finetune",
    "selection": {"trainable_units": ["encoder.block10", "encoder.block11"]}
  }
}
```

## PFT 深度的含义

PFT 是 partial fine-tuning，即只解冻 backbone 的一部分。`unfreeze_last_n=3` 的“3”由 adapter 自己解释：

- 当前 Bonito adapter 中，一个单元是一个后端 LSTM block；
- Transformer 基座中可以是一个 Transformer block；
- CNN 或混合模型中可以用具名 stage，未必适合使用“末端 N 层”。

当前 legacy Stone + Bonito 的 validation 消融结果是：PFT-1 `0.8431`、PFT-2 `0.8444`、PFT-3 `0.8480`、PFT-5 `0.8391`，所以 **PFT-3 只作为这一 Bonito 权重和预处理组合的默认值**。新模型必须重新在 validation 上选择适配单元，不能继承这个结论。

## 安全约束

1. checkpoint 必须保存 backbone ID、adapter locator、预处理 profile、可训练单元和 label map。
2. adapter 必须拒绝不兼容的预处理 profile。
3. 冻结/解冻深度只用 validation 选择，test 不参与。
4. 外部模型先跑 fixture 和 benchmark-mini；未显示增益时不扩大。
5. 当前通用 adapter/registry 契约已经实现；Bonito PFT raw-cache 是已验证实现。任意 adapter 的通用训练编排和直接 CCF5 cache 命令仍是 V1.1 工作，不应宣称已经全部完成。
