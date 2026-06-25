# Trained Models

本目录包含基于 wekws 框架训练的关键词唤醒（KWS）模型。

## 模型列表

### kws_model (第一次训练)
- **训练日期**: 2026-06-12
- **关键词**: 
  - 嗨小问 (Hi Xiaowen)
  - 你好问问 (Nihao Wenwen)
- **模型架构**: DS-TCN (Depthwise Separable Temporal Convolutional Network)
- **基础模型**: ModelScope `damo/speech_charctc_kws_phone-xiaoyun` (spec_aug)
- **训练配置**: 
  - 学习率: 1e-4
  - Batch size: 32
  - Epochs: 30
  - 数据集: 嗨小问 39条 + 你好问问 56条 + 负样例

**文件说明**:
- `avg_5.pt` - 模型权重（取最佳5个checkpoint平均）
- `config.yaml` - 模型配置
- `global_cmvn` - 特征归一化参数
- `words.txt` - 关键词列表

---

### kws_model_v2 (第二次重训练)
- **训练日期**: 2026-06-15
- **关键词**: 
  - 嗨小问 (Hi Xiaowen)
  - 你好问问 (Nihao Wenwen)
- **模型架构**: DS-TCN (Depthwise Separable Temporal Convolutional Network)
- **基础模型**: ModelScope `damo/speech_charctc_kws_phone-xiaoyun` (spec_aug)
- **训练配置**: 同 kws_model
- **改进**: 使用相同的数据集重新训练，验证训练流程的稳定性

**文件说明**:
- `avg_5.pt` - 模型权重（取最佳5个checkpoint平均）
- `config.yaml` - 模型配置
- `global_cmvn` - 特征归一化参数
- `words.txt` - 关键词列表
- `model_architecture.txt` - 详细的模型结构说明

---

## 使用方法

### 流式推理
```python
from wekws import load_model, stream_inference

# 加载模型
model = load_model('trained_models/kws_model_v2/avg_5.pt', 
                   'trained_models/kws_model_v2/config.yaml')

# 流式推理
for chunk in audio_stream:
    keyword, score = stream_inference(model, chunk)
    if score > 0.5:
        print(f"检测到关键词: {keyword}, 置信度: {score}")
```

### 批量推理
参考 `examples/hi_xiaowen/s0/kws_infer2.py`

---

## 性能指标

| 模型 | 训练数据量 | 验证集准确率 | 平均推理延迟 |
|------|-----------|------------|------------|
| kws_model | 652条 | - | ~50ms |
| kws_model_v2 | 652条 | - | ~50ms |

---

## 训练数据集

- **正样本**: 
  - 嗨小问: 39条录音
  - 你好问问: 56条录音
- **负样本**: 557条（来自预训练数据集）
- **数据增强**: SpecAugment
- **数据划分**: Train 80% / Dev 10% / Test 10%

---

## 许可证

本模型基于 wekws 框架训练，遵循 Apache 2.0 许可证。
