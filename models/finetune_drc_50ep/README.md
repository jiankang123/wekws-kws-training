# finetune_drc_50ep - KWS Model (DRC处理数据)

DS-TCN 关键词唤醒模型，基于 DRC (Dynamic Range Compression) 处理后的数据微调训练。

## 模型信息

- **架构**: DS-TCN (Depthwise Separable Temporal Convolutional Network)
- **参数量**: ~287k
- **模型大小**: 1.2 MB
- **关键词**: 你好问问 (0), 嗨小问 (1)
- **训练轮次**: 50 epochs
- **预训练模型**: kws_wenwen_dstcn (avg_30.pt)

## 数据集

- **来源**: data_drc_processed (DRC处理后的音频数据)
- **总样本数**: 46
  - 你好问问: 30 个样本
  - 嗨小问: 16 个样本
- **数据划分**:
  - Train: 36 样本 (78.3%)
  - Dev: 4 样本 (8.7%)
  - Test: 6 样本 (13.0%)

## 训练结果

- **最终 CV Loss**: 0.001205
- **最终 CV Accuracy**: 80%
- **模型**: avg_5.pt (Epoch 45-49 平均)

## 使用方法



## 训练日期

2026-07-01
