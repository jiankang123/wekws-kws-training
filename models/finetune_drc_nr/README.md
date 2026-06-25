# KWS Model: finetune_drc_nr

DRC+降噪联合处理数据微调的关键词唤醒模型。

## 训练数据
- 你好问问 (Nihao_Wenwen): 30 个 WAV，先DRC再降噪处理
- 嗨小问 (Hi_Xiaowen): 16 个 WAV，先DRC再降噪处理
- 共 46 个正例（16kHz, 16-bit, 单声道）
- 数据集划分: train=36, dev=4, test=6

## 训练配置
- 预训练模型: kws_wenwen_dstcn (DS-TCN)
- Epochs: 30
- LR: 0.0001
- Batch size: 32
- 模型平均: avg_5 (epoch 25-29)

## 训练指标
- 最终 cv_loss (Epoch 29): 0.9606
- 最佳5 epoch avg cv_loss: ~1.0
- 注: dev集仅4个样本，cv_acc波动大

## 文件说明
-  - 平均后的最终模型权重 (~1.2MB)
-  - 模型配置文件
-  - 关键词映射 (Nihao_Wenwen=0, Hi_Xiaowen=1)
-  - 全局均值方差归一化参数

## 训练时间
2026-06-25
