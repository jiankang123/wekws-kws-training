# KWS DS-TCN 模型 - DRC 去噪数据微调

## 模型说明
基于 DS-TCN 架构的关键词唤醒模型，使用 DRC（动态范围压缩）去噪后的音频数据微调。

## 关键词
- 你好问问 (label 0)
- 嗨小问 (label 1)

## 使用方法
1. 将 `avg_5.pt`、`config.yaml`、`words.txt`、`global_cmvn` 放在同一目录下
2. 使用 wekws 框架的推理脚本进行检测

## 文件列表
| 文件 | 说明 |
|------|------|
| `avg_5.pt` | 模型权重（287,490 参数，1.1MB） |
| `config.yaml` | 模型配置 |
| `words.txt` | 关键词列表 |
| `global_cmvn` | CMVN 归一化参数 |
| `TRAINING_REPORT.md` | 详细训练报告 |
