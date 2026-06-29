# DS-TCN 语音唤醒模型 - 你好问问 & 嗨小问

本目录包含基于 DS-TCN 架构微调训练的双关键词语音唤醒模型。

## 📦 模型文件

| 文件 | 说明 |
|------|------|
| `avg_5.pt` | 模型权重（1.1MB，287K 参数） |
| `config.yaml` | 模型配置（特征提取、网络结构） |
| `words.txt` | 关键词标签映射 |
| `global_cmvn` | 特征归一化参数 |

## 🎯 支持的关键词

- **嗨小问** (Hi_Xiaowen) - Label 0
- **你好问问** (Nihao_Wenwen) - Label 1

## 📊 模型性能

- ✅ **验证准确率**: 80.0%
- ✅ **验证 Loss**: 0.0117
- ✅ **训练轮数**: 50 epochs
- ✅ **数据集**: 46 个音频样本（嗨小问 16 + 你好问问 30）

详细训练报告请查看 [TRAINING_REPORT.md](./TRAINING_REPORT.md)

## 🚀 快速使用

### 推理示例（Python）

```python
import torch
from wekws.model import KWS_Model

# 加载模型
checkpoint = torch.load('avg_5.pt', map_location='cpu')
model = KWS_Model(config_dict)
model.load_state_dict(checkpoint)
model.eval()

# 流式检测
with torch.no_grad():
    logits = model(audio_features)
    scores = torch.softmax(logits, dim=-1)
    
    # 检查是否检测到关键词
    if scores[0] > 0.3:  # 嗨小问
        print("检测到: 嗨小问")
    elif scores[1] > 0.3:  # 你好问问
        print("检测到: 你好问问")
```

### 推荐阈值

- **检测阈值**: 0.3
- **典型得分**: 正确关键词 > 0.5，错误关键词 < 0.1

## 🔧 系统要求

- **音频格式**: 16kHz, 单声道, 16-bit PCM WAV
- **推理环境**: PyTorch 1.x+
- **依赖框架**: WeKWS

## 📝 训练信息

- **基础模型**: kws_wenwen_dstcn（预训练）
- **训练方法**: 微调（fine-tuning）
- **学习率**: 0.0001
- **批次大小**: 32
- **训练时长**: ~47 秒

## 📄 引用

如需使用本模型，请参考 WeKWS 项目：
- GitHub: https://github.com/wenet-e2e/wekws

---

**生成日期**: 2026-06-29  
**框架**: WeKWS + DS-TCN  
**训练服务器**: mobvoi-rhea-06
