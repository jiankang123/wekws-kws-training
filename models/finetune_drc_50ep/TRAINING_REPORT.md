# DS-TCN KWS 微调训练报告

## 训练概况

| 项目 | 值 |
|------|-----|
| 模型架构 | DS-TCN (Depthwise Separable Temporal Convolutional Network) |
| 训练方式 | 基于预训练模型微调 |
| 训练轮数 | 50 epochs |
| 学习率 | 0.0001 |
| Batch Size | 32 |
| GPU | 单卡 (CUDA 0) |
| 训练时长 | ~2 分钟 |

## 数据集

| 关键词 | 文件数 | 训练集 | 验证集 | 测试集 |
|--------|--------|--------|--------|--------|
| 你好问问 | 30 | 24 | 3 | 3 |
| 嗨小问 | 16 | 12 | 1 | 3 |
| **合计** | **46** | **36** | **4** | **6** |

数据来源：DRC（动态范围压缩）处理后的去噪音频。

## 训练结果

### 最终 epoch (49) 指标
- **cv_loss**: 0.2748
- **cv_acc**: 80%

### 最后 5 个 epoch 趋势

| Epoch | cv_loss | cv_acc |
|-------|---------|--------|
| 45 | 0.3290 | 60% |
| 46 | 0.3146 | 60% |
| 47 | 0.2986 | 60% |
| 48 | 0.2911 | 60% |
| 49 | 0.2748 | 80% |

### 模型平均
- 选取 cv_loss 最低的 5 个 checkpoint（epoch 45-49）进行平均
- 输出文件：`avg_5.pt`

## 模型文件

| 文件 | 说明 |
|------|------|
| `avg_5.pt` | 模型权重（平均后） |
| `config.yaml` | 模型配置 |
| `words.txt` | 关键词列表 |
| `global_cmvn` | CMVN 归一化参数 |

## 关键词映射

```
你好问问 0
嗨小问 1
```

## 训练环境

- 服务器：mobvoi-rhea-06 (10.1.205.24)
- Python：/home/jiankang.wang/miniconda3/envs/wenet/bin/python
- 框架：wekws (DS-TCN)
- 预训练模型：kws_wenwen_dstcn/avg_30.pt
