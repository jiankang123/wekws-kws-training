# KWS Model: finetune_nr1_50ep

## Quick Info
- **Keywords**: 你好问问, 嗨小问
- **Architecture**: DS-TCN (~287k params, ~1.2MB)
- **Pretrained Base**: kws_wenwen_dstcn (avg_30.pt)
- **Data**: 46 WAV files, noise-reduction level 1 (30 你好问问 + 16 嗨小问)
- **Data Split**: Train=36, Dev=4, Test=6 (8:1:1)
- **Training**: 50 epochs, Adam, lr=0.001
- **Date**: 2026-07-01

## Results
| Metric | Value |
|--------|-------|
| Train Accuracy | 100% |
| CV Loss | 0.000768 |
| CV Accuracy | 80% |
| Best CV Loss | 0.000768 (Epoch 49) |

## Files
| File | Description |
|------|-------------|
| `avg_5.pt` | Deployment model (average of epochs 45-49) |
| `config.yaml` | Model & training configuration |
| `words.txt` | Keyword vocabulary |
| `global_cmvn` | CMVN normalization parameters |

## Usage
```python
import torch
model = torch.load("avg_5.pt", map_location="cpu")
```
