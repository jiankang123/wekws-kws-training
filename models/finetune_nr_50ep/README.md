# KWS Model: finetune_nr_50ep

## Quick Info
- **Keywords**: 你好问问, 嗨小问
- **Architecture**: DS-TCN (Depthwise Separable Temporal Convolutional Network)
- **Parameters**: ~287k
- **Model Size**: ~1.2MB
- **Pretrained Base**: kws_wenwen_dstcn (avg_30.pt)
- **Training Date**: 2026-06-29
- **Data**: 105 DRC+Noise-Reduction WAV files (56 你好问问 + 49 嗨小问)
- **CV Accuracy**: 71.43% | **CV Loss**: 0.2094 | **Train Accuracy**: 100%

## Files
| File | Description |
|------|-------------|
| `avg_5_last.pt` | Deployment model (average of epochs 45-49) |
| `config.yaml` | Model & training configuration |
| `words.txt` | Keyword vocabulary |
| `global_cmvn` | CMVN normalization parameters |
| `TRAINING_REPORT.md` | Detailed training report |

## Usage
```python
import torch
model = torch.load("avg_5_last.pt", map_location="cpu")
```

See `TRAINING_REPORT.md` for full training details and recommendations.
