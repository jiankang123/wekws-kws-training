# KWS Model: finetune_drc_nr_20260629_105p

## Quick Info
- **Keywords**: 你好问问, 嗨小问
- **Model**: DS-TCN (287k params, ~1.2MB)
- **Accuracy**: 71.43% CV accuracy
- **Date**: 2026-06-29

## Files
- `avg_5_last.pt` - Recommended deployment model (average of last 5 epochs)
- `config.yaml` - Model and training configuration
- `words.txt` - Keyword vocabulary
- `global_cmvn` - CMVN normalization parameters
- `TRAINING_REPORT.md` - Detailed training report

## Usage
```python
# Load model for inference
import torch
model = torch.load("avg_5_last.pt")
```

See `TRAINING_REPORT.md` for full training details.
