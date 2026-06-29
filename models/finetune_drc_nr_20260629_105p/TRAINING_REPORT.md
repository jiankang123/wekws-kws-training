# KWS Training Report - finetune_drc_nr_20260629_105p

## Experiment Info
- **Date**: 2026-06-29
- **Keywords**: 你好问问 (Nihao Wenwen), 嗨小问 (Hi Xiaowen)
- **Data Source**: DRC + Noise Reduction processed audio from project eb81d6e5-19bd-4fc3-bd1b-e1670b535970
- **Total Samples**: 105 (56 你好问问 + 49 嗨小问)
- **Data Split**: Train=84, Dev=10, Test=11 (8:1:1 ratio)

## Model Architecture
- **Framework**: wekws (WeChat KWS)
- **Architecture**: DS-TCN (Depthwise Separable Temporal Convolutional Network)
- **Parameters**: ~287k
- **Model Size**: ~1.2MB
- **Pretrained Model**: kws_wenwen_dstcn (avg_30.pt)

## Training Configuration
- **Epochs**: 50 (0-49)
- **Batch Size**: 256
- **Optimizer**: Adam
- **Initial Learning Rate**: 0.001
- **Weight Decay**: 0.0001
- **Feature**: 40-dim FBank
- **Sample Rate**: 16kHz, 16-bit PCM, mono
- **GPU**: CUDA device 0
- **Scheduler**: warmup LR with 1000 steps

## Training Results
### Final Epoch (49)
- **Training Accuracy**: 100%
- **CV Loss**: 0.2094
- **CV Accuracy**: 71.43% (5/7 correct on dev set)

### Best Performance
- **Best CV Loss**: 0.1895 (Epoch 40)
- **Best CV Accuracy**: 71.43% (stable from epoch ~15 onwards)

### Model Outputs
- **All Checkpoints**: 0.pt ~ 49.pt (50 checkpoints total)
- **Averaged Model**: avg_5_last.pt (average of epochs 45-49)
- **Final Model**: final.pt (symlink to 49.pt)

## Issues Encountered
- Some WAV files failed to read during training (warnings logged):
  - haixiaowen_ch_split_13/14/17/19/20/21.wav
  - nihaowenwen_jym_split_11/12/15/17/19.wav
- This may indicate corrupted audio or formatting issues with ~10 files

## Recommendations
1. Investigate and fix/replace the failing WAV files
2. Consider increasing dev/test set size for more robust validation
3. Current 71% CV accuracy suggests model may benefit from:
   - More training data
   - Data augmentation
   - Longer training or adjusted hyperparameters

## Files
- train.log: Full training log
- config.yaml: Training configuration
- words.txt: Keyword vocabulary
- avg_5_last.pt: Recommended model for deployment (averaged last 5 epochs)
- *.pt: Individual epoch checkpoints

Training completed successfully on 2026-06-29 17:11:42 (Asia/Shanghai)
