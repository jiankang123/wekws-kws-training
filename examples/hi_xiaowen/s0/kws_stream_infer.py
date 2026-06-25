#!/usr/bin/env python3
"""
流式 KWS 推理脚本（基于 avg_30.pt，逐帧 cache 传递）
模拟真实设备流式唤醒词检测，检测音频中每次触发的时刻

用法:
  python kws_stream_infer.py \
    --config exp/ds_tcn/config.yaml \
    --checkpoint exp/ds_tcn/avg_30.pt \
    --wav data/kws_test_long.wav \
    --keywords_dict exp/ds_tcn/words.txt \
    --threshold 0.5
"""

import argparse
import sys
import yaml
import torch
import torchaudio
import numpy as np


def load_wav(wav_path, target_sr=16000):
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # [1, T]


def compute_fbank(waveform, sample_rate=16000, num_mel_bins=40):
    # wekws processor 里对 waveform 做了 * (1<<15) 的缩放
    # 必须保持一致，否则特征值范围不对，模型输出异常
    waveform_scaled = waveform * (1 << 15)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_scaled,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        sample_frequency=sample_rate,
        dither=0.0,
        energy_floor=0.0,   # wekws processor 用的是 0.0
    )  # [T, 40]
    return fbank


def stream_infer(model, fbank, chunk_size=1, threshold=0.5,
                 keywords=None, smooth_window=5, min_interval=50):
    """
    流式推理：逐 chunk 送入模型，传递 cache，检测触发事件

    Args:
        model: KWSModel (加载自 avg_30.pt)
        fbank: [T, D] 特征张量
        chunk_size: 每次送入帧数（1帧=10ms）
        threshold: 触发阈值
        keywords: 输出 index 对应的关键词名称列表
        smooth_window: 平滑窗口（取窗口内最大值）
        min_interval: 同关键词两次触发最小间隔帧数（防重复）
    """
    T, D = fbank.shape
    num_kws = len(keywords) if keywords else 1

    cache = torch.zeros(0, 0, 0, dtype=torch.float)

    all_scores = []
    score_buf = {i: [] for i in range(num_kws)}
    last_trigger = {}
    trigger_events = []

    print(f"流式推理：总帧数={T}（{T*0.01:.2f}s），"
          f"chunk={chunk_size}帧/{chunk_size*10}ms，"
          f"阈值={threshold}，平滑={smooth_window}帧")
    print("-" * 60)

    with torch.no_grad():
        frame_idx = 0
        out_frame = 0
        while frame_idx < T:
            end = min(frame_idx + chunk_size, T)
            chunk = fbank[frame_idx:end].unsqueeze(0)  # [1, c, D]

            score_out, cache = model(chunk, cache)     # [1, c', num_kws]

            for t in range(score_out.shape[1]):
                frame_scores = score_out[0, t, :].numpy()
                all_scores.append(frame_scores.copy())
                cur_out = out_frame
                out_frame += 1

                for ki in range(min(num_kws, len(frame_scores))):
                    score_buf[ki].append(float(frame_scores[ki]))
                    if len(score_buf[ki]) > smooth_window:
                        score_buf[ki].pop(0)

                    smooth_score = max(score_buf[ki])
                    if smooth_score >= threshold:
                        last = last_trigger.get(ki, -min_interval)
                        if cur_out - last >= min_interval:
                            last_trigger[ki] = cur_out
                            kw_name = keywords[ki] if keywords and ki < len(keywords) else f'kw_{ki}'
                            event_time = cur_out * 0.01
                            trigger_events.append({
                                'keyword': kw_name,
                                'frame': cur_out,
                                'time': event_time,
                                'score': smooth_score,
                            })
                            print(f"  🔔 [{kw_name}]  @{event_time:.2f}s  得分={smooth_score:.4f}")

            frame_idx = end

    return np.array(all_scores) if all_scores else np.zeros((0, num_kws)), trigger_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--wav', required=True)
    parser.add_argument('--keywords_dict', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--chunk_size', type=int, default=1,
                        help='每次送入帧数，1=10ms（默认1）')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='平滑窗口帧数（默认5=50ms）')
    parser.add_argument('--min_interval', type=int, default=50,
                        help='同关键词两次触发最小间隔帧（默认50=0.5s）')
    args = parser.parse_args()

    # 读取关键词列表（id>=0 的才是真实关键词）
    kw_map = {}
    with open(args.keywords_dict, 'r') as f:
        for line in f:
            arr = line.strip().split()
            if arr:
                name, idx = arr[0], int(arr[1])
                if idx >= 0:
                    kw_map[idx] = name
    num_kws = max(kw_map.keys()) + 1 if kw_map else 0
    keywords = [kw_map.get(i, f'kw_{i}') for i in range(num_kws)]
    print(f"关键词: {keywords}")

    # 加载 config
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # 加载模型（用 Python 类，不用 JIT）
    sys.path.insert(0, '/data1/jiankang.wang/cbbot/wekws')
    from wekws.model.kws_model import init_model
    from wekws.utils.checkpoint import load_checkpoint

    model = init_model(configs['model'])
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(f"模型加载成功: {args.checkpoint}")

    # 加载音频 & 提取特征
    print(f"\n加载音频: {args.wav}")
    waveform = load_wav(args.wav)
    duration = waveform.shape[1] / 16000
    print(f"时长: {duration:.2f}s")

    fbank = compute_fbank(waveform)
    print(f"FBank: {fbank.shape[0]} 帧")

    # 流式推理
    print()
    all_scores, events = stream_infer(
        model=model,
        fbank=fbank,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        keywords=keywords,
        smooth_window=args.smooth_window,
        min_interval=args.min_interval,
    )

    # 汇总
    print()
    print("=" * 60)
    print("触发事件汇总：")
    if not events:
        print(f"  ❌ 未触发（当前阈值={args.threshold}，建议降低试试）")
    else:
        for ev in events:
            print(f"  ✅ [{ev['keyword']}]  @{ev['time']:.2f}s  得分={ev['score']:.4f}")

    print()
    print("各关键词得分统计：")
    for ki, kw in enumerate(keywords):
        if ki >= all_scores.shape[1]:
            continue
        ks = all_scores[:, ki]
        cnt = sum(1 for ev in events if ev['keyword'] == kw)
        print(f"  [{kw}]  max={ks.max():.4f}  mean={ks.mean():.4f}  触发次数={cnt}")


if __name__ == '__main__':
    main()
