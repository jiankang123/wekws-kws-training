#!/usr/bin/env python3
"""
独立 KWS 推理脚本（基于 JIT 导出的 model.zip）
不依赖 wenet/wekws 模块，仅需 torch 和 torchaudio
用法:
  python kws_infer.py --model model.zip --wav test.wav --keywords "hi xiaowen,nihaowenwen"
"""

import argparse
import torch
import torchaudio
import numpy as np
import sys


def load_wav(wav_path, target_sr=16000):
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # [1, T]


def compute_fbank(waveform, sample_rate=16000, num_mel_bins=40,
                  frame_length=25, frame_shift=10):
    """计算 FBank 特征，与 wekws 保持一致"""
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate,
        dither=0.0,
        energy_floor=1.0,
    )  # [T, 40]
    return fbank


def apply_cmvn(fbank, cmvn_file):
    """加载 global_cmvn 并做均值方差归一化"""
    mean = []
    istd = []
    with open(cmvn_file, 'r') as f:
        lines = f.readlines()
    # 格式：第一行 <AddShift>，第二行偏移值，第三行 <Rescale>，第四行缩放值
    # 或者 kaldi ark 格式
    # 尝试解析 kaldi 文本 CMVN
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith(']'):
            continue
        vals = line.split()
        try:
            fvals = [float(v) for v in vals]
            if not mean:
                mean = fvals
            elif not istd:
                istd = fvals
                break
        except ValueError:
            continue

    if mean and istd:
        mean_t = torch.tensor(mean[:fbank.shape[1]], dtype=torch.float32)
        istd_t = torch.tensor(istd[:fbank.shape[1]], dtype=torch.float32)
        fbank = (fbank + mean_t) * istd_t
    return fbank


def sliding_window_infer(model, fbank, window_size=128, window_shift=5):
    """
    滑动窗口推理，输出每帧的关键词得分
    window_size: 窗口帧数（约1.28秒@100fps after subsampling）
    window_shift: 滑动步长（帧）
    """
    T = fbank.shape[0]
    scores_list = []
    timestamps = []

    with torch.no_grad():
        i = 0
        while i + window_size <= T:
            chunk = fbank[i:i+window_size].unsqueeze(0)  # [1, W, 40]
            lengths = torch.tensor([window_size])
            score = model(chunk, lengths)  # [1, W', num_keywords]
            # 取窗口中间帧附近的最大分数
            s = score[0].max(dim=0).values  # [num_keywords]
            scores_list.append(s.numpy())
            timestamps.append(i * 0.01)  # 10ms per frame
            i += window_shift

        # 处理最后一段不足 window_size 的部分
        if T > 0 and i < T:
            chunk = fbank[max(0, T-window_size):T].unsqueeze(0)
            lengths = torch.tensor([min(window_size, T)])
            score = model(chunk, lengths)
            s = score[0].max(dim=0).values
            scores_list.append(s.numpy())
            timestamps.append(i * 0.01)

    return np.array(scores_list), timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='JIT model.zip 路径')
    parser.add_argument('--wav', required=True, help='输入 wav 文件')
    parser.add_argument('--cmvn', default=None, help='global_cmvn 文件路径')
    parser.add_argument('--keywords', default='hi xiaowen,nihaowenwen',
                        help='关键词列表，逗号分隔，顺序与模型输出对应')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='触发阈值（默认0.5）')
    parser.add_argument('--window_size', type=int, default=200,
                        help='滑动窗口大小（帧数，默认200≈2s）')
    parser.add_argument('--window_shift', type=int, default=10,
                        help='滑动步长（帧数，默认10=0.1s）')
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(',')]
    print(f"关键词: {keywords}")
    print(f"触发阈值: {args.threshold}")

    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = torch.jit.load(args.model)
    model.eval()
    print("模型加载成功")

    # 加载音频
    print(f"\n加载音频: {args.wav}")
    waveform = load_wav(args.wav)
    duration = waveform.shape[1] / 16000
    print(f"音频时长: {duration:.2f}s，采样点数: {waveform.shape[1]}")

    # 提取特征
    fbank = compute_fbank(waveform)
    print(f"FBank 特征: {fbank.shape}  (帧数 x 特征维度)")

    # CMVN 归一化
    if args.cmvn:
        fbank = apply_cmvn(fbank, args.cmvn)
        print(f"已应用 CMVN 归一化")

    # 滑动窗口推理
    print(f"\n开始推理（窗口={args.window_size}帧/{args.window_size*0.01:.1f}s，步长={args.window_shift}帧/{args.window_shift*0.01:.2f}s）...")
    scores, timestamps = sliding_window_infer(
        model, fbank,
        window_size=args.window_size,
        window_shift=args.window_shift
    )

    print(f"\n推理完成，共 {len(scores)} 个窗口\n")
    print("=" * 60)

    # 找触发点（峰值检测）
    triggered = {kw: [] for kw in keywords}
    for kw_idx, kw in enumerate(keywords):
        if kw_idx >= scores.shape[1]:
            continue
        kw_scores = scores[:, kw_idx]
        max_score = kw_scores.max()
        max_time = timestamps[kw_scores.argmax()]
        print(f"[{kw}] 最高得分: {max_score:.4f}  出现时刻: {max_time:.2f}s")

        # 找所有超过阈值的峰值
        above = kw_scores >= args.threshold
        in_peak = False
        peak_start = 0
        peak_score = 0.0
        peak_time = 0.0
        for i, (ts, sc, ab) in enumerate(zip(timestamps, kw_scores, above)):
            if ab and not in_peak:
                in_peak = True
                peak_start = ts
                peak_score = sc
                peak_time = ts
            elif ab and in_peak:
                if sc > peak_score:
                    peak_score = sc
                    peak_time = ts
            elif not ab and in_peak:
                in_peak = False
                triggered[kw].append((peak_time, peak_score))
        if in_peak:
            triggered[kw].append((peak_time, peak_score))

    print()
    print("=" * 60)
    print("触发结果汇总：")
    any_triggered = False
    for kw in keywords:
        if triggered[kw]:
            any_triggered = True
            for t, s in triggered[kw]:
                print(f"  ✅ [{kw}] 触发 @ {t:.2f}s  得分={s:.4f}")
        else:
            print(f"  ❌ [{kw}] 未触发（最高得分未超过阈值 {args.threshold}）")

    if not any_triggered:
        # 降低阈值打印 top-5 得分
        print(f"\n（未检测到触发，以下为各关键词 Top-5 得分时刻）")
        for kw_idx, kw in enumerate(keywords):
            if kw_idx >= scores.shape[1]:
                continue
            kw_scores = scores[:, kw_idx]
            top5_idx = np.argsort(kw_scores)[-5:][::-1]
            print(f"  [{kw}]:", ', '.join(
                f"{timestamps[i]:.2f}s={kw_scores[i]:.4f}" for i in top5_idx
            ))


if __name__ == '__main__':
    main()
