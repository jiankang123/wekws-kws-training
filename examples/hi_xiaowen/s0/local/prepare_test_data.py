#!/usr/bin/env python3
"""
准备测试集的 data.list 文件，用于 wekws 推理
从 mobvoi_hotword_dataset 的 json 标注生成 wekws 格式的 data.list
"""

import os
import json
import wave
import argparse


def get_wav_duration(wav_path):
    """获取 wav 文件时长（秒）"""
    try:
        with wave.open(wav_path, 'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception as e:
        return -1.0


def prepare_test_list(wav_dir, p_json, n_json, output_dir, max_files=None):
    """
    wav_dir: wav 文件目录
    p_json: 正例 json 文件（唤醒词）
    n_json: 负例 json 文件（非唤醒词）
    output_dir: 输出目录
    max_files: 每类最多使用多少条（None=全部）
    """
    os.makedirs(output_dir, exist_ok=True)

    entries = []

    # 处理正例（keyword_id=0 对应 "hi xiaowen"）
    with open(p_json, 'r', encoding='utf-8') as f:
        p_data = json.load(f)

    print(f"正例总数: {len(p_data)}")
    if max_files:
        p_data = p_data[:max_files]
    print(f"使用正例: {len(p_data)}")

    for item in p_data:
        utt_id = item['utt_id']
        wav_path = os.path.join(wav_dir, utt_id + '.wav')
        if not os.path.exists(wav_path):
            continue
        duration = get_wav_duration(wav_path)
        if duration < 0:
            continue
        # keyword_id=0 -> "hi xiaowen"，keyword_id=1 -> "nihaowenwen"
        keyword_id = item.get('keyword_id', 0)
        if keyword_id == 0:
            txt = 'hi xiaowen'
        elif keyword_id == 1:
            txt = 'nihaowenwen'
        else:
            txt = '<SILENCE>'
        entries.append(dict(key=utt_id, txt=txt, duration=round(duration, 3), wav=wav_path))

    # 处理负例（非唤醒词）
    with open(n_json, 'r', encoding='utf-8') as f:
        n_data = json.load(f)

    print(f"负例总数: {len(n_data)}")
    if max_files:
        n_data = n_data[:max_files]
    print(f"使用负例: {len(n_data)}")

    for item in n_data:
        utt_id = item['utt_id']
        wav_path = os.path.join(wav_dir, utt_id + '.wav')
        if not os.path.exists(wav_path):
            continue
        duration = get_wav_duration(wav_path)
        if duration < 0:
            continue
        entries.append(dict(key=utt_id, txt='<SILENCE>', duration=round(duration, 3), wav=wav_path))

    # 写 data.list
    output_list = os.path.join(output_dir, 'data.list')
    with open(output_list, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n生成完成: {output_list}")
    print(f"总条数: {len(entries)}")
    return output_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', required=True, help='wav 文件目录')
    parser.add_argument('--p_json', required=True, help='正例 JSON 文件')
    parser.add_argument('--n_json', required=True, help='负例 JSON 文件')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--max_files', type=int, default=None, help='每类最多条数，默认全部')
    args = parser.parse_args()

    prepare_test_list(
        wav_dir=args.wav_dir,
        p_json=args.p_json,
        n_json=args.n_json,
        output_dir=args.output_dir,
        max_files=args.max_files
    )
