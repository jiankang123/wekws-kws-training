#!/usr/bin/env python3
"""
准备微调数据集：将自定义正例 + 现有负例 整理成 wekws data.list 格式
正例来自：嗨小问(id=0)、你好问问(id=1)
负例来自：mobvoi_hotword_dataset 的 negative 样本

用法：
  python prepare_finetune_data.py
"""

import os
import json
import wave
import random
import argparse

random.seed(42)


def get_duration(wav_path):
    try:
        with wave.open(wav_path, 'r') as w:
            return w.getnframes() / w.getframerate()
    except:
        return -1.0


def make_entries(wav_dir, label_id, label_name):
    """扫描目录，生成 data list 条目"""
    entries = []
    for fname in sorted(os.listdir(wav_dir)):
        if not fname.endswith('.wav'):
            continue
        wav_path = os.path.join(wav_dir, fname)
        dur = get_duration(wav_path)
        if dur <= 0:
            continue
        key = os.path.splitext(fname)[0]
        entries.append({
            'key': key,
            'txt': label_id,
            'duration': round(dur, 3),
            'wav': wav_path,
        })
    print(f"  [{label_name}] id={label_id}: {len(entries)} 条")
    return entries


def split_train_dev_test(entries, train_ratio=0.8, dev_ratio=0.1):
    """随机划分 train/dev/test"""
    random.shuffle(entries)
    n = len(entries)
    n_train = max(1, int(n * train_ratio))
    n_dev = max(1, int(n * dev_ratio))
    return entries[:n_train], entries[n_train:n_train+n_dev], entries[n_train+n_dev:]


def write_list(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f"  -> 写入 {len(entries)} 条到 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--haixiaowen_dir',
                        default='/data1/jiankang.wang/cbbot/kws_data/嗨小问')
    parser.add_argument('--nihaowenwen_dir',
                        default='/data1/jiankang.wang/cbbot/kws_data/你好问问')
    parser.add_argument('--negative_dir',
                        default='/data1/jiankang.wang/cbbot/wekws/examples/hi_xiaowen/s0/data/local/mobvoi_hotword_dataset')
    parser.add_argument('--negative_json',
                        default='/data1/jiankang.wang/cbbot/wekws/examples/hi_xiaowen/s0/data/download/mobvoi_hotword_dataset_resources/n_train.json',
                        help='负例标注json，用于区分正负例wav')
    parser.add_argument('--output_dir',
                        default='/data1/jiankang.wang/cbbot/wekws/examples/hi_xiaowen/s0/data/finetune')
    args = parser.parse_args()

    print("=== 准备正例数据 ===")
    hxw_entries = make_entries(args.haixiaowen_dir, 0, '嗨小问 Hi_Xiaowen')
    nhww_entries = make_entries(args.nihaowenwen_dir, 1, '你好问问 Nihao_Wenwen')

    print("\n=== 准备负例数据 ===")
    # 从 mobvoi 数据集中取负例（n_train.json 里记录了 utt_id）
    neg_utt_ids = set()
    if os.path.exists(args.negative_json):
        with open(args.negative_json, 'r') as f:
            n_data = json.load(f)
        neg_utt_ids = {item['utt_id'] for item in n_data}
        print(f"  n_train.json 中负例数: {len(neg_utt_ids)}")

    neg_entries = []
    if os.path.exists(args.negative_dir):
        all_wavs = [f for f in os.listdir(args.negative_dir) if f.endswith('.wav')]
        for fname in sorted(all_wavs):
            utt_id = os.path.splitext(fname)[0]
            # 如果有标注，只用标注为负例的；没有标注则全部用
            if neg_utt_ids and utt_id not in neg_utt_ids:
                continue
            wav_path = os.path.join(args.negative_dir, fname)
            dur = get_duration(wav_path)
            if dur <= 0:
                continue
            neg_entries.append({
                'key': utt_id,
                'txt': -1,
                'duration': round(dur, 3),
                'wav': wav_path,
            })
    print(f"  [负例 filler] id=-1: {len(neg_entries)} 条")

    # 按 8:1:1 划分每类
    print("\n=== 划分 train/dev/test ===")
    hxw_train, hxw_dev, hxw_test = split_train_dev_test(hxw_entries)
    nhww_train, nhww_dev, nhww_test = split_train_dev_test(nhww_entries)
    neg_train, neg_dev, neg_test = split_train_dev_test(neg_entries)

    train = hxw_train + nhww_train + neg_train
    dev   = hxw_dev   + nhww_dev   + neg_dev
    test  = hxw_test  + nhww_test  + neg_test

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    print(f"  train: {len(train)} (嗨小问={len(hxw_train)}, 你好问问={len(nhww_train)}, 负例={len(neg_train)})")
    print(f"  dev:   {len(dev)}   (嗨小问={len(hxw_dev)},  你好问问={len(nhww_dev)},  负例={len(neg_dev)})")
    print(f"  test:  {len(test)}  (嗨小问={len(hxw_test)}, 你好问问={len(nhww_test)}, 负例={len(neg_test)})")

    print("\n=== 写入 data.list ===")
    write_list(train, os.path.join(args.output_dir, 'train', 'data.list'))
    write_list(dev,   os.path.join(args.output_dir, 'dev',   'data.list'))
    write_list(test,  os.path.join(args.output_dir, 'test',  'data.list'))

    print("\n完成！数据已写入:", args.output_dir)


if __name__ == '__main__':
    main()
