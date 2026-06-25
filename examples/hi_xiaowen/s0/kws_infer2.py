#!/usr/bin/env python3
"""
独立 KWS 推理脚本 - 仅依赖 wekws 自身 dataset，不依赖 wenet
用法:
  python kws_infer2.py \
    --config exp/ds_tcn/config.yaml \
    --checkpoint exp/ds_tcn/avg_30.pt \
    --test_data data/test_custom/data.list \
    --keywords_dict exp/ds_tcn/words.txt \
    --score_file exp/ds_tcn/score.txt \
    --gpu -1
"""

import argparse
import sys
import os
import json
import yaml
import torch

# 直接用 wekws 自己的 dataset，不走 init_dataset（避免 wenet 依赖）
from wekws.dataset.dataset import Dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--keywords_dict', required=True)
    parser.add_argument('--score_file', required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()

    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')

    # 读取 config
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # 读取关键词列表
    keywords = []
    with open(args.keywords_dict, 'r') as f:
        for line in f:
            arr = line.strip().split()
            if arr:
                keywords.append(arr[0])
    print(f'关键词: {keywords}')

    # 构建模型
    model = init_model(configs['model'])
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()
    print('模型加载成功')

    # 构建数据集（wekws 自带 dataset）
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_conf'] = {'batch_size': args.batch_size}
    dataset_conf['shuffle'] = False
    dataset_conf['filter_conf'] = {'max_length': 102400, 'min_length': 0}

    dataset = Dataset(args.test_data, dataset_conf)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
    )

    print(f'开始推理，输出到: {args.score_file}')
    os.makedirs(os.path.dirname(args.score_file) or '.', exist_ok=True)

    with open(args.score_file, 'w') as fout:
        for batch_idx, batch in enumerate(dataloader):
            # wekws dataset padding 返回 tuple:
            # (keys, feats, labels, feats_lengths, label_lengths)
            keys, feats, labels, feats_lengths, label_lengths = batch
            feats = feats.to(device)           # [B, T, D]
            feats_lengths = feats_lengths.to(device)  # [B]

            with torch.no_grad():
                # wekws 模型接口: (x, cache) -> (score, out_cache)
                in_cache = torch.zeros(0, 0, 0, device=device)
                score, _ = model(feats, in_cache)   # [B, T', num_kws]

            for i, key in enumerate(keys):
                T = score.shape[1]
                for keyword_i, keyword in enumerate(keywords):
                    if keyword_i >= score.shape[2]:
                        continue
                    keyword_scores = score[i, :, keyword_i]
                    score_frames = ' '.join(
                        ['{:.6f}'.format(x) for x in keyword_scores.tolist()])
                    fout.write('{} {} {}\n'.format(key, keyword, score_frames))

            if batch_idx % 5 == 0:
                print(f'Progress batch {batch_idx}')
                sys.stdout.flush()

    print('\n推理完成！')

    # 简单统计：打印每个关键词的最高分
    print('\n===== 得分统计 =====')
    kw_max = {kw: (0.0, '') for kw in keywords}
    with open(args.score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            key, kw = parts[0], parts[1]
            scores = [float(s) for s in parts[2:]]
            mx = max(scores)
            if mx > kw_max[kw][0]:
                kw_max[kw] = (mx, key)

    for kw, (mx, key) in kw_max.items():
        print(f'  [{kw}] 最高帧得分: {mx:.4f}  (来自: {key})')


if __name__ == '__main__':
    main()
