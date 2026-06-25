#!/bin/bash
# 微调脚本：基于预训练 DS_TCN 模型，在自定义数据上 finetune
# 预训练模型：/data1/jiankang.wang/cbbot/kws_wenwen_dstcn/avg_30.pt

. ./path.sh

# ===== 路径配置 =====
pretrained_model=/data1/jiankang.wang/cbbot/kws_wenwen_dstcn/avg_30.pt
dir=exp/finetune_ds_tcn
data_dir=data/finetune

# ===== 训练配置 =====
config=conf/finetune_ds_tcn.yaml
gpus="0"
num_workers=4

# ===== 解析参数 =====
stage=${1:-0}
stop_stage=${2:-4}

. tools/parse_options.sh || exit 1

export PYTHONPATH=/data1/jiankang.wang/cbbot/wekws:$PYTHONPATH

echo "stage=$stage, stop_stage=$stop_stage"
echo "pretrained: $pretrained_model"
echo "output dir: $dir"

mkdir -p $dir

# ===== Stage 0: 准备数据 =====
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: 准备微调数据..."
    python local/prepare_finetune_data.py \
        --haixiaowen_dir /data1/jiankang.wang/cbbot/kws_data/嗨小问 \
        --nihaowenwen_dir /data1/jiankang.wang/cbbot/kws_data/你好问问 \
        --negative_dir /data1/jiankang.wang/cbbot/wekws/examples/hi_xiaowen/s0/data/local/mobvoi_hotword_dataset \
        --negative_json /data1/jiankang.wang/cbbot/wekws/examples/hi_xiaowen/s0/data/download/mobvoi_hotword_dataset_resources/n_train.json \
        --output_dir $data_dir
    echo "Stage 0 done."
fi

# ===== Stage 1: 复制预训练配置，准备微调 config =====
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: 准备微调 config..."
    cp $pretrained_model $dir/init_model.pt
    cp /data1/jiankang.wang/cbbot/kws_wenwen_dstcn/global_cmvn $dir/global_cmvn
    cp /data1/jiankang.wang/cbbot/kws_wenwen_dstcn/words.txt $dir/words.txt
    # config 已在 conf/finetune_ds_tcn.yaml，直接使用
    cp $config $dir/config.yaml
    # 修复 cmvn 路径为绝对路径
    sed -i "s|cmvn_file:.*|cmvn_file: $dir/global_cmvn|" $dir/config.yaml
    echo "Stage 1 done."
fi

# ===== Stage 2: 微调训练 =====
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: 开始微调训练..."
    python wekws/bin/train.py \
        --config $dir/config.yaml \
        --train_data $data_dir/train/data.list \
        --cv_data $data_dir/dev/data.list \
        --gpus $gpus \
        --num_workers $num_workers \
        --checkpoint $pretrained_model \
        --model_dir $dir 2>&1 | tee $dir/train.log
    echo "Stage 2 done."
fi

# ===== Stage 3: 模型平均 =====
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: 模型平均..."
    num_average=5
    avg_model=$dir/avg_${num_average}.pt
    python wekws/bin/average_model.py \
        --dst_model $avg_model \
        --src_path $dir \
        --num $num_average \
        --val_best
    echo "平均模型: $avg_model"
    echo "Stage 3 done."
fi

# ===== Stage 4: 流式推理测试 =====
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: 流式推理测试..."
    num_average=5
    avg_model=$dir/avg_${num_average}.pt
    if [ ! -f $avg_model ]; then
        echo "模型文件不存在: $avg_model，跳过测试"
    else
        python kws_stream_infer.py \
            --config $dir/config.yaml \
            --checkpoint $avg_model \
            --wav data/kws_test_long.wav \
            --keywords_dict $dir/words.txt \
            --threshold 0.5
    fi
    echo "Stage 4 done."
fi

echo "All stages done!"
