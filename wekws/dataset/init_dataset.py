# 替换版 init_dataset.py
# 使用 wekws 自己的 dataset（不依赖 wenet），兼容 PyTorch 1.13
# 原始版本依赖新版 wenet 的 Dataset/CharTokenizer，与旧版 PyTorch 不兼容

from wekws.dataset.dataset import Dataset


def init_dataset(data_list_file, conf, data_type='raw', partition=True):
    """
    初始化数据集（使用 wekws 自带 dataset，不依赖 wenet）

    Args:
        data_list_file: data.list 文件路径
        conf: dataset_conf 配置字典
        data_type: 'raw' 或 'shard'（目前只支持 raw），默认 'raw'
        partition: 是否按 rank 分片（分布式训练时使用）

    Returns:
        Dataset 对象
    """
    if data_type == 'shard':
        raise NotImplementedError("shard 格式暂不支持，请使用 raw 格式 (data.list)")

    # 处理 cv（验证集）配置：关闭 shuffle 和 spec_aug
    if not partition:
        import copy
        conf = copy.deepcopy(conf)
        conf['shuffle'] = False
        conf['spec_aug'] = False
        conf['speed_perturb'] = False

    dataset = Dataset(data_list_file, conf)
    return dataset
