# %%
import os
os.environ["SDAS_MODE"] = "RANDOM"

from utils import cfg, logger, print_b
import utils
import numpy as np
import torch

# special configs for medmnist
from medmnist import INFO

utils.init(default_config_file="configs/PathMNIST_random.yaml")

logger.info(cfg)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in ['bloodmnist', 'dermamnist', 'octmnist', 'pathmnist'], f"{cfg.DATASET.NAME} is not a supported MedMNIST dataset"


info = INFO[cfg.DATASET.NAME]
num_classes = len(info['label'])
train_memory_dataset, train_memory_loader = utils.train_memory_medmnist(
    dataname = cfg.DATASET.NAME,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, root="../datasets")

# 收集所有样本的标签
all_labels = []
for batch_data in train_memory_loader:
    batch_labels = batch_data[1]  # 获取批次的标签
    labels_numpy = batch_labels.numpy()
    all_labels.extend(labels_numpy)

# 转换为torch tensor用于后续处理
labels_array = np.array(all_labels)
labels_tensor = torch.tensor(labels_array.astype(int)).squeeze()
print("Labels shape:", labels_tensor.shape)
print("Labels dtype:", labels_tensor.dtype)

# %%
def get_selection_fn(data_len, seed, final_sample_num):
    np.random.seed(seed)
    selected_inds = np.random.choice(np.arange(data_len), size=final_sample_num, replace=False)
    return selected_inds

# %%
num_labeled = cfg.RANDOM.NUM_SELECTED_SAMPLES
data_len = len(train_memory_dataset)
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP

for seed in cfg.RANDOM.SEEDS:
    print_b(f"Running random selection with seed {seed}")

    selected_inds = utils.get_selection(get_selection_fn, data_len=data_len, final_sample_num=num_labeled, recompute=recompute_num_dependent, save=True, seed=seed, pass_seed=True)

    counts = np.bincount(labels_tensor[selected_inds])

    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))

# %%
