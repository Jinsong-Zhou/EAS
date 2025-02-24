# %%
import os
os.environ["SDAS_MODE"] = "USL"

import numpy as np
import torch
import models.resnet_medmnist as resnet_medmnist
import utils
from utils import cfg, logger, print_b
from medmnist import INFO

utils.init(default_config_file="configs/BloodMNIST_sdas.yaml")

logger.info(cfg)

print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)
assert cfg.DATASET.NAME in ['bloodmnist', 'dermamnist', 'pathmnist', 'octmnist'], f"{cfg.DATASET.NAME} is not a supported MedMNIST dataset"

# model
model = resnet_medmnist.__dict__[cfg.MODEL.ARCH]().cuda()
state_dict = utils.single_model(checkpoint["train_model"])
mismatch = model.load_state_dict(state_dict, strict=False)
logger.warning(
    f"Key mismatches: {mismatch} (extra contrastive keys are intended)")
model.eval()    

# dataset
print_b("Loading dataset")
info = INFO[cfg.DATASET.NAME]
num_classes = len(info['label'])
train_memory_dataset, train_memory_loader = utils.train_memory_medmnist(
    dataname = cfg.DATASET.NAME,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME)

# 收集所有样本的标签
all_labels = []
for batch_data in train_memory_loader:
    batch_labels = batch_data[1]  # 获取批次的标签
    labels_numpy = batch_labels.numpy()
    all_labels.extend(labels_numpy)

# 转换为torch tensor用于后续处理
labels_array = np.array(all_labels)
labels_tensor = torch.tensor(labels_array.astype(int))

# %%
print_b("Loading feat list")
print(train_memory_dataset)
feats_list = utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
score_first_order = 1/neighbors_dist

# %%
num_centroids, final_sample_num = utils.get_sample_info_cifar(class_num=cfg.CLASS_NUM,
    chosen_sample_num=cfg.USL.NUM_SELECTED_SAMPLES)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))

# %%
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
cluster_labels = np.load(cfg.CLUSTERING_LABEL)

print_b("Getting selections with regularization")
selected_inds, selected_scores = utils.get_selection(utils.get_selection_with_sdas, feats_list, neighbors_dist,
                                                        cluster_labels, num_centroids,
                                                        final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS,
                                                        w=cfg.USL.REG.W,
                                                        momentum=cfg.USL.REG.MOMENTUM,
                                                        horizon_dist=cfg.USL.REG.HORIZON_DIST, alpha=cfg.USL.REG.ALPHA,
                                                        verbose=True, 
                                                        recompute=recompute_num_dependent, 
                                                        # a = cfg.USL.A,
                                                        save=True)

counts = np.bincount(labels_tensor[selected_inds].flatten())


print("Class counts:", sum(counts > 0))
print(counts.tolist())

print("max: {}, min: {}".format(counts.max(), counts.min()))

print("Number of selected indices:", len(selected_inds))
print("Selected IDs:")
print(repr(selected_inds))