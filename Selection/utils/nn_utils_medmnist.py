import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from .config_utils import logger
from .nn_utils import get_transform
import medmnist
from medmnist import INFO

transform_init = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))
 ])

def get_sample_info_cifar(class_num, chosen_sample_num):
    num_centroids = class_num
    final_sample_num = chosen_sample_num

    # We get one more centroid to take empty clusters into account
    if chosen_sample_num == 2500:
        num_centroids = 2501
        final_sample_num = 2500
        logger.warning("Returning 2501 as the number of centroids")

    return num_centroids, final_sample_num


def get_selection_ksdas(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
    selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=num_centroids)
    
    for _ in tqdm(range(iters)):
        selected_inds = []
        selected_inds_max = []
        selected_scores = []
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue
            
            # Scores in the selection process
            scores = 1 / neighbors_dist[match_arr] - \
                w * selection_regularizer[match_arr]
                
            scores_list = scores.tolist()
            n = int(final_sample_num/num_centroids)
            min_dist_ind = pd.Series(scores_list).sort_values(ascending = False).index[:n]
            min_dist_ind_max = scores.argmax()
            selected_inds_max.append(match[min_dist_ind_max])
            for i in min_dist_ind:
                selected_inds.append(match[i])
                selected_scores.append(scores[i])

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]
        selected_inds_max = np.array(selected_inds_max)
        selected_data_max = data[selected_inds_max]
        selected_scores = np.array(selected_scores)
        zipped = zip(selected_inds, selected_scores)
        sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse = True)
        result = zip(*sort_zipped)
        selected_inds, selected_scores = [list(x) for x in result]   
        

        new_selection_regularizer = (
            (data[:, None, :] - selected_data_max[None, :, :]) ** 2).sum(dim=-1)
        

        if verbose:
            logger.info(
                f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

        new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
            new_selection_regularizer + selected_ids_comparison_mask * 1e6
        

        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()} Min: {new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e6

        # selection_regularizer: N_full
        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)

        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)
            
    assert len(
        selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds, selected_scores


def get_selection_with_sdas(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, a=1.5, verbose=False):
    n = int(final_sample_num/num_centroids)
    mul_hot_data = np.zeros((len(cluster_labels), final_sample_num))
    for i, category in enumerate(cluster_labels):
        start_idx = category * n
        end_idx = (category + 1) * n
        mul_hot_data[i, start_idx:end_idx] = 1
    cluster_labels = torch.tensor(cluster_labels)
    mul_hot_data = torch.tensor(mul_hot_data)
    selection_regularizer = torch.zeros_like(neighbors_dist)
    intra_selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=num_centroids)

    for _ in tqdm(range(iters)):
        selected_inds = []
        selected_inds_max = []
        selected_scores = []

        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # Scores in the selection process
            scores = 1 / neighbors_dist[match_arr] - a * intra_selection_regularizer[match_arr] - \
                w * selection_regularizer[match_arr]
            
            scores_list = scores.tolist()
            min_dist_ind = pd.Series(scores_list).sort_values(ascending = False).index[:n]
            min_dist_ind_max = scores.argmax()
            selected_inds_max.append(match[min_dist_ind_max])
            for i in min_dist_ind:
                selected_inds.append(match[i])
                selected_scores.append(scores[i])
        

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]

        selected_inds_max = np.array(selected_inds_max)
        selected_data_max = data[selected_inds_max]
        selected_scores = np.array(selected_scores)
        zipped = zip(selected_inds, selected_scores)
        sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse = True)
        result = zip(*sort_zipped)
        selected_inds, selected_scores = [list(x) for x in result]

        new_selection_regularizer = (
            (data[:, None, :] - selected_data[None, :, :]) ** 2).sum(dim=-1)
        intra_new_selection_regularizer = (
            (data[:, None, :] - selected_data[None, :, :]) ** 2).sum(dim=-1)
        
        if verbose:
            logger.info(
                f"Max: {intra_new_selection_regularizer.max()} Mean: {intra_new_selection_regularizer.mean()}")
            logger.info(
                f"Selected_scores: {selected_scores}")


        new_selection_regularizer = (1 - mul_hot_data) * new_selection_regularizer 
        new_selection_regularizer[new_selection_regularizer == 0] = 1e6
        
        intra_new_selection_regularizer = mul_hot_data * intra_new_selection_regularizer 
        intra_new_selection_regularizer[intra_new_selection_regularizer == 0] = 1e6


        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Max: {intra_new_selection_regularizer.max()} Mean: {intra_new_selection_regularizer.mean()} Min: {intra_new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e6
            
            intra_new_selection_regularizer[intra_new_selection_regularizer >=
                                      horizon_dist] = 1e6

        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)
        intra_new_selection_regularizer = (
            1 / intra_new_selection_regularizer ** alpha).sum(dim=1)
        
        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)
            
        intra_selection_regularizer = intra_selection_regularizer * \
            momentum + intra_new_selection_regularizer * (1 - momentum)

    assert len(
        selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds, selected_scores


def train_memory_medmnist(dataname, transform_name, batch_size=128, workers=2, with_val=False, root='../datasets'):

    transform_test = get_transform(transform_name)
    info = INFO[dataname]
    DataClass = getattr(medmnist, info['python_class'])
    train_memory_dataset =  DataClass(split='train', transform=transform_test, download=True, as_rgb=True, root=root, size=28)
    if with_val:
        val_memory_dataset = DataClass(split='train+val', transform=transform_test, download=True, as_rgb=True, root=root, size=28)

    train_memory_loader = torch.utils.data.DataLoader(
        train_memory_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)

    if with_val:
        val_memory_loader = torch.utils.data.DataLoader(
            val_memory_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False)
        return train_memory_dataset, train_memory_loader, val_memory_dataset, val_memory_loader
    else:
        return train_memory_dataset, train_memory_loader
    