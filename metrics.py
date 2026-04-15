import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
import data_utils
from skimage.morphology import skeletonize


###################### Utility functions for multi-label metrics ######################

def get_num_class(nb_classes):
    real_nb = 1
    overlapping=0

    while real_nb + overlapping != nb_classes:
        overlapping += real_nb
        real_nb += 1

        if overlapping + real_nb > nb_classes:
            print("Wrong number of class")
            return

    return real_nb

def map_overlapping_classes(nb_classes):
    """
    Function to get the overlapping classes for the Dice coefficient.
    Args:
        nb_classes (int): Number of classes in the dataset.
        include_background (bool): whether background must be counted as a class. If False (default), 
    Returns:
        list: mapping of overlapping classes. Each element is the list of overlapping classes for the corresponding class.
    Example:
        >>> get_overlapping_classes(2)
        [[3], [3]]
        >>> get_overlapping_classes(3)
        [[4, 5], [4, 6], [5, 6]]
    """
    overlap_mapping = [[] for _ in range(nb_classes)]
    for i in range(nb_classes-1):
        for j in range(i+1, nb_classes):
            overlap_mapping[i].append(nb_classes+i+j)
            overlap_mapping[j].append(nb_classes+i+j)
    return overlap_mapping

def multi2onehot_tensor(x:torch.Tensor, # Non one-hot encoded targs
        dim:int=2 # The axis to stack for encoding (class dimension)
    ) -> torch.Tensor:
        "Creates one binary mask per class"
        return torch.stack([torch.where((x==1) | (x==3), 1, 0), torch.where((x==2) | (x==3), 1, 0)], dim=dim)

def to_one_hot(preds, targets, mode="one_hot"):
    """
    Convert preds and targets to (B, C, H, W) binary tensors
    """

    if mode == "one_hot":
        if isinstance(preds, list):
            preds = preds[-1]

        if preds.dtype != torch.bool:
            preds = torch.sigmoid(preds) > 0.5

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        return preds.bool(), targets.bool()

    elif mode == "multi_label":
        preds = torch.argmax(preds, dim=1)

        num_classes = get_num_class(preds.max().item())

        overlap_mapping = map_overlapping_classes(num_classes)

        pred_channels = []
        target_channels = []

        for cls in range(1, num_classes + 1):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)

            for overlap_cls in overlap_mapping[cls - 1]:
                pred_cls |= (preds == overlap_cls)
                target_cls |= (targets == overlap_cls)

            pred_channels.append(pred_cls)
            target_channels.append(target_cls)

        return torch.stack(pred_channels, dim=1), torch.stack(target_channels, dim=1)

    else:
        raise ValueError(mode)

###################### Sensitivity ######################

def sensitivity_score(preds, targets, return_per_class=False, eps=1e-8):
    B, C = preds.shape[:2]

    preds = preds.view(B, C, -1)
    targets = targets.view(B, C, -1)

    tp = (preds & targets).sum(dim=2)
    fn = ((~preds) & targets).sum(dim=2)

    sens = tp / (tp + fn + eps)

    if return_per_class:
        return sens.mean(dim=0).cpu().numpy()

    return sens.mean()

def sensitivity(preds, targets, mode="one_hot", return_per_class=False):
    preds, targets = to_one_hot(preds, targets, mode)
    return sensitivity_score(preds, targets, return_per_class)

###################### Hausdorff Distance ######################

def hausdorff_percentile_optimized(A, B, percentile=95):
    """
    Compute the 95% Hausdorff distance (HD95) between two point sets A and B using KD-Trees.
    
    Parameters:
    A, B : ndarray
        Arrays of shape (N, D) and (M, D), representing N and M points in D-dimensional space.

    Returns:
    float
        The 95th percentile Hausdorff distance.
    """
    # Build KD-Trees for fast nearest-neighbor search
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    # Compute the nearest-neighbor distances from A to B and B to A
    min_dist_A = tree_A.query(B, k=1)[0]  # Closest distance from each B to A
    min_dist_B = tree_B.query(A, k=1)[0]  # Closest distance from each A to B

    # Combine both sets of distances
    all_distances = np.concatenate([min_dist_A, min_dist_B])

    # Compute the 95th percentile
    try:
        HD = np.percentile(all_distances, percentile)
    except IndexError:
        # If there are no distances (e.g., one of the sets is empty), return a large distance
        HD = float('inf')
    # HD = np.percentile(all_distances, percentile)
    
    return HD

def hausdorff_score(preds, targets, return_per_class=False, percentile=95):
    scores = []

    for c in range(preds.shape[1]):
        class_scores = []

        for b in range(preds.shape[0]):
            A = np.argwhere(preds[b, c].cpu().numpy())
            B = np.argwhere(targets[b, c].cpu().numpy())

            if len(A) == 0 or len(B) == 0:
                class_scores.append(np.nan)
                continue

            class_scores.append(
                hausdorff_percentile_optimized(A, B, percentile)
            )

        scores.append(np.nanmean(class_scores))

    if return_per_class:
        return scores

    return np.nanmean(scores)

def hausdorff_distance(preds, targets, mode="one_hot", return_per_class=False, percentile=95):
    preds, targets = to_one_hot(preds, targets, mode)
    return hausdorff_score(preds, targets, return_per_class, percentile)

###################### Dice ######################

def dice_score(preds, targets, return_per_class=False, eps=1e-8):
    B, C = preds.shape[:2]

    preds = preds.view(B, C, -1)
    targets = targets.view(B, C, -1)

    intersection = (preds & targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection + eps) / (union + eps)

    if return_per_class:
        return dice.mean(dim=0).cpu().numpy()

    return dice.mean()

def dice(preds, targets, mode="one_hot", return_per_class=False):
    preds, targets = to_one_hot(preds, targets, mode)
    return dice_score(preds, targets, return_per_class)

###################### clDice ######################

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def cldice_score(preds, targets, return_per_class=False):
    scores = []

    for c in range(preds.shape[1]):
        class_scores = []

        for b in range(preds.shape[0]):
            pred = preds[b, c].cpu().numpy()
            target = targets[b, c].cpu().numpy()

            tprec = cl_score(pred, skeletonize(target))
            tsens = cl_score(target, skeletonize(pred))

            class_scores.append(2 * tprec * tsens / (tprec + tsens + 1e-8))

        scores.append(np.mean(class_scores))

    if return_per_class:
        return scores

    return np.mean(scores)

def clDice(preds, targets, mode="one_hot", return_per_class=False, rrwnet=False):
    preds, targets = to_one_hot(preds, targets, mode)
    return cldice_score(preds, targets, return_per_class)