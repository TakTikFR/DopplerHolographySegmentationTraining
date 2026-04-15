import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
import data_utils
from skimage.morphology import skeletonize


def multi2onehot_tensor(x:torch.Tensor, # Non one-hot encoded targs
        dim:int=2 # The axis to stack for encoding (class dimension)
    ) -> torch.Tensor:
        "Creates one binary mask per class"
        return torch.stack([torch.where((x==1) | (x==3), 1, 0), torch.where((x==2) | (x==3), 1, 0)], dim=dim)

def preprocess_tensors(preds, targets):
    if isinstance(preds, list):
        preds = preds[-1]
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    return preds, targets

###################### Sensitivity ######################

def sensitivity(preds, targets, return_per_class=False, eps=1e-8):
    preds, targets = preprocess_tensors(preds, targets)

    B, C = preds.shape[:2]

    preds = preds.view(B, C, -1)
    targets = targets.view(B, C, -1)

    tp = (preds & targets).sum(dim=2)
    fn = ((~preds) & targets).sum(dim=2)

    sens = tp / (tp + fn + eps)

    if return_per_class:
        return sens.mean(dim=0).cpu().numpy()

    return sens.mean()

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

def hausdorff_distance(preds, targets, return_per_class=False, percentile=95):
    preds, targets = preprocess_tensors(preds, targets)
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

###################### Dice ######################

def dice(preds, targets, return_per_class=False, eps=1e-8):
    preds, targets = preprocess_tensors(preds, targets)

    B, C = preds.shape[:2]

    preds = preds.view(B, C, -1)
    targets = targets.view(B, C, -1)

    intersection = (preds & targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection + eps) / (union + eps)

    if return_per_class:
        return dice.mean(dim=0).cpu().numpy()

    return dice.mean()

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

def clDice(preds, targets, return_per_class=False, rrwnet=False):
    preds, targets = preprocess_tensors(preds, targets)

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