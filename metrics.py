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

###################### Sensitivity ######################

def sensitivity(p, t):
    tp = (p & t).float().sum()
    fn = ((~p) & t).float().sum()
    return tp / (tp + fn + 1e-8)

def sensitivity_one_hot(p,t, convert_to_one_hot=False, rrwnet=False):
    if convert_to_one_hot:
        t = multi2onehot_tensor(t, dim=1)
    if rrwnet:
        p = p[:,:2,:,:]
    p = torch.sigmoid(p) > 0.5
    return sensitivity(p,t)

def sensitivity_multi(preds, targets, return_class_score=False):
    num_classes = get_num_class(preds.shape[1]-1)

    preds = torch.argmax(preds, dim=1)  # Shape: (batch, H, W)

    sensitivities = []
    target_cls_sizes = []

    overlap_mapping = map_overlapping_classes(num_classes)

    for cls in range(1, num_classes+1):
        # We merge the classes for overlap with their corresponding classes, since it represents the intersection of the two classes.
        # Counting it as a separate class would lead to overestimation.
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        for overlap_cls in overlap_mapping[cls-1]:
            pred_cls |= (preds == overlap_cls)
            target_cls |= (targets == overlap_cls)

        pred_cls = pred_cls.to(bool).flatten()
        target_cls = target_cls.to(bool).flatten()
        
        target_cls_sizes.append(target_cls.sum().item())

        sensitivities.append(sensitivity(pred_cls, target_cls).item())
    
    if return_class_score:
        return sensitivities
    return np.average(sensitivities, weights=target_cls_sizes)

###################### Hausdorff Distance ######################

def hausdorff_distance(A, B):
    """
    Computes the Hausdorff Distance between two sets of points A and B.

    Parameters:
    A : t
        An (N, D) array representing N points in D-dimensional space.
    B : t
        An (M, D) array representing M points in D-dimensional space.

    Returns:
    float
        The Hausdorff Distance between A and B.
    """
    d_AB = directed_hausdorff(A, B)[0]
    d_BA = directed_hausdorff(B, A)[0]
    return max(d_AB, d_BA)

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

def hausdorff_dist_multi(preds, targets, return_class_hd = False, percentile=95):
    num_classes=get_num_class(preds.shape[1]-1)
    
    # Convert logits to class predictions (argmax)
    preds = torch.argmax(preds, dim=1)  # Shape: (batch, H, W)
    
    # Ignore background (assume class 0 is background)
    hd_scores = []
    target_cls_sizes = []

    overlap_mapping = map_overlapping_classes(num_classes)

    for cls in range(1, num_classes+1):
        # We merge the classes for overlap with their corresponding classes, since it represents the intersection of the two classes.
        # Counting it as a separate class would lead to overestimation.
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        for overlap_cls in overlap_mapping[cls-1]:
            pred_cls |= (preds == overlap_cls)
            target_cls |= (targets == overlap_cls)

        for i in range(pred_cls.shape[0]):
            if percentile == 100:
                hd = hausdorff_distance(np.argwhere(pred_cls[i].cpu().numpy()==1), np.argwhere(target_cls[i].cpu().numpy()==1))
            else:
                hd = hausdorff_percentile_optimized(np.argwhere(pred_cls[i].cpu().numpy()==1), np.argwhere(target_cls[i].cpu().numpy()==1), percentile=percentile)
            hd_scores.append(hd)
            target_cls_sizes.append(target_cls[i].sum().item())
    # Return mean Hausdorff Distance over foreground classes
    if return_class_hd:
        return hd_scores

    return np.average(hd_scores, weights=target_cls_sizes)

def hausdorff_dist_one_hot(preds, targets, return_class_hd = False, percentile=95, rrwnet=False):
    if type(preds) == list:
        preds = preds[-1]
    if rrwnet:
        targets = data_utils.multi2onehot_tensor(targets, dim=1)
        targets = targets[:,:2,:,:]
        preds = preds[:,:2,:,:]
        
    hd_scores = []
    if preds.unique() != [0,1]:
        preds = torch.sigmoid(preds) > 0.5
            # If there is no channel dimension, add one
    if len(targets.shape) == 3:
        targets = targets.view(targets.shape[0], preds.shape[1], targets.shape[1], targets.shape[2])
    try :
        for cls in range(preds.shape[1]):
            if percentile == 100:
                hd = hausdorff_distance(np.argwhere(preds[:,cls,:,:].cpu().numpy()==1), np.argwhere(targets[:,cls,:,:].cpu().numpy()==1))
            else:
                hd = hausdorff_percentile_optimized(np.argwhere(preds[:,cls,:,:].cpu().numpy()==1), np.argwhere(targets[:,cls,:,:].cpu().numpy()==1), percentile=percentile)
            hd_scores.append(hd)
    except Exception as e:
        print(f"Error computing Hausdorff Distance: {e}")
        hd_scores.append(np.nan)

    # Return mean Hausdorff Distance over foreground classes
    if return_class_hd:
        return hd_scores
    return np.nanmean(hd_scores)

###################### Dice ######################

def dice_multi(preds, targets, return_class_score = False, smooth=1e-6):
    """
    Function to calculate the Dice coefficient, when the input is a one-hot encoded tensor, with overlapping classes predicted as a separate class.
    The function returns the Dice coefficient for each class, and the mean Dice coefficient over all classes.
    It assumes the first tensor is the background.
        Args:
            preds (torch.Tensor): Predicted tensor with shape (batch, num_classes, H, W).
            targets (torch.Tensor): Target tensor with shape (batch, num_classes, H, W).
            return_class_score (bool): If True, return Dice coefficient for each class.
            smooth (float): Smoothing factor to avoid division by zero.
        Returns:
            float: Mean Dice coefficient over all classes.
    """
    num_classes=get_num_class(preds.shape[1]-1)
    # Convert logits to class predictions (argmax)
    preds = torch.argmax(preds, dim=1)  # Shape: (batch, H, W)

    # Ignore background (assume class 0 is background)
    dice_scores = []
    target_cls_sizes = []

    overlap_mapping = map_overlapping_classes(num_classes)

    for cls in range(1, num_classes+1):
        # We merge the third class with other classes, since it represents the intersection of the two classes.
        # Counting it as a separate class would lead to overestimation of the Dice score.
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        for overlap_cls in overlap_mapping[cls-1]:
            pred_cls |= (preds == overlap_cls)
            target_cls |= (targets == overlap_cls)
        
        pred_cls = pred_cls.to(torch.uint8).flatten()
        target_cls = target_cls.to(torch.uint8).flatten()
        
        # For weighting the mean score
        target_cls_sizes.append(target_cls.sum().item())
        
        intersection = (pred_cls & target_cls).sum()
        target_sum = target_cls.sum()
        pred_sum = pred_cls.sum()

        # If class absent in GT → ignore
        if target_sum == 0:
            dice_scores.append(torch.nan)
        else:
            dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
            dice_scores.append(dice.item())

    # Return mean Dice over foreground classes
    if return_class_score:
        return dice_scores
    
    return np.average(dice_scores, weights=target_cls_sizes)

def dice_one_hot(pred, target, rrwnet=False):
    if type(pred) == list:
        pred = pred[-1]
    if rrwnet:
        target = data_utils.multi2onehot_tensor(target, dim=1)
        target = target[:,:2,:,:]
        pred = pred[:,:2,:,:]

    if pred.unique() != [0,1]:
        pred = torch.sigmoid(pred) > 0.5
    # print(pred.unique())
    # Flatten spatial dimensions (B, C, H, W) → (B, C, H*W)
    pred = pred.view(pred.shape[0], pred.shape[1], -1)
    
    # If there is no channel dimension, add one
    if len(target.shape) == 3:
        target = target.view(target.shape[0], pred.shape[1], -1)
    else:
        target = target.view(target.shape[0], target.shape[1], -1)
    
    # Compute intersection and union
    intersection = (pred * target).sum(dim=2)
    union = pred.sum(dim=2) + target.sum(dim=2)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (union + 1e-8)  # Add small epsilon to avoid division by zero
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

def clDice(p, target):
    """[this function computes the cldice metric]

    Args:
        pred ([bool]): [predicted image]
        target ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec, tsens = 1e-8, 1e-8
    # print(p.shape)
    if len(p.shape) == 4:
        if p.shape[1] == 1:
            pred = p[0,0,:,:]
        else:
            pred = p[0,0,:,:]  # Remove channel dimension
    else : 
        pred = p
    target = torch.squeeze(target).cpu().numpy()
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred = (pred > 0.5).float().detach().cpu().numpy()  # Binarize predictions
    if len(pred.shape)==2:
        tprec = cl_score(pred,skeletonize(target))
        tsens = cl_score(target,skeletonize(pred))
    elif len(pred.shape)==3:
        tprec = cl_score(pred,skeletonize(target, method='lee'))
        tsens = cl_score(target,skeletonize(pred, method='lee'))

    # print(f"TPrec: {tprec}, TRecall: {tsens}")
    return 2*tprec*tsens/(tprec+tsens)

def multi_label_clDice(pred, target, return_class_score=False):
    """[this function computes the cldice metric for multiple labels]
    Args:
        pred ([bool]): [predicted image of shape BxCxHxW]
        target ([bool]): [ground truth image of shape BxCxHxW]
    Returns:
        [float]: [cldice metric]
    """
    num_classes=get_num_class(pred.shape[1]-1)
    pred = torch.argmax(pred, dim=1)  # Shape: (batch, H, W)

    overlap_mapping = map_overlapping_classes(num_classes)
    cldice_scores = []
    target_cls_sizes = []

    for cls in range(1, num_classes+1):
        # We merge the classes for overlap with their corresponding classes, since it represents the intersection of the two classes.
        # Counting it as a separate class would lead to overestimation.
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        for overlap_cls in overlap_mapping[cls-1]:
            pred_cls |= (pred == overlap_cls)
            target_cls |= (target == overlap_cls)

        for i in range(pred_cls.shape[0]):
            cldice_scores.append(clDice(pred_cls[i], target_cls[i]))
            # For weighting the mean score
            target_cls_sizes.append(target_cls[i].sum().item())


    if return_class_score:
        return cldice_scores
    
    return np.average(cldice_scores, weights=target_cls_sizes)

def clDice_one_hot(p, t, rrwnet=False):
    """[this function computes the cldice metric]

    Args:
        pred ([bool]): [predicted image]
        target ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if type(p) == list:
        p = p[-1]
    if rrwnet:
        t = data_utils.multi2onehot_tensor(t, dim=1)
        t = t[:,:2,:,:]
        p = p[:,:2,:,:]

    tprec, tsens = 1e-8, 1e-8
    # print(p.shape, t.shape)
    nb_batches = p.shape[0]
    nb_classes = p.shape[1]

    if len(t.shape) < len(p.shape):
        t = t.unsqueeze(1)

    classes_score = [0] * nb_classes
    for b in range(nb_batches):
        for c in range(nb_classes):
            pred = p[b,c,:,:]  # Remove channel dimension
            target = t[b,c,:,:].cpu().numpy()
            pred = torch.sigmoid(pred)  # Convert logits to probabilities
            pred = (pred > 0.5).float().detach().cpu().numpy()  # Binarize predictions
            if len(pred.shape)==2:
                tprec = cl_score(pred,skeletonize(target))
                tsens = cl_score(target,skeletonize(pred))
            elif len(pred.shape)==3:
                tprec = cl_score(pred,skeletonize(target, method='lee'))
                tsens = cl_score(target,skeletonize(pred, method='lee'))
            classes_score[c] += 2*tprec*tsens/(tprec+tsens)
    # print(f"TPrec: {tprec}, TRecall: {tsens}")
    return np.mean(np.array(classes_score)/nb_batches)