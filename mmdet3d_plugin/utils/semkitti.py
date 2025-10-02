import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import normal


semantic_kitti_class_frequencies = np.array(
    [
        5.41773033e09,
        1.57835390e07,
        1.25136000e05,
        1.18809000e05,
        6.46799000e05,
        8.21951000e05,
        2.62978000e05,
        2.83696000e05,
        2.04750000e05,
        6.16887030e07,
        4.50296100e06,
        4.48836500e07,
        2.26992300e06,
        5.68402180e07,
        1.57196520e07,
        1.58442623e08,
        2.06162300e06,
        3.69705220e07,
        1.15198800e06,
        3.34146000e05,
    ]
)

kitti_class_names = [
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, non_empty_idx]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != ignore_index
    nonempty_target = ssc_target != non_empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    eps = 1e-5
    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / (nonempty_probs.sum()+eps)
    recall = intersection / (nonempty_target.sum()+eps)
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, ignore_index=255):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != ignore_index
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

def CE_ssc_loss_balance(pred, target, class_weights=None, ignore_index=255):
    """
    逐类计算 CrossEntropyLoss 再平均。
    
    参数:
      pred          (Tensor): 预测，形状 [B, C, ...]  
      target        (Tensor): 真实标签，形状 [B, ...]，每个值 in [0, C-1] 或 ==ignore_index  
      class_weights (Tensor): 长度 C 的一维 Tensor，用于最终跨类加权（可选）  
      ignore_index  (int):    在 target 中被忽略的标签值
    
    返回:
      loss (Tensor): 标量，总损失
    """
    # 1) 先计算 per-pixel 的 CE loss, shape [B, ...]
    
    #    如果提供了 class_weights，这里不传入 weight，后面再做跨类加权
    per_pixel_loss = F.cross_entropy(
        pred, 
        target.long(), 
        weight=None, 
        ignore_index=ignore_index, 
        reduction='none'
    )  # [B, H, W, ...] or [B, D1, D2, ...]
    
    # 2) 对每个 class 分别做 mask & 平均
    C = pred.shape[1]
    device = pred.device
    dtype = pred.dtype

    class_losses = []
    class_ws = []  # 如果要加权，可以把对应 weight 存到这里

    # 把 target 展成一维以便 indexing
    flat_loss = per_pixel_loss.reshape(-1)
    flat_target = target.reshape(-1)
    
    for cls in range(C):
        if cls == ignore_index:
            continue
        mask = (flat_target == cls)
        if mask.sum() == 0:
            # 该类在当前批次中没有样本，跳过
            continue
        
        # 该类所有像素的 average loss
        cls_loss = flat_loss[mask].mean()
        class_losses.append(cls_loss)
        
        if class_weights is not None:
            # 记录该类的权重
            class_ws.append(class_weights[cls].item())
        else:
            class_ws.append(1.0)

    if len(class_losses) == 0:
        # 如果整个 batch 没有任何有效类（大概率不应该发生）
        return torch.tensor(0.0, device=device, dtype=dtype)

    class_losses = torch.stack(class_losses)  # [K], K=当前批次包含的类数
    class_ws = torch.tensor(class_ws, device=device, dtype=dtype)  # [K]

    # 3) 跨类加权平均
    loss = (class_losses * class_ws).sum() / class_ws.sum()
    return loss

def CE_ssc_loss_corr(pred, target, class_weights=None, ignore_index=255, corr_dist=[0.05,0.05], gt_dist=0.0):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="none"
    )
    B, X, Y, Z = target.shape
    cls_mask = target == 1
    
    gt_dist_norm = gt_dist.float().clone()
    gt_dist_norm[:, 0:2,:, :, :] = gt_dist_norm[:, 0:2, :,:, :] / X
    gt_dist_norm[:, 2:4, :,:, :] = gt_dist_norm[:, 2:4,:, :, :] / Y
    gt_dist_norm[:, 4:6,:, :, :] = gt_dist_norm[:, 4:6, :,:, :] / Z


    len_x = gt_dist_norm[:,0,:,:,:] + gt_dist_norm[:,1,:,:,:]
    len_y = gt_dist_norm[:,2,:,:,:] + gt_dist_norm[:,3,:,:,:]

    x_mask = len_x > corr_dist[0]
    y_mask = len_y > corr_dist[1]

    sp_mask = (x_mask + y_mask).bool()
    corr_mask = sp_mask  & cls_mask

    len_x = gt_dist[:,0,:,:,:] + gt_dist[:,1,:,:,:]
    len_y = gt_dist[:,2,:,:,:] + gt_dist[:,3,:,:,:]

    loss = criterion(pred, target.long())
    loss[corr_mask] *=0

    loss = loss.mean()

    return loss

# len_x = gt_dist[:,0,:,:,:] + gt_dist[:,1,:,:,:]
# len_y = gt_dist[:,2,:,:,:] + gt_dist[:,3,:,:,:]

# car = len_x[(target==1)] 
# print(car.max(),car.float().mean(),car.min())


# car = len_x[(target==1).unsqueeze(1).repeat(1,6,1,1,1)] 
def Focal_CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = ClassBalancedFocalLoss(ignore_index=ignore_index)

    loss = criterion(pred, target.long(),semantic_kitti_class_frequencies)

    return loss

def BLV_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """
    criterion = BlvLoss(cls_num_list=torch.from_numpy(semantic_kitti_class_frequencies).float().cuda())

    loss = criterion(pred, target.long(),ignore_index=ignore_index)

    return loss  
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    gt_sorted: 1D Tensor of sorted ground truth (binary) values (0 or 1).
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    if gts == 0:
        return torch.zeros_like(gt_sorted)
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        grad = torch.cat((jaccard[0:1], jaccard[1:] - jaccard[:-1]))
    else:
        grad = jaccard
    return grad

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Flattened Lovasz-Softmax loss.
    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1).
    labels: [P] Tensor, ground truth labels (0...C-1)
    """
    if probas.numel() == 0:
        # only void pixels, the loss is 0
        return probas * 0.
    
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == 'present' else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        # probas for class c
        class_pred = probas[:, c]
        errors = (torch.abs(fg - class_pred))
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if losses:
        return sum(losses) / len(losses)
    else:
        # if no classes present, return zero
        return torch.tensor(0., device=probas.device)

def lovasz_softmax_loss(pred, target, ignore_index=None):
    """
    Lovász-Softmax loss.
    
    Args:
        pred (torch.Tensor): Logits from the network with shape [BS, C, D, H, W] for 3D segmentation.
        target (torch.Tensor): Ground truth labels with shape [BS, D, H, W].
        `ignore_index (int or None): Label to ignore.
        
    Returns:
        torch.Tensor: Computed Lovász-Softmax loss.
    """
    # reshape predictions and targets
    # [BS, C, D, H, W] -> [N, C];  [BS, D, H, W] -> [N]
    bs, C = pred.size(0), pred.size(1)
    pred = pred.permute(0, 2, 3, 4, 1).contiguous()
    pred_flat = pred.view(-1, C)
    target_flat = target.view(-1)
    
    if ignore_index is not None:
        valid = target_flat != ignore_index
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]
    
    probas = F.softmax(pred_flat, dim=1)
    loss = lovasz_softmax_flat(probas, target_flat, classes='present')
    return loss





class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for voxel segmentation/classification tasks.

    Args:
        beta (float): Hyperparameter for effective number calculation, usually close to 1 (e.g., 0.9999).
        gamma (float): Focusing parameter of Focal Loss to adjust the weighting on hard examples (commonly 2.0).
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
    """
    def __init__(self, beta=0.9999, gamma=2.0, ignore_index=255):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred, target, samples_per_class):
        """
        Args:
            pred (torch.Tensor): Prediction tensor of shape [BS, C, ...]. For example, [3, 20, 256, 256, 32].
            target (torch.Tensor): Ground truth labels of shape matching the spatial dimensions of pred (e.g., [BS, D, H, W]).
            samples_per_class (array-like): An array or list with the sample counts for each of the C classes.
        
        Returns:
            torch.Tensor: The average loss computed over all voxels.
        """
        device = pred.device
        num_classes = pred.size(1)

        # Calculate effective numbers per class using the formula:
        # effective_num = 1 - beta^(n_i)
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)  # Add a small epsilon to avoid division by zero
        
        # Normalize the weights such that the total sum equals the number of classes
        weights = weights / np.sum(weights) * num_classes
        
        # Convert weights to a tensor
        class_weights = torch.tensor(weights, dtype=pred.dtype, device=device)

        # Compute standard cross-entropy loss (without reduction) per voxel:
        ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            ignore_index=self.ignore_index, 
            reduction="none"
        )(pred, target.long())

        # Compute p_t using the relation: ce_loss = -log(p_t)  ==>  p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Compute the Focal Loss term:
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Return the mean loss over all voxels
        return focal_loss.mean()


def vel_loss(pred, gt):
    return F.l1_loss(pred, gt)
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

class BlvLoss(nn.Module):
#cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_num_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name



    def forward(self, pred, target, ignore_index, weight=None,  avg_factor=None, reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 4, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 4, 1, 2,3)

        loss = F.cross_entropy(pred, target, reduction='none',  ignore_index=ignore_index)

        if weight is not None:
            weight = weight.float()

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


