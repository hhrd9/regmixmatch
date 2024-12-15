# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn.functional as F
import torch
import numpy as np
from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook
from torch.nn.functional import interpolate

class FreeMatchThresholingHook(MaskingHook):
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p 
    

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        valid_indices = max_probs >= 0.99
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask, valid_indices

def resizemix(img,gt_label,gpu,crop_ratio=0.9,scope=(0.1, 0.8),alpha=1.0,lam=None,use_alpha=False,interpolate_mode="nearest",is_bias=False):
    # normal mixup process
    rand_index = torch.randperm(img.size(0)).cuda(gpu)
    img_resize = img.clone()
    img_resize = img_resize[rand_index].cuda(gpu)
    _, _, h, w = img.size()
    shuffled_gt = gt_label[rand_index]

    # generate tao
    tao = np.random.beta(alpha, alpha)
    tao = np.sqrt(tao)
    tao = scope[0] + tao*(scope[1]-scope[0])

    # random box
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(img.size(), tao)
    # center crop first
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    img_resize = img_resize[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
    # resize
    img_resize = interpolate(
        img_resize, (bby2 - bby1, bbx2 - bbx1), mode=interpolate_mode
    )

    img[:, :, bby1:bby2, bbx1:bbx2] = img_resize
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    mixed_y = lam * gt_label + (1 - lam) * shuffled_gt

    return img, mixed_y, lam

def one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


def rand_bbox_tao(size, tao):
    """ generate random box by tao (scale) """
    W = size[2]
    H = size[3]
    cut_w = int(W * tao)
    cut_h = int(H * tao)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


        
