# MIT License
#
# Copyright (c) 2021 TorchSSL
#
# Modifications by RegMixMatch



import torch
import torch.nn.functional as F
from train_utils import ce_loss
import numpy as np
from torch.nn.functional import interpolate

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value



def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val



def entropy_loss(mask, logits_s, logits_w, prob_model, label_hist):
    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_w.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()



def consistency_loss(dataset, logits_s, logits_w, p_target, time_p, p_model, name='ce', tau_m=0.999, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        p_cutoff = time_p
        p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]
        threshold = p_cutoff * p_model_cutoff[max_idx]
        if dataset == 'svhn':
            threshold = torch.clamp(threshold, min=0.9, max=0.95)
        valid_indices = max_probs >= tau_m
        mask = max_probs.ge(threshold)
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
        return masked_loss.mean(), mask, max_idx, valid_indices, pseudo_label, max_probs

    else:
        assert Exception('Not Implemented consistency_loss')

        
def one_hot(targets, nClass, gpu):
    logits = torch.zeros(targets.size(0), nClass).cuda(gpu)
    return logits.scatter_(1, targets.unsqueeze(1), 1)


def l2_loss(logits_w, y):
    return F.mse_loss(torch.softmax(logits_w, dim=-1), y, reduction='mean')


# Copyright 2021-2023 CAIRI AI Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copyright: https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/augments/resizemix.py
# This code is modified version of resizemix, for resizemix of regmixmatch.
def resizemix(img,gt_label,gpu,crop_ratio=0.9,scope=(0.1, 0.8),alpha_h=1.0,lam=None,use_alpha=False,interpolate_mode="nearest",is_bias=False):
    # normal mixup process
    rand_index = torch.randperm(img.size(0)).cuda(gpu)
    img_resize = img.clone()
    img_resize = img_resize[rand_index].cuda(gpu)
    _, _, h, w = img.size()
    shuffled_gt = gt_label[rand_index]

    # generate tao
    tao = np.random.beta(alpha_h, alpha_h)
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


def resizemix_cam(x, y, x_h, y_h, gpu, crop_ratio=0.9,scope=(0.1, 0.8),alpha_l=1.0,interpolate_mode="nearest"):

    x_resize = x_h.clone()
    _, _, h, w = x.size()
    tao = np.random.beta(alpha_l, alpha_l)
    tao = np.sqrt(tao)
    tao = scope[0] + tao*(scope[1]-scope[0])
    # tao = np.random.uniform(scope[0], scope[1])

    # random box
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(x.size(), tao)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    # center crop first
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    x_resize = x_resize[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
    # resize
    x_resize = interpolate(
        x_resize, (bby2 - bby1, bbx2 - bbx1), mode=interpolate_mode
    )
    

    labels = torch.argmax(y, dim=-1)  # Convert from one-hot to integer labels
    labels_h = torch.argmax(y_h, dim=-1)
    for class_idx in labels.unique():
        # Find indices of all samples belonging to the current class
        class_mask = labels == class_idx
        class_mask_h = labels_h == class_idx
        class_indices = class_mask.nonzero(as_tuple=False).view(-1)
        class_indices_h = class_mask_h.nonzero(as_tuple=False).view(-1)
        if(len(class_indices_h)>0):
            # Randomly shuffle indices within this class
            shuffled_indices_within_class_h = class_indices_h[torch.randint(0, len(class_indices_h), (len(class_indices),))].view(-1)
            # Select samples within this class for mixup
            x[class_indices, :, bby1:bby2, bbx1:bbx2] = x_resize[shuffled_indices_within_class_h]
            y[class_indices] = lam * y[class_indices] + (1 - lam) * y_h[shuffled_indices_within_class_h]
        
    return x, y, lam


def resizemix_l(x, y, x_h, y_h, gpu, crop_ratio=0.9,scope=(0.1, 0.8),alpha=1.0,interpolate_mode="nearest"):
    x_resize = x_h.clone()
    bs, _, h, w = x.size()
    tao = np.random.beta(alpha, alpha)
    tao = np.sqrt(tao)
    tao = scope[0] + tao*(scope[1]-scope[0])
    # tao = np.random.uniform(scope[0], scope[1])

    # random box
    bbx1, bby1, bbx2, bby2 = rand_bbox_tao(x.size(), tao)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    # center crop first
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    x_resize = x_resize[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
    # resize
    x_resize = interpolate(
        x_resize, (bby2 - bby1, bbx2 - bbx1), mode=interpolate_mode
    )
    class_indices = torch.randint(0, len(x_h), (len(x),)).view(-1)
    x[:, :, bby1:bby2, bbx1:bbx2] = x_resize[class_indices]
    y = lam * y + (1 - lam) * y_h[class_indices]
        
    return x, y, lam
