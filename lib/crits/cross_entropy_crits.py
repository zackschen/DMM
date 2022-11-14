from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyCriterion(nn.Module):

    def __init__(self, reg_lambda):
        super(CrossEntropyCriterion, self).__init__()
        self.eps = 1e-8
        self.reg_lambda = reg_lambda
        self.softmax = nn.Softmax(dim=1)
        self.smooth_l1_loss_multi = nn.SmoothL1Loss(size_average=False)
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, att_scores, kld_labels, gt_reg, is_multi=False, reg_inweight=None,
                reg_all=None, num_reg=None):

        batch_size, num_props = att_scores.size(0), att_scores.size(1)
        att_logits = att_scores[:, :, 0].view(batch_size, num_props)
        softmax_scores = self.softmax(att_logits)

        y_true = torch.clamp(kld_labels, self.eps, 1.0)
        y_pred = torch.clamp(softmax_scores, self.eps, 1.0)
        loss_vec = torch.sum(y_true * torch.log(y_true/y_pred), dim=-1)
        loss_cls = torch.mean(loss_vec)

        if is_multi:
            reg_weight = reg_inweight.view(batch_size*num_props, 1).expand(-1, 4).contiguous()
            reg_logits = att_scores[:, :, 1:].view(-1, 4)
            reg_logits = reg_logits * reg_weight
            reg_all = reg_all.view(-1, 4)
            loss_reg = self.smooth_l1_loss_multi(reg_logits, reg_all) / torch.sum(reg_inweight)
        else:
            pred_label = torch.argmax(att_logits, dim=1).view(-1, 1).to(torch.int64)
            row_index = torch.range(0, batch_size-1).view(-1, 1).to(torch.int64)
            pred_reg = att_scores[:, :, 1:][[row_index, pred_label]]
            loss_reg = self.smooth_l1_loss(pred_reg, gt_reg)

        loss = loss_cls + self.reg_lambda * loss_reg

        return loss, loss_vec
