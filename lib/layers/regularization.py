################################
# This code is reffered by
# https://github.com/GT-RIPL/Continual-Learning-Benchmark
################################

import logging
import random
from tqdm import tqdm
import numpy as np
import torch
from layers.joint_match import JointMatching


class EWC(JointMatching):

    def __init__(self, opt):
        super(EWC, self).__init__(opt)
        self.opt = opt
        self.fisher = None
        self.old_params = None
        self.ewc_lambda = 5.0

    def get_older_params(self):
        return {n: p.clone().detach() for n, p in self.get_ewc_params()}

    def get_ewc_params(self):
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        return params.items()
    
    
    def fisher_matrix_diag_origin(self, loader, category_i, split, mm_crit, att_crit):
        self.older_params = self.get_older_params()
        self.train()
        fisher = {n: torch.zeros(p.shape).cuda() for n, p in self.get_ewc_params() if p.requires_grad}
        
        while True:
            data = loader.getCategoryBatch(split, self.opt)
            # Set mini-batch dataset
            bs = len(data['ref_ids'])
            
            Feats = data['Feats']
            labels = data['labels']

            # add [neg_vis, neg_lang]
            if self.opt['visual_rank_weight'] > 0:
                Feats = loader.combine_feats(Feats, data['neg_Feats'])
                labels = torch.cat([labels, data['labels']])
            if self.opt['lang_rank_weight'] > 0:
                Feats = loader.combine_feats(Feats, data['Feats'])
                labels = torch.cat([labels, data['neg_labels']])

            att_labels = data['att_labels']
            if 'select_ixs' in data.keys():
                select_ixs = data['select_ixs']
            else:
                select_ixs = torch.tensor(0, dtype=torch.int).cuda()

            scores, _, sub_attn, loc_attn, rel_attn, _, _, att_scores = \
                    self.forward(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                            Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
            # print (scores)
            loss = mm_crit(scores)
            if select_ixs.numel() > 0:
                loss += self.opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                                    att_labels.index_select(0, select_ixs))
            loss.backward()

            for n, p in self.get_ewc_params():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * bs

            if data['bounds']['wrapped']:
                break

        fisher = {n: (p / bs) for n, p in fisher.items()}
        if category_i == 1:
            self.fisher = fisher
        else:
            for n in self.fisher.keys():
                self.fisher[n] = self.fisher[n] + fisher[n]
        return fisher

    def regularizer(self):
        loss_reg = 0
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in self.get_ewc_params():
            if n in self.fisher.keys():
                loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
        # print(loss_reg*self.lamb)
        return loss_reg

