from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import sys
import math
import time
import random
from pprint import pprint

# model
import _init_paths
import models.utils as model_utils
import models.eval_easy_utils as eval_utils

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Weight_Regularized_Adam(optim.Adam):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Weight_Regularized_Adam, self).__init__(params, lr=lr, betas=betas,eps=eps,weight_decay=weight_decay,amsgrad=amsgrad)

    def __setstate__(self, state):
        super(Weight_Regularized_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.

        Arguments:
            reg_params: omega of all the params
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        reg_lambda = reg_params.get('lambda')

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                #MAS PART CODE GOES HERE
                #if this param has an omega to use for regulization
                if p in reg_params:

                    reg_param = reg_params.get(p)
                    #get omega for this parameter
                    omega = reg_param.get('omega')
                    #initial value when the training start
                    init_val = reg_param.get('init_val')

                    curr_wegiht_val = p.data
                    #move the tensors to cuda
                    init_val = init_val.cuda()
                    omega = omega.cuda()

                    #get the difference
                    weight_dif = curr_wegiht_val.add(-1, init_val)
                    #compute the MAS penalty
                    regulizer = weight_dif.mul(2*reg_lambda*omega)
                    del weight_dif
                    del curr_wegiht_val
                    del omega
                    del init_val
                    #add the MAS regulizer to the gradient
                    grad.add_(regulizer)
                    del regulizer
                #MAS PARAT CODE ENDS

                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = (group['lr'] if isinstance(group['lr'], float) else group['lr']['lr']) * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class MAS_Omega_update(optim.Adam):
    """
    Update the parameter importance using the gradient of the function output norm. To be used at deployment
        time.
    reg_params:parameters omega to be updated
    batch_index, batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MAS_Omega_update, self).__init__(params, **defaults)

    def __setstate__(self, state):
        super(MAS_Omega_update, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, reg_params, prev_size, cur_size, closure=None):
        """
        Performs a single parameters importance update step.
        """

        #print ('********************************DOING A STEP********************************')

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            #if the parameter has an omega to be updated
            for p in group['params']:

                #print ('******************************ONE PARAM******************************')

                if p.grad is None:
                    continue

                if p in reg_params:
                    grad = p.grad.data

                    #HERE MAS IMPORTANCE UPDATE GOES
                    #get the gradient
                    unreg_dp = p.grad.data.clone()
                    reg_param = reg_params.get(p)

                    zero = torch.FloatTensor(p.data.size()).zero_()
                    #get parameter omega
                    omega = reg_param.get('omega')
                    omega = omega.cuda()

                    #sum up the magnitude of the gradient
                    omega = omega.mul(prev_size)

                    omega = omega.add(unreg_dp.abs_())
                    #update omega value
                    omega = omega.div(cur_size)
                    if omega.equal(zero.cuda()):
                        print ('omega after zero')

                    reg_param['omega'] = omega

                    reg_params[p] = reg_param
                    #HERE MAS IMPORTANCE UPDATE ENDS

        return loss#HAS NOTHING TO DO


class MAS_Omega_Vector_Grad_update(optim.Adam):
    """
    Update the parameter importance using the gradient of the function output norm. To be used at deployment
        time.
    reg_params:parameters omega to be updated
    batch_index, batch_size:used to keep a running average over the seen samples
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MAS_Omega_Vector_Grad_update, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MAS_Omega_Vector_Grad_update, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, reg_params, intermediate, prev_size, cur_size, closure=None):
        """
        Performs a single parameters importance update step.
        """

        #print ('********************************DOING A STEP********************************')

        loss = None
        if closure is not None:
            loss = closure()
        index = 0

        for group in self.param_groups:
            for p in group['params']:

                #print ('******************************ONE PARAM******************************')

                if p.grad is None:
                    continue

                if p in reg_params:
                    grad = p.grad.data
                    unreg_dp = p.grad.data.clone()
                    #HERE MAS CODE GOES
                    reg_param = reg_params.get(p)

                    zero = torch.FloatTensor(p.data.size()).zero_()
                    omega = reg_param.get('omega')
                    omega = omega.cuda()

                    #get the magnitude of the gradient
                    if intermediate:
                        if 'w' in reg_param.keys():
                            w = reg_param.get('w')
                        else:
                            w = torch.FloatTensor(p.data.size()).zero_()
                        w=w.cuda()
                        w=w.add(unreg_dp.abs_())
                        reg_param['w'] = w
                    else:
                        #sum up the magnitude of the gradient
                        w = reg_param.get('w')
                        omega = omega.mul(prev_size)
                        omega = omega.add(w)
                        omega = omega.div(cur_size)
                        reg_param['w'] = zero.cuda()

                        if omega.equal(zero.cuda()):
                            print ('omega after zero')

                    reg_param['omega'] = omega
                    #pdb.set_trace()
                    reg_params[p] = reg_param
                    #HERE MAS IMPORTANCE UPDATE ENDS
                index += 1

        return loss#HAS NOTHING TO DO


# train one iter
def lossFun(loader, current_buffer, last_buffer, split, optimizer, model, mm_crit, att_crit, opt, iter,
            rel_buffer=None, loc_buffer=None, rel_last_buffer=None, loc_last_buffer=None):
    # set mode
    model.train()

    # zero gradient
    optimizer.zero_grad()

    # time
    T = {}

    # load one batch of data
    tic = time.time()
    data = loader.getCategoryBatch(split, opt, False)
    ref_ids = data['ref_ids']
    ref_sent_ids = data['ref_sent_ids']
    Feats = data['Feats']
    labels = data['labels']
    neg_labels = data['neg_labels']

    #################################################################
    # combine loder_list
    # undercontructing
    if opt['multi_buffer'] > 0:
        sub_loader_list = last_buffer.transform_loader_list(opt['buffer_sample_number']/3)
        loc_loader_list = loc_last_buffer.transform_loader_list(opt['buffer_sample_number']/3)
        rel_loader_list = rel_last_buffer.transform_loader_list(opt['buffer_sample_number']/3)
        loader_list = sub_loader_list + loc_loader_list + rel_loader_list
    else:
        loader_list = last_buffer.transform_loader_list(opt['buffer_sample_number'])

    buffer_data = loader.getBufferBatch(loader_list, opt, False)
    #################################################################
    """
    buffer_data = loader.getBufferBatch(last_buffer.transform_loader_list(opt['buffer_sample_number']),
                                        opt, False)
    """
    buffer_labels = buffer_data['labels']
    buffer_neg_labels = buffer_data['neg_labels']

    # chunk pos_labels and neg_labels using max_len
    max_len = max([(labels != 0).sum(1).max().item(),
                   (neg_labels != 0).sum(1).max().item(),
                   (buffer_labels != 0).sum(1).max().item(),
                   (buffer_neg_labels != 0).sum(1).max().item()])
    labels = labels[:, :max_len]
    neg_labels = neg_labels[:, :max_len]
    buffer_labels = buffer_labels[:, :max_len]
    buffer_neg_labels = buffer_neg_labels[:, :max_len]

    clone_labels = labels.clone()

    Feats = loader.combine_feats(Feats, buffer_data['Feats'])
    labels = torch.cat([labels, buffer_labels])

    # add [neg_vis, neg_lang]
    if opt['visual_rank_weight'] > 0:
        Feats = loader.combine_feats(Feats, data['neg_Feats'])
        labels = torch.cat([labels, clone_labels])
        Feats = loader.combine_feats(Feats, buffer_data['neg_Feats'])
        labels = torch.cat([labels, buffer_labels])
    if opt['lang_rank_weight'] > 0:
        Feats = loader.combine_feats(Feats, data['Feats'])
        labels = torch.cat([labels, neg_labels])
        Feats = loader.combine_feats(Feats, buffer_data['Feats'])
        labels = torch.cat([labels, buffer_neg_labels])

    att_labels, select_ixs = data['att_labels'], data['select_ixs']
    buffer_att_labels, buffer_select_ixs = buffer_data['att_labels'], buffer_data['select_ixs']

    att_labels = torch.cat([att_labels, buffer_att_labels])
    select_ixs = torch.cat([select_ixs, buffer_select_ixs])
    T['data'] = time.time()-tic

    # forward
    tic = time.time()
    scores, _, sub_attn, loc_attn, rel_attn, _, weights, att_scores = \
            model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                  Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
    """
    scores, _, = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], labels)
    """
    # print (scores)
    loss, loss_batch = mm_crit(scores)
    if select_ixs.numel() > 0:
        loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                             att_labels.index_select(0, select_ixs))

    # if iter < 500:
    #   num_pos = len(data['ref_ids'])
    #   loss += 0.1*model.sub_rel_kl(sub_attn[:num_pos], rel_attn[:num_pos], labels[:num_pos])

    loss.backward()
    model_utils.clip_gradient(optimizer, opt['grad_clip'])
    optimizer.step(model.reg_params)
    T['model'] = time.time()-tic

    """
    # update
    #################################################################
    # update L, R
    if opt['multi_buffer'] > 0:
        batch_size = loss_batch.size()[0]
        loc_weights = weights[:, 1][:batch_size]
        rel_weights = weights[:, 2][:batch_size]
        loc_loss_batch = loss_batch * loc_weights
        rel_loss_batch = loss_batch * rel_weights

        loc_loss_batch = loc_loss_batch.cpu().detach().tolist()[:-opt['buffer_sample_number']]
        loc_buffer.update(ref_ids, ref_sent_ids, loc_loss_batch)
        rel_loss_batch = rel_loss_batch.cpu().detach().tolist()[:-opt['buffer_sample_number']]
        rel_buffer.update(ref_ids, ref_sent_ids, rel_loss_batch)
    #################################################################


    loss_batch = loss_batch.cpu().detach().tolist()[:-opt['buffer_sample_number']]
    current_buffer.update(ref_ids, ref_sent_ids, loss_batch)
    """
    if opt['subject_flag'] > 0:
        batch_size = loss_batch.size()[0]
        sub_weights = weights[:, 0][:batch_size]
        loss_batch = loss_batch * sub_weights

    return loss.item(), T, data['bounds']['wrapped']

def id_to_sentids(loader_list):
    loader_sent_ids = []
    for id in loader_list:
        loader_sent_ids.append(id['sent_id'])
    return loader_sent_ids

def map_two_list(source, target):
    map_dict = {}
    for i, sent_id in enumerate(source):
        map_dict[i] = target.index(sent_id)
    return map_dict

def update_buffer_loss(loader, current_buffer, model, mm_crit, att_crit, opt, iter,
                       rel_buffer=None, loc_buffer=None):
    # set mode
    model.eval()

    # time
    T = {}

    # load one batch of data
    tic = time.time()

    ########################################################
    # combine loader list
    # underconstruct
    if opt['multi_buffer'] > 0:
        sub_loader_list = current_buffer.transform_loader_list()
        loc_loader_list = loc_buffer.transform_loader_list()
        rel_loader_list = rel_buffer.transform_loader_list()
        loader_list = sub_loader_list + loc_loader_list + rel_loader_list
        loader_sent_ids = id_to_sentids(loader_list)
    else:
        loader_list = current_buffer.transform_loader_list()

    data = loader.getBufferBatch(loader_list, opt)
    ########################################################
    """
    data = loader.getBufferBatch(current_buffer.transform_loader_list(), opt)
    """

    ref_sent_ids = data['ref_sent_ids']
    if opt['multi_buffer'] > 0:
        map_dict = map_two_list(loader_sent_ids, ref_sent_ids)
    Feats = data['Feats']
    labels = data['labels']

    # add [neg_vis, neg_lang]
    if opt['visual_rank_weight'] > 0:
        Feats = loader.combine_feats(Feats, data['neg_Feats'])
        labels = torch.cat([labels, data['labels']])
    if opt['lang_rank_weight'] > 0:
        Feats = loader.combine_feats(Feats, data['Feats'])
        labels = torch.cat([labels, data['neg_labels']])

    att_labels, select_ixs = data['att_labels'], data['select_ixs']

    T['data'] = time.time()-tic

    # forward
    tic = time.time()
    scores, _, sub_attn, loc_attn, rel_attn, _, weights, att_scores = \
            model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                  Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
    """
    scores, _, = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], labels)
    """
    # print (scores)
    loss, loss_batch = mm_crit(scores)
    if select_ixs.numel() > 0:
        loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                             att_labels.index_select(0, select_ixs))

    T['model'] = time.time()-tic

    # update
    ######################################################
    if opt['multi_buffer'] > 0:
        loader_loss_batch = []
        loader_weights = []
        loss_batch = loss_batch.cpu().detach().tolist()
        weights = weights.cpu().detach().tolist()
        for i in range(len(loader_sent_ids)):
            loader_loss_batch.append(loss_batch[map_dict[i]])
            loader_weights.append(weights[map_dict[i]])
        loader_loss_batch = np.array(loader_loss_batch)
        loader_weights = np.array(loader_weights)

        sub_size, loc_size, rel_size = int(current_buffer.size), int(loc_buffer.size), int(rel_buffer.size)
        sub_weights = loader_weights[:sub_size, 0]
        loc_weights = loader_weights[sub_size:sub_size+loc_size, 1]
        rel_weights = loader_weights[sub_size+loc_size:sub_size+loc_size+rel_size, 2]

        if opt['subject_flag'] > 0:
            sub_loss_batch = loader_loss_batch[:sub_size] * sub_weights
        else:
            sub_loss_batch = loader_loss_batch[:sub_size]
        loc_loss_batch = loader_loss_batch[sub_size:sub_size+loc_size] * loc_weights
        rel_loss_batch = loader_loss_batch[sub_size+loc_size:sub_size+loc_size+rel_size] * rel_weights

        current_buffer.update_loss(loader_sent_ids[:sub_size], sub_loss_batch.tolist())
        loc_buffer.update_loss(loader_sent_ids[sub_size:sub_size+loc_size], loc_loss_batch.tolist())
        rel_buffer.update_loss(loader_sent_ids[sub_size+loc_size:sub_size+loc_size+rel_size], rel_loss_batch.tolist())
    else:
        if opt['subject_flag'] > 0:
            batch_size = loss_batch.size()[0]
            sub_weights = weights[:, 0][:batch_size]
            loss_batch = loss_batch * sub_weights

        loss_batch = loss_batch.cpu().detach().tolist()
        current_buffer.update_loss(ref_sent_ids, loss_batch)
    ######################################################
    """
    if opt['subject_flag'] > 0:
        batch_size = loss_batch.size()[0]
        sub_weights = weights[:, 0][:batch_size]
        loss_batch = loss_batch * sub_weights

    loss_batch = loss_batch.cpu().detach().tolist()
    current_buffer.update_loss(ref_sent_ids, loss_batch)
    """

    return T


def train_model(model, criterion, att_crit, current_buffer, last_buffer, optimizer, loader, split, opt,
                model_dir, category, log, log_category, resume, rel_buffer=None, loc_buffer=None,
                rel_last_buffer=None, loc_last_buffer=None):

    infos = {}
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    cat_to_ix = loader.cat_to_ix
    supcat_to_ix = loader.supcat_to_ix
    category_structure = loader.category_structure

    model_category_dir = osp.join(model_dir, category)
    if not osp.isdir(model_category_dir): os.makedirs(model_category_dir)

    log.write(' ---------------------------------------------------------------------------------\n')
    log.write(' ----------------------------------%s\n' % (split))
    log.write(' ---------------------------------------------------------------------------------\n')
    log_category.write(' ---------------------------------------------------------------------------------\n')
    log_category.write(' ----------------------------------%s\n' % (split))
    log_category.write(' ---------------------------------------------------------------------------------\n')

    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)

    # move to GPU
    if opt['gpuid'] >= 0:
        model.cuda()
        criterion.cuda()

    # start training
    data_time, model_time = 0, 0
    buffer_data_time, buffer_model_time = 0, 0
    lr = opt['learning_rate']
    best_predictions = None
    while True:
        # run one iteration
        loss, T, wrapped = lossFun(loader, current_buffer, last_buffer, split, optimizer, model, criterion,
                                   att_crit, opt, iter, rel_buffer=rel_buffer, loc_buffer=loc_buffer,
                                   rel_last_buffer=rel_last_buffer, loc_last_buffer=loc_last_buffer)
        data_time += T['data']
        model_time += T['model']

        # update iter and epoch
        iter += 1
        if wrapped:
            epoch += 1
        """
        # update buffer inside
        if epoch >= opt['buffer_start_epoch']:
            T = update_buffer_loss(loader, current_buffer, model, criterion, att_crit, opt, iter,
                                   rel_buffer=rel_buffer, loc_buffer=loc_buffer)
            buffer_data_time += T['data']
            buffer_model_time += T['model']
        """

        # write the training loss summary
        if iter % opt['losses_log_every'] == 0:
            loss_history[iter] = loss
            # print stats
            log_toc = time.time()

            if epoch >= opt['buffer_start_epoch']:
                print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter,\
                      buffer_data:%.2fs/iter, buffer_model:%.2fs/iter' \
                      % (iter, epoch, loss, lr, data_time/opt['losses_log_every'],
                         model_time/opt['losses_log_every'], buffer_data_time/opt['losses_log_every'],
                         buffer_model_time/opt['losses_log_every']))
                data_time, model_time = 0, 0
                buffer_data_time, buffer_model_time = 0, 0
            else:
                print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
                      % (iter, epoch, loss, lr, data_time/opt['losses_log_every'],
                         model_time/opt['losses_log_every']))
                data_time, model_time = 0, 0

        # decay the learning rates
        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor =  0.1 ** frac
            lr = opt['learning_rate'] * decay_factor
            # update optimizer's learning rate
            model_utils.set_lr(optimizer, lr)

        # eval loss and save checkpoint
        # if iter % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:
        # if iter % opt['save_checkpoint_every'] == 0 or epoch == opt['max_category_epoch']:
        if wrapped:
        # if iter == 1:
            # val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, None, 'val', opt)
            if 'flickr' in opt['dataset_splitBy']:
                val_loss, val_category_loss, acc, val_category_acc, predictions =\
                        eval_utils.eval_split(loader, model, None, 'test', opt)
                print('[%s] loss=%.2f%% acc=%.2f%%' % (category, val_category_loss[category],
                                                   val_category_acc[category]))
            else:
                val_loss, val_category_loss, val_supercategory_loss, acc, category_acc, supercategory_acc,\
                        category_loss_evals, supercategory_loss_evals, predictions =\
                        eval_utils.eval_split(loader, model, None, 'val', opt)

            log.write('%d/%d: %.6f, %.2f%%\n' % (iter, epoch, val_supercategory_loss[category], supercategory_acc[category]))
            log_category.write('%d/%d: %.6f, %.2f%%\n' % (iter, epoch, val_loss, acc*100))
            for supercategory in supercategory_loss_evals.keys():
                log_category.write('----%s[%d], loss=%.6f, acc= %.2f%%\n' % (supercategory,
                                                                             int(supercategory_loss_evals[supercategory]),
                                                                             val_supercategory_loss[supercategory],
                                                                             supercategory_acc[supercategory]))
                """
                subcategory = category_structure[supercategory]
                for sub_i in subcategory:
                    category = sub_i['name']
                    log_category.write('+ + + + %s[%d], loss= %.6f, acc= %.2f%%\n'%(category,
                                                                                    int(category_loss_evals[category]),
                                                                                    val_category_loss[category],
                                                                                    category_acc[category]))
                """

            """
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
            val_accuracies += [(iter, acc)]
            print('validation loss: %.2f' % val_loss)
            print('validation acc : %.2f%%\n' % (acc*100.0))
            """
            val_loss_history[iter] = val_supercategory_loss[category]
            val_result_history[iter] = {'loss': val_supercategory_loss[category], 'accuracy': supercategory_acc[category]}
            val_accuracies += [(iter, supercategory_acc[category])]
            print('validation loss: %.2f' % (val_supercategory_loss[category]))
            print('validation acc : %.2f%%\n' % (supercategory_acc[category]))
            # print('validation precision : %.2f%%' % (overall['precision']*100.0))
            # print('validation recall    : %.2f%%' % (overall['recall']*100.0))
            # print('validation f1        : %.2f%%' % (overall['f1']*100.0))

            # save model if best
            current_score = supercategory_acc[category]
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_predictions = predictions
                best_model_name = osp.join(model_category_dir, opt['id'] + '_best_model.pth.tar')
                best_file_name = osp.join(model_category_dir, opt['id'] + '.pth.tar')
                torch.save(model, best_model_name)
                save_checkpoint({
                    'epoch':epoch,
                    'model':model,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'opt':opt,
                }, best_file_name)
                checkpoint = {}
                checkpoint['model'] = model
                """
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)
                """

            # write json report
            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_predictions'] = predictions if best_predictions is None else best_predictions
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            infos['word_to_ix'] = loader.word_to_ix
            with open(osp.join(model_category_dir, opt['id']+'.json'), 'w') as io:
                json.dump(infos, io)

        # if iter >= opt['max_iters'] and opt['max_iters'] > 0:
        if epoch >= opt['max_category_epoch'] and opt['max_category_epoch'] > 0:
            break

    return checkpoint['model']


def compute_importance_l2(criterion, att_crit, model, optimizer, loader, split, opt):
    """
    Mimic the depoloyment setup where the model is applied on some samples and those are used to update the
        importance params
    Uses the L2norm of the function output. This is what we MAS uses as default
    """
    # set model
    if opt['gpuid'] >= 0:
        model.cuda()
        criterion.cuda()
        att_crit.cuda()
    model.train()

    # initialize
    # split  = split + '_img'
    loader.resetIterator(split)
    visual_rank = opt['visual_rank_weight'] > 0
    lang_rank = opt['lang_rank_weight'] > 0
    prev_size = 0
    cur_size = 0

    while True:

        # zero the parameter gradients
        optimizer.zero_grad()

        data = loader.getCategoryBatch(split, opt)
        Feats = data['Feats']
        labels = data['labels']

        wrapped = data['bounds']['wrapped']

        if opt['visual_rank_weight'] > 0:
            Feats = loader.combine_feats(Feats, data['neg_Feats'])
            labels = torch.cat([labels, data['labels']])
        if opt['lang_rank_weight'] > 0:
            Feats = loader.combine_feats(Feats, data['Feats'])
            labels = torch.cat([labels, data['neg_labels']])

        att_labels, select_ixs = data['att_labels'], data['select_ixs']


        scores, _, sub_attn, loc_attn, rel_attn, _, _, att_scores = \
                model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                      Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
        """
        scores, _, = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], labels)
        """
        # att_scores = F.sigmoid(att_scores)

        N = scores.size(0)
        batch_size = 0
        if visual_rank and not lang_rank:
            batch_size = N//2
        elif not visual_rank and lang_rank:
            batch_size = N//2
        elif visual_rank and lang_rank:
            batch_size = N//3

        cur_size = cur_size + batch_size

        # compute the L2 norm of output
        Target_zeros = torch.zeros(scores.size())
        Target_zeros = Variable(Target_zeros.cuda())
        # note no averaging is happening here
        loss = torch.nn.MSELoss(size_average=False)

        # compute the L2 norm of attribute
        Target_attr_zeros = torch.zeros(att_scores.size())
        Target_attr_zeros = Variable(Target_attr_zeros.cuda())
        # note no averaging is happening here
        attr_loss = torch.nn.MSELoss(size_average=False)

        targets = loss(scores, Target_zeros)
        if select_ixs.numel() > 0:
            targets += opt['att_weight'] * attr_loss(att_scores.index_select(0, select_ixs),
                                                     Target_attr_zeros.index_select(0, select_ixs))
        """
        targets += opt['att_weight'] * attr_loss(att_scores, Target_attr_zeros)
        """
        # compute the gradients
        targets.backward()

        # update the parameters importance
        optimizer.step(model.reg_params, prev_size, cur_size)
        # necesscary size to keep the running average
        prev_size = cur_size

        if wrapped:
            break

    return model


def compute_importance(criterion, att_crit, model, optimizer, loader, split, opt):
    """
    Mimic the depoloyment setup where the model is applied on some samples and those are used to update the
        importance params
    Uses the L1norm of the function output.
    """
    # set model
    if opt['gpuid'] >= 0:
        model.cuda()
        criterion.cuda()
        att_crit.cuda()
    model.train()

    # initialize
    # split  = split + '_img'
    loader.resetIterator(split)
    vis_rank = opt['visual_rank_weight'] > 0
    lang_rank = opt['lang_rank_weight'] > 0
    prev_size = 0
    cur_size = 0

    while True:

        # zero the parameter gradients
        optimizer.zero_grad()

        data = loader.getCategoryBatch(split, opt)
        Feats = data['Feats']
        labels = data['labels']

        wrapped = data['bounds']['wrapped']

        if opt['visual_rank_weight'] > 0:
            Feats = loader.combine_feats(Feats, data['neg_Feats'])
            labels = torch.cat([labels, data['labels']])
        if opt['lang_rank_weight'] > 0:
            Feats = loader.combine_feats(Feats, data['Feats'])
            labels = torch.cat([labels, data['neg_labels']])

        att_labels, select_ixs = data['att_labels'], data['select_ixs']

        scores, _, sub_attn, loc_attn, rel_attn, _, _, att_scores = \
                model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                      Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
        """
        scores, _, = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], labels)
        """
        # att_scores = F.sigmoid(att_scores)

        N = scores.size(0)
        batch_size = 0
        if visual_rank and not lang_rank:
            batch_size = N//2
        elif not visual_rank and lang_rank:
            batch_size = N//2
        elif visual_rank and lang_rank:
            batch_size = N//3

        cur_size = cur_size + batch_size

        # compute the L1 norm of output
        Target_zeros = torch.zeros(scores.size())
        Target_zeros = Variable(Target_zeros.cuda(), requires_grad=False)
        # note no averaging is happening here
        loss = torch.nn.L1Loss(size_average=False)

        # compute the L2 norm of attribute
        Target_attr_zeros = torch.zeros(att_scores.size())
        Target_attr_zeros = Variable(Target_attr_zeros.cuda())
        # note no averaging is happening here
        attr_loss = torch.nn.L1Loss(size_average=False)

        targets = loss(scores, Target_zeros)
        if select_ixs.numel() > 0:
            targets += opt['att_weight'] * attr_loss(att_scores.index_select(0, select_ixs),
                                                     Target_attr_zeros.index_select(0, select_ixs))
        """
        targets += opt['att_weight'] * attr_loss(att_scores, Target_attr_zeros)
        """
        # compute the gradients
        targets.backward()

        # update the parameters importance
        optimizer.step(model.reg_params, prev_size, cur_size)
        # necesscary size to keep the running average
        prev_size = cur_size

        if wrapped:
            break

    return model


def compute_importance_gradient_vector(criterion, att_crit, model, optimizer, loader, split, opt):
    """
    Mimic the depoloyment setup where the model is applied on some samples and those are used to update the
        importance params
    Uses the gradient of the origin loss(have label).
    """
    # set model
    if opt['gpuid'] >= 0:
        model.cuda()
        criterion.cuda()
        att_crit.cuda()
    model.train()

    # initialize
    # split  = split + '_img'
    loader.resetIterator(split)
    vis_rank = opt['visual_rank_weight'] > 0
    lang_rank = opt['lang_rank_weight'] > 0
    prev_size = 0
    cur_size = 0

    att_crit = nn.BCEWithLogitsLoss(loader.get_attribute_weights(), size_average=False)

    while True:

        # zero the parameter gradients
        optimizer.zero_grad()

        data = loader.getCategoryBatch(split, opt)
        Feats = data['Feats']
        labels = data['labels']

        wrapped = data['bounds']['wrapped']

        if opt['visual_rank_weight'] > 0:
            Feats = loader.combine_feats(Feats, data['neg_Feats'])
            labels = torch.cat([labels, data['labels']])
        if opt['lang_rank_weight'] > 0:
            Feats = loader.combine_feats(Feats, data['Feats'])
            labels = torch.cat([labels, data['neg_labels']])

        att_labels, select_ixs = data['att_labels'], data['select_ixs']

        scores, _, sub_attn, loc_attn, rel_attn, _, _, att_scores = \
                model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                      Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
        """
        scores, _, = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], labels)
        """
        # att_scores = F.sigmoid(att_scores)

        N = scores.size(0)
        batch_size = 0
        if visual_rank and not lang_rank:
            batch_size = N//2
        elif not visual_rank and lang_rank:
            batch_size = N//2
        elif visual_rank and lang_rank:
            batch_size = N//3

        cur_size = cur_size + batch_size

        # compute the mm_crit of output
        targets = criterion(scores) * batch_size
        if select_ixs.numel() > 0:
            targets += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                                    Target_attr_zeros.index_select(0, select_ixs))

        # compute the gradients
        targets.backward()

        # update the parameters importance
        optimizer.step(model.reg_params, prev_size, cur_size)
        # necesscary size to keep the running average
        prev_size = cur_size

        if wrapped:
            break

    return model


def initialize_reg_params(model, freeze_layers=[]):
    """initialize an omega for each parameter to zero"""

    reg_params = {}
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            print ('initializing param', name)
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            # initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param

    return reg_params


def initialize_store_reg_params(model, freeze_layers=[]):
    """
    set omega to zero but after storing its value in a temp omega in which later we can accumulate them both
    """

    reg_params = model.reg_params
    for name, param in model.named_parameters():
        # in case there some layers that are not trained
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print ('storing previous omega', name)
                prev_omega = reg_param.get('omega')
                new_omega = torch.FloatTensor(param.size()).zero_()
                init_val = param.data.clone()
                reg_param['prev_omega'] = prev_omega
                reg_param['omega'] = new_omega

                # initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param] = reg_param

        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print ('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]

    return reg_params


def accumulate_reg_params(model, freeze_layers=[]):
    """accumulate the newly computed omega with the previously stored one from the old previous tasks"""

    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print ('restoring previous omega', name)
                prev_omega = reg_param.get('prev_omega')
                prev_omega = prev_omega.cuda()

                new_omega = (reg_param.get('omega')).cuda()
                acc_omega = torch.add(prev_omega, new_omega)

                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega

                reg_params[param] = reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print ('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]

    return reg_params


def weighted_module_reg_params(model, opt, sub_layers=[], loc_layers=[], rel_layers=[], freeze_layers=[]):
    """weighted reg_param by module"""

    reg_params = model.reg_params
    sub_param, loc_param, rel_param = 0, 0 ,0
    sub_num, loc_num, rel_num = 0, 0, 0
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if name in sub_layers:
                if param in reg_params:
                    reg_param = reg_params.get(param)
                    print ('storing sub omega', name)
                    omega = reg_param.get('omega')
                    sub_param += omega.sum()
                    sub_num += omega.numel()

            if name in loc_layers:
                if param in reg_params:
                    reg_param = reg_params.get(param)
                    print ('storing loc omega', name)
                    omega = reg_param.get('omega')
                    loc_param += omega.sum()
                    loc_num += omega.numel()

            if name in rel_layers:
                if param in reg_params:
                    reg_param = reg_params.get(param)
                    print ('storing rel omega', name)
                    omega = reg_param.get('omega')
                    rel_param += omega.sum()
                    rel_num += omega.numel()

    print('sub_num ', sub_num)
    print('loc_num ', loc_num)
    print('rel_num ', rel_num)
    if opt['module_sum'] > 0:
        ############################
        if opt['module_normalize'] > 0:
            # 1 sum normalize
            sum_param = sub_param + loc_param + rel_param
            sub_param_weight = sub_param / sum_param
            loc_param_weight = loc_param / sum_param
            rel_param_weight = rel_param / sum_param
        else:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++')
            # 2 sum
            sub_param_weight = sub_param
            loc_param_weight = loc_param
            rel_param_weight = rel_param
        ############################

    else:
        sub_param = sub_param / sub_num
        loc_param = loc_param / loc_num
        rel_param = rel_param / rel_num

        ############################
        if opt['module_normalize'] > 0:
            # 3 mean normalize
            sum_param = sub_param + loc_param + rel_param
            sub_param_weight = sub_param / sum_param
            loc_param_weight = loc_param / sum_param
            rel_param_weight = rel_param / sum_param
        else:
            # 4 mean
            sub_param_weight = sub_param
            loc_param_weight = loc_param
            rel_param_weight = rel_param
        ############################

    print ('sub_param_weight ', sub_param_weight)
    print ('loc_param_weight ', loc_param_weight)
    print ('rel_param_weight ', rel_param_weight)

    for name, param in model.named_parameters():
        if name not in freeze_layers:
            if opt['sub_module'] > 0:
                if name in sub_layers:
                    if param in reg_params:
                        reg_param = reg_params.get(param)
                        print ('weighted sub omega', name)
                        omega = (reg_param.get('omega')).cuda()

                        w_omega = omega * sub_param_weight
                        reg_param['omega'] = w_omega
                        reg_params[param] = reg_param

            if name in loc_layers:
                if param in reg_params:
                    reg_param = reg_params.get(param)
                    print ('weighted loc omega', name)
                    omega = (reg_param.get('omega')).cuda()

                    w_omega = omega * loc_param_weight
                    reg_param['omega'] = w_omega
                    reg_params[param] = reg_param

            if name in rel_layers:
                if param in reg_params:
                    reg_param = reg_params.get(param)
                    print ('weighted rel omega', name)
                    omega = (reg_param.get('omega')).cuda()

                    w_omega = omega * rel_param_weight
                    reg_param['omega'] = w_omega
                    reg_params[param] = reg_param

    return reg_params


def weighted_decay_accumulate_reg_params(model, previous_ratio, freeze_layers=[]):
    """accumulate the newly computed omega with the previously stored one from the old previous tasks"""

    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print ('restoring previous omega', name)
                prev_omega = reg_param.get('prev_omega')
                prev_omega = prev_omega.cuda()

                new_omega = (reg_param.get('omega')).cuda()

                # weighted decay
                prev_omega = prev_omega.mul_(previous_ratio)
                new_omega = new_omega.mul_(1 - previous_ratio)

                # accu
                acc_omega = torch.add(prev_omega, new_omega)

                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega

                reg_params[param] = reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print ('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]

    return reg_params


def save_checkpoint(state, filename=None):
  torch.save(state, filename)
