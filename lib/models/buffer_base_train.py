from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint

# model
import _init_paths
import models.utils as model_utils
import models.eval_easy_utils as eval_utils
from crits.max_margin_crit import MaxMarginCriterion
from opt import parse_opt

# torch
import torch
import torch.nn as nn
from torch.autograd import Variable

def save_checkpoint(state, filename=None):
  torch.save(state, filename)

# train one iter
def lossFun(loader, current_buffer, split, optimizer, model, mm_crit, att_crit, opt, iter, last_buffer=None):

  has_buffer = last_buffer is not None
  # set mode
  model.train()
  # zero gradient
  optimizer.zero_grad()
  # time
  T = {}
  # load one batch of data
  tic = time.time()

  if has_buffer:
    data = loader.getCategoryBatch(split, opt, False)
    ref_ids = data['ref_ids']
    ref_sent_ids = data['ref_sent_ids']
    Feats = data['Feats']
    labels = data['labels']
    neg_labels = data['neg_labels']

    buffer_data = loader.getBufferBatch(last_buffer.transform_loader_list(opt['buffer_sample_number']),
                                        opt, False)
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

  else:
    data = loader.getCategoryBatch(split, opt)
    ref_ids = data['ref_ids']
    ref_sent_ids = data['ref_sent_ids']
    Feats = data['Feats']
    labels = data['labels']
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

  loss, loss_batch = mm_crit(scores)
  if select_ixs.numel() > 0:
    loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                         att_labels.index_select(0, select_ixs))

  loss.backward()
  model_utils.clip_gradient(optimizer, opt['grad_clip'])
  optimizer.step()
  T['model'] = time.time()-tic

  return loss.item(), T, data['bounds']['wrapped']


def update_buffer_loss(loader, current_buffer, model, mm_crit, att_crit, opt, iter):
  # set mode
  model.eval()

  # time
  T = {}

  # load one batch of data
  tic = time.time()
  data = loader.getBufferBatch(current_buffer.transform_loader_list(), opt)
  ref_sent_ids = data['ref_sent_ids']
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

  # print (scores)
  loss, loss_batch = mm_crit(scores)
  if select_ixs.numel() > 0:
    loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                         att_labels.index_select(0, select_ixs))

  T['model'] = time.time()-tic

  if opt['subject_flag'] > 0:
    batch_size = loss_batch.size()[0]
    sub_weights = weights[:, 0][:batch_size]
    loss_batch = loss_batch * sub_weights

  # update
  loss_batch = loss_batch.cpu().detach().tolist()
  current_buffer.update_loss(ref_sent_ids, loss_batch)

  return T


def train_model(model, criterion, att_crit, loader, split, opt, model_dir, category, log, log_category,
                current_buffer, last_buffer=None):

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

  # set up optimizer
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=opt['learning_rate'],
                               betas=(opt['optim_alpha'], opt['optim_beta']),
                               eps=opt['optim_epsilon'])
  # move to GPU
  if opt['gpuid'] >= 0:
    model.cuda()
    criterion.cuda()
    att_crit.cuda()

  # start training
  data_time, model_time = 0, 0
  buffer_data_time, buffer_model_time = 0, 0
  lr = opt['learning_rate']
  best_predictions = None
  while True:
    # run one iteration
    loss, T, wrapped = lossFun(loader, current_buffer, split, optimizer, model, criterion, att_crit, opt,
                               iter, last_buffer)
    data_time += T['data']
    model_time += T['model']
    # current_buffer._print()

    # update iter and epoch
    iter += 1
    if wrapped:
      epoch += 1

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
      # val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, None, 'val', opt)
      val_loss, val_category_loss, val_supercategory_loss, acc, category_acc, supercategory_acc,\
              category_loss_evals, supercategory_loss_evals, predictions =\
              eval_utils.eval_split(loader, model, None, 'val', opt)

      log.write('%d/%d: %.6f, %.2f%%\n' % (iter, epoch, val_supercategory_loss[category], supercategory_acc[category]))
      log_category.write('%d/%d: %.6f, %.2f%%\n' % (iter, epoch, loss, acc*100))
      for supercategory in supercategory_loss_evals.keys():
        log_category.write('----%s[%d], loss=%.6f, acc= %.2f%%\n' % (supercategory,
                                                                     int(supercategory_loss_evals[supercategory]),
                                                                     val_supercategory_loss[supercategory],
                                                                     supercategory_acc[supercategory]))

      val_loss_history[iter] = val_supercategory_loss[category]
      val_result_history[iter] = {'loss': val_supercategory_loss[category], 'accuracy': supercategory_acc[category]}
      val_accuracies += [(iter, supercategory_acc[category])]
      print('validation loss: %.2f' % (val_supercategory_loss[category]))
      print('validation acc : %.2f%%\n' % (supercategory_acc[category]))

      # save model if best
      # current_score = acc
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
          'model_state_dict':model.state_dict(),
          'optimizer_state_dict':optimizer.state_dict(),
          'opt':opt,
          }, best_file_name)

        checkpoint = {}
        checkpoint['model'] = model
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

    if epoch >= opt['max_category_epoch'] and opt['max_category_epoch'] > 0:
      break

  return checkpoint['model']

def load_first_model(model, criterion, att_crit, loader, split, opt, model_dir, f_model_dir, category,
                     log, log_category, current_buffer, rel_buffer=None, loc_buffer=None, first_flag=True):

    if first_flag:
        previous_model_path = osp.join(f_model_dir, category, opt['f_id'] + '_best_model.pth.tar')
    else:
        previous_model_path = osp.join(f_model_dir, category, opt['id']+ '_best_model.pth.tar')
    model = torch.load(previous_model_path)

    model_category_dir = osp.join(model_dir, category)
    if not osp.isdir(model_category_dir): os.makedirs(model_category_dir)

    # set model
    if opt['gpuid'] >= 0:
      model.cuda()
      criterion.cuda()
      att_crit.cuda()
    model.eval()

    # initialize
    loader.resetIterator(split)

    ii = 0
    while True:
        data = loader.getCategoryBatch(split, opt)
        ref_ids = data['ref_ids']
        ref_sent_ids = data['ref_sent_ids']

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

        scores, _, sub_attn, loc_attn, rel_attn, _, weights, att_scores = \
                model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                      Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)

        # print (scores)
        loss, loss_batch = criterion(scores)
        if select_ixs.numel() > 0:
            loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                                 att_labels.index_select(0, select_ixs))

        # update L, R
        if opt['multi_buffer'] > 0:
            batch_size = loss_batch.size()[0]
            loc_weights = weights[:, 1][:batch_size]
            rel_weights = weights[:, 2][:batch_size]
            loc_loss_batch = loss_batch * loc_weights
            rel_loss_batch = loss_batch * rel_weights

            loc_loss_batch = loc_loss_batch.cpu().detach().tolist()
            loc_buffer.update(ref_ids, ref_sent_ids, loc_loss_batch)
            rel_loss_batch = rel_loss_batch.cpu().detach().tolist()
            rel_buffer.update(ref_ids, ref_sent_ids, rel_loss_batch)

        # update
        batch_size = loss_batch.size()[0]
        sub_weights = weights[:, 0][:batch_size].cpu().detach().numpy().tolist()
        if opt['subject_flag'] > 0:
            loss_batch = loss_batch * weights[:, 0][:batch_size]

        loss_batch = loss_batch.cpu().detach().tolist()
        current_buffer.update(ref_ids, ref_sent_ids, loss_batch, sub_weights)

        if wrapped:
            break

    val_loss, val_category_loss, val_supercategory_loss, acc, category_acc, supercategory_acc,\
            category_loss_evals, supercategory_loss_evals, predictions =\
            eval_utils.eval_split(loader, model, None, 'val', opt)

    log.write('pretrain_model: %.6f, %.2f%%\n' % (val_supercategory_loss[category],
                                                  supercategory_acc[category]))
    log_category.write('pretrain_model: %.6f, %.2f%%\n' % (loss, acc*100))
    for supercategory in supercategory_loss_evals.keys():
        log_category.write('----%s[%d], loss=%.6f, acc= %.2f%%\n' % (supercategory,
                                                                     int(supercategory_loss_evals[supercategory]),
                                                                     val_supercategory_loss[supercategory],
                                                                     supercategory_acc[supercategory]))
    return model
