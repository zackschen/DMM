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
def lossFun(loader, split, optimizer, model, mm_crit, att_crit, opt, iter, category_i):
  # set mode
  model.train()

  # zero gradient
  optimizer.zero_grad()

  # time
  T = {}

  # load one batch of data
  tic = time.time()
  data = loader.getCategoryBatch(split, opt)
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
  scores, _, sub_attn, loc_attn, rel_attn, _, _, att_scores = \
          model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                Feats['cxt_fc7'], Feats['cxt_lfeats'], labels)
  # print (scores)
  loss = mm_crit(scores)
  if select_ixs.numel() > 0:
    loss += opt['att_weight'] * att_crit(att_scores.index_select(0, select_ixs),
                                         att_labels.index_select(0, select_ixs))
  
  if category_i > 0:
    ewc_reg = model.regularizer() * model.ewc_lambda
    loss += ewc_reg
    
  loss.backward()
  model_utils.clip_gradient(optimizer, opt['grad_clip'])
  optimizer.step()
  T['model'] = time.time()-tic

  # return
  return loss.item(), T, data['bounds']['wrapped'],len(labels)


def train_model(model, criterion, att_crit, loader, split, opt, model_dir, category_i, category, log, log_category):

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
  lr = opt['learning_rate']
  best_predictions = None
  batch_count = 0
  while True:
    # run one iteration
    loss, T, wrapped,length = lossFun(loader, split, optimizer, model, criterion, att_crit, opt, iter, category_i)
    batch_count += length
    data_time += T['data']
    model_time += T['model']

    # update iter and epoch
    iter += 1
    if wrapped:
      epoch += 1
      batch_count = 0

    # write the training loss summary
    if iter % opt['losses_log_every'] == 0:
      loss_history[iter] = loss
      # print stats
      log_toc = time.time()
      print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
            % (iter, epoch, loss, lr, data_time/opt['losses_log_every'], model_time/opt['losses_log_every']))
      data_time, model_time = 0, 0
    # print('batch count : %s' % batch_count)
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
      infos['att_to_ix'] = loader.att_to_ix
      with open(osp.join(model_category_dir, opt['id']+'.json'), 'w') as io:
        json.dump(infos, io)

    # if iter >= opt['max_iters'] and opt['max_iters'] > 0:
    if epoch >= opt['max_category_epoch'] and opt['max_category_epoch'] > 0:
      break

  return checkpoint['model']
