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
from loaders.gt_mrcn_loader import GtMRCNLoader
from loaders.gt_mrcn_flickr_loader import GtMRCNFlickrLoader
from layers.match import JointMatching
import models.utils as model_utils
import models.eval_det_cross_utils as eval_utils
# from crits.max_margin_crit import MaxMarginCriterion
from crits.cross_entropy_crits import CrossEntropyCriterion
from opt import parse_opt

# torch
import torch
import torch.nn as nn
from torch.autograd import Variable

# train one iter
def lossFun(loader, optimizer, model, ce_crit, opt, iter):
  # set mode
  model.train()

  # zero gradient
  optimizer.zero_grad()

  # time
  T = {}

  # load one batch of data
  tic = time.time()
  data = loader.getCrossBatch('train_ref', opt)
  Feats = data['Feats']
  labels = data['labels']

  bbx_labels = data['bbx_labels']
  kld_labels = data['kld_labels']
  bbx_reg = data['bbx_reg']
  reg_inweight = data['reg_inweight']
  reg_all = data['reg_all']
  T['data'] = time.time()-tic

  # forward
  tic = time.time()
  scores, sub_grid_attn = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], labels, True)
  # print (scores)
  loss, loss_vec = ce_crit(scores, kld_labels, bbx_reg, True, reg_inweight, reg_all)

  # if iter < 500:
  #   num_pos = len(data['ref_ids'])
  #   loss += 0.1*model.sub_rel_kl(sub_attn[:num_pos], rel_attn[:num_pos], labels[:num_pos])

  loss.backward()
  model_utils.clip_gradient(optimizer, opt['grad_clip'])
  optimizer.step()
  T['model'] = time.time()-tic

  scores = scores.data.cpu().numpy()
  cur_accu, _, _ = eval_utils.eval_cur_batch(bbx_labels, scores, True)

  return loss.data[0], cur_accu, T, data['bounds']['wrapped']


def main(args):

  opt = vars(args)

  # initialize
  if opt['dataset'] == 'flickr30k':
    opt['dataset_splitBy'] = opt['dataset']
  else:
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
  checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'])
  if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)
  log_path = osp.join(opt['log_path'], opt['id']+'.log')
  log_category_path = osp.join(opt['log_path'], opt['id']+'_category.log')
  log = open(log_path, 'w', 0)
  log_category = open(log_category_path, 'w', 0)

  # set random seed
  torch.manual_seed(opt['seed'])
  random.seed(opt['seed'])

  # set up loader
  data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
  data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.h5')
  det_feats_dir = osp.join('cache/feats', opt['dataset_splitBy'], 'det_feats/mask-rcnn-res101')
  if 'flickr' in opt['dataset_splitBy']:
    data_split = osp.join('cache/prepro', opt['dataset_splitBy'], 'data_split.json')
    loader = GtMRCNFlickrLoader(data_h5=data_h5, data_json=data_json, data_split=data_split,
                                det_feats_dir=det_feats_dir)
  else:
    loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)
  # prepare feats
  feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
  head_feats_dir=osp.join('cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)
  loader.prepare_mrcn(head_feats_dir, args)
  # ann_feats = osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn',
  #                      '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
  # loader.loadFeats({'ann': ann_feats})

  # set up model
  opt['vocab_size']= loader.vocab_size
  opt['fc7_dim']   = loader.fc7_dim
  opt['pool5_dim'] = loader.pool5_dim
  # opt['num_atts']  = loader.num_atts
  model = JointMatching(opt)

  # resume from previous checkpoint
  infos = {}
  if opt['start_from'] is not None:
    pass
  iter = infos.get('iter', 0)
  epoch = infos.get('epoch', 0)
  val_accuracies = infos.get('val_accuracies', [])
  loss_history = infos.get('loss_history', {})
  loader.iterators = infos.get('iterators', loader.iterators)
  cat_to_ix = loader.cat_to_ix

  if opt['load_best_score'] == 1:
    best_val_score = infos.get('best_val_score', None)

  # set up criterion
  ce_crit = CrossEntropyCriterion(opt['reg_lambda'])

  # move to GPU
  if opt['gpuid'] >= 0:
    model.cuda()
    ce_crit.cuda()
    # att_crit.cuda()

  # set up optimizer
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=opt['learning_rate'],
                               betas=(opt['optim_alpha'], opt['optim_beta']),
                               eps=opt['optim_epsilon'])

  # start training
  data_time, model_time = 0, 0
  lr = opt['learning_rate']
  best_predictions, best_overall = None, None
  while True:
    # run one iteration
    loss, cur_acc, T, wrapped = lossFun(loader, optimizer, model, ce_crit, opt, iter)
    data_time += T['data']
    model_time += T['model']

    # write the training loss summary
    if iter % opt['losses_log_every'] == 0:
      loss = loss.item()
      loss_history[iter] = loss
      # print stats
      log_toc = time.time()
      print('iter[%s](epoch[%s]), train_loss=%.3f, acc=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' %
            (iter, epoch, loss, cur_acc, lr, data_time/opt['losses_log_every'],
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
    if iter % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:
      # val_loss, acc, predictions, overall = eval_utils.eval_split(loader, model, None, 'val', opt)
      """
      if 'flickr' in opt['dataset_splitBy']:
        val_loss, val_category_loss, acc, val_category_acc, predictions = eval_utils.eval_split(loader, model, None, 'test', opt)
      else:
        val_loss, _, acc, _, predictions = eval_utils.eval_split(loader, model, None, 'val', opt)
      """
      # ------------------------- preparing -------------------------------
      acc, category_acc, num_category_cor, num_category_cnt, predictions = \
              eval_utils.eval_split(loader, model, None, 'test', opt)
      log.write('%d/%d: %.4f, %.2f%%\n' % (iter+1, epoch, loss, acc*100))
      log_category.write('%d/%d: %.4f, %.2f%%\n' % (iter+1, epoch, loss, acc*100))
      for category, category_id in cat_to_ix.items():
        log_category.write('----%s:%.2f%%, %.2f/%.2f\n' % (category, category_acc[category_id]*100,
                                                           num_category_cor[category_id], num_category_cnt[category_id]))
      # ------------------------- preparing -------------------------------
      val_accuracies += [(iter, acc)]
      # print('validation precision : %.2f%%' % (overall['precision']*100.0))
      # print('validation recall    : %.2f%%' % (overall['recall']*100.0))
      # print('validation f1        : %.2f%%' % (overall['f1']*100.0))

      # save model if best
      current_score = acc
      if best_val_score is None or current_score > best_val_score:
        best_val_score = current_score
        best_predictions = predictions
        # best_overall = overall
        checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
        checkpoint = {}
        checkpoint['model'] = model
        checkpoint['opt'] = opt
        torch.save(checkpoint, checkpoint_path)
        print('model saved to %s' % checkpoint_path)

      # write json report
      infos['iter'] = iter
      infos['epoch'] = epoch
      infos['iterators'] = loader.iterators
      infos['loss_history'] = loss_history
      infos['val_accuracies'] = val_accuracies
      infos['best_val_score'] = best_val_score
      infos['best_predictions'] = predictions if best_predictions is None else best_predictions
      # infos['best_overall'] = overall if best_overall is None else best_overall
      infos['opt'] = opt
      with open(osp.join(checkpoint_dir, opt['id']+'.json'), 'wb') as io:
        json.dump(infos, io)

    # update iter and epoch
    iter += 1
    if wrapped:
      epoch += 1
    if iter >= opt['max_iters'] and opt['max_iters'] > 0:
      break


if __name__ == '__main__':

  args = parse_opt()
  main(args)

