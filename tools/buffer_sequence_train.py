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
from layers.joint_match import JointMatching
import models.utils as model_utils
import models.eval_easy_utils as eval_utils
# import models.finetune_base_train as base_train
import models.buffer_base_train as base_train
from crits.max_margin_crit import MaxMarginCriterionBatch
from opt import parse_opt

# torch
import torch
import torch.nn as nn
from torch.autograd import Variable

CATEGORY_TO_ID_5task = {0:'person', 1:'kitchen', 2:'animal', 3:'indoor', 4:'outdoor'}

CATEGORY_TO_ID_10task = {0: 'food', 1: 'indoor', 2: 'sports', 3: 'person',4: 'animal',\
  5: 'vehicle',6: 'furniture', 7: 'accessory', 8: 'electronic', 9: 'kitchen'}

def main(args):

  opt = vars(args)

  isDebug = True if sys.gettrace() else False

  # initialize
  opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
  checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['id'])
  if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

  if opt['buffer_type'] == "low":
    from buffer.replay_buffer_low import Buffer,BufferPool
  elif opt['buffer_type'] == "high":
    from buffer.replay_buffer import Buffer,BufferPool
  else:
    from buffer.replay_buffer_random import Buffer,BufferPool
  # first model_dir
  f_model_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['f_id'])
  pre_model_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['id'])

  log_path = osp.join(opt['log_path'], opt['dataset'], opt['id'] + '.log')
  log_category_path = osp.join(opt['log_path'], opt['dataset'], opt['id'] + '_category.log')
  log = open(log_path, 'a', buffering=1)
  log_category = open(log_category_path, 'a', buffering=1)

  # set random seed
  torch.manual_seed(opt['seed'])
  random.seed(opt['seed'])

  # set up loader
  if opt["task"] == 5:
    data_json = osp.join('cache/prepro', opt['dataset_splitBy'], '5task.json')
    data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], '5task.h5')
  else:
    data_json = osp.join('cache/prepro', opt['dataset_splitBy'], '10task.json')
    data_h5 = osp.join('cache/prepro', opt['dataset_splitBy'], '10task.h5')
  loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json, opt = opt)
  # prepare feats
  feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
  head_feats_dir=osp.join('cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)
  loader.prepare_mrcn(head_feats_dir, args)
  ann_feats = osp.join('cache/feats', opt['dataset_splitBy'], 'mrcn',
                       '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
  loader.loadFeats({'ann': ann_feats})

  # set up model
  opt['vocab_size']= loader.vocab_size
  opt['fc7_dim']   = loader.fc7_dim
  opt['pool5_dim'] = loader.pool5_dim
  opt['num_atts']  = loader.num_atts
  model = JointMatching(opt)

  # set up criterion
  att_crit = nn.BCEWithLogitsLoss(loader.get_attribute_weights())

  # set buffer pool
  buffer_pool = BufferPool(opt['buffer_size'])
  CATEGORY_TO_ID = CATEGORY_TO_ID_5task if opt["task"] == 5 else CATEGORY_TO_ID_10task
  num_category = len(CATEGORY_TO_ID)
  for category_i in range(num_category):
    category_ii = num_category - category_i - 1
    category = CATEGORY_TO_ID[category_i]
    # initialize
    split = 'train_' + category
    loader.resetIterator(split)
    current_len = loader.split_len(split)

    # update buffer_pool
    buffer_pool.add(category, current_len)

    # initialize current buffer
    current_size = buffer_pool.get_buffer_size(category)
    current_buffer = Buffer(current_size)  # for memory representative data in current data

    if category_i == 0:
      if num_category == 5 and not isDebug: opt["max_category_epoch"] = 40
      # set up criterion, for first
      mm_crit = MaxMarginCriterionBatch(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])

      model = base_train.train_model(model, mm_crit, att_crit, loader, split, opt, checkpoint_dir, category,
                                     log, log_category, current_buffer)
      base_train.load_first_model(model, mm_crit, att_crit, loader, split, opt, checkpoint_dir,
                                          f_model_dir, category, log, log_category, current_buffer)
      # set up criterion, for other
      mm_crit = MaxMarginCriterionBatch(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'],
                                        opt['buffer_sample_number'])
    else:
      if num_category == 5 and not isDebug: opt["max_category_epoch"] = 20
      model = base_train.train_model(model, mm_crit, att_crit, loader, split, opt, checkpoint_dir, category,
                                     log, log_category, current_buffer, last_buffer)
      mm_crit_b = MaxMarginCriterionBatch(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])
      base_train.load_first_model(model, mm_crit_b, att_crit, loader, split, opt, checkpoint_dir,
                                  pre_model_dir, category, log, log_category, current_buffer,
                                  first_flag=False)

    # current_buffer._print()
    buffer_pool.add_buffer(category, current_buffer)
    last_buffer = buffer_pool.get_last_buffer()
    buffer_pool._print()

    buffer_dir = osp.join(checkpoint_dir, category, 'buffer.json')
    buffer_pool.save(buffer_dir)

  log.close()
  log_category.close()

if __name__ == '__main__':

  args = parse_opt()
  main(args)

