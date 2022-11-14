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
# from buffer.replay_buffer import Buffer
# from buffer.replay_buffer import BufferPool
from buffer.replay_buffer_low import Buffer
from buffer.replay_buffer_low import BufferPool
from layers.joint_match import JointMatching
import models.utils as model_utils
import models.eval_easy_utils as eval_utils
# import models.finetune_base_train as base_train
import models.buffer_base_train as base_train
from crits.max_margin_crit import MaxMarginCriterionBatch
from mas.mas_buffer import MAS
from mas.mas_buffer import MAS_Omega_Accumulation
from opt import parse_opt

# torch
import torch
import torch.nn as nn
from torch.autograd import Variable

CATEGORY_TO_ID_5task = {0:'person', 1:'kitchen', 2:'animal', 3:'indoor', 4:'outdoor'}

CATEGORY_TO_ID_10task = {0: 'food', 1: 'indoor', 2: 'sports', 3: 'person',4: 'animal',\
  5: 'vehicle',6: 'furniture', 7: 'accessory', 8: 'electronic', 9: 'kitchen'}


sub_encoder_layers = [
    'sub_encoder.pool5_normalizer.weight', 'sub_encoder.fc7_normalizer.weight',
    'sub_encoder.att_normalizer.weight', 'sub_encoder.phrase_normalizer.weight',
    'sub_encoder.att_fuse.0.weight', 'sub_encoder.att_fuse.0.bias',
    'sub_encoder.att_fuse.1.weight', 'sub_encoder.att_fuse.1.bias',
    'sub_encoder.att_fc.weight', 'sub_encoder.att_fc.bias',
    'sub_encoder.attn_fuse.0.weight', 'sub_encoder.attn_fuse.0.bias',
    'sub_encoder.attn_fuse.2.weight', 'sub_encoder.attn_fuse.2.bias',
]
loc_encoder_layers = [
    'loc_encoder.lfeat_normalizer.weight', 'loc_encoder.dif_lfeat_normalizer.weight',
    'loc_encoder.fc.weight', 'loc_encoder.fc.bias',
]
rel_encoder_layers = [
    'rel_encoder.vis_feat_normalizer.weight', 'rel_encoder.lfeat_normalizer.weight',
    'rel_encoder.fc.weight', 'rel_encoder.fc.bias',
]

sub_matching_layers = [
    'sub_matching.vis_emb_fc.0.weight', 'sub_matching.vis_emb_fc.0.bias',
    'sub_matching.vis_emb_fc.1.weight', 'sub_matching.vis_emb_fc.1.bias',
    'sub_matching.vis_emb_fc.4.weight', 'sub_matching.vis_emb_fc.4.bias',
    'sub_matching.vis_emb_fc.5.weight', 'sub_matching.vis_emb_fc.5.bias',
    'sub_matching.lang_emb_fc.0.weight', 'sub_matching.lang_emb_fc.0.bias',
    'sub_matching.lang_emb_fc.1.weight', 'sub_matching.lang_emb_fc.1.bias',
    'sub_matching.lang_emb_fc.4.weight', 'sub_matching.lang_emb_fc.4.bias',
    'sub_matching.lang_emb_fc.5.weight', 'sub_matching.lang_emb_fc.5.bias',
]
loc_matching_layers = [
    'loc_matching.vis_emb_fc.0.weight', 'loc_matching.vis_emb_fc.0.bias',
    'loc_matching.vis_emb_fc.1.weight', 'loc_matching.vis_emb_fc.1.bias',
    'loc_matching.vis_emb_fc.4.weight', 'loc_matching.vis_emb_fc.4.bias',
    'loc_matching.vis_emb_fc.5.weight', 'loc_matching.vis_emb_fc.5.bias',
    'loc_matching.lang_emb_fc.0.weight', 'loc_matching.lang_emb_fc.0.bias',
    'loc_matching.lang_emb_fc.1.weight', 'loc_matching.lang_emb_fc.1.bias',
    'loc_matching.lang_emb_fc.4.weight', 'loc_matching.lang_emb_fc.4.bias',
    'loc_matching.lang_emb_fc.5.weight', 'loc_matching.lang_emb_fc.5.bias',
]
rel_matching_layers = [
    'rel_matching.vis_emb_fc.0.weight', 'rel_matching.vis_emb_fc.0.bias',
    'rel_matching.vis_emb_fc.1.weight', 'rel_matching.vis_emb_fc.1.bias',
    'rel_matching.vis_emb_fc.4.weight', 'rel_matching.vis_emb_fc.4.bias',
    'rel_matching.vis_emb_fc.5.weight', 'rel_matching.vis_emb_fc.5.bias',
    'rel_matching.lang_emb_fc.0.weight', 'rel_matching.lang_emb_fc.0.bias',
    'rel_matching.lang_emb_fc.1.weight', 'rel_matching.lang_emb_fc.1.bias',
    'rel_matching.lang_emb_fc.4.weight', 'rel_matching.lang_emb_fc.4.bias',
    'rel_matching.lang_emb_fc.5.weight', 'rel_matching.lang_emb_fc.5.bias',
]

sub_attn_layers = [
    'sub_attn.fc.weight', 'sub_attn.fc.bias',
]
loc_attn_layers = [
    'loc_attn.fc.weight', 'loc_attn.fc.bias',
]
rel_attn_layers = [
    'rel_attn.fc.weight', 'rel_attn.fc.bias',
]

weight_fc_layers = [
    'weight_fc.weight', 'weight_fc.bias',
]

freeze_1 = loc_encoder_layers + rel_encoder_layers
freeze_2 = freeze_1 + loc_matching_layers + rel_matching_layers

freeze_3 = loc_attn_layers + rel_attn_layers
freeze_4 = freeze_3 + sub_attn_layers

freeze_5 = weight_fc_layers

freeze_6 = freeze_4 + freeze_5

freeze_layers_dict = {0:[], 1:freeze_1, 2:freeze_2, 3:freeze_3, 4:freeze_4, 5:freeze_5, 6:freeze_6}

sub_layers = sub_encoder_layers + sub_matching_layers
loc_layers = loc_encoder_layers + loc_matching_layers
rel_layers = rel_encoder_layers + rel_matching_layers


def main(args):

  opt = vars(args)

  isDebug = True if sys.gettrace() else False

  # initialize
  opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
  checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'], opt['id'])
  if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

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
  loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json,opt = opt)
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
  old_category = 'person'
  old_split = 'train_person'

  # old_data_size, data_size = 0, 0
  # previous_ratio = 0.5

  freeze_layers = freeze_layers_dict.get(opt['freeze_id'])
  for category_i in range(num_category):

    category_ii = num_category - category_i - 1
    category = CATEGORY_TO_ID[category_i]

    # initialize
    split = 'train_' + category
    loader.resetIterator(split)
    current_len = loader.split_len(split)

    buffer_pool.add(category, current_len)

    # initialize current buff
    current_size = buffer_pool.get_buffer_size(category)
    current_buffer = Buffer(current_size) # for memory representative data in current data

    if category_i == 0:
      # set up criterion, for first
      if num_category == 5 and not isDebug: opt["max_category_epoch"] = 40
      mm_crit = MaxMarginCriterionBatch(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])
      model = base_train.train_model(model, mm_crit, att_crit, loader, split, opt, checkpoint_dir,
                                    category, log, log_category, current_buffer)
      base_train.load_first_model(model, mm_crit, att_crit, loader, split, opt, checkpoint_dir,
                                    f_model_dir, category, log, log_category, current_buffer)
      # set up criterion, for other
      mm_crit = MaxMarginCriterionBatch(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'],
                                        opt['buffer_sample_number'])
    elif category_i == 1:
      if num_category == 5 and not isDebug: opt["max_category_epoch"] = 20
      model = MAS(mm_crit, att_crit, current_buffer, last_buffer, loader, old_split, split, opt,
                  checkpoint_dir, f_model_dir, old_category, category, log, log_category, freeze_layers,
                  sub_layers=sub_layers, loc_layers=loc_layers, rel_layers=rel_layers)
      mm_crit_b = MaxMarginCriterionBatch(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])
      base_train.load_first_model(model, mm_crit_b, att_crit, loader, split, opt, checkpoint_dir,
                                  pre_model_dir, category, log, log_category, current_buffer,
                                  first_flag=False)
    else:
      if num_category == 5 and not isDebug: opt["max_category_epoch"] = 20
      # previous_ratio = old_data_size * 1.0 / (data_size * 1.0)
      model = MAS_Omega_Accumulation(mm_crit, att_crit, current_buffer, last_buffer, loader, old_split,
                                      split, opt, checkpoint_dir, old_category, category, log, log_category,
                                      freeze_layers, sub_layers=sub_layers, loc_layers=loc_layers,
                                      rel_layers=rel_layers)
      base_train.load_first_model(model, mm_crit_b, att_crit, loader, split, opt, checkpoint_dir,
                                  pre_model_dir, category, log, log_category, current_buffer,
                                  first_flag=False)
    #                                previous_ratio=previous_ratio)

    # current_buffer._print()
    buffer_pool.add_buffer(category, current_buffer)
    last_buffer = buffer_pool.get_last_buffer()
    buffer_pool._print()

    buffer_dir = osp.join(checkpoint_dir, category, 'buffer.json')
    buffer_pool.save(buffer_dir)

    # update data size
    # current_size = loader.split_len(split)
    # old_data_size = data_size
    # data_size = data_size + current_size

    old_category = category
    old_split = 'train_' + category

  log.close()
  log_category.close()

if __name__ == '__main__':

  args = parse_opt()
  main(args)

