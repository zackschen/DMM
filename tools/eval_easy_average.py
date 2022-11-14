from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse

# model
import _init_paths
from layers.joint_match import JointMatching
from loaders.gt_mrcn_loader import GtMRCNLoader
import models.eval_easy_utils as eval_utils

# torch
import torch
import torch.nn as nn



def load_model(checkpoint_path, opt):
  tic = time.time()
  model = JointMatching(opt)
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'].state_dict())
  model.eval()
  model.cuda()
  print('model loaded in %.2f seconds' % (time.time()-tic))
  return model

def evaluate(params):
  # set up loader
  if params["task"] == 5:
    CATEGORYS = ['person','kitchen','animal','indoor','outdoor']
    data_json = osp.join('cache/prepro', params['dataset_splitBy'], '5task.json')
    data_h5 = osp.join('cache/prepro', params['dataset_splitBy'], '5task.h5')
  else:
    CATEGORYS = ['food','indoor','sports','person','animal','vehicle','furniture','accessory', 'electronic','kitchen']
    data_json = osp.join('cache/prepro', params['dataset_splitBy'], '10task.json')
    data_h5 = osp.join('cache/prepro', params['dataset_splitBy'], '10task.h5')
  loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json, opt = params)

  log_path = osp.join('eval_log', params['dataset'], 'combine_average_' + params['id'] + '.log')
  log_category_path = osp.join('eval_log', params['dataset'], 'combine_average_' + params['id']+'_category.log')
  log = open(log_path, 'a', buffering=1)
  log_category = open(log_category_path, 'a', buffering=1)

  if params['random']:
    CATEGORYS = ['food']

  for category in CATEGORYS:
    
    loader.test_iterators = 0

    if params['random']:
      model_prefix = osp.join('output', params['id'])
      infos = json.load(open(model_prefix+'.json'))
      model_opt = infos['opt']
      model_opt['dataset'] = params['dataset']
      model_opt['splitBy'] = params['splitBy']
      model_opt['dataset_splitBy'] = params['dataset_splitBy']
      model_opt['task'] = params['task']
      model_path = model_prefix + '.pth.tar'
      model = load_model(model_path, model_opt)
    else:
      model_prefix = osp.join('output', params['dataset_splitBy'], params['id'], category, params['id'])
      infos = json.load(open(model_prefix+'.json'))
      model_opt = infos['opt']
      # model_path = model_prefix + '.pth'
      model_path = model_prefix + '.pth.tar'
      model = load_model(model_path, model_opt)

    # loader's feats
    feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
    args.imdb_name = model_opt['imdb_name']
    args.net_name = model_opt['net_name']
    args.tag = model_opt['tag']
    args.iters = model_opt['iters']
    loader.prepare_mrcn(head_feats_dir=osp.join('cache/feats/', model_opt['dataset_splitBy'], 'mrcn', feats_dir),
                        args=args)
    ann_feats = osp.join('cache/feats', model_opt['dataset_splitBy'], 'mrcn',
                        '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
    loader.loadFeats({'ann': ann_feats})

    # check model_info and params
    assert model_opt['dataset'] == params['dataset']
    assert model_opt['splitBy'] == params['splitBy']

    # evaluate on the split,
    # predictions = [{sent_id, sent, gd_ann_id, pred_ann_id, pred_score, sub_attn, loc_attn, weights}]
    split = params['split']
    model_opt['num_sents'] = params['num_sents']
    model_opt['verbose'] = params['verbose']
    crit = None
    # combine testA and testB
    val_loss, val_category_loss, val_supercategory_loss, acc, category_acc, supercategory_acc,\
            category_loss_evals, supercategory_loss_evals, predictions, rightnum =\
            eval_utils.eval_union_split(loader, model, crit, split, model_opt) 
    log.write(' ---------------------------------------------------------------------------------\n')
    log.write(' ------------------------------%s----%s\n' % (category,split))
    log.write(' ---------------------------------------------------------------------------------\n')
    log_category.write(' ---------------------------------------------------------------------------------\n')
    log_category.write(' ------------------------------%s----%s\n' % (category,split))
    log_category.write(' ---------------------------------------------------------------------------------\n')
    log.write('%.6f, %.2f%%\n' % (val_loss, acc*100))
    log_category.write('%.2f%%\n' % (acc*100))
    for supercategory in supercategory_loss_evals.keys():
        log_category.write('%s: all_num=[%d], right_num=%d, loss= %.6f, acc= %.2f%%\n' % (supercategory,
                                                                      int(supercategory_loss_evals[supercategory]),
                                                                      int(rightnum[supercategory]),
                                                                      val_supercategory_loss[supercategory],
                                                                      supercategory_acc[supercategory]))
  log.close()
  log_category.close()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
  parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
  parser.add_argument('--split', type=str, default='test', help='split: testAB or val, etc')
  parser.add_argument('--id', type=str, default='0', help='model id name')
  parser.add_argument('--num_sents', type=int, default=-1, help='how many sentences to use when periodically evaluating the loss? (-1=all)')
  parser.add_argument('--verbose', type=int, default=0, help='if we want to print the testing progress')
  parser.add_argument('--task', type=int, default=5, help='number of tasks')
  parser.add_argument('--random', type=bool, default=False, help='the random model')
  args = parser.parse_args()
  params = vars(args)

  # make other options
  params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
  evaluate(params)


