from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable

CATEGORY_TO_MERGECATEGORY_5task = {'person':'person', 'animal':'animal', 'food':'kitchen', 'vehicle':'outdoor',
                             'furniture':'indoor', 'kitchen':'kitchen', 'indoor':'indoor',
                             'accessory':'outdoor', 'electronic':'indoor', 'outdoor':'outdoor',
                             'appliance':'indoor', 'sports':'outdoor'}
MERGECATEGORY_5task = {'person', 'animal', 'kitchen', 'indoor', 'outdoor'}

CATEGORY_TO_MERGECATEGORY_10task = {'person':'person', 'animal':'animal', 'food':'food', 'vehicle':'vehicle',
                             'furniture':'furniture', 'kitchen':'kitchen', 'indoor':'indoor',
                             'accessory':'accessory', 'electronic':'electronic', 'outdoor':'outdoor',
                             'appliance':'appliance', 'sports':'sports'}
MERGECATEGORY_10task = {'person', 'animal', 'food', 'vehicle', 'furniture','kitchen','indoor','accessory','electronic','sports'}


def compute_overall(predictions):
  """
  check precision and recall for predictions.
  Input: predictions = [{ref_id, cpts, pred}]
  Output: overall = {precision, recall, f1}
  """
  overall = {'precision': -1, 'recall': -1, 'f1': -1}
  NP, NR, NC = 0, 0, 0  # num of pred, num of recall, num of correct
  for item in predictions:
    cpts, pred = item['gd_att_wds'], item['pred_att_wds']
    inter = list(set(cpts).intersection(set(pred)))
    # add to overall
    NC += len(inter)
    NP += len(pred)
    NR += len(cpts)
  if NP > 0:
    overall['precision'] = float(NC)/NP
  if NR > 0:
    overall['recall'] = float(NC)/NR
  if NP > 0 and NR > 0:
    overall['f1'] = 2*overall['precision']*overall['recall']/(overall['precision']+overall['recall'])
  return overall


def eval_attributes(loader, model, split, opt):
  # set mode
  model.eval()

  # initialize
  loader.resetIterator(split)

  # predict
  predictions = []
  while True:
    data = loader.getAttributeBatch(split)

    # forward sub_encoder
    Feats = data['Feats']
    fake_phrase_emb = Variable(torch.zeros(len(data['ref_ids']), opt['word_vec_size']).float().cuda())
    _, _, att_scores = model.sub_encoder(Feats['pool5'], Feats['fc7'], fake_phrase_emb)
    att_scores = F.sigmoid(att_scores)  # (num_anns, num_atts)

    # predict attributes
    for i, ref_id in enumerate(data['ref_ids']):
      if len(loader.Refs[ref_id]['att_wds']) > 0:
        pred_att_wds = []
        for j, sc in enumerate(list(att_scores[i].data.cpu().numpy())):
          if sc >= .5:
            pred_att_wds.append(loader.ix_to_att[j])
        entry = {}
        entry['gd_att_wds'] = loader.Refs[ref_id]['att_wds']
        entry['pred_att_wds'] = pred_att_wds
        predictions += [entry]
        print('ref_id%s: [pred]%s, [gd]%s' % \
              (ref_id, ' '.join(entry['pred_att_wds']), ' '.join(entry['gd_att_wds'])))

    # if we wrapped around the split
    if data['bounds']['wrapped']:
      break

  # evaluate
  overall = compute_overall(predictions)
  print(overall)
  return overall


def eval_split(loader, model, crit, split, opt):
  verbose = opt.get('verbose', True)
  num_sents = opt.get('num_sents', -1)
  # assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'
  global CATEGORY_TO_MERGECATEGORY
  global MERGECATEGORY
  CATEGORY_TO_MERGECATEGORY = CATEGORY_TO_MERGECATEGORY_5task if opt["task"] == 5 else CATEGORY_TO_MERGECATEGORY_10task
  MERGECATEGORY = MERGECATEGORY_5task if opt["task"] == 5 else MERGECATEGORY_10task
  # set mode
  model.eval()

  # initialize
  loader.resetIterator(split)
  cat_to_ix = loader.cat_to_ix
  ix_to_cat = loader.ix_to_cat
  supcat_to_ix = loader.supcat_to_ix
  ix_to_supcat = loader.ix_to_supcat
  Sentences = loader.Sentences
  Anns = loader.Anns
  category_acc, category_loss_evals, category_loss_sum = {}, {}, {}
  for category in cat_to_ix:
      category_acc[category] = 0
      category_loss_sum[category] = 0
      category_loss_evals[category] = 1e-10
  supercategory_acc, supercategory_loss_evals, supercategory_loss_sum = {}, {}, {}
  # for supercategory in supcat_to_ix:
  for supercategory in MERGECATEGORY:
      supercategory_acc[supercategory] = 0
      supercategory_loss_sum[supercategory] = 0
      supercategory_loss_evals[supercategory] = 1e-10
  n = 0
  loss_sum = 0
  loss_evals = 0
  acc = 0
  predictions = []
  finish_flag = False
  model_time = 0


  data_dict = {}
  while True:

    data = loader.getTestBatch(split, opt)
    ann_ids = data['ann_ids']
    sent_ids = data['sent_ids']
    Feats = data['Feats']
    labels = data['labels']

    gd_ixs = data['gd_ixs']

    for i, sent_id in enumerate(sent_ids):
      supercategory = ix_to_supcat[Anns[ann_ids[gd_ixs[i]]]['supercategory_id']]
      supercategory = CATEGORY_TO_MERGECATEGORY[supercategory]  # merge
      category = ix_to_cat[Anns[ann_ids[data['gd_ixs'][i]]]['category_id']]

      # expand labels
      label = labels[i:i+1]      # (1, label.size(1))
      max_len = (label != 0).sum().item()
      label = label[:, :max_len] # (1, max_len)
      expanded_labels = label.expand(len(ann_ids), max_len) # (n, max_len)

      # forward
      tic = time.time()
      scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores = \
        model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
              Feats['cxt_fc7'], Feats['cxt_lfeats'],
              expanded_labels)
      scores = scores.data.cpu().numpy()
      att_scores = F.sigmoid(att_scores) # (n, num_atts)
      rel_ixs = rel_ixs.data.cpu().numpy().tolist() # (n, )

      # save weights
      weights_d = weights[0].data.cpu().numpy().tolist()
      data_dict[sent_id] = {}
      data_dict[sent_id]['weights'] = weights_d

      # compute loss
      pred_ix = np.argmax(scores)
      gd_ix = data['gd_ixs'][i]
      pos_sc = scores[gd_ix]
      scores[gd_ix] = -1e5
      max_neg_sc = np.max(scores)
      loss = max(0, opt['margin']+max_neg_sc-pos_sc)
      loss_sum += loss
      category_loss_sum[category] += loss
      supercategory_loss_sum[supercategory] += loss
      loss_evals += 1
      category_loss_evals[category] += 1
      supercategory_loss_evals[supercategory] += 1

      # compute accuracy
      if pred_ix == gd_ix:
        acc += 1
        category_acc[category] += 1
        supercategory_acc[supercategory] += 1

      # relative ann_id
      rel_ix = rel_ixs[pred_ix]

      # add info
      entry = {}
      entry['sent_id'] = sent_id
      entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0] # gd-truth sent
      entry['gd_ann_id'] = data['ann_ids'][gd_ix]
      entry['pred_ann_id'] = data['ann_ids'][pred_ix]
      entry['pred_score'] = scores.tolist()[pred_ix]
      entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist() # list of 49 attn
      predictions.append(entry)
      toc = time.time()
      model_time += (toc - tic)

      # if used up
      if num_sents > 0 and loss_eval >= num_sents:
        finish_flag = True
        break

    # print
    ix0 = data['bounds']['it_pos_now']
    ix1 = data['bounds']['it_max']
    if verbose:
      print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
            (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))
    model_time = 0

    # if we already wrapped around the split
    if finish_flag or data['bounds']['wrapped']:
      break

  for supercategory in MERGECATEGORY:
    print ('%s: all_num=[%d], right_num=%d  acc=%.2f%%' % (supercategory, int(supercategory_loss_evals[supercategory]),
    int(supercategory_acc[supercategory]),supercategory_acc[supercategory]*100.0/supercategory_loss_evals[supercategory]))

  for category in cat_to_ix:
    category_acc[category] = category_acc[category]*100.0 / category_loss_evals[category]
    category_loss_sum[category] = category_loss_sum[category] / category_loss_evals[category]
  for supercategory in MERGECATEGORY:
    supercategory_acc[supercategory] = supercategory_acc[supercategory]*100.0 / supercategory_loss_evals[supercategory]
    supercategory_loss_sum[supercategory] = supercategory_loss_sum[supercategory] / supercategory_loss_evals[supercategory]
  # return loss_sum/loss_evals, acc/loss_evals, predictions, overall
  return loss_sum/loss_evals, category_loss_sum, supercategory_loss_sum, acc/loss_evals, category_acc,\
          supercategory_acc,  category_loss_evals, supercategory_loss_evals, predictions
          
# combine testA and testB
def eval_union_split(loader, model, crit, split, opt):
  verbose = opt.get('verbose', False)
  num_sents = opt.get('num_sents', -1)
  # assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'
  global CATEGORY_TO_MERGECATEGORY
  global MERGECATEGORY
  CATEGORY_TO_MERGECATEGORY = CATEGORY_TO_MERGECATEGORY_5task if opt["task"] == 5 else CATEGORY_TO_MERGECATEGORY_10task
  MERGECATEGORY = MERGECATEGORY_5task if opt["task"] == 5 else MERGECATEGORY_10task
  # set mode
  model.eval()

  # initialize
  loader.resetIterator(split)
  cat_to_ix = loader.cat_to_ix
  ix_to_cat = loader.ix_to_cat
  supcat_to_ix = loader.supcat_to_ix
  ix_to_supcat = loader.ix_to_supcat
  Sentences = loader.Sentences
  Anns = loader.Anns
  category_acc, category_loss_evals, category_loss_sum = {}, {}, {}
  for category in cat_to_ix:
      category_acc[category] = 0
      category_loss_sum[category] = 0
      category_loss_evals[category] = 1e-10
  supercategory_acc, supercategory_loss_evals, supercategory_loss_sum = {}, {}, {}
  # for supercategory in supcat_to_ix:
  for supercategory in MERGECATEGORY:
      supercategory_acc[supercategory] = 0
      supercategory_loss_sum[supercategory] = 0
      supercategory_loss_evals[supercategory] = 1e-10
  n = 0
  loss_sum = 0
  loss_evals = 0
  acc = 0
  predictions = []
  finish_flag = False
  model_time = 0


  data_dict = {}
  while True:
    if split == 'test' and opt['dataset'] in ['refcoco','refcoco+']:
      data = loader.getUnionTestBatch(opt)
    else:
      data = loader.getTestBatch(split,opt)
    ann_ids = data['ann_ids']
    sent_ids = data['sent_ids']
    Feats = data['Feats']
    labels = data['labels']

    gd_ixs = data['gd_ixs']

    for i, sent_id in enumerate(sent_ids):
      supercategory = ix_to_supcat[Anns[ann_ids[gd_ixs[i]]]['supercategory_id']]
      supercategory = CATEGORY_TO_MERGECATEGORY[supercategory]  # merge
      category = ix_to_cat[Anns[ann_ids[data['gd_ixs'][i]]]['category_id']]

      # expand labels
      label = labels[i:i+1]      # (1, label.size(1))
      max_len = (label != 0).sum().item()
      label = label[:, :max_len] # (1, max_len)
      expanded_labels = label.expand(len(ann_ids), max_len) # (n, max_len)

      # forward
      tic = time.time()
      scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores = \
        model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
              Feats['cxt_fc7'], Feats['cxt_lfeats'],
              expanded_labels)
      scores = scores.data.cpu().numpy()
      att_scores = F.sigmoid(att_scores) # (n, num_atts)
      rel_ixs = rel_ixs.data.cpu().numpy().tolist() # (n, )

      # save weights
      weights_d = weights[0].data.cpu().numpy().tolist()
      data_dict[sent_id] = {}
      data_dict[sent_id]['weights'] = weights_d

      # compute loss
      pred_ix = np.argmax(scores)
      gd_ix = data['gd_ixs'][i]
      pos_sc = scores[gd_ix]
      scores[gd_ix] = -1e5
      max_neg_sc = np.max(scores)
      loss = max(0, opt['margin']+max_neg_sc-pos_sc)
      loss_sum += loss
      category_loss_sum[category] += loss
      supercategory_loss_sum[supercategory] += loss
      loss_evals += 1
      category_loss_evals[category] += 1
      supercategory_loss_evals[supercategory] += 1

      # compute accuracy
      if pred_ix == gd_ix:
        acc += 1
        category_acc[category] += 1
        supercategory_acc[supercategory] += 1

      # relative ann_id
      rel_ix = rel_ixs[pred_ix]

      # add info
      entry = {}
      entry['sent_id'] = sent_id
      entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0] # gd-truth sent
      entry['gd_ann_id'] = data['ann_ids'][gd_ix]
      entry['pred_ann_id'] = data['ann_ids'][pred_ix]
      entry['pred_score'] = scores.tolist()[pred_ix]
      entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist() # list of 49 attn
      predictions.append(entry)
      toc = time.time()
      model_time += (toc - tic)

      # if used up
      if num_sents > 0 and loss_eval >= num_sents:
        finish_flag = True
        break

    # print
    ix0 = data['bounds']['it_pos_now']
    ix1 = data['bounds']['it_max']
    if verbose:
      print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
            (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))
    model_time = 0

    # if we already wrapped around the split
    if finish_flag or data['bounds']['wrapped']:
      break

  for supercategory in MERGECATEGORY:
    print ('%s: all_num=[%d], right_num=%d  acc=%.2f%%' % (supercategory, int(supercategory_loss_evals[supercategory]),
    int(supercategory_acc[supercategory]),supercategory_acc[supercategory]*100.0/supercategory_loss_evals[supercategory]))

  for category in cat_to_ix:
    category_acc[category] = category_acc[category]*100.0 / category_loss_evals[category]
    category_loss_sum[category] = category_loss_sum[category] / category_loss_evals[category]
  
  rightnum = {}
  for supercategory in MERGECATEGORY:
    rightnum[supercategory] = supercategory_acc[supercategory]
    supercategory_acc[supercategory] = supercategory_acc[supercategory]*100.0 / supercategory_loss_evals[supercategory]
    supercategory_loss_sum[supercategory] = supercategory_loss_sum[supercategory] / supercategory_loss_evals[supercategory]
  # return loss_sum/loss_evals, acc/loss_evals, predictions, overall
  return loss_sum/loss_evals, category_loss_sum, supercategory_loss_sum, acc/loss_evals, category_acc,\
          supercategory_acc,  category_loss_evals, supercategory_loss_evals, predictions, rightnum


def eval_dets_split(loader, model, crit, split, opt):
  verbose = opt.get('verbose', True)
  num_sents = opt.get('num_sents', -1)
  assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'

  # set mode
  model.eval()

  # evaluate attributes
  overall = eval_attributes(loader, model, split, opt)

  # initialize
  loader.resetIterator(split)
  n = 0
  loss_sum = 0
  loss_evals = 0
  acc = 0
  predictions = []
  finish_flag = False
  model_time = 0

  while True:

    data = loader.getTestBatch(split, opt)
    det_ids = data['det_ids']
    sent_ids = data['sent_ids']
    Feats = data['Feats']
    labels = data['labels']

    for i, sent_id in enumerate(sent_ids):

      # expand labels
      label = labels[i:i+1]      # (1, label.size(1))
      max_len = (label != 0).sum().item()
      label = label[:, :max_len] # (1, max_len)
      expanded_labels = label.expand(len(det_ids), max_len) # (n, max_len)

      # forward
      # scores  : overall matching score (n, )
      # sub_grid_attn : (n, 49) attn on subjective's grids
      # sub_attn: (n, seq_len) attn on subjective words of expression
      # loc_attn: (n, seq_len) attn on location words of expression
      # rel_attn: (n, seq_len) attn on relation words of expression
      # rel_ixs : (n, ) selected context object
      # weights : (n, 2) weights on subj and loc
      # att_scores: (n, num_atts)
      tic = time.time()
      scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores = \
        model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
              Feats['cxt_fc7'], Feats['cxt_lfeats'],
              expanded_labels)
      scores = scores.data.cpu().numpy()
      att_scores = F.sigmoid(att_scores) # (n, num_atts)
      rel_ixs = rel_ixs.data.cpu().numpy().tolist() # (n, )

      # compute loss
      pred_ix = np.argmax(scores)
      gd_ix = data['gd_ixs'][i]
      pos_sc = scores[gd_ix]
      scores[gd_ix] = -1e5
      max_neg_sc = np.max(scores)
      loss = max(0, opt['margin']+max_neg_sc-pos_sc)
      loss_sum += loss
      loss_evals += 1

      # compute accuracy
      if pred_ix == gd_ix:
        acc += 1

      # relative det_id
      rel_ix = rel_ixs[pred_ix]

      # predict attribute on the predicted object
      pred_atts = []
      pred_att_scores = att_scores[pred_ix].data.cpu().numpy()
      top_att_ixs = pred_att_scores.argsort()[::-1][:5] # check top 5 attributes
      for k in top_att_ixs:
        pred_atts.append((loader.ix_to_att[k], float(pred_att_scores[k])))

      # add info
      entry = {}
      entry['sent_id'] = sent_id
      entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0] # gd-truth sent
      entry['gd_det_id'] = data['det_ids'][gd_ix]
      entry['pred_det_id'] = data['det_ids'][pred_ix]
      entry['pred_score'] = scores.tolist()[pred_ix]
      entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist() # list of 49 attn
      entry['sub_attn'] = sub_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
      entry['loc_attn'] = loc_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
      entry['rel_attn'] = rel_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
      entry['rel_det_id'] = data['cxt_det_ids'][pred_ix][rel_ix]        # rel det_id
      entry['weights'] = weights[pred_ix].data.cpu().numpy().tolist()   # list of 2 weights
      entry['pred_atts'] = pred_atts # list of (att_wd, score)
      predictions.append(entry)
      toc = time.time()
      model_time += (toc - tic)

      # if used up
      if num_sents > 0 and loss_eval >= num_sents:
        finish_flag = True
        break

    # print
    ix0 = data['bounds']['it_pos_now']
    ix1 = data['bounds']['it_max']
    if verbose:
      print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, (%.4f), model time (per sent) is %.2fs' % \
            (split, ix0, ix1, acc*100.0/loss_evals, loss, model_time/len(sent_ids)))
    model_time = 0

    # if we already wrapped around the split
    if finish_flag or data['bounds']['wrapped']:
      break

  return loss_sum/loss_evals, acc/loss_evals, predictions, overall
