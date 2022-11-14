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


def calc_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    iou = 0.0
    if box1[2] <= box1[0] or box1[3] <= box1[1]:
        return iou
    if box2[2] <= box2[0] or box2[3] <= box2[1]:
        return iou
    if box1[2] <= box2[0] or box1[0] >= box2[2]:
        return iou
    if box1[3] <= box2[1] or box1[1] >= box2[3]:
        return iou

    xl_min = min(box1[0], box2[0])
    xl_max = max(box1[0], box2[0])
    xr_min = min(box1[2], box2[2])
    xr_max = max(box1[2], box2[2])

    yl_min = min(box1[1], box2[1])
    yl_max = max(box1[1], box2[1])
    yr_min = min(box1[3], box2[3])
    yr_max = max(box1[3], box2[3])

    inter = float(xr_min-xl_max)*float(yr_min-yl_max)
    union = float(xr_max-xl_min)*float(yr_max-yl_min)

    iou = float(inter) / float(union)
    if iou < 0:
        iou = 0.0
    return iou

def calc_iou_by_reg_feat(gt, pred, reg_feat, ht, wt):
    # gt, pred: [xmin, ymin, xmax, ymax]
    # reg_feat:	[t_x, t_y, t_w, t_h]
    # t_x = (x_gt - x_pred) / w_pred (center point x)
    # t_y = (y_gt - y_pred) / h_pred (center point y)
    # t_w = log(w_gt / w_pred)
    # t_h = log(h_gt / h_pred)

    pred = np.array(pred).astype('float')
    pred_w = pred[2]-pred[0]+1.0
    pred_h = pred[3]-pred[1]+1.0
    reg_w = np.exp(reg_feat[2])*pred_w-1.0
    reg_h = np.exp(reg_feat[3])*pred_h-1.0

    if reg_w > 0.0 and reg_h > 0.0:
        reg = np.zeros(4).astype('float32')
        reg[0] = (pred[0]+pred[2])/2.0+pred_w*reg_feat[0]-reg_w/2.0
        reg[1] = (pred[1]+pred[3])/2.0+pred_h*reg_feat[1]-reg_h/2.0
        reg[2] = (pred[0]+pred[2])/2.0+pred_w*reg_feat[0]+reg_w/2.0
        reg[3] = (pred[1]+pred[3])/2.0+pred_h*reg_feat[1]+reg_h/2.0
        reg[0] = max(0.0, reg[0])
        reg[1] = max(0.0, reg[1])
        reg[2] = min(wt, reg[2])
        reg[3] = min(ht, reg[3])
        return calc_iou(gt, reg), reg
    else:
        return 0.0, None

def eval_cur_batch(gt_label, cur_logits, is_train=True, type_batch=None, num_type=0, num_sample=0,
                   pos_or_reg=None, bbx_loc=None, gt_loc_all=None, ht=1.0, wt=1.0):

    accu = 0.0
    type_accu = np.zeros(9)
    if is_train:
        res_prob = cur_logits[:, :, 0]
        # res_prob = cur_logits
        res_label = np.argmax(res_prob, axis=1)
        accu = float(np.sum(res_label == gt_label)) / float(len(gt_label))
    else:
        num_bbx = len(bbx_loc)
        res_prob = cur_logits[:, :num_bbx, 0]
        # res_prob = cur_logits[:, :num_bbx]
        res_label = np.argmax(res_prob, axis=1)
        for gt_id in range(len(pos_or_reg)):
            cur_gt_pos = gt_label[gt_id]
            success = False

            cur_gt = gt_loc_all[gt_id]
            if np.any(cur_gt):
                cur_bbx = bbx_loc[res_label[gt_id]]
                cur_reg = cur_logits[gt_id, res_label[gt_id], 1:]
                # cur_reg = np.zeros(4).astype('float32')
                # print 'IOU Stats: ', cur_gt, cur_bbx, cur_reg
                iou, _ = calc_iou_by_reg_feat(cur_gt, cur_bbx, cur_reg, ht, wt)
                # cur_bbx = np.reshape(cur_bbx, (-1))
                # iou = calc_iou(cur_gt, cur_bbx)
                if iou > 0.5:
                    success = True
            if success:
                accu += 1.0
                type_accu[type_batch[gt_id]] += 1.0

        accu /= float(num_sample)
        type_accu = type_accu * 1.0 / (num_type+1e-10)

    return accu, type_accu, res_label


def eval_split(loader, model, crit, split, opt):
  verbose = opt.get('verbose', True)
  num_sents = opt.get('num_sents', -1)
  assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'

  # set mode
  model.eval()

  # evaluate attributes
  """
  overall = eval_attributes(loader, model, split, opt)
  """
  # initialize
  loader.resetIterator(split)
  cat_to_ix = loader.cat_to_ix
  ix_to_cat = loader.ix_to_cat
  Sentences = loader.Sentences

  num_corr_all = 0.0
  num_cnt_all = 0.0
  num_category_cnt = np.zeros(len(cat_to_ix))
  num_category_cor = np.zeros(len(cat_to_ix))
  predictions = []
  model_time = 0

  while True:

    data = loader.getTestCrossBatch(split, opt)
    image_id = data['image_id']
    ann_ids = data['ann_ids']
    category_ids = data['category_ids']
    num_category = data['num_category']
    sent_ids = data['sent_ids']
    Feats = data['Feats']
    labels = data['labels']

    num_sample = data['num_sample']
    pos_or_reg = data['pos_or_reg']
    bbx_loc = data['bbx_loc']
    gt_pos_all = data['gt_pos_all']
    gt_loc_all = data['gt_loc_all']
    ht, wt = data['h'], data['w']

    # forward
    # scores  : overall matching score (n, )
    # sub_grid_attn : (n, 49) attn on subjective's grids
    tic = time.time()
    scores, sub_grid_attn = model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], labels, False)
    scores = scores.data.cpu().numpy()
    sub_grid_attn = sub_grid_attn.data.cpu().numpy()

    """
    print ('image_id', image_id)
    print ('ann_ids', ann_ids)
    print ('category_ids', category_ids)
    print ('num_category', num_category)
    print ('gt_pos_all', np.array(gt_pos_all).shape)
    print (gt_pos_all)
    print ('gt_loc_all', np.array(gt_loc_all).shape)
    print (gt_loc_all)
    print ('bbx_loc', np.array(bbx_loc).shape)
    print (bbx_loc)
    print ('num_sample', num_sample)
    print ('pos_or_reg', np.array(pos_or_reg).shape)
    print (pos_or_reg)
    exit()
    """

    cur_accuracy, cur_category_accuracy, pred_ix = \
            eval_cur_batch(gt_pos_all, scores, False, category_ids, num_category, num_sample,
                           pos_or_reg, bbx_loc, gt_loc_all, ht, wt)

    num_valid = np.sum(np.all(gt_loc_all, 1))

    num_corr_all = num_corr_all + cur_accuracy * num_sample
    num_cnt_all = num_cnt_all + float(num_sample)
    num_category_cor = num_category_cor + num_category * 1.0 * cur_category_accuracy
    num_category_cnt = num_category_cnt + num_category

    gt_sub_grid_attn = []

    for i, index in enumerate(pred_ix):
        gt_sub_grid_attn.append(sub_grid_attn[i, index])
    # add info
    entry = {}
    entry['image_id'] = image_id
    entry['ann_ids'] = ann_ids
    entry['sub_grid_attn'] = np.array(gt_sub_grid_attn).tolist() # list of 49 attn
    predictions.append(entry)
    toc = time.time()
    model_time += (toc - tic)

    # print
    ix0 = data['bounds']['it_pos_now']
    ix1 = data['bounds']['it_max']
    if verbose:
      print('evaluating [%s] ... image[%d/%d]\'s sents[%d/%d], acc=%.2f%%, model time (per sent) is %.2fs' % \
            (split, ix0, ix1, num_valid, len(gt_pos_all), cur_accuracy*100, model_time/len(sent_ids)))
      # for category in cat_to_ix:
      #   print ('%s[%d],  acc=%.2f%%' % (category, int(category_loss_evals[category]), category_acc[cat_to_ix[category]]*100.0/category_loss_evals[category]))
    model_time = 0

    # if we already wrapped around the split
    if data['bounds']['wrapped']:
      break

  accu = num_corr_all / num_cnt_all
  category_accu = num_category_cor / (num_category_cnt + 1e-10)

  print('validation acc : %.2f%%\n' % (accu*100.0))
  for category, category_id in cat_to_ix.items():
    print ('%s:%.4f%%, %.2f/%.2f' % (category, category_accu[category_id]*100, num_category_cor[category_id],
                                     num_category_cnt[category_id]))

  return accu, category_accu, num_category_cor, num_category_cnt, predictions


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
      max_len = (label != 0).sum().data[0]
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
