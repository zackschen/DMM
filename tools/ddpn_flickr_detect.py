"""
Run detection on all images and save detected bounding box (with category_name and score).

cache/detections/refcoco_unc/{net}_{imdb}_{tag}_dets.json has
0. dets: list of {det_id, box, image_id, category_id, category_name, score}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import argparse
import json
import torch
from scipy.misc import imread
this_dir = osp.dirname(__file__)

# add paths
import _init_paths
from model.nms_wrapper import nms
from mrcn import inference, inference_no_imdb

# dataloader
sys.path.insert(0, osp.join(this_dir, '../lib/loaders'))
from loader import Loader


def cls_to_detections(scores, boxes, imdb, nms_thresh, conf_thresh):
  # run nms and threshold for each class detection
  cls_to_dets = {}
  num_dets = 0
  for cls_ind, class_name in enumerate(imdb.classes[1:]):
    cls_ind += 1  # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind+1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(torch.from_numpy(dets), nms_thresh)
    dets = dets[keep.numpy(), :]
    inds = np.where(dets[:, -1] >= conf_thresh)[0]
    dets = dets[inds, :]
    cls_to_dets[class_name] = dets
    num_dets += dets.shape[0]

  return cls_to_dets, num_dets


def main(args):

  # Image Directory
  params = vars(args)
  if params['dataset'] == 'flickr30k':
    dataset_splitBy = params['dataset']
  else:
    dataset_splitBy = params['dataset'] + '_' + params['splitBy']

  if 'flickr' in dataset_splitBy:
    IMAGE_DIR = 'data/images/flickr30k/images'
  elif 'coco' or 'combined' in dataset_splitBy:
    IMAGE_DIR = 'data/images/mscoco/images/train2014'
  elif 'clef' in dataset_splitBy:
    IMAGE_DIR = 'data/images/saiapr_tc-12'
  else:
    print('No image directory prepared for ', args.dataset)
    sys.exit(0)

  # make save dir
  save_dir = osp.join('cache/detections', dataset_splitBy)
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)
  print(save_dir)

  # import refer
  from refer import REFER
  data_root = params['data_root']
  refer = REFER(data_root, 'refcoco', 'unc')
  cat_name_to_cat_ix = {category_name: category_id for category_id, category_name in refer.Cats.items()}
  print (cat_name_to_cat_ix)

  # load dataset
  data_json = osp.join('cache/prepro', dataset_splitBy, 'data.json')
  data_h5 = osp.join('cache/prepro', dataset_splitBy, 'data.h5')
  loader = Loader(data_json, data_h5)
  images = loader.images

  # detect and prepare dets.json
  dets = []
  det_id = 0
  h5_id = 0
  cnt = 0
  for i, image in enumerate(images):
    image_id = image['image_id']
    det_bbox = image['det_bbox']

    for detection in det_bbox:
      x1, y1, x2, y2 = detection
      det = {'det_id': det_id,
             'h5_id' : det_id,  # we make h5_id == det_id
             'box': [x1, y1, x2-x1+1, y2-y1+1],
             'image_id': image_id,
             'category_id': cat_name_to_cat_ix['cake'],
             'category_name': 'cake',
             'score': 0.5}
      dets += [det]
      det_id += 1

    cnt += 1
    print('%s/%s done.' % (cnt, len(images)))

  # save dets.json = [{det_id, box, image_id, score}]
  # to cache/detections/
  save_path = osp.join(save_dir, '%s_%s_%s_dets.json' % (args.net_name, args.imdb_name, args.tag))
  with open(save_path, 'w') as f:
    json.dump(dets, f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_root', default='data', type=str, help='data folder containing images and four datasets.')
  parser.add_argument('--dataset', default='flickr30k', type=str, help='refcoco/refcoco+/refcocog')
  parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')

  parser.add_argument('--imdb_name', default='vg_minus_ddpn', help='image databased trained on.')
  parser.add_argument('--net_name', default='res101')
  parser.add_argument('--iters', default=1250000, type=int)
  parser.add_argument('--tag', default='notime')

  parser.add_argument('--nms_thresh', default=0.3, help='NMS threshold')
  parser.add_argument('--conf_thresh', default=0.65, help='confidence threshold')

  args = parser.parse_args()

  main(args)


