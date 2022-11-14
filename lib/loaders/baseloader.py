"""
data_json has
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data as data
import os.path as osp
import numpy as np
import h5py
import json
import random
from loaders.loader import Loader

import torch
from torch.autograd import Variable

# mrcn path
from mrcn import inference_no_imdb

CATEGORY_TO_MERGECATEGORY = {'person':'person', 'animal':'animal', 'food':'food', 'vehicle':'vehicle',
                             'furniture':'furniture', 'kitchen':'kitchen', 'indoor':'indoor',
                             'accessory':'accessory', 'electronic':'electronic', 'outdoor':'outdoor',
                             'appliance':'appliance', 'sports':'sports'}

"""
CATEGORY_TO_MERGECATEGORY = {'person':'person', 'animal':'other', 'food':'other', 'vehicle':'other',
                             'furniture':'other', 'kitchen':'other', 'indoor':'other',
                             'accessory':'other', 'electronic':'other', 'outdoor':'other',
                             'appliance':'other', 'sports':'other'}
"""

# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


class baseloader(data.Dataset):
  def __init__(self, opt, data_json, data_h5, train = True):
    self.opt = opt
    self.train = train
    # parent loader instance
    print('Loader loading data.json: ', data_json)
    self.info = json.load(open(data_json))
    self.word_to_ix = self.info['word_to_ix']
    self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
    print('vocab size is ', self.vocab_size)
    self.cat_to_ix = self.info['cat_to_ix']
    self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
    print('object cateogry size is ', len(self.ix_to_cat))
    self.supcat_to_ix = self.info['supcat_to_ix']
    self.ix_to_supcat = {ix: supcat for supcat, ix in self.supcat_to_ix.items()}
    self.images = self.info['images']
    self.anns = self.info['anns']
    self.refs = self.info['refs']
    self.sentences = self.info['sentences']
    self.category_structure = self.info['category_structure']
    print('we have %s images.' % len(self.images))
    print('we have %s anns.' % len(self.anns))
    print('we have %s refs.' % len(self.refs))
    print('we have %s sentences.' % len(self.sentences))
    print('label_length is ', self.label_length)

    # construct mapping
    self.Refs = {ref['ref_id']: ref for ref in self.refs}
    self.Images = {image['image_id']: image for image in self.images}
    self.Anns = {ann['ann_id']: ann for ann in self.anns}
    self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
    self.annToRef = {ref['ann_id']: ref for ref in self.refs}
    self.sentToRef = {sent_id: ref for ref in self.refs for sent_id in ref['sent_ids']}

    # read data_h5 if exists
    self.data_h5 = None
    if data_h5 is not None:
      print('Loader loading data.h5: ', data_h5)
      self.data_h5 = h5py.File(data_h5, 'r')
      assert self.data_h5['labels'].shape[0] == len(self.sentences), 'label.shape[0] not match sentences'
      assert self.data_h5['labels'].shape[1] == self.label_length, 'label.shape[1] not match label_length'
    
    # prepare attributes
    self.att_to_ix = self.info['att_to_ix']
    self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
    self.num_atts = len(self.att_to_ix)
    self.att_to_cnt = self.info['att_to_cnt']

    # img_iterators for each split
    self.split_ix = {}
    self.split_supercategory_ix = {}
    self.iterators = {}
    for image_id, image in self.Images.items():
      # we use its ref's split (there is assumption that each image only has one split)
      split = self.Refs[image['ref_ids'][0]]['split']
      if split not in self.split_ix:
        self.split_ix[split] = []
        self.iterators[split] = 0

      # supercategory split
      ref_ids = image['ref_ids']
      for ref_id in ref_ids:
        supercategory_id = self.Refs[ref_id]['supercategory_id']
        supercategory = self.ix_to_supcat[supercategory_id]
        supercategory = CATEGORY_TO_MERGECATEGORY[supercategory]

        split_supercategory = split + '_' + supercategory
        if split_supercategory not in self.split_supercategory_ix:
          self.split_supercategory_ix[split_supercategory] = []
          self.iterators[split_supercategory] = 0

        self.split_supercategory_ix[split_supercategory] += [ref_id]

        split_supercategory_img = split + '_' + supercategory + '_img'
        if split_supercategory_img not in self.split_supercategory_ix:
          self.split_supercategory_ix[split_supercategory_img] = []
          self.iterators[split_supercategory_img] = 0
        if image_id not in self.split_supercategory_ix[split_supercategory_img]:
          self.split_supercategory_ix[split_supercategory_img] += [image_id]

      self.split_ix[split] += [image_id]
    for k, v in self.split_ix.items():
      print('assigned %d images to split %s' % (len(v), k))
    for k, v in self.split_supercategory_ix.items():
      print('assigned %d images to split %s' % (len(v), k))

  def prepare_mrcn(self, head_feats_dir, args):
    """
    Arguments:
        head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
        args: imdb_name, net_name, iters, tag
    """
    self.head_feats_dir = head_feats_dir
    self.mrcn = inference_no_imdb.Inference(args)
    assert args.net_name == 'res101'
    self.pool5_dim = 1024
    self.fc7_dim = 2048
    
  # load different kinds of feats
  def loadFeats(self, Feats):
    # Feats = {feats_name: feats_path}
    self.feats = {}
    self.feat_dim = None
    for feats_name, feats_path in Feats.items():
      if osp.isfile(feats_path):
        self.feats[feats_name] = h5py.File(feats_path, 'r')
        self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
        assert self.feat_dim == self.fc7_dim
        print('FeatLoader loading [%s] from %s [feat_dim %s]' % \
              (feats_name, feats_path, self.feat_dim))