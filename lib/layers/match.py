from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from layers.lang_encoder import RNNEncoder, PhraseAttention
from layers.visual_encoder import LocationEncoder, SubjectEncoder, RelationEncoder

"""
Simple Matching function for
- visual_input (n, vis_dim)
- lang_input (n, vis_dim)
forward them through several mlp layers and finally inner-product, get cossim
"""

class Normalize_Scale(nn.Module):
  def __init__(self, dim, init_norm=20):
    super(Normalize_Scale, self).__init__()
    self.init_norm = init_norm
    self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

  def forward(self, bottom):
    # input is variable (n, dim)
    assert isinstance(bottom, Variable), 'bottom must be variable'
    bottom_normalized = nn.functional.normalize(bottom, p=2, dim=-1)
    bottom_normalized_scaled = bottom_normalized * self.weight
    return bottom_normalized_scaled


class JointMatching(nn.Module):

  def __init__(self, opt):
    super(JointMatching, self).__init__()
    self.fc7_dim = opt['fc7_dim']
    self.pool5_dim = opt['pool5_dim']
    self.cemb_dim = opt['cemb_dim']
    self.loc_dim = opt['loc_dim']
    self.num_props = opt['num_props']
    self.end_points = {}

    num_layers = opt['rnn_num_layers']
    hidden_size = opt['rnn_hidden_size']
    num_dirs = 2 if opt['bidirectional'] > 0 else 1
    jemb_dim = opt['jemb_dim']

    # language rnn encoder
    self.rnn_encoder = RNNEncoder(vocab_size=opt['vocab_size'],
                                  word_embedding_size=opt['word_embedding_size'],
                                  word_vec_size=opt['word_vec_size'],
                                  hidden_size=opt['rnn_hidden_size'],
                                  bidirectional=opt['bidirectional']>0,
                                  input_dropout_p=opt['word_drop_out'],
                                  dropout_p=opt['rnn_drop_out'],
                                  n_layers=opt['rnn_num_layers'],
                                  rnn_type=opt['rnn_type'],
                                  variable_lengths=opt['variable_lengths']>0)
    # text mapping
    self.text_map = nn.Sequential(nn.Linear(opt['rnn_hidden_size']*2, opt['cemb_dim']),
                                  nn.LeakyReLU(negative_slope=0.25),
                                  Normalize_Scale(opt['cemb_dim'], opt['visual_init_norm']))
    # image mapping   no kernel regularizer
    self.img_map = nn.Sequential(nn.Conv2d(opt['pool5_dim'], opt['cemb_dim'], 3, padding=1),
                                 nn.LeakyReLU(negative_slope=0.25))
    # build attention
    self.a_s_normalize = Normalize_Scale(opt['cemb_dim'], opt['visual_init_norm'])
    # matching
    self.v_bn = nn.BatchNorm1d(opt['cemb_dim']+opt['loc_dim'])
    self.e_bn = nn.BatchNorm1d(opt['cemb_dim'])
    self.att_fuse = nn.Sequential(nn.Conv2d(opt['cemb_dim']*2+opt['loc_dim'], opt['jemb_dim'], 1, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(opt['jemb_dim'], 5, 1, 1))

    """
    # phrase attender
    self.sub_attn = PhraseAttention(hidden_size * num_dirs)

    # visual matching
    self.sub_encoder = SubjectEncoder(opt)
    self.sub_matching = Matching(opt['fc7_dim'], opt['word_vec_size'],
                                 opt['jemb_dim'], opt['jemb_drop_out'])
    """

  def forward(self, pool5, fc7, lfeats, labels, is_train):
    """
    Inputs:
    - pool5       : (n*num_props, pool5_dim, 7, 7)
    - fc7         : (n*num_props, fc7_dim, 7, 7)
    - lfeats      : (n*num_props, 5)
    - labels      : (n, seq_len)
    Output:
    - scores        : (n, num_props, 5)
    - sub_grid_attn : (n, 49)
    - sub_attn      : (n, seq_len) attn on subjective words of expression
    """
    batch, conv_size = int(fc7.size(0)/self.num_props), fc7.size(2)

    # print('Building Text model')
    # expression encoding hidden (n, 1024)
    context, hidden, embedded = self.rnn_encoder(labels)

    # print('Common space mapping ...')
    sen_raw = self._text_mapping(hidden)

    vis_data = pool5.view(-1, self.pool5_dim, conv_size, conv_size)
    vis_raw = self._image_mapping(vis_data)
    vis_raw = vis_raw.view(batch, self.num_props, self.cemb_dim, -1).transpose(2, 3).contiguous()

    # print('Common space attention ...')
    self.end_points.update(self._build_attention(sen_raw, vis_raw))

    # print('Feature Fusing and Predict')
    vis_raw = self.end_points['visual_attend']     # (n, num_props, d)
    lfeats = lfeats.view(batch, self.num_props, self.loc_dim)
    vis_raw = torch.cat([vis_raw, lfeats], 2)
    vis_raw = vis_raw.view(-1, self.cemb_dim+self.loc_dim)
    matching_scores = self._matching(sen_raw, vis_raw, is_train) # (n, num_props, 5)

    sub_grid_attn = self.end_points['heatmap']

    return matching_scores, sub_grid_attn

  def _text_mapping(self, e_s):

    e_s = self.text_map(e_s)

    return e_s

  def _image_mapping(self, feat_map):

    feat_map = self.img_map(feat_map)

    return feat_map

  def _build_attention(self, e_s, v):
    # e_s (B, D)
    # v   (B, N, TxT, D)

    ## sentence level ##
    attn_outs = {}
    # heatmap pool
    h_s = F.relu(torch.einsum('bj,bntj->bnt', (e_s,v)))
    end_point = 'heatmap'
    attn_outs[end_point] = h_s
    # attention
    a_s = torch.einsum('bnt,bntj->bnj', (h_s,v))
    a_s = self.a_s_normalize(a_s)
    end_point = 'visual_attend'
    attn_outs[end_point] = a_s
    """
    # score (sentence level)
    R_s = torch.einsum('bnj,bj->bn', (a_s,e_s))
    end_point = 'score_sentence'
    attn_outs[end_point] = R_s
    """

    return attn_outs

  def _matching(self, e, v, is_train=False):
    # e (batch, d)
    # v (batchxn, d+5)

    # preprocess
    e_bn = self.e_bn(e)
    v_bn = self.v_bn(v)
    e = e_bn.view(-1, 1, 1, self.cemb_dim).expand(-1, self.num_props, 1, self.cemb_dim).contiguous()
    v = v_bn.view(-1, self.num_props, 1, self.cemb_dim+self.loc_dim)

    # fuse and predict
    feat_concat = torch.cat([e, v], 3).permute(0, 3, 1, 2).contiguous()
    att_scores = self.att_fuse(feat_concat)
    att_scores = att_scores.view(-1, 5, self.num_props).transpose(1, 2).contiguous()

    return att_scores
