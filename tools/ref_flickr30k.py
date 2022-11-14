"""
data_json has
0. refs        : list of {ref_id, ann_id, box, image_id, category_id, sent_ids}
1. images      : list of {image_id, sen_ids, ref_ids, ann_ids, det_bbox, file_name, width, height, h5_id}
2. anns        : list of {ann_id, category_id, image_id, box, h5_id} new {kld_label, pos_id,
                          gt_pos_all, bbx_reg, bbx_reg_all}
3. sentences   : list of {sent_id, sen_id, category_id, tokens, first_idx, idx, h5_id} new {ref_id}
4. real_sens   : list of {sen_id, image_id, sent_ids, tokens, raw, idx, h5_id}
4: word_to_ix  : word->ix
5: cat_to_ix   : cat->ix
6: label_length: L
7: sen_label_length: SL
no h5_id
"""

import os
import sys
import json
import argparse
import string
import os.path as osp
import operator
import random
import math

import h5py
import numpy as np
import xml.dom.minidom
import scipy.io as sio
import PIL.Image as Image

CATEGORY_TO_ID = {'people':0, 'bodyparts':1, 'clothing':2, 'animals':3, 'vehicles':4, 'instruments':5,
                  'scene':6, 'other':7, 'notvisual':8}

def all_lower(L):

    return [s.lower() for s in L]

def load_img_id_list(file_list):

    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')

    return img_list

def xywh_to_xyxy(box):

    x, y, w, h = box

    return [x, y, x+w-1, y+h-1]

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

def cc(box):

    # box: [xmin, ymin, xmax, ymax]
    x = (box[2]-box[0]) / 2.0 + box[0]
    y = (box[3]-box[1]) / 2.0 + box[1]
    c = [x,y]

    return c #c: [x,y]

def calc_bbx_reg(box_gt, box_pos):
    #box: [xmin, ymin, xmax, ymax]

    x_gt, y_gt = cc(box_gt)
    x_pos, y_pos = cc(box_pos)
    w_pos = box_pos[2] - box_pos[0]
    h_pos = box_pos[3] - box_pos[1]
    w_gt = box_gt[2] - box_gt[0]
    h_gt = box_gt[3] - box_gt[1]

    tx_gt = (x_gt-x_pos) * 1.0 / w_pos
    ty_gt = (y_gt-y_pos) * 1.0 / h_pos
    tw_gt = math.log(w_gt * 1.0 / w_pos)
    th_gt = math.log(h_gt * 1.0 / h_pos)

    t_gt = [tx_gt, ty_gt, tw_gt, th_gt]

    return t_gt

def merge_box(box1, box2):
    # box : (xmin, ymin, xmax, ymax)

    box = [0.0, 0.0, 0.0, 0.0]
    box[0] = box1[0] if box1[0]<box2[0] else box2[0]
    box[1] = box1[1] if box1[1]<box2[1] else box2[1]
    box[2] = box1[2] if box1[2]>box2[2] else box2[2]
    box[3] = box1[3] if box1[3]>box2[3] else box2[3]

    return box

def load_ann_dict(ann_name):

    DOMTree = xml.dom.minidom.parse(ann_name)
    collection = DOMTree.documentElement
    objects = collection.getElementsByTagName('object')

    # solve mulitiple regions
    flag = True
    gts = {}

    for obi in objects:
        name = obi.getElementsByTagName('name')
        xmin = obi.getElementsByTagName('xmin')
        ymin = obi.getElementsByTagName('ymin')
        xmax = obi.getElementsByTagName('xmax')
        ymax = obi.getElementsByTagName('ymax')
        if len(xmin) == 0:
            continue
        for name_i in range(len(name)):
            gt = {}
            phrase_id = int(name[name_i].childNodes[0].data)
            gt[phrase_id] = {}
            gt[phrase_id]['box'] = []
            # gt.append(int(name[name_i].childNodes[0].data))
            gt[phrase_id]['box'].append(int(xmin[0].childNodes[0].data))
            gt[phrase_id]['box'].append(int(ymin[0].childNodes[0].data))
            gt[phrase_id]['box'].append(int(xmax[0].childNodes[0].data))
            gt[phrase_id]['box'].append(int(ymax[0].childNodes[0].data))
            gt[phrase_id]['num_object'] = 1
            if phrase_id in gts:
                #merge
                gts[phrase_id]['box'] = merge_box(gts[phrase_id]['box'], gt[phrase_id]['box'])
                gts[phrase_id]['num_object'] += 1
                flag = False
            if flag is True:
                gts.update(gt)
            flag = True

    for phrase_id in gts:
        x1, y1, x2, y2 = gts[phrase_id]['box']
        gts[phrase_id]['box'] = [x1, y1, x2-x1+1, y2-y1+1]

    if gts == {}:
        gts = None
    """
    print ("gts:")
    print (gts)
    print ("-----------------------------------------------")
    """
    return gts

def load_sen_dict(sen_name):
   # sent{idx:{sen_idx, category_id, tokens, first_idx, phrase_id}}
   # sen {sen_idx:{tokens, raw}}

    sen_data = sio.loadmat(sen_name)['sentenceData']
    sent, sen = {}, {}

    for sen_i in range(len(sen_data)):
        # clean data: no phrase
        if sen_data[sen_i]['phrases'][0].tolist() == []:
            continue

        sen_idx = sen_i + 1
        sen[sen_idx] = {}

        sen_raw = sen_data[sen_i]['sentence'][0][0]
        sen_token = sen_raw.split()
        sen[sen_idx]['raw'] = sen_raw
        sen[sen_idx]['tokens'] = all_lower(sen_token)

        phrase_data = sen_data[sen_i]['phrases'][0][0]
        phrase_id_data = sen_data[sen_i]['phraseID'][0][0]
        phrase_category_data = sen_data[sen_i]['phraseType'][0][0]
        phrase_first_idx_data = sen_data[sen_i]['phraseFirstWordIdx'][0]
        for phrase_i in range(len(phrase_data)):
            phrase_idx = sen_idx*100 + phrase_i + 1
            sent[phrase_idx] = {}

            phrase_token = []
            for word_i in range(len(phrase_data[phrase_i])):
                phrase_token.append(all_lower(phrase_data[phrase_i][word_i][0])[0])
            phrase_first_idx = int(phrase_first_idx_data[phrase_i][0])
            phrase_category = phrase_category_data[phrase_i][0][0][0]
            phrase_id = int(phrase_id_data[phrase_i][0])

            sent[phrase_idx]['sen_idx'] = sen_idx
            sent[phrase_idx]['tokens'] = phrase_token
            sent[phrase_idx]['first_idx'] = phrase_first_idx
            sent[phrase_idx]['category_id'] = CATEGORY_TO_ID[phrase_category]
            sent[phrase_idx]['phrase_id'] = phrase_id
    """
    print ("sen:")
    print (sen)
    print ("-----------------------------------------------")
    print ("sent:")
    print (sent)
    print ("-----------------------------------------------")
    """
    return sen, sent

def get_det_bbox_ann(gts, det_bbox):
    """
    gts     : { phrase_id{box, num_objects} } (xmin, ymin, w, h)
    det_bbox: (n, 4) (xmin, ymin, xmax, ymax)

    Return:
    det_bbox_ann : { phrase_id{kld_label, pos_id, gt_pos_all, bbx_reg, bbx_reg_all} }
    """
    num_props, threshold, eps = 100, 0.5, 1e-10

    det_bbox_ann = {}
    for phrase_id in gts:
        gt_box = xywh_to_xyxy(gts[phrase_id]['box'])

        kld_label = np.zeros(num_props)
        pos_id = -1
        gt_pos_all = []
        bbx_reg = [0.0] * 4
        bbx_reg_all = []

        iou_max = threshold
        pos_id_index = -1
        gt_pos_all_index = -1
        for i, det_box in enumerate(det_bbox):
            iou = calc_iou(gt_box, det_box)
            if iou >= threshold:
                kld_label[i] = iou
                gt_pos_all.append(i)
                gt_pos_all_index = gt_pos_all_index + 1
                bbx_reg_all.append(calc_bbx_reg(gt_box, det_box))

                if iou >= iou_max:
                    iou_max = iou
                    pos_id = i
                    pos_id_index = gt_pos_all_index

        kld_label = list(kld_label / (np.sum(kld_label) + eps))
        if pos_id != -1:
            bbx_reg = bbx_reg_all[pos_id_index]

        det_bbox_ann[phrase_id] = {}
        det_bbox_ann[phrase_id]['kld_label'] = kld_label
        det_bbox_ann[phrase_id]['pos_id'] = pos_id
        det_bbox_ann[phrase_id]['gt_pos_all'] = gt_pos_all
        det_bbox_ann[phrase_id]['bbx_reg'] = bbx_reg
        det_bbox_ann[phrase_id]['bbx_reg_all'] = bbx_reg_all

    return det_bbox_ann


def print_data(data_refs, data_images, data_anns, data_sentences, data_real_sens):

    print ('data_refs:')
    for i in data_refs:
        print(i)

    print ('data_images:')
    for i in data_images:
        print(i)

    print ('data_anns:')
    for i in data_anns:
        print(i)

    print ('data_sentences:')
    for i in data_sentences:
        print(i)

    print ('data_real_sens:')
    for i in data_real_sens:
        print(i)

def build_vocab(sent, sen, params):
    """
    Our vocabulary will add flickr30k categories, <UNK>, PAD, BOS, EOS
    """
    # remove bad words, and return final sentences (sent_id -> final)
    count_thr = params['word_count_threshold']

    # count up the number of words
    word2count = {}
    for sen_i in sen:
        tokens = sen_i['tokens']
        for wd in tokens:
            word2count[wd] = word2count.get(wd, 0) + 1

    # print some stats
    total_words = sum(word2count.values())
    bad_words = [wd for wd, n in word2count.items() if n <= count_thr]
    good_words= [wd for wd, n in word2count.items() if n > count_thr]
    bad_count = sum([word2count[wd] for wd in bad_words])
    print('number of good words: %d' % len(good_words))
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(word2count), len(bad_words)*100.0/len(word2count)))
    print('number of UNKs in sentences: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
    vocab = good_words

    # add category words
    for cat_name in CATEGORY_TO_ID:
        if wd not in word2count or word2count[wd] <= count_thr:
            word2count[wd] = 1e5
            vocab.append(wd)
            print('category word [%s] added to vocab.' % wd)

    # add UNK, BOS, EOS, PAD
    if bad_count > 0:
        vocab.append('<UNK>')
    vocab.append('<BOS>')
    vocab.append('<EOS>')
    vocab.insert(0, '<PAD>')  # add PAD to the very front

    # lets now produce final tokens
    for i, sen_i in enumerate(sen):
        tokens = sen_i['tokens']
        final = [wd if word2count[wd] > count_thr else '<UNK>' for wd in tokens]
        sen[i]['tokens'] = final
    for i, sent_i in enumerate(sent):
        tokens = sent_i['tokens']
        final = [wd if word2count[wd] > count_thr else '<UNK>' for wd in tokens]
        sent[i]['tokens'] = final

    return vocab, sent, sen

def check_sent_length(sent_to_final):

    sent_lengths = {}
    for sent in sent_to_final:
        tokens = sent['tokens']
        nw = len(tokens)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length of sentence in raw data is %d' % max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    acc = 0  # accumulative distribution
    for i in range(max_len+1):
        acc += sent_lengths.get(i, 0)
        print('%2d: %10d %.3f%% %.3f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len, acc*100.0/sum_len))

def encode_captions(sentences, wtoi, max_length):
    """
    sentences = [{sent_id, tokens, h5_id}]
    """

    M = len(sentences)
    L = np.zeros((M, max_length), dtype=np.int32)
    for i, sent in enumerate(sentences):
        h5_id = sent['h5_id']
        assert h5_id == i
        tokens = sent['tokens']
        for j, w in enumerate(tokens):
            if j < max_length:
                L[h5_id, j] = wtoi[w]

    return L

def main(params):

    data_root, dataset = params['data_root'], params['dataset']

    # max_length
    if params['max_length'] == None:
        params['max_length'] = 15
        params['max_sen_length'] = 30

    # mkdir and write json file
    if not osp.isdir(osp.join('cache/prepro', dataset)):
        os.makedirs(osp.join('cache/prepro', dataset))

    # create flickr30k structure
    root_path = osp.join(data_root, dataset)
    train_val_lst_path = osp.join(root_path, 'flickr30k_train_val.lst')
    test_lst_path = osp.join(root_path, 'flickr30k_test.lst')
    img_path = osp.join(root_path, 'flickr30k-images')
    ann_path = osp.join(root_path, 'Flickr30kEntities/Annotations')
    sen_path = osp.join(root_path, 'Flickr30kEntities/Sentences_mat')
    det_path = osp.join(root_path, 'feature/bottom-up-feats')

    train_val_lst = load_img_id_list(train_val_lst_path)
    test_lst = load_img_id_list(test_lst_path)
    file_lst = np.concatenate((train_val_lst, test_lst), axis=0)

    data_refs, data_images, data_anns, data_sentences, data_real_sens = [], [], [], [], []
    sent_id_lst = random.sample(range(1, 10000000), 9999999)
    sen_id_lst = random.sample(range(1, 1000000), 999999)

    image_h5_id, anns_h5_id, sent_h5_id, sen_h5_id = 0, 0, 0, 0
    for i, file_id in enumerate(file_lst):
        # print (file_id)
        # img
        img_name = osp.join(img_path, str(file_id)+'.jpg')
        image = Image.open(img_name)
        width, height = image.size[:2]

        # acquire gt annotation : ann{phraseid:box, num_objects}
        ann_name = osp.join(ann_path, str(file_id)+'.xml')
        gts = load_ann_dict(ann_name)
        if gts is None:
            continue    # clean data

        # acquire sentence and real_sen information :
        #   sent{idx:{sen_idx, category_id, tokens, first_idx, phrase_id}}
        #   sen {sen_idx:{tokens, raw}}
        sen_name = osp.join(sen_path, str(file_id)+'.mat')
        sen, sent = load_sen_dict(sen_name)

        # acquire det_bbox
        det_name = osp.join(det_path, str(file_id)+'.jpg.npz')
        det_bbox = np.load(det_name)['bbox'].tolist()   # (xmin, ymin, xmax, ymax)

        # acquire ann_det_bbox{phrase_id:{kld_label, pos_id, gt_pos_all, bbx_reg, bbx_reg_all}}
        # 2. anns : list of {ann_id, category_id, image_id, box, h5_id}
        # new {kld_label, pos_id, gt_pos_all, bbx_reg, bbx_reg_all}
        det_bbox_ann = get_det_bbox_ann(gts, det_bbox)

        """
        0. refs        : list of {ref_id, ann_id, box, num_object, image_id, category_id, sent_ids}
        1. images      : list of {image_id, sen_ids, ref_ids, ann_ids, det_bbox, file_name, width,
                                  height, h5_id}
        2. anns        : list of {ann_id, category_id, image_id, box, h5_id} new {kld_label, pos_id,
                                  gt_pos_all, bbx_reg, bbx_reg_all}
        3. sentences   : list of {sent_id, sen_id, category_id, tokens, first_idx, idx, h5_id} new {ref_id}
        4. real_sens   : list of {sen_id, image_id, sent_ids, tokens, raw, idx, h5_id}
        """

        # prepare data
        # intialize sent_id, sen_id
        for sent_idx in sent:
            sent[sent_idx]['sent_id'] = sent_id_lst[sent_h5_id]
            sent[sent_idx]['h5_id'] = sent_h5_id
            sent_h5_id = sent_h5_id + 1
        for sen_idx in sen:
            sen[sen_idx]['sen_id'] = sen_id_lst[sen_h5_id]
            sen[sen_idx]['h5_id'] = sen_h5_id
            sen_h5_id = sen_h5_id + 1

        # prepare refs and anns
        ref_ids = []
        for phrase_id in gts:
            gts_i = gts[phrase_id]
            sent_ids = []
            category_id = 0
            for sent_idx in sent:
                sent_i = sent[sent_idx]
                if phrase_id == sent_i['phrase_id']:
                    sent_ids.append(sent_i['sent_id'])
                    category_id = sent_i['category_id']

            if sent_ids == []:
                print (file_id, phrase_id)
                pass
            else:
                ref_ids += [phrase_id]
                data_refs += [{'ref_id':phrase_id, 'ann_id':phrase_id, 'box':gts_i['box'],
                               'num_object':gts_i['num_object'], 'image_id':file_id,
                               'category_id':category_id, 'sent_ids':sent_ids}]

            det_bbox_ann_i = det_bbox_ann[phrase_id]
            data_anns += [{'ann_id':phrase_id, 'category_id':category_id, 'image_id':file_id,
                           'box':gts_i['box'], 'kld_label':det_bbox_ann_i['kld_label'],
                           'pos_id':det_bbox_ann_i['pos_id'], 'gt_pos_all':det_bbox_ann_i['gt_pos_all'],
                           'bbx_reg':det_bbox_ann_i['bbx_reg'], 'bbx_reg_all':det_bbox_ann_i['bbx_reg_all'],
                           'h5_id':anns_h5_id}]
            anns_h5_id = anns_h5_id + 1

        # prepare images
        sen_ids = [sen[sen_idx]['sen_id'] for sen_idx in sen]
        ann_ids = ref_ids

        data_images += [ {'image_id':file_id, 'sen_ids':sen_ids, 'ref_ids':ref_ids, 'ann_ids':ref_ids,
                          'det_bbox':det_bbox, 'file_name':str(file_id)+'.jpg', 'width':width,
                          'height':height, 'h5_id':image_h5_id} ]
        image_h5_id = image_h5_id + 1

        # prepare sentences
        for sent_idx in sent:
            sent_i = sent[sent_idx]
            sen_id = sen[sent_i['sen_idx']]['sen_id']

            data_sentences += [{'sent_id':sent_i['sent_id'], 'ref_id':sent_i['phrase_id'], 'sen_id':sen_id,
                                'category_id':sent_i['category_id'], 'tokens':sent_i['tokens'],
                                'first_idx':sent_i['first_idx'], 'idx':sent_idx, 'h5_id':sent_i['h5_id']}]

        # prepare real_sens
        for sen_idx in sen:
            sen_i = sen[sen_idx]
            sent_ids = []
            for sent_idx in sent:
                sent_i = sent[sent_idx]
                if sen_idx == sent_i['sen_idx']:
                    sent_ids.append(sent_i['sent_id'])

            data_real_sens += [{'sen_id':sen_i['sen_id'], 'image_id':file_id, 'sent_ids':sent_ids,
                                'tokens':sen_i['tokens'], 'raw':sen_i['raw'], 'idx':sen_idx,
                                'h5_id':sen_i['h5_id']}]

        if (i+1) % 5000 == 0:
            print ('iteratrion : %d / %d')%(i+1, 31783)

        # print_data(data_refs, data_images, data_anns, data_sentences, data_real_sens)
        # break
    vocab, data_sentences, data_real_sens = build_vocab(data_sentences, data_real_sens, params)
    itow = {i: w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    print ('sent_length:')
    check_sent_length(data_sentences)
    print ('sen_length:')
    check_sent_length(data_real_sens)

    # write json
    json.dump({'refs': data_refs,
               'images': data_images,
               'anns': data_anns,
               'sentences': data_sentences,
               'real_sen': data_real_sens,
               'word_to_ix': wtoi,
               'cat_to_ix': CATEGORY_TO_ID,
               'label_length': params['max_length'],
               'label_sen_length': params['max_sen_length'],},
              open(osp.join('cache/prepro/', dataset, params['output_json']), 'w'))
    print('%s written.' % osp.join('cache/prepro', params['output_json']))

    # write h5 file which contains /sentences
    f = h5py.File(osp.join('cache/prepro', dataset, params['output_h5']), 'w')
    L = encode_captions(data_sentences, wtoi, params['max_length'])
    L_SEN = encode_captions(data_real_sens, wtoi, params['max_sen_length'])
    f.create_dataset("labels", dtype='int32', data=L)
    f.create_dataset("labels_sen", dtype='int32', data=L_SEN)
    f.close()
    print('%s writtern.' % osp.join('cache/prepro', params['output_h5']))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
    parser.add_argument('--data_root', default='data', type=str, help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', default='flickr30k', type=str, help='flickr30k')
    # parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
    parser.add_argument('--max_length', type=int, help='max length of a caption')  # refcoco 10, refclef 10, refcocog 20
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')

    # argparse
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))

    # call main
    main(params)

