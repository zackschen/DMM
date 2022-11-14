"""
split_json
images  : list of {image_id, split}
"""

import os.path as osp
import json

import numpy as np

def load_img_id_list(file_list):

    img_list = []
    with open(file_list) as fin:
        for img_id in fin.readlines():
            img_list.append(int(img_id.strip()))
    img_list = np.array(img_list).astype('int')

    return img_list

if __name__ == '__main__':

    root_path = 'data/flickr30k'
    train_val_lst_path = osp.join(root_path, 'flickr30k_train_val.lst')
    test_lst_path = osp.join(root_path, 'flickr30k_test.lst')

    train_val_lst = load_img_id_list(train_val_lst_path)
    test_lst = load_img_id_list(test_lst_path)

    data_split = []
    split = 'train'
    for i, file_id in enumerate(train_val_lst):
        data_split += [{'image_id':file_id, 'split':split}]
        if (i+1) % 500 == 0:
            print ('iteratrion : %d / %d')%(i+1, len(train_val_lst))
            print (data_split[-1])

    split = 'test'
    for i, file_id in enumerate(test_lst):
        data_split += [{'image_id':file_id, 'split':split}]
        if (i+1) % 500 == 0:
            print ('iteratrion : %d / %d')%(i+1, len(test_lst))
            print (data_split[-1])

    # write json
    json.dump(data_split, open(osp.join('cache/prepro', 'flickr30k', 'data_split.json'), 'w'))
