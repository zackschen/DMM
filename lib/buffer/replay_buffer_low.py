from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

import numpy as np
import json
from collections import OrderedDict

class Buffer(object):

    def __init__(self, buffer_size, _buffer=None, last_loss=0):

        if _buffer is None:
            self.buffer = OrderedDict()
            self.size = buffer_size
            self.last_loss = 0
        else:
            self.buffer = OrderedDict(sorted(_buffer.items(), key=lambda obj:obj[1]['loss'], reverse=False))
            self.size = buffer_size
            self.last_loss = last_loss


    def add(self, other_Buffer):

        self.buffer.update(other_Buffer.buffer)
        self.buffer = OrderedDict(sorted(self.buffer.items(), key=lambda obj:obj[1]['loss'], reverse=False))

        self.size = self.size + other_Buffer.size
        self.last_loss = min(self.last_loss, other_Buffer.last_loss)


    def decrease(self, size):

        self.size = size
        while self.full():
            self.buffer.popitem(last=True)

        # update last_loss
        buffer_list = list(self.buffer.values())
        if len(buffer_list) > 0:
            self.last_loss = buffer_list[-1]['loss']


    def clear(self, size_flag=False):

        self.buffer = OrderedDict()
        self.last_loss = 0
        if size_flag is False:
            self.size = 0


    def full(self):

        return len(self.buffer) > self.size


    def empty(self):

        return len(self.buffer) == 0


    def transform_loader_list(self, sample_number=-1):

        # get whole buffer
        if sample_number == -1:
            sample_number = self.size

        sample_number = int(sample_number)
        # buffer_list : [{ref_id, sent_id}]
        buffer_list = []
        buffer_values = list(self.buffer.values())
        sample_number = min(sample_number,len(buffer_values))
        sample_buffer = random.sample(buffer_values, sample_number)
        buffer_list += [buffer_value['id'] for buffer_value in sample_buffer]

        return buffer_list


    def update(self, ref_ids, sent_ids, loss, sub_weights = None):

        # add
        if self.full() and (self.last_loss >= max(loss)):
            return
        else:
            for i, sent_id in enumerate(sent_ids):
                self.buffer[sent_id] = {}
                self.buffer[sent_id]['id'] = {}
                self.buffer[sent_id]['id']['sent_id'] = sent_id
                self.buffer[sent_id]['id']['ref_id'] = ref_ids[i]
                self.buffer[sent_id]['loss'] = loss[i]
                self.buffer[sent_id]['sub_weights'] = sub_weights[i]

        # del
        self.buffer = OrderedDict(sorted(self.buffer.items(), key=lambda obj:obj[1]['loss'], reverse=False))
        while self.full():
            self.buffer.popitem(last=True)

        # update last_loss
        buffer_list = list(self.buffer.values())
        if len(buffer_list) > 0:
            self.last_loss = buffer_list[-1]['loss']

        return


    def update_loss(self, sent_ids, loss):

        for i, sent_id in enumerate(sent_ids):
            self.buffer[sent_id]['loss'] = loss[i]

        self.buffer = OrderedDict(sorted(self.buffer.items(), key=lambda obj:obj[1]['loss'], reverse=False))

        # update last_loss
        buffer_list = list(self.buffer.values())
        if len(buffer_list) > 0:
            self.last_loss = buffer_list[-1]['loss']


    def _print(self, sample_number=5):
        sample_number = int(sample_number)
        # buffer_list : [{ref_id, sent_id}]
        buffer_list = []
        buffer_values = list(self.buffer.values())
        sample_number = min(sample_number,len(buffer_values))
        sample_list = random.sample(buffer_values, sample_number)
        print ('sent_ids : ', sample_list)
        sample_list = list(self.buffer.keys())
        print ('len, ', len(sample_list))
        print ('buffer_size, ', self.size)

    def save(self, save_dir):

        save_pool = {}
        save_pool['buffer'] = self.buffer
        save_pool['last_loss'] = self.last_loss
        save_pool['buffer_size'] = self.size

        with open(save_dir, 'w') as f:
            json.dump(save_pool, f)

"""
buffer_pool:{category:{size, ratio}}
"""

class BufferPool(object):

    def __init__(self, buffer_size):

        self.buffer_pool = {}
        self.buffer_size = buffer_size
        self.data_size = 0
        self.num_category = 0
        self.eps = 1e-8
        self.last_buffer = Buffer(0)


    def update_ratio(self):

        last_buffer_size = 0
        for i, category in enumerate(self.buffer_pool.keys()):
            category_size = self.buffer_pool[category]['data_size']
            ratio = category_size / (self.data_size + self.eps)
            buffer_size = int(round(self.buffer_size * ratio, 0))

            self.buffer_pool[category]['ratio'] = ratio
            if i <  (self.num_category - 1):
                last_buffer_size += buffer_size
                self.buffer_pool[category]['buffer_size'] = buffer_size
            else:
                buffer_size = self.buffer_size - last_buffer_size
                self.buffer_pool[category]['buffer_size'] = buffer_size


    def update_buffer(self):

        for category in self.buffer_pool.keys():
            buffer_size = self.buffer_pool[category]['buffer_size']
            self.buffer_pool[category]['buffer'].decrease(buffer_size)


    def add(self, category, size):

        self.data_size += size
        self.buffer_pool[category] = {}
        self.buffer_pool[category]['data_size'] = size
        self.num_category = len(self.buffer_pool)
        self.update_ratio()


    def add_buffer(self, category, buffer):

        self.buffer_pool[category]['buffer'] = buffer
        self.update_buffer()


    def get_buffer_size(self, category):

        return self.buffer_pool[category]['buffer_size']


    def get_last_buffer(self):

        self.last_buffer.clear()
        for category in self.buffer_pool.keys():
            self.last_buffer.add(self.buffer_pool[category]['buffer'])

        return self.last_buffer


    def _print(self):
        print ('buffer_size [%d], data_size [%d], num_category [%d]' % (self.buffer_size, self.data_size,
                                                                        self.num_category))
        print ('----------------------------------------------------------------------------------------')
        for category in self.buffer_pool.keys():
            buffer_size = self.buffer_pool[category]['buffer_size']
            data_size = self.buffer_pool[category]['data_size']
            ratio = self.buffer_pool[category]['ratio']
            real_buffer_size = self.buffer_pool[category]['buffer'].size
            last_loss = self.buffer_pool[category]['buffer'].last_loss
            print ('%s : buffer_size [%d], data_size [%d], ratio [%.4f], real_buffer_size [%d]' %
                   (category, buffer_size, data_size, ratio, real_buffer_size))
            self.buffer_pool[category]['buffer']._print()
        print ('----------------------------------------------------------------------------------------')


    def save(self, save_dir):

        save_pool = {}
        for category in self.buffer_pool.keys():
            save_pool[category] = {}
            save_pool[category]['buffer'] = self.buffer_pool[category]['buffer'].buffer
            save_pool[category]['last_loss'] = self.buffer_pool[category]['buffer'].last_loss
            save_pool[category]['buffer_size'] = self.buffer_pool[category]['buffer_size']
            save_pool[category]['data_size'] = self.buffer_pool[category]['data_size']
            save_pool[category]['ratio'] = self.buffer_pool[category]['ratio']

        with open(save_dir, 'w') as f:
            json.dump(save_pool, f)

    def load(self, buffer_pool_path):

        with open(buffer_pool_path, 'r') as f:
            load_pool = json.load(f)

        for category in load_pool.keys():
            self.data_size += load_pool[category]['data_size']
            self.buffer_pool[category] = {}
            self.buffer_pool[category]['buffer'] = Buffer(load_pool[category]['buffer_size'],
                                                          load_pool[category]['buffer'],
                                                          load_pool[category]['last_loss'])
            self.buffer_pool[category]['buffer_size'] = load_pool[category]['buffer_size']
            self.buffer_pool[category]['data_size'] = load_pool[category]['data_size']
            self.buffer_pool[category]['ratio'] = load_pool[category]['ratio']

        self.num_category = len(self.buffer_pool)
        self.update_ratio()
        self.update_buffer()

