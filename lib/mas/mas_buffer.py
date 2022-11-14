from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import sys

from mas.mas_based_buffer_training import *

import torch



def sanitycheck(model):

    for name, param in model.named_parameters():

        print (name)
        if param in model.reg_params:

            reg_param = model.reg_params.get(param)
            omega = reg_param.get('omega')

            print ('omega max is', omega.max())
            print ('omega min is', omega.min())
            print ('omega mean is', omega.mean())


def MAS(criterion, att_crit, current_buffer, last_buffer, loader, pre_split, split, opt, model_dir,
        f_model_dir, previous_category, category, log, log_category, freeze_layers=[], b1=False, reg_lambda=1,
        norm='L2', rel_buffer=None, loc_buffer=None, rel_last_buffer=None, loc_last_buffer=None,
        sub_layers=[], loc_layers=[], rel_layers=[]):

    # previous_model_path = osp.join(model_dir, previous_category, opt['id'] + '_best_model.pth.tar')
    previous_model_path = osp.join(f_model_dir, previous_category, opt['f_id'] + '_best_model.pth.tar')
    model_ft = torch.load(previous_model_path)

    """
    # update omega value
    if b1:
        update_batch_size = 1
    else:
        update_batch_size = batch_size
    """

    model_ft = update_weights_params(criterion, att_crit, loader, pre_split, opt, model_ft, freeze_layers,
                                     norm, sub_layers, loc_layers, rel_layers)

    #set the lambda value gor the MAS
    model_ft.reg_params['lambda'] = reg_lambda

    #check the computed omega
    sanitycheck(model_ft)

    if opt['gpuid'] >= 0:
        model_ft.cuda()
        criterion.cuda()
        att_crit.cuda()

    #call the MAS optimizer
    optimizer_ft = Weight_Regularized_Adam(model_ft.parameters(),
                                           lr=opt['learning_rate'],
                                           betas=(opt['optim_alpha'], opt['optim_beta']),
                                           eps=opt['optim_epsilon'])

    model_category_dir = osp.join(model_dir, category)
    if not osp.isdir(model_category_dir): os.makedirs(model_category_dir)

    resume = osp.join(model_category_dir, opt['id'] + '.pth.tar')
    #trian model
    #this training function passes the reg params to the optimizer to be used for
    #penalizing changes on important params
    model_ft = train_model(model_ft, criterion, att_crit, current_buffer, last_buffer, optimizer_ft, loader,
                           split, opt, model_dir, category, log, log_category, resume,
                           rel_buffer=rel_buffer, loc_buffer=loc_buffer,
                           rel_last_buffer=rel_last_buffer, loc_last_buffer=loc_last_buffer)

    return model_ft


def MAS_Omega_Accumulation(criterion, att_crit, current_buffer, last_buffer, loader, pre_split, split, opt,
                           model_dir, previous_category, category, log, log_category, freeze_layers=[],
                           b1=False, reg_lambda=1, norm='L2', previous_ratio=0.5,
                           rel_buffer=None, loc_buffer=None, rel_last_buffer=None, loc_last_buffer=None,
                           sub_layers=[], loc_layers=[], rel_layers=[]):

    """
    In case of accumulating omega for the different tasks in the sequence, baisically to mimic the setup of
        standard methods where the regularizer is computed on the training set. Note that this doesn't
        consider our adaptation
    """

    previous_model_path = osp.join(model_dir, previous_category, opt['id'] + '_best_model.pth.tar')
    model_ft = torch.load(previous_model_path)

    """
    # update omega value
    if b1:
        update_batch_size = 1
    else:
        update_batch_size = batch_size
    """

    if opt['mas_weighted_decay']:
        model_ft = weighted_decay_accumulate_MAS_weights(criterion, att_crit, loader, split, opt, model_ft,
                                                         freeze_layers, norm, previous_ratio)
    else:
        model_ft = accumulate_MAS_weights(criterion, att_crit, loader, pre_split, opt, model_ft,
                                          freeze_layers, norm, sub_layers, loc_layers, rel_layers)

    #set the lambda value gor the MAS
    model_ft.reg_params['lambda'] = reg_lambda

    #check the computed omega
    sanitycheck(model_ft)

    if opt['gpuid'] >= 0:
        model_ft.cuda()
        criterion.cuda()

    #call the MAS optimizer
    optimizer_ft = Weight_Regularized_Adam(model_ft.parameters(),
                                           lr=opt['learning_rate'],
                                           betas=(opt['optim_alpha'], opt['optim_beta']),
                                           eps=opt['optim_epsilon'])

    model_category_dir = osp.join(model_dir, category)
    if not osp.isdir(model_category_dir): os.makedirs(model_category_dir)

    resume = osp.join(model_category_dir, opt['id'] + '.pth.tar')
    #trian model
    #this training function passes the reg params to the optimizer to be used for
    #penalizing changes on important params
    model_ft = train_model(model_ft, criterion, att_crit, current_buffer, last_buffer, optimizer_ft,
                           loader, split, opt, model_dir, category, log, log_category, resume,
                           rel_buffer=rel_buffer, loc_buffer=loc_buffer,
                           rel_last_buffer=rel_last_buffer, loc_last_buffer=loc_last_buffer)

    return model_ft


def update_weights_params(criterion, att_crit, loader, split, opt, model_ft, freeze_layers=[], norm='L2',
                          sub_layers=[], loc_layers=[], rel_layers=[]):
    """
    update the importance weights based on the samples included in the reg_set.Assume starting from
    zero omega
    model_ft: the model trained on the previous task
    """

    #initialize the importance params, omega, to zero
    reg_params = initialize_reg_params(model_ft, freeze_layers)
    model_ft.reg_params = reg_params

    #define he importance weight optimizer. Actually it is only one step. It can be integrated at the end of\
    #the first task
    optimizer_ft = MAS_Omega_update(model_ft.parameters(),
                                    lr=opt['learning_rate'],
                                    betas=(opt['optim_alpha'], opt['optim_beta']),
                                    eps=opt['optim_epsilon'])

    if norm == 'L2':
        print ('***************************MAS with L2 norm*****************************')
        #compute the importance params
        model_ft = compute_importance_l2(criterion, att_crit, model_ft, optimizer_ft, loader, split, opt)
    else:
        if norm == 'vector':
            optimizer_ft = MAS_Omega_Vector_Grad_update(model_ft.parameters(),
                                                        lr=opt['learing_rate'],
                                                        betas=(opt['optim_alpha'], opt['optim_beta']),
                                                        eps=opt['optim_epsilon'])
            model_ft = compute_importance_gradient_vector(criterion, att_crit, model_ft, optimizer_ft, loader,
                                                          split, opt)
        else:
            model_ft = compute_importance(criterion, att_crit, model_ft, optimizer_ft, loader, split, opt)

    reg_params = weighted_module_reg_params(model_ft, opt, sub_layers, loc_layers, rel_layers)
    model_ft.reg_params = reg_params

    return model_ft


def accumulate_MAS_weights(criterion, att_crit, loader, split, opt, model_ft, freeze_layers=[], norm='L2',
                           sub_layers=[], loc_layers=[], rel_layers=[]):
    """
    accumulate the importance params: stores the previously computed omega, compute omega on the last
        previous task and accumulate omega resulting on importance params for all the previous tasks
    model_ft: the model trained on the previous task
    """

    #initialize the importance params, omega, to zero
    reg_params = initialize_store_reg_params(model_ft, freeze_layers)
    model_ft.reg_params = reg_params

    #define he importance weight optimizer. Actually it is only one step. It can be integrated at the end of\
    #the first task
    optimizer_ft = MAS_Omega_update(model_ft.parameters(),
                                    lr=opt['learning_rate'],
                                    betas=(opt['optim_alpha'], opt['optim_beta']),
                                    eps=opt['optim_epsilon'])

    if norm == 'L2':
        print ('***************************MAS with L2 norm*****************************')
        #compute the importance params
        model_ft = compute_importance_l2(criterion, att_crit, model_ft, optimizer_ft, loader, split, opt)
    else:
        model_ft = compute_importance(criterion, att_crit, model_ft, optimizer_ft, loader, split, opt)

    # accumulate the new importance params with the previously stored ones (previous omega)
    reg_params = accumulate_reg_params(model_ft, freeze_layers)
    model_ft.reg_params = reg_params
    reg_params = weighted_module_reg_params(model_ft, opt, sub_layers, loc_layers, rel_layers)
    model_ft.reg_params = reg_params
    sanitycheck(model_ft)

    return model_ft


def weighted_decay_accumulate_MAS_weights(criterion, att_crit, loader, split, opt, model_ft,
                                          freeze_layers=[], norm='L2', previous_ratio=0.5):
    """
    accumulate the importance params: stores the previously computed omega, compute omega on the last
        previous task and accumulate omega resulting on importance params for all the previous tasks
    model_ft: the model trained on the previous task
    """

    #initialize the importance params, omega, to zero
    reg_params = initialize_store_reg_params(model_ft, freeze_layers)
    model_ft.reg_params = reg_params

    #define he importance weight optimizer. Actually it is only one step. It can be integrated at the end of\
    #the first task
    optimizer_ft = MAS_Omega_update(model_ft.parameters(),
                                    lr=opt['learning_rate'],
                                    betas=(opt['optim_alpha'], opt['optim_beta']),
                                    eps=opt['optim_epsilon'])

    if norm == 'L2':
        print ('***************************MAS with L2 norm*****************************')
        #compute the importance params
        model_ft = compute_importance_l2(criterion, att_crit, model_ft, optimizer_ft, loader, split, opt)
    else:
        model_ft = compute_importance(criterion, att_crit, model_ft, optimizer_ft, loader, split, opt)

    # accumulate the new importance params with the previously stored ones (previous omega)
    reg_params = weighted_decay_accumulate_reg_params(model_ft, previous_ratio, freeze_layers)
    model_ft.reg_params = reg_params
    sanitycheck(model_ft)

    return model_ft
