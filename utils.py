# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:48:15 2019

@author: T_ESTIENNE
"""
import os
import keras.models as models
import numpy as np

# My package
from miccai_public import diffeomorphicTransformer as Transformer
from miccai_public import losses
from miccai_public import blocks

main_path = os.path.abspath(__file__)
n = main_path.find('Python')
if n > 0:
    main_path = main_path[:n] + 'Python/'
else:
    n = main_path.find('workspace')
    main_path = main_path[:n]
    print(main_path)


def choice(models):

    print('***** Models found : *****')
    for i, model in enumerate(models):
        print('[{}/{}] : {}'.format(i, len(models) - 1, model))

    print('***** Models selection ? *****')
    model_selected = input()

    return model_selected


def search_model(args):
    '''
        This function search in the list of the different models witch
        one we want to load
    '''

    save_path = args.model_path

    models = [file for file in os.listdir(save_path)
              if args.load_name in file]

    models.sort()

    model_selected = int(choice(models))

    return save_path + models[model_selected], models[model_selected]


def load(args):

    if not hasattr(args, 'lcc_loss_coeff'):
        args.lcc_loss_coeff = 0

    if not hasattr(args, 'mse_loss_coeff'):
        args.mse_loss_coeff = 1

    if not hasattr(args, 'dice_loss_coeff'):
        args.dice_loss_coeff = 0

    if not hasattr(args, 'l2_loss_coeff'):
        args.l2_loss_coeff = 0

    # Loss
    def loss(y_true, y_pred):
        return losses.custom_loss(y_true, y_pred, args=args)

    def l2_loss(y_true, y_pred):
        return losses.l2(y_true, y_pred, 1)

    # Metrics

    def mse(y_true, y_pred):
        return losses.mse(y_true, y_pred, 1)

    def dice(y_true, y_pred):
        return losses.dice(y_true, y_pred, 1)

    def lcc(y_true, y_pred):
        return losses.lcc(y_true, y_pred, 1)

    def l2(y_true, y_pred):
        return losses.l2(y_true, y_pred, 1)

    def generalised_dice(y_true, y_pred):
        return losses.generalised_dice(y_true, y_pred, 1)

    model_path, model_name = search_model(args)

    custom_objects = {'DefReg': blocks.DefReg,
                      'SqueezeExcitation': blocks.SqueezeExcitation,
                      'diffeomorphicTransformer3D': Transformer.diffeomorphicTransformer3D,
                      'intergral3DGrid': Transformer.intergral3DGrid,
                      'l2_loss': l2_loss,
                      'loss': loss,
                      'mse': mse,
                      'dice': dice,
                      'lcc': lcc,
                      'l2': l2,
                      'generalised_dice': generalised_dice}

    epoch = int(choice(os.listdir(model_path)))
    model_epoch = os.listdir(model_path)[epoch]

    print('Loading model....')

    model = models.load_model(model_path + '/' + model_epoch,
                              custom_objects=custom_objects)

    print('Model load :')
    print('Session : {}'.format(model_name))
    print('Epoch : {}'.format(model_epoch))

    return model, model_name, model_epoch


def save(X, y, gt, save_path, args):

    if args.use_mask:
        deformed, displacements, deformed_mask = y
        moving, reference, moving_mask = X
        reference_mask = gt[2]

        np.save(save_path + 'deformed_mask.npy', deformed_mask)
        np.save(save_path + 'reference_mask.npy', reference_mask)
        np.save(save_path + 'moving_mask', moving_mask)
        np.save(save_path + 'reference_mask', reference_mask)

    elif args.segmentation:
        deformed, displacements, deformed_mask, seg = y
        moving, reference = X
        reference_mask = gt[2]
        moving_mask = gt[3]

        np.save(save_path + 'deformed_mask.npy', deformed_mask)
        np.save(save_path + 'seg.npy', seg)
        np.save(save_path + 'reference_mask.npy', reference_mask)
        np.save(save_path + 'moving_mask.npy', moving_mask)

    else:
        deformed, displacements = y
        moving, reference = X

    np.save(save_path + 'moving.npy', moving)
    np.save(save_path + 'reference.npy', reference)
    np.save(save_path + 'deformed.npy', deformed)
    np.save(save_path + 'grid.npy', displacements)
