# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:17:59 2019

@author: T_ESTIENNE

This file is used to calculate the dice score for every brain structures from the freesurfer dataset
"""

import numpy as np


all_labels = {2: 'Left_Cerebral_White_Matter', 3: 'Left_Cerebral_Cortex',
              4: 'Left_Lateral_Ventricle',
              7: 'Left_Cerebellum_White_Matter', 8: 'Left_Cerebellum_Cortex',
              10: 'Left_Thalamus_Proper', 11: 'Left_Caudate',
              12: 'Left_Putamen', 13: 'Left_Pallidum',
              14: 'Third_Ventricle', 15: 'Fourth_Ventricle',
              16: 'Brain_Stem', 17: 'Left_Hippocampus', 18: 'Left_Amygdala',
              24: 'CSF', 28: 'Left_VentralDC',
              41: 'Right_Cerebral_White_Matter',
              42: 'Right_Cerebral_Cortex', 43: 'Right_Lateral_Ventricle',
              46: 'Right_Cerebellum_White_Matter',
              47: 'Right_Cerebellum_Cortex', 49: 'Right_Thalamus_Proper',
              50: 'Right_Caudate', 51: 'Right_Putamen', 52: 'Right_Pallidum',
              53: 'Right_Hippocampus', 54: 'Right_Amygdala',
              60: 'Right_VentralDC'
              }

three_label = {1: 'White_Matter', 2: 'Ventricles', 3: 'Grey_matter'}


def dice(y_pred, y_true):

    im1 = np.asarray(y_pred).astype(np.bool)
    im2 = np.asarray(y_true).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return 0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def multi_dice(y_pred, y_true):
    '''
    1 is white matter
    2 is ventricles
    3 is grey matter
    '''
    n_classes = y_pred.shape[-1]
    pred = np.argmax(y_pred, axis=-1)
    gt = np.argmax(y_true, axis=-1)

    dices = [dice(gt == i, pred == i) for i in range(1, n_classes)]

    return dices


def evaluate(X, y, gt, args):
    
    if args.all_label:
        labels = all_labels
    else:
        labels = three_label

    dice_list = []

    if args.use_mask:
        deformed, displacements, deformed_mask = y
        moving, reference, moving_mask = X
        reference_mask = gt[2]

        for k in range(moving.shape[0]):

            y_pred = np.argmax(deformed_mask[k, ...], axis=-1)
            gt = np.argmax(reference_mask[k, ...], axis=-1)

            y_pred = deformed_mask[k, ...]
            gt = reference_mask[k, ...]

            dice_dict = {}

            for i, label in enumerate(labels.values()):

                dice_dict['target_' + label] = dice(gt[..., (i+1)] > 0,
                                                    y_pred[..., (i+1)] > 0)

            dice_list.append(dice_dict)

    elif args.segmentation:
        deformed, displacements, deformed_mask, seg = y
        moving, reference = X
        reference_mask = gt[2]
        moving_mask = gt[3]

        for k in range(moving.shape[0]):

            y_pred = np.argmax(deformed_mask[k, ...], axis=-1)
            gt = np.argmax(reference_mask[k, ...], axis=-1)

            dice_dict = {}

            for i, label in enumerate(labels.values()):

                dice_dict['target_' + label] = dice(gt == (i + 1),
                                                    y_pred == (i + 1))

            y_pred = np.argmax(seg[k, ...], axis=-1)
            gt = np.argmax(moving_mask[k, ...], axis=-1)

            for i, label in enumerate(labels.values()):

                dice_dict['source_' + label] = dice(gt == (i + 1),
                                                    y_pred == (i + 1))

            dice_list.append(dice_dict)
    else:
        print('No evaluation')

    return dice_list
