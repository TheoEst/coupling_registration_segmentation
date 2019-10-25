#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:27:58 2019

@author: theoestienne

Losses functions. Some functions are taken from the voxelmorph github repo
https://github.com/voxelmorph/voxelmorph/blob/master/src/losses.py
"""

# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import keras.losses


def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
	
	https://github.com/voxelmorph/voxelmorph/blob/master/src/losses.py
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + np.arange(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


class NCC():
    """
    local (over window) normalized cross correlation
	
	https://github.com/voxelmorph/voxelmorph/blob/master/src/losses.py
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [
            1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)


# %%

def generalised_dice(y_true, y_pred, k):

    if k == 0:
        return 0
    else:
        dice = 0

        # White matter
        wm_true = K.cast(y_true[..., 1], 'float32')
        wm_pred = K.cast(y_pred[..., 1], 'float32')

        ventricles_true = K.cast(y_true[..., 2], 'float32')
        ventricles_pred = K.cast(y_pred[..., 2], 'float32')

        gm_true = K.cast(y_true[..., 3], 'float32')
        gm_pred = K.cast(y_pred[..., 3], 'float32')

        dice += binary_dice(ventricles_true, ventricles_pred)
        dice += binary_dice(wm_true, wm_pred)
        dice += binary_dice(gm_true, gm_pred)

        return k * dice


def dice(y_true, y_pred, k):
    '''
        Dice loss between the ground-truth brain and the predicted deformed brain
    '''
    if k == 0:
        return 0
    else:

        brain_true = K.cast(y_true > 0, 'float32')
        brain_pred = K.cast(y_pred > 0, 'float32')

        dice = binary_dice(brain_true, brain_pred)

        return k * dice


def lcc(y_true, y_pred, k):

    if k == 0:

        return 0

    else:

        lcc = NCC().loss(y_true, y_pred)

        return k * lcc


def mse(y_true, y_pred, k):

    mse = keras.losses.mean_squared_error(y_true, y_pred)

    return k * mse


def l2(y_true, y_pred, k):

    mse = keras.losses.mean_squared_error(y_true, y_pred)

    return k * mse


def custom_loss(y_true, y_pred, args):

    loss = 0

    if args.dice_loss_coeff > 0:

        loss += dice(y_true, y_pred, args.dice_loss_coeff)

    if args.lcc_loss_coeff > 0:

        loss += lcc(y_true, y_pred, args.lcc_loss_coeff)

    if args.mse_loss_coeff > 0:

        loss += mse(y_true, y_pred, args.mse_loss_coeff)

    return loss
