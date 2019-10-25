# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:32:10 2019

@author: T_ESTIENNE
"""

import keras.layers as layers
import keras.regularizers as regularizers
from keras import backend as K


def ConvBlock(x, channels):

    out = layers.Conv3D(channels, (2, 2, 2),
                        padding='valid', strides=2)(x)
    out = layers.LeakyReLU()(out)
    out = layers.Conv3D(channels, (3, 3, 3),
                        padding='same', strides=1)(out)
    out = layers.LeakyReLU()(out)

    return out

def SkipDeconvBlock(x, skip_x, channels):

    out = layers.Conv3DTranspose(channels // 2, (2, 2, 2),
                                 padding='valid', strides=2)(x)
    out = layers.LeakyReLU()(out)
    cat = layers.Concatenate(axis=-1)([out, skip_x])
    out = layers.Conv3D(channels, (3, 3, 3), padding='same', strides=1)(cat)

    out = layers.LeakyReLU()(out)
    out = layers.Add()([out, cat])

    return out


class DefReg(regularizers.Regularizer):
    """Regularizer for the deformation field
    # Arguments
        alpha: Float; regularization factor.
        value: Float; penalize if different than value
    """

    def __init__(self, alpha=1e-5, value=0):
        self.alpha = K.cast_to_floatx(alpha)
        self.value = K.cast_to_floatx(value)

    def __call__(self, x):
        regularization = self.alpha*K.sum(K.abs(x-self.value))
        return regularization

    def get_config(self):
        return {'alpha': float(self.alpha)}


def SqueezeExcitation(in_block, ch, ratio=10):

    x = layers.GlobalAveragePooling3D()(in_block)
    x = layers.Dense(ch//ratio, activation='relu')(x)
    x = layers.Dense(ch, activation='sigmoid')(x)

    return layers.Multiply()([in_block, x])


def SegmentationBlock(out, channels, all_label):

    seg = layers.Conv3D(channels, (3, 3, 3), padding='same')(out)
    seg = layers.LeakyReLU()(seg)

    seg = layers.Conv3D(channels, (3, 3, 3), padding='same')(seg)
    seg = layers.LeakyReLU()(seg)
    
    if all_label:
        out_channels = 28
    else:
        out_channels = 4
        
    seg = layers.Conv3D(out_channels, (3, 3, 3), padding='same')(seg)

    seg = layers.Activation('softmax', name='seg')(seg)

    return seg


def DeformableBlock(out, channels, deform_regularisation, non_rigid_trainable):

    deformable = layers.Conv3D(channels, (3, 3, 3), padding='same')(out)
    deformable = layers.LeakyReLU()(deformable)

    deformable = layers.Conv3D(channels, (3, 3, 3), padding='same')(deformable)
    deformable = layers.LeakyReLU()(deformable)

    deformable = layers.Conv3D(3, (3, 3, 3),
                               kernel_initializer='zeros',
                               bias_initializer='zeros',
                               trainable=non_rigid_trainable,
                               padding='same',
                               activity_regularizer=DefReg(alpha=deform_regularisation,
                                                           value=0))(deformable)

    deformable = layers.Activation('sigmoid', name='mask')(deformable)

    return deformable


def AffineBlock(out, affine_regularisation, affine_trainable):

    # Affine
    affine = layers.GlobalAveragePooling3D()(out)

    affine = layers.Dense(12,
                          kernel_initializer='zeros',
                          bias_initializer='zeros',
                          trainable=affine_trainable,
                          activity_regularizer=DefReg(alpha=affine_regularisation,
                                                      value=0))(affine)

    affine = layers.Activation('linear')(affine)

    return affine


def InputBlock(x, channels, input_channels=2):

    out = layers.Conv3D(channels, (3, 3, 3),
                        padding='same', strides=1)(x)
    out = layers.LeakyReLU()(out)

    n_cat = int(channels / input_channels)

    i = layers.Concatenate(axis=-1)([x for k in range(n_cat)])

    out = layers.Add()([i, out])

    return out


def DecoderBlock(skip_x, pool_blocks, channels):

    out = skip_x[-1]
    
    for i in range(pool_blocks):
        out = SkipDeconvBlock(out, skip_x[-i-2],
                              channels[-i-1])

    return out


def DecoderSegmentationBlock(skip_x, pool_blocks, channels):

    out = skip_x[-1]

    for i in range(pool_blocks):
        out = SkipDeconvBlock(out, skip_x[-i-2],
                              channels[-i-1])

    return out
