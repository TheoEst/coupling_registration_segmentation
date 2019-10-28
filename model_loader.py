#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:37:51 2019

@author: theoestienne

Creation of the model
"""

import keras.layers as layers
import keras.models as models
from coupling_registration_segmentation import diffeomorphicTransformer as Transformer
from coupling_registration_segmentation import blocks


def Encoder(pool_blocks, channels, dim):

    dim = (*dim, 2)
    input_channel = 2

    input_layer = layers.Input(shape=dim)

    enc16 = blocks.InputBlock(input_layer, channels[0], input_channel)

    skip_x = [enc16]

    for i in range(pool_blocks):
        skip_x.append(blocks.ConvBlock(skip_x[-1], channels[i+1]))

    return models.Model(input_layer, skip_x)


def getVNetModel(X, affine, deform_regularisation,
                 affine_regularisation,
                 channel_division=1,
                 pool_blocks=4,
                 segmentation=False,
                 use_mask=False,
                 non_rigid_trainable=True,
                 affine_trainable=True,
                 dim=(128, 128, 128),
                 all_label=False
                 ):

    if use_mask:
        moving, reference, moving_mask = X
    else:
        moving, reference = X

    channels = [16, 32, 64, 128, 256]
    channels = channels[:pool_blocks+1]
    channels = [int(channel / channel_division) for channel in channels]

    # Encoder

    encoder = Encoder(pool_blocks, channels, dim)

    i = layers.Concatenate(axis=-1)([moving, reference])
    skip_x = encoder(i)

    # Decoder
    out = blocks.DecoderBlock(skip_x, pool_blocks, channels)

    # Deformable part
    deformable = blocks.DeformableBlock(out, channels[1],
                                        deform_regularisation,
                                        non_rigid_trainable)

    # Affine part
    x = [moving, deformable]

    if affine:
        affine = blocks.AffineBlock(out,
                                    affine_regularisation,
                                    affine_trainable)
        x.append(affine)

    defgrid = Transformer.intergral3DGrid(name='defgrid')(x)

    deformed = Transformer.diffeomorphicTransformer3D(name='deformed')(
        [moving, defgrid])

    # Segmentation part
    if segmentation:

        out = blocks.DecoderSegmentationBlock(skip_x, pool_blocks,
                                              channels)

        seg = blocks.SegmentationBlock(out, channels[1], all_label)

        deformed_mask = Transformer.diffeomorphicTransformer3D(name='deformed_mask')(
            [seg, defgrid])

        return deformed, defgrid, deformed_mask, seg

    elif use_mask:
        deformed_mask = Transformer.diffeomorphicTransformer3D(name='deformed_mask')(
            [moving_mask, defgrid])
        return deformed, defgrid, deformed_mask

    else:
        return deformed, defgrid
