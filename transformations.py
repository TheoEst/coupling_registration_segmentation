# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:24:34 2019

@author: T_ESTIENNE

Transformations applied to the batch
"""
import numpy as np


def translate(img, translation=None):

    if translation is None:

        if len(img.shape) == 4:
            x, y, z, _ = np.where(img > 0)
        else:
            x, y, z = np.where(img > 0)

        x_center = int(np.mean(x))
        y_center = int(np.mean(y))
        z_center = int(np.mean(z))

        center = [int(k // 2) for k in img.shape]
        translation = (center[0] - x_center, center[1] -
                       y_center, center[2] - z_center)

    if len(img.shape) == 3:
        n, m, p = img.shape
        new_img = np.zeros((3*n, 3*m, 3*p), img.dtype)
        new_img[n:2*n, m:2*m, p:2*p] = img
    else:
        n, m, p, q = img.shape
        new_img = np.zeros((3*n, 3*m, 3*p, q), img.dtype)
        new_img[n:2*n, m:2*m, p:2*p, :] = img

    x = n-translation[0]
    y = m-translation[1]
    z = p-translation[2]

    if len(img.shape) == 3:
        new_img = new_img[x:x+n, y:y+m, z:z+p]
    else:
        new_img = new_img[x:x+n, y:y+m, z:z+p, :]

    return new_img, translation


def center_crop(array, output_size):

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    depth, height, width, _ = array.shape

    if depth == output_size[0]:
        depth_min = 0
        depth_max = depth
    else:
        depth_min = int((depth - output_size[0])/2)
        depth_max = -(depth - output_size[0] - depth_min)

    if height == output_size[1]:
        height_min = 0
        height_max = height
    else:
        height_min = int((height - output_size[1])/2)
        height_max = -(height - output_size[1] - height_min)

    if width == output_size[2]:
        width_min = 0
        width_max = width
    else:
        width_min = int((width - output_size[2])/2)
        width_max = -(width - output_size[2] - width_min)

    crop = array[depth_min:depth_max,
                 height_min:height_max,
                 width_min:width_max,
                 :]

    return crop


def random_crop(array, output_size):

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    depth, height, width, _ = array.shape

    if depth == output_size[0]:
        i = 0
    else:
        i = np.random.randint(0, depth - output_size[0])

    if height == output_size[1]:
        j = 0
    else:
        j = np.random.randint(0, height - output_size[1])

    if width == output_size[2]:
        k = 0
    else:
        k = np.random.randint(0, width - output_size[2])

    array = array[i:i + output_size[0],
                  j:j + output_size[1],
                  k:k + output_size[2],
                  :]

    return array


def normalize(array):

    mean = np.mean(array[array > 0])
    std = np.std(array[array > 0])

    array = (array - mean) / std
    array = np.clip(array, -5, 5)

    mini = np.min(array)
    maxi = np.max(array)

    array = (array - mini) / (maxi - mini)

    return array
