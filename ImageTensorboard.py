#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:50:17 2019

@author: theoestienne

https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

This files is used to plot and save after each epoch the moving, reference and deformed image.

"""

import tensorflow as tf
from PIL import Image
import io
import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_result(moving, reference, deformed, grid, batch):

    kwargs = {'cmap': 'gray'}

    fig, ax = plt.subplots(3, 4, gridspec_kw={'wspace': 0, 'hspace': 0.02,
                                              'top': 0.93, 'bottom': 0.01,
                                              'left': 0.01, 'right': 0.99})

    x_slice = int(moving.shape[1] // 2)
    y_slice = int(moving.shape[2] // 2)
    z_slice = int(moving.shape[3] // 2)

    ax[0, 0].imshow(reference[batch, x_slice, :, :, 0], **kwargs)
    ax[1, 0].imshow(reference[batch, :, y_slice, :, 0], **kwargs)
    ax[2, 0].imshow(reference[batch, :, :, z_slice, 0], **kwargs)

    ax[0, 1].imshow(moving[batch, x_slice, :, :, 0], **kwargs)
    ax[1, 1].imshow(moving[batch, :, y_slice, :, 0], **kwargs)
    ax[2, 1].imshow(moving[batch, :, :, z_slice, 0], **kwargs)

    ax[0, 2].imshow(deformed[batch, x_slice, :, :, 0], **kwargs)
    ax[1, 2].imshow(deformed[batch, :, y_slice, :, 0], **kwargs)
    ax[2, 2].imshow(deformed[batch, :, :, z_slice, 0], **kwargs)

    dx, dy, dz = (grid[batch, :, :, :, 0],
                  grid[batch, :, :, :, 1],
                  grid[batch, :, :, :, 2])

    ax[0, 3].contour(dy[x_slice, ::-1, :], 100, alpha=0.90, linewidths=0.5)
    ax[0, 3].contour(dz[x_slice, ::-1, :], 100, alpha=0.90, linewidths=0.5)

    ax[1, 3].contour(dx[:, y_slice, :], 100, alpha=0.90, linewidths=0.5)
    ax[1, 3].contour(dz[:, y_slice, :], 100, alpha=0.90, linewidths=0.5)

    ax[2, 3].contour(dx[:, :, z_slice], 100, alpha=0.90, linewidths=0.5)
    ax[2, 3].contour(dy[:, :, z_slice], 100, alpha=0.90, linewidths=0.5)

    for i in range(3):
        for j in range(4):
            ax[i, j].grid(False)
            ax[i, j].axis('off')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    ax[0, 0].set_title('Target')
    ax[0, 1].set_title('Source')
    ax[0, 2].set_title('Deformed')
    ax[0, 3].set_title('Grid')

    fig.canvas.draw()

    plt.close()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """

    width, height,  _ = tensor.shape

    image = Image.frombytes("RGBA", (width, height), tensor.tostring())
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=3,
                            encoded_image_string=image_string)


class TensorBoardImage(keras.callbacks.Callback):

    def __init__(self, validation_generator,
                 log_path):

        super().__init__()
        self.validation_generator = validation_generator
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs={}):

        # Load image
        X, _ = self.validation_generator.__getitem__(0)

        if len(X) == 3:  # Use Mask as input
            moving, reference, moving_mask = X
        else:
            moving, reference = X

        # Do something to the image
        y = self.model.predict(X)

        if len(y) == 2:
            deformed, displacements = y
        elif len(y) == 3:
            deformed, displacements, deformed_mask = y
        elif len(y) == 4:
            deformed, displacements, deformed_mask, seg = y

        batch_size = deformed.shape[0]

        writer = tf.summary.FileWriter(self.log_path)

        for batch in range(batch_size):

            tensor = plot_result(
                moving, reference, deformed, displacements, batch)

            tensor = make_image(tensor)

            summary = tf.Summary(
                value=[tf.Summary.Value(tag='Validation : ' + str(batch),
                                        image=tensor)])

            writer.add_summary(summary, epoch)

        writer.close()

        return
