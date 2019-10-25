#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:39:30 2019

@author: theoestienne

Plot for the miccai article
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# My package
from miccai_public import utils


def plot_contour(X, y, gt, plot_path, args):

    if args.use_mask:
        deformed, displacements, deformed_mask = y
        moving, reference, moving_mask = X
        reference_mask = gt[2]

    elif args.segmentation:
        deformed, displacements, deformed_mask, seg = y
        moving, reference = X
        reference_mask = gt[2]
        moving_mask = gt[3]

    kwargs = {'cmap': 'gray'}

    reference_mask = np.argmax(reference_mask, axis=-1)
    moving_mask = np.argmax(moving_mask, axis=-1)
    deformed_mask = np.argmax(deformed_mask, axis=-1)

    x_slice = int(moving.shape[1] // 2)
    y_slice = int(moving.shape[2] // 2)
    z_slice = int(moving.shape[3] // 2)

    n_col = 3

    reference = np.squeeze(reference)
    moving = np.squeeze(moving)
    deformed = np.squeeze(deformed)

    if args.plot_grid:
        n_col = 4

    for batch in range(args.batch_size):

        fig, ax = plt.subplots(3, n_col,
                               gridspec_kw={'wspace': 0, 'hspace': 0.02,
                                            'top': 0.93, 'bottom': 0.01,
                                            'left': 0.01, 'right': 0.99})

        ax[0, 0].imshow(reference[batch, x_slice, :, :], **kwargs)
        ax[1, 0].imshow(reference[batch, :, y_slice, :], **kwargs)
        ax[2, 0].imshow(reference[batch, :, :, z_slice], **kwargs)

        ax[0, 1].imshow(moving[batch, x_slice, :, :], **kwargs)
        ax[1, 1].imshow(moving[batch, :, y_slice, :], **kwargs)
        ax[2, 1].imshow(moving[batch, :, :, z_slice], **kwargs)

        ax[0, 2].imshow(deformed[batch, x_slice, :, :], **kwargs)
        ax[1, 2].imshow(deformed[batch, :, y_slice, :], **kwargs)
        ax[2, 2].imshow(deformed[batch, :, :, z_slice], **kwargs)

        contour_kwargs = {'levels': 5, 'alpha': 0.9, 'linewidths': 0.5}

        ax[0, 0].contour(reference_mask[batch, x_slice, :, :],
                         **contour_kwargs)
        ax[1, 0].contour(reference_mask[batch, :, y_slice, :],
                         **contour_kwargs)
        ax[2, 0].contour(reference_mask[batch, :, :, z_slice],
                         **contour_kwargs)

        ax[0, 1].contour(moving_mask[batch, x_slice, :, :],
                         **contour_kwargs)
        ax[1, 1].contour(moving_mask[batch, :, y_slice, :],
                         **contour_kwargs)
        ax[2, 1].contour(moving_mask[batch, :, :, z_slice],
                         **contour_kwargs)

        ax[0, 2].contour(deformed_mask[batch, x_slice, :, :],
                         **contour_kwargs)
        ax[1, 2].contour(deformed_mask[batch, :, y_slice, :],
                         **contour_kwargs)
        ax[2, 2].contour(deformed_mask[batch, :, :, z_slice],
                         **contour_kwargs)

        # Grille
        if args.plot_grid:
            contour_kwargs = {'levels': 100, 'alpha': 0.9, 'linewidths': 0.5}

            dx, dy, dz = (displacements[batch, :, :, :, 0],
                          displacements[batch, :, :, :, 1],
                          displacements[batch, :, :, :, 2])

            ax[0, 3].contour(dy[x_slice, ::-1, :], **contour_kwargs)
            ax[0, 3].contour(dz[x_slice, ::-1, :], **contour_kwargs)

            ax[1, 3].contour(dx[:, y_slice, :], **contour_kwargs)
            ax[1, 3].contour(dz[:, y_slice, :], **contour_kwargs)

            ax[2, 3].contour(dx[:, :, z_slice], **contour_kwargs)
            ax[2, 3].contour(dy[:, :, z_slice], **contour_kwargs)

        for i in range(3):
            for j in range(n_col):
                ax[i, j].grid(False)
                ax[i, j].axis('off')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

        ax[0, 0].set_title('Target')
        ax[0, 1].set_title('Source')
        ax[0, 2].set_title('Deformed')
        ax[0, 3].set_title('Grid')

        fig.canvas.draw()

        fig.savefig(plot_path + 'plot_' + str(batch))

        plt.close(fig)


def plot_boxplot():

    results_path = main_path + '/save/miccai_results/'

    results = os.listdir(results_path)

    indexes = utils.choice(results).split()

    colors = ['red', 'blue', 'green', 'black', 'pink', 'yellow']
    labels = ['Grey_matter', 'Ventricles', 'White Matter']

    index_list = ['target_gm_dice', 'target_ventricles_dice',
                  'target_wm_dice']

    fig, ax = plt.subplots(1, 1, gridspec_kw={'wspace': 0.05, 'hspace': 0,
                                              'top': 0.87, 'bottom': 0.01,
                                              'left': 0.01, 'right': 0.99}
                           )

    nb_data = len(indexes)
    positions = [[0.5 + i + (nb_data+1)*j for j in range(3)]
                 for i in range(nb_data)]

    boxplot_list = []
    results_name_list = []

    for i, index in enumerate(indexes):

        results_name = results[int(index)]
        dataframe = pd.read_csv(results_path + results_name, index_col=0)

        data = [dataframe[index] for index in index_list]
        bp = ax.boxplot(data, positions=positions[i], patch_artist=True)

        boxplot_list.append(bp)
        results_name_list.append(results_name)

    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(positions) - 1, np.max(positions) + 1)

    ax.set_xticks(positions[1])
    ax.set_xticklabels(labels)

    # fill with colors
    for bplot, color in zip(boxplot_list, colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    ax.legend([bp["boxes"][0] for bp in boxplot_list], results_name_list,
              loc='upper right')
    return fig
