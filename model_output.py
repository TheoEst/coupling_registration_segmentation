# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:54:05 2019

@author: T_ESTIENNE

Model inference and save of the predictions
"""
import os
import numpy as np
import argparse
import keras.utils

# My package
from miccai.keras import Dataset
from miccai.keras import utils
from miccai.keras import main
from miccai.keras import plot

main_path = os.path.abspath(__file__)
n = main_path.find('Python')

if n > 0:
    main_path = main_path[:n] + 'Python/'
else:
    n = main_path.find('workspace')
    main_path = main_path[:n]
    print(main_path)


def parse_args():

    parser_main = main.parse_args(add_help=False)

    parser = argparse.ArgumentParser(
        description='Keras automatic registration',
        parents=[parser_main])

    parser.add_argument('--test', action='store_true',
                        help='Calculation of the metrics on the set set')

    parser.add_argument('--n-img', type=int, default=1,
                        help='Number of batch to plot')

    parser.add_argument('--plot', action='store_true')

    parser.add_argument('--plot-grid', action='store_true',
                        help='Plot the grid')
    
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--load-segmentation', action='store_true')
        
    args = parser.parse_args()
    
    return args


def predict(args):
    data_path = main_path + 'data/' + args.folder
    model_path = main_path + 'save/models/'

    save_path = main_path + 'save/miccai_output/'
    plot_path = main_path + 'save/miccai_plot/'

    args.model_path = model_path

    dim = (160, 176, 208)
    
    # Parameters
    params = {'data_path': data_path,
              'dim': dim,
              'batch_size': args.batch_size,
              'shuffle': True,
              'translation':args.translation}

    # Datasets
    if args.create_new_split:

        if args.use_mask or args.segmentation:
            files = Dataset.load_freesurfer_datasets(data_path)
        else:
            files = Dataset.load_datasets(data_path)

        (files_train, files_validation,
         files_test) = Dataset.create_dataset(files, data_path)

    else:
        (files_train, files_validation,
         files_test) = Dataset.load_existing_dataset(data_path)

    # Generators
    if args.use_mask or args.segmentation:
        DataGen = Dataset.FreeSurferDataGenerator
        params['use_mask'] = args.use_mask
        params['segmentation'] = args.segmentation
        params['all_label'] = args.all_label
    else:
        DataGen = Dataset.DataGenerator

    if args.test:
        validation_generator = DataGen(files_test, validation=True, **params)
    else:
        validation_generator = DataGen(files_validation, validation=True, **params)
    
    if args.pretrained:
        load_model, model_name, model_epoch = utils.load(args)

        model, base_model = main.design_model(dim, args)
        model.compile(optimizer='adam')

        if args.load_segmentation:
            args.segmentation, args.all_label = True, False
            seg_model, base_seg_model = main.design_model(dim, args)
            args.segmentation, args.all_label  = False, True
            seg_model.set_weights(load_model.get_weights())

            base_model.set_weights(base_seg_model.get_weights())
        else:
            base_model.set_weights(load_model.get_weights())

    else:
        load_model, model_name, model_epoch = utils.load(args)
        model, base_model = main.design_model(dim, args)
        model.set_weights(load_model.get_weights())

    if args.parallel:
        model = keras.utils.multi_gpu_model(model, gpus=args.nb_gpu)
        
    save_path += model_name + '/' + model_epoch + '/'
    plot_path += model_name + '/' + model_epoch + '/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    for i in range(args.n_img):

        X, gt = validation_generator.__getitem__(i)

        y = model.predict(X, batch_size=args.batch_size)

        if args.save:
            utils.save(X, y, gt, save_path, args)

        if args.plot:
            plot.plot_contour(X, y, gt, plot_path, args)


if __name__ == '__main__':

    args = parse_args()

    predict(args)
