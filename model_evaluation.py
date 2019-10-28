# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:08:00 2019

@author: T_ESTIENNE

Model inference and calculation of the metrics
"""
import os
import argparse
import keras.utils
from tqdm import tqdm
import pandas as pd

# My package
from coupling_registration_segmentation import Dataset
from coupling_registration_segmentation import utils
from coupling_registration_segmentation import main
from coupling_registration_segmentation import metrics

main_path = os.path.abspath(__file__)
n = main_path.find('Python')
if n > 0:
    main_path = main_path[:n] + 'Python/'
else:
    n = main_path.find('workspace')
    main_path = main_path[:n]
    print(main_path)

repo_name = 'coupling_registration_segmentation/'

def parse_args():

    parser_main = main.parse_args(add_help=False)

    parser = argparse.ArgumentParser(
        description='Keras automatic registration',
        parents=[parser_main])

    parser.add_argument('--test', action='store_true',
                        help='Calculation of the metrics on the set set')

    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--load-segmentation', action='store_true')

    args = parser.parse_args()

    return args


def predict(args):
    
    data_path = main_path + 'data/' + args.folder
    dataset_path = main_path + repo_name + 'datasets/'
    save_path = main_path + repo_name +  'save/'
    args.model_path = save_path + 'models/'
            
    for folder in [save_path, args.model_path, dataset_path]:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    
    dim = (160, 176, 208)

    # Parameters
    params = {'data_path': data_path,
              'dim': dim,
              'batch_size': args.batch_size,
              'shuffle': True,
              'translation': args.translation}

    # Datasets
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
        generator = DataGen(files_test, validation=True, **params)
    else:
        generator = DataGen(files_validation, validation=True, **params)

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

    dice_dict = []

    for X, gt in tqdm(generator):
        y = model.predict(X, batch_size=args.batch_size)

        dice_dict += metrics.evaluate(X, y, gt, args)

    results = pd.DataFrame(dice_dict)

    results.to_csv(save_path + model_name + '_' + model_epoch + '.csv')


if __name__ == '__main__':

    args = parse_args()

    predict(args)
