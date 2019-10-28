#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:29:50 2019

@author: theoestienne
"""
import os
import keras.layers as layers
import argparse
import keras.utils
import keras.models as models
import keras.callbacks as callbacks
import keras.losses
import keras.optimizers as optimizers
import time
import functools
import math

# My package
from coupling_registration_segmentation import model_loader
from coupling_registration_segmentation import Dataset
from coupling_registration_segmentation import ImageTensorboard
from coupling_registration_segmentation import losses
from coupling_registration_segmentation import log
from coupling_registration_segmentation import utils

main_path = os.path.abspath(__file__)
n = main_path.find('Python')
if n > 0:
    main_path = main_path[:n] + 'Python/'
else:
    n = main_path.find('workspace')
    main_path = main_path[:n]
    print(main_path)

repo_name = 'coupling_registration_segmentation/'

def parse_args(add_help=True):

    parser = argparse.ArgumentParser(
        description='Keras automatic registration',
        add_help=add_help)

    parser.add_argument('--gpu', '-g', default=0, type=int, metavar='N',
                        help='Index of GPU used for calcul (default: 0)')

    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run (default: 20)')

    parser.add_argument('--batch-size', '-b', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')

    parser.add_argument('--lr', '--learning-rate', default=.001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')

    parser.add_argument('--tensorboard', default=True, action='store_false',
                        help='use tensorboard_logger to save data')

    parser.add_argument('--parallel', action='store_false', default=True,
                        help='Use data parallel in CUDA')

    parser.add_argument('--nb-gpu', default=4, type=int, metavar='N',
                        help='Number of gpu in case of parallel calculation')

    parser.add_argument('--save', '-s', action='store_false', default=True,
                        help='Save the model during training')

    parser.add_argument('--folder', default='oasis/', type=str,
                        help='Folder where to find the data')

    parser.add_argument('--workers', '-w', default=4, type=int,
                        help='Use multiprocessing for dataloader')

    parser.add_argument('--use-affine', action='store_false', default=True,
                        help='Use affine transformation in addition to deformable deformation')

    parser.add_argument('--deform-reg', type=float, default=1e-11,
                        help='Regularisation of the deformation layer')

    parser.add_argument('--affine-reg', type=float, default=1e-1,
                        help='Regularisation of the affine layer')

    parser.add_argument('--crop-size', type=float, nargs='+', default=240,
                        )

    parser.add_argument('--channel-division', type=int, default=4,
                        help='Divide the number of channels of each convolution')

    parser.add_argument('--pool-blocks', type=int, default=2,
                        help='Number of pooling block (Minimum 2)')

    parser.add_argument('--early-stopping', action='store_true',
                        help='Use early stopping')

    parser.add_argument('--dice-loss-coeff', type=float, default=0,
                        help='Coefficient to the dice Loss')

    parser.add_argument('--lcc-loss-coeff', type=float, default=1,
                        help='Coefficient to the Local Cross Correlation Loss')

    parser.add_argument('--mse-loss-coeff', type=float, default=1,
                        help='Coefficient to the Mean Square Error Loss')

    parser.add_argument('--session-name', type=str, default='',
                        help='Give a name to the session')

    parser.add_argument('--lr-decrease', action='store_true',
                        help='Reduce the learning rate')

    parser.add_argument('--lr-epochs-drop', type=int, default=20,
                        help='Number of epochs before decreasing the learning rate')

    parser.add_argument('--lr-drop', type=int, default=0.5,
                        help='Drop factor of the learning rate')

    parser.add_argument('--L2-loss-coeff', type=float, default=0,
                        help='L2 loss on the product grid')

    parser.add_argument('--segmentation', action='store_true',
                        help='Predict the segmentation with a bi task network')

    parser.add_argument('--use-mask', action='store_true',
                        help='Calculate the deformation of the mask')

    parser.add_argument('--deformed-mask-loss-weights', type=float, default=1,
                        help='Weights for the dice loss deformed mask')

    parser.add_argument('--seg-loss-weights', type=float, default=1,
                        help='Weights fot the dice loss segmentation')

    parser.add_argument('--create-new-split', action='store_true',
                        help='Create a train, validation, test split')

    parser.add_argument('--freeze-non-rigid', action='store_false',
                        help='Freeze the non rigid part of the network')

    parser.add_argument('--freeze-affine', action='store_false',
                        help='Freeze the affine part of the network')

    parser.add_argument('--translation', action='store_false', default=True,
                        help='Do translation on dataset')

    parser.add_argument('--load-model', action='store_true',
                        help='Load a trained model')

    parser.add_argument('--load-name', type=str,
                        help='Name of model loaded')

    parser.add_argument('--all-label', action='store_true',
                        help='Use all label of the aseg files')
       
    parser.add_argument('--w2', type=float, default=1,
                        help='Coefficient for the source and reference dice loss [see article]')
    return parser


# learning rate schedule
def step_decay(epoch, args):

    initial_lrate = args.lr
    drop = args.lr_drop
    epochs_drop = args.lr_epochs_drop
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate


def design_model(dim, args):

    moving = layers.Input((*dim, 1))
    reference = layers.Input((*dim, 1))

    X = [moving, reference]
    
    if args.all_label:
        out_channels = 29
    else:
        out_channels = 4
        
    if args.use_mask:
        moving_mask = layers.Input((*dim, out_channels))
        X.append(moving_mask)

    y = model_loader.getVNetModel(X,
                                  affine=args.use_affine,
                                  deform_regularisation=args.deform_reg,
                                  affine_regularisation=args.affine_reg,
                                  channel_division=args.channel_division,
                                  pool_blocks=args.pool_blocks,
                                  segmentation=args.segmentation,
                                  use_mask=args.use_mask,
                                  non_rigid_trainable=args.freeze_non_rigid,
                                  affine_trainable=args.freeze_affine,
                                  dim=dim,
                                  all_label=args.all_label
                                  )

    if args.segmentation:
        deformed, displacements, deformed_mask, seg = y

        model = models.Model(input=[moving, reference],
                             output=[deformed, displacements,
                                     deformed_mask, seg])

    elif args.use_mask:
        deformed, displacements, deformed_mask = y

        model = models.Model(input=[moving, reference, moving_mask],
                             output=[deformed, displacements, deformed_mask])

    else:
        deformed, displacements = y
        model = models.Model(input=[moving, reference],
                             output=[deformed, displacements])

    base_model = models.Model(input=[moving, reference],
                              output=[deformed, displacements])
    return model, base_model


def design_loss(args):

    def loss(y_true, y_pred):
        return losses.custom_loss(y_true, y_pred, args=args)

    def l2_loss(y_true, y_pred):
        return losses.l2(y_true, y_pred, args.L2_loss_coeff)

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

    loss_dict = {}
    loss_weights_dict = {}
    metrics_dict = {}

    if args.segmentation:

        loss_dict['deformed_mask'] = generalised_dice
        metrics_dict['deformed_mask'] = generalised_dice
        loss_weights_dict['deformed_mask'] = args.w2 * args.deformed_mask_loss_weights
        loss_dict['seg'] = generalised_dice
        metrics_dict['seg'] = generalised_dice
        loss_weights_dict['seg'] = args.w2 * args.seg_loss_weights

    elif args.use_mask:

        loss_dict['deformed_mask'] = generalised_dice
        metrics_dict['deformed_mask'] = generalised_dice
        loss_weights_dict['deformed_mask'] = args.deformed_mask_loss_weights

    loss_dict['deformed'] = loss
    loss_dict['defgrid'] = l2_loss
    metrics_dict['deformed'] = [mse, dice, lcc]
    metrics_dict['defgrid'] = l2

    loss_weights_dict['deformed'] = 1
    loss_weights_dict['defgrid'] = 1

    return loss_dict, loss_weights_dict, metrics_dict


def main(args):

    data_path = main_path + 'data/' + args.folder
    dataset_path = main_path + repo_name + 'datasets/'
    save_path = main_path + repo_name +  'save/'
    args.model_path = main_path + repo_name + 'models/'

    session_name = args.session_name + '_' + time.strftime('%m.%d %Hh%M')
    
    # Log
    log_folder = save_path + 'training_log/'
    log_path = log_folder + session_name + '.log'

    for folder in [save_path, args.model_path, log_folder, dataset_path]:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    
    logging = log.set_logger(log_path)
    
    for arg, value in sorted(vars(args).items()):
        logging.info("%s: %r", arg, value)
        
    dim = (160, 176, 208)

    # DataGen Parameters
    params = {'data_path': data_path,
              'dim': dim,
              'batch_size': args.batch_size,
              'shuffle': True,
              'translation': args.translation}

    # Datasets
    if args.create_new_split:

        if args.use_mask or args.segmentation:
            files = Dataset.load_freesurfer_datasets(data_path)
        else:
            files = Dataset.load_datasets(data_path)

        (files_train, files_validation,
         files_test) = Dataset.create_dataset(files, dataset_path)

    else:
        (files_train, files_validation,
         files_test) = Dataset.load_existing_dataset(dataset_path)

    # Generators
    if args.use_mask or args.segmentation:
        DataGen = Dataset.FreeSurferDataGenerator
        params['use_mask'] = args.use_mask
        params['segmentation'] = args.segmentation
        params['all_label'] = args.all_label
    else:
        DataGen = Dataset.DataGenerator

    training_generator = DataGen(files_train, **params)
    validation_generator = DataGen(files_validation, validation=True, **params)

    # Design Loss and metrics
    loss_dict, loss_weights_dict, metrics_dict = design_loss(args)

    # Design model
    model, base_model = design_model(dim, args)

    if args.load_name:

        loaded_model, model_folder, model_name = utils.load(args)

        base_model.set_weights(loaded_model.get_weights())

    print(model.summary())

    # Callbacks
    callbacks_list = []

    if args.tensorboard:
        log_path = save_path + 'logs/' + session_name + '/'
        tensorboard = callbacks.TensorBoard(log_dir=log_path,
                                            update_freq='batch',
                                            histogram_freq=0,
                                            write_grads=True)

        callbacks_list.append(tensorboard)

        tensorboard_image = ImageTensorboard.TensorBoardImage(validation_generator=validation_generator,
                                                              log_path=log_path
                                                              )

        callbacks_list.append(tensorboard_image)

    if args.save:
        model_path = save_path + 'models/' + session_name + '/'

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        save_callback = callbacks.ModelCheckpoint(model_path +
                                                  'model.{epoch:02d}--{val_loss:.3f}.hdf5',
                                                  save_weights_only=False,
                                                  period=5)
        callbacks_list.append(save_callback)

    if args.early_stopping:

        early_stopping = callbacks.EarlyStopping(patience=15)

        callbacks_list.append(early_stopping)

    if args.lr_decrease:

        lr_decrease = callbacks.LearningRateScheduler(
            functools.partial(step_decay,
                              args=args)
        )

        callbacks_list.append(lr_decrease)

    # Parallel
    if args.parallel:
        # Replicates `model` on 8 GPUs.
        # This assumes that your machine has 8 available GPUs.
        model = keras.utils.multi_gpu_model(model, gpus=args.nb_gpu)

    # Optimizer and compile
    optim = optimizers.Adam(lr=args.lr)

    model.compile(optimizer=optim,
                  loss=loss_dict,
                  metrics=metrics_dict,
                  loss_weights=loss_weights_dict
                  )

    train(model, training_generator, validation_generator, callbacks_list, args)


def train(model, training_generator, validation_generator, callbacks, args):

    fit_kwargs = {'epochs': args.epochs,
                  'callbacks': callbacks,
                  'workers' : args.workers
                  }

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  validation_steps=len(
                                      validation_generator),
                                  **fit_kwargs
                                  )

    return history


if __name__ == '__main__':

    parser = parse_args()
    args = parser.parse_args()

    main(args)
