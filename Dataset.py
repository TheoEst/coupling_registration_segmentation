#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:27:15 2019

@author: theoestienne
"""

import numpy as np
import keras
import os
import SimpleITK as sitk
import nibabel.freesurfer.mghformat as mgh
import sklearn.model_selection as model_selection
from sklearn import preprocessing

from coupling_registration_segmentation import transformations


def aseg_label(all_label=False):

    if all_label:
        return [0, 16, 10, 49, 47, 8, 2, 41, 7, 46, 12, 51, 28, 60, 13, 52, 11,
                50, 4, 43, 17, 53, 14, 15, 18, 54, 3, 42, 24]
    else:

        return [0, 2, 41, 3, 42, 4, 43]


def label_encoder(aseg_label):
    le = preprocessing.LabelEncoder()

    le.fit(aseg_label)

    return le


def load_nifti(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def load_mgh(path):
    return mgh.load(path).get_data()


def load_datasets(path):

    n = len(path)

    nifti = [os.path.join(root, name)[n:]
             for root, dirs, files in os.walk(path)
             for name in files
             if name.endswith('.npy')
             or name.endswith('.nii.gz')
             or name.endswith('orig.mgz')]

    return nifti


def load_freesurfer_datasets(path):

    files = [file[:-7]
             for file in os.listdir(path) if file.endswith('_T1.mgz')]

    return files


def load_existing_dataset(path):

    files_train = np.loadtxt(path + 'train.txt', dtype=str)
    files_val = np.loadtxt(path + 'val.txt', dtype=str)
    files_test = np.loadtxt(path + 'test.txt', dtype=str)

    return files_train, files_val, files_test


def create_dataset(files, path):

    (files_train,
     files_validation) = model_selection.train_test_split(files,
                                                          test_size=0.3,
                                                          random_state=42)

    (files_test,
     files_validation) = model_selection.train_test_split(files_validation,
                                                          test_size=0.3,
                                                          random_state=42)

    np.savetxt(path + 'train.txt', files_train, fmt='%s')
    np.savetxt(path + 'val.txt', files_validation, fmt='%s')
    np.savetxt(path + 'test.txt', files_test, fmt='%s')

    return files_train, files_validation, files_test


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, data_path, batch_size=4, dim=(64, 64, 64),
                 n_channels=1, shuffle=True, validation=False,
                 translation=False):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.n_channels = n_channels

        self.validation = validation
        self.shuffle = shuffle
        self.on_epoch_end()

        if self.validation:
            self.validation_index()

        ones = np.ones((batch_size, *dim))

        self.identity_grid = np.stack([np.cumsum(ones, axis=1),
                                       np.cumsum(ones, axis=2),
                                       np.cumsum(ones, axis=3)],
                                      axis=-1)

        self.translation = translation

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def init_data(self):

        if self.list_IDs[0].endswith('.nii.gz'):

            self.data = {self.data_path + ID: load_nifti(self.data_path + ID)
                         for ID in self.list_IDs
                         }

        elif self.list_IDs[0].endswith('.npy'):

            self.data = {self.data_path + ID: np.load(self.data_path + ID)[1, :, :, :]
                         for ID in self.list_IDs
                         }

        else:
            print('Wrong files')
            self.data = {}

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_moving = self.indexes_moving[index *
                                             self.batch_size:(index+1)*self.batch_size]
        indexes_reference = self.indexes_reference[index *
                                                   self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(self.list_IDs[i], self.list_IDs[j]) for i, j
                         in zip(indexes_moving, indexes_reference)]

        # Generate data
        moving, reference = self.__data_generation(list_IDs_temp)

        return [moving, reference], [reference, self.identity_grid]

    def validation_index(self):

        self.indexes_moving = list(range(len(self.list_IDs)))
        self.indexes_reference = list(range(1, len(self.list_IDs))) + [0]

    def on_epoch_end(self):
        'Updates indexes after each epoch for training'

        if not self.validation:
            self.indexes_moving = np.arange(len(self.list_IDs))
            self.indexes_reference = np.arange(len(self.list_IDs))

            if self.shuffle:
                np.random.shuffle(self.indexes_moving)
                np.random.shuffle(self.indexes_reference)

    def __data_generation(self, list_IDs_temp):

        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        moving = np.empty((self.batch_size, *self.dim, self.n_channels))
        reference = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, (ID_moving, ID_reference) in enumerate(list_IDs_temp):
            # Store sample
            array = self.load(ID_moving)
            moving[i, :, :, :, :], _ = self.transform(array)

            # Store class
            array = self.load(ID_reference)
            reference[i, :, :, :, :], _ = self.transform(array)

        return moving, reference

    def transform(self, array, normalize=True, translation=None):

        if self.translation:
            array, translation = transformations.translate(array, translation)
        else:
            translation = None

        if len(array.shape) == 3:
            array = array[:, :, :, np.newaxis]

        array = transformations.center_crop(array, self.dim)

        if normalize:
            array = transformations.normalize(array)

        return array, translation

    def load(self, ID):

        path = self.data_path + ID

        if path.endswith('.nii.gz'):
            return load_nifti(path)

        elif path.endswith('.npy'):
            return np.load(path)[1, :, :, :]

        elif path.endswith('.mgz'):
            brain = load_mgh(path)
            return brain
        elif 'MR' in path:
            brain = load_mgh(path + '_orig.mgz')
            return brain
        else:
            print('Wrong files')
            return None


class FreeSurferDataGenerator(DataGenerator):

    def __init__(self, list_IDs, data_path, batch_size=4, dim=(64, 64, 64),
                 n_channels=1, shuffle=True, validation=False,
                 use_mask=True, segmentation=False,
                 translation=False, all_label=False):

        super(FreeSurferDataGenerator, self).__init__(list_IDs, data_path,
                                                      batch_size, dim,
                                                      n_channels, shuffle,
                                                      validation,
                                                      translation)

        self.use_mask = use_mask
        self.segmentation = segmentation
        self.aseg_label = aseg_label(all_label)
        self.label_encoder = label_encoder(self.aseg_label)
        self.all_label = all_label

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes_moving = self.indexes_moving[index *
                                             self.batch_size:(index+1)*self.batch_size]
        indexes_reference = self.indexes_reference[index *
                                                   self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(self.list_IDs[i], self.list_IDs[j]) for i, j
                         in zip(indexes_moving, indexes_reference)]

        # Generate data
        (moving, reference,
         moving_mask, reference_mask) = self.__data_generation(
            list_IDs_temp)

        if self.use_mask:
            return [moving, reference, moving_mask], [reference,
                                                      self.identity_grid,
                                                      reference_mask]
        elif self.segmentation:
            return [moving, reference], [reference, self.identity_grid,
                                         reference_mask, moving_mask]
        else:
            print('Problem')
            return None

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'

        if self.all_label:
            out_channels = 29
        else:
            out_channels = 4

        # Initialization
        moving = np.empty((self.batch_size, *self.dim, self.n_channels))
        moving_mask = np.empty((self.batch_size, *self.dim, out_channels))
        reference = np.empty((self.batch_size, *self.dim, self.n_channels))
        reference_mask = np.empty((self.batch_size, *self.dim, out_channels))

        # Generate data
        for i, (ID_moving, ID_reference) in enumerate(list_IDs_temp):
            # Moving sample
            array, array_mask = self.load(ID_moving)
            moving[i, :, :, :, :], translation = self.transform(array)
            moving_mask[i, :, :, :, :], _ = self.transform(
                array_mask, normalize=False, translation=translation)

            # Reference class
            array, array_mask = self.load(ID_reference)
            reference[i, :, :, :, :], translation = self.transform(array)
            reference_mask[i, :, :, :, :], _ = self.transform(
                array_mask, normalize=False, translation=translation)

        return moving, reference, moving_mask, reference_mask

    def load(self, ID):

        ID = ID[:-9]
        path = self.data_path + ID
        brain = load_mgh(path + '_orig.mgz')

        aseg = load_mgh(path + '_aseg.mgz')

        aseg[~np.isin(aseg, self.aseg_label)] = 0

        if self.all_label:
            aseg = aseg.flatten()
            aseg = self.label_encoder.transform(aseg)
            n_class = len(self.label_encoder.classes_)
            aseg = np.reshape(aseg, (256, 256, 256))

        else:

            aseg[np.isin(aseg, [2, 41])] = 1
            aseg[np.isin(aseg, [3, 42])] = 2
            aseg[np.isin(aseg, [4, 43])] = 3
            n_class = 4

        one_hot = np.eye(n_class)[aseg]


        return brain, one_hot
