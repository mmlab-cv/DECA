import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import re
import cv2
from random import shuffle

from utils import compute_distances
from utils import *
from layers import *

import pytorch_lightning as pl


class CapsulePoseDataModule(pl.LightningDataModule):

    def __init__(self, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS

    def prepare_data(self):
        dataset_path = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        if os.path.isfile(dataset_path+'train_files.npy'):
            train_files = np.load(dataset_path+'train_files.npy')
            valid_files = np.load(dataset_path+'valid_files.npy')
            test_files = np.load(dataset_path+'test_files.npy')
            print('Files have been loaded')
        else:
            train_files = [f for f in os.listdir(
                os.path.join(dataset_path, 'train')) if not re.search(r'.npy', f)] # --> npy = (32,3)
            np.save('train_files', train_files)
            valid_files = [f for f in os.listdir(
                os.path.join(dataset_path, 'validation')) if not re.search(r'.npy', f)]
            np.save('valid_files', valid_files)
            test_files = [f for f in os.listdir(
                os.path.join(dataset_path, 'validation')) if not re.search(r'.npy', f)]
            np.save('test_files', test_files)
            train_files = np.load('train_files.npy')
            valid_files = np.load('valid_files.npy')
            test_files = np.load('test_files.npy')
            print('Files have been created')

        num_images = len(train_files)
        print('Loaded Training samples: ' + str(num_images))

        self.num_train_examples = len(train_files)
        self.train_indices = list(range(self.num_train_examples))
        self.num_valid_examples = len(valid_files)
        self.valid_indices = list(range(self.num_valid_examples))
        self.num_test_examples = len(test_files)
        self.test_indices = list(range(self.num_test_examples))

        shuffle(self.train_indices)
        shuffle(self.valid_indices)
        shuffle(self.test_indices)

        self.train_files = train_files[self.train_indices]
        self.validation_files = valid_files[self.valid_indices]

    def train_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'train')

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(
                brightness=self.FLAGS.brightness, contrast=self.FLAGS.contrast, hue=self.FLAGS.hue),
            transforms.ToTensor(),
            Standardize()])

        dataset = poseDATA(self.FLAGS, dataset_path, self.train_files,
                           self.train_indices, transform)

        capsulepose_train = FastDataLoader(dataset, shuffle=True, pin_memory=True,
                                           num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return capsulepose_train

    def val_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'validation')

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            Standardize()])

        dataset = poseDATA(self.FLAGS, dataset_path, self.validation_files,
                           self.valid_indices, transform)

        capsulepose_val = FastDataLoader(dataset, shuffle=False, pin_memory=True,
                                         num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return capsulepose_val

    def test_dataloader(self):
        working_dir = os.path.join(self.FLAGS.dataset_dir, self.FLAGS.dataset)
        dataset_path = os.path.join(working_dir, 'validation')

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            Standardize()])

        dataset = poseDATA(self.FLAGS, dataset_path, self.validation_files,
                           self.valid_indices, transform)

        capsulepose_test = FastDataLoader(dataset, shuffle=False, pin_memory=True,
                                          num_workers=self.FLAGS.num_workers, batch_size=self.FLAGS.batch_size, drop_last=True)

        return capsulepose_test


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class poseDATA(Dataset):
    ''' In:
            data_path (string): path to the dataset split folder, i.e. train/valid/test
            transform (callable, optional): transform to be applied on a sample.
        Out:
            sample (dict): sample data and respective label'''

    def __init__(self, FLAGS, data_path, input_list, indices, transform):

        self.data_path = data_path
        self.data, self.labels = [], []
        self.num_images = len(input_list)
        self.list_images = input_list
        self.list_masks = []
        self.indices = indices
        self.dataset_iterations = FLAGS.dataset_iterations
        self.art_select_ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18]
        self.split_size = self.num_images // FLAGS.num_workers
        self.starting_point = -1
        self.idx = -1
        self.FLAGS = FLAGS
        self.transform = transform

        for f in self.list_images:
            name = f[:-4] + '.npy' 
            self.list_masks.append(name)

        self.c = list(zip(self.list_images, self.list_masks))
        print("Data path: ", self.data_path)
        if('test' in self.data_path):
            print("TODO: implement test loading!")

        if('train' in self.data_path):
            shuffle(self.c)
            self.list_images, self.list_masks = zip(*(self.c))
            print('Training list has been shuffled (' +
                  str(self.num_images) + ' images)')

        else:
            self.list_images, self.list_masks = zip(*(self.c))
            print('Validation list has not been shuffled (' +
                  str(self.num_images) + ' images)')

    def __len__(self):
        return self.num_images * self.dataset_iterations * self.FLAGS.batch_size // self.FLAGS.n_epochs

    def __getitem__(self, idx):

        if(self.starting_point == -1):
            self.starting_point = (
                torch.utils.data.get_worker_info().id) * self.split_size

        if(self.idx == -1):
            next_id = self.starting_point
        else:
            next_id = (self.idx + 1)

        if(self.idx >= self.starting_point + self.split_size - 1):
            self.starting_point = (
                torch.utils.data.get_worker_info().id) * self.split_size
            self.idx = self.idx - self.split_size
        else:
            self.idx = next_id

        im = cv2.imread(self.data_path + '/' +
                        self.list_images[self.indices[self.idx]])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(self.data_path + 'depth/' +
                        self.list_images[self.indices[self.idx]].replace("render", "depth"))
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        msk = np.float32(
            np.load(self.data_path + '/' + self.list_masks[self.indices[self.idx]]))
        bias = np.repeat(np.reshape(msk[0, :], [1, 3]), 19, axis=0)
        msk = msk - bias
        msk = rotate(msk)   
        msk = msk[self.art_select_, :]  

        msk2d = np.float32(
            np.load(self.data_path + '2d' + '/' + self.list_masks[self.indices[self.idx]])) # --> (32,2)
        msk2d = msk2d[self.art_select_, :]


        if(self.transform):
            image = self.transform(im)

        msk = np.reshape(
            msk, [self.FLAGS.n_classes, 3]) / 100.
			
        msk2d = np.reshape(
            msk2d, [self.FLAGS.n_classes, 2]) / 256.

        depth = 1. - (depth / 255.)

        label = {'msk': msk,
                #  'rm': rm, 
                 'msk2d': msk2d,
                 'depth': depth}


        return image, label, self.indices[self.idx]  # (X, Y)
