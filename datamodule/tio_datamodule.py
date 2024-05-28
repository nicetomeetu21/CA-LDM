# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import torchio as tio
from torchvision.utils import save_image
import numpy as np
import torch
from natsort import natsorted
import json


def image_reader(path):
    data = np.load(path)
    data = torch.from_numpy(data).float()
    data /= 255.
    data = data.unsqueeze(0)
    affine = np.eye(4)
    return data, affine


def label_reader(path):
    data = np.load(path)
    data = torch.from_numpy(data).long()
    data //= 255
    data = data.unsqueeze(0)
    affine = np.eye(4)
    return data, affine


class testTioDatamodule(pl.LightningDataModule):
    def __init__(self, image_npy_root, test_data_list=None):
        super().__init__()
        self.image_root = image_npy_root

        if test_data_list is None:
            self.test_names = natsorted(os.listdir(image_npy_root))

    def prepare_data(self):
        self.test_subjects = []
        print(self.test_names)
        for name in self.test_names:
            # print(name, os.path.join(self.image_root, name))
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, name),
                                      reader=image_reader),
                name=name
            )
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1), in_min_max=(0, 1)),
            tio.EnsureShapeMultiple(8),
        ])
        return preprocess

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False)


class TioDatamodule(pl.LightningDataModule):
    def __init__(self, image_npy_root, train_name_json, test_name_json, image_size, batch_size, num_workers, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_root = image_npy_root

        self.image_size = image_size

        with open(train_name_json, "r") as f:
            self.train_image_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_image_names = json.load(f)

    def prepare_data(self):
        # print(self.train_image_names, self.test_image_names)
        self.train_subjects = []
        for name in self.train_image_names:
            # print(os.path.join(self.image_root, name))
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, name+'.npy'),
                                      reader=image_reader),
                name=name
            )
            # print(f'load {name}')
            # subject.load()
            self.train_subjects.append(subject)
        # print('train loaded')
        self.test_subjects = []
        for name in self.test_image_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, name+'.npy'),
                                      reader=image_reader),
                name=name
            )
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.Resize(self.image_size),
            tio.RescaleIntensity((-1, 1), in_min_max=(0, 1)),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 2)),
        ])
        return augment

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.train_set = tio.SubjectsDataset(self.train_subjects, transform=tio.Compose([preprocess, augment]))
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)



class PatchTioDatamodule(pl.LightningDataModule):
    def __init__(self, image_npy_root, train_name_json, test_name_json, image_size, batch_size, num_workers, patch_per_size,
                 queue_length, samples_per_volume, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_root = image_npy_root

        self.image_size = image_size
        self.patch_size = patch_per_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume

        with open(train_name_json, "r") as f:
            self.train_image_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_image_names = json.load(f)

    def prepare_data(self):
        # print(self.train_image_names, self.test_image_names)
        self.train_subjects = []
        for name in self.train_image_names:
            # print(os.path.join(self.image_root, name))
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, name+'.npy'),
                                      reader=image_reader),
                name=name
            )
            self.train_subjects.append(subject)

        self.test_subjects = []
        for name in self.test_image_names:
            subject = tio.Subject(
                image=tio.ScalarImage(os.path.join(self.image_root, name+'.npy'),
                                      reader=image_reader),
                name=name
            )
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.Resize(self.image_size),
            tio.RescaleIntensity((-1, 1), in_min_max=(0, 1)),
        ])
        return preprocess

    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 2)),
        ])
        return augment

    def setup(self, stage=None):
        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        train_set = tio.SubjectsDataset(self.train_subjects, transform=tio.Compose([preprocess, augment]))
        self.patch_train_set = tio.Queue(
            train_set,
            self.queue_length,
            self.samples_per_volume,
            tio.data.UniformSampler(self.patch_size),
            num_workers=self.num_workers,
            shuffle_patches=True,
            shuffle_subjects=True
        )
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=preprocess)

    def train_dataloader(self):
        return DataLoader(self.patch_train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
