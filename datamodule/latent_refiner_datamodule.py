# -*- coding:utf-8 -*-
import json
import os

import cv2 as cv
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from natsort import natsorted
import numpy as np


class trainDatamodule(pl.LightningDataModule):
    def __init__(self,  latent_1_root, latent_2_root, train_name_json, test_name_json, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.latent_1_root = latent_1_root
        self.latent_2_root = latent_2_root
        with open(train_name_json, "r") as f:
            self.train_cube_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_cube_names = json.load(f)

    def setup(self, stage=None):
        train_latent_1_pathes = []
        train_latent_2_pathes = []
        for cube_name in self.train_cube_names:
            img_names = natsorted(os.listdir(os.path.join(self.latent_1_root, cube_name)))
            for img_name in img_names:
                train_latent_1_pathes.append(os.path.join(self.latent_1_root, cube_name, img_name))
                train_latent_2_pathes.append(os.path.join(self.latent_2_root, cube_name, img_name))

        print(f'train len: {len(train_latent_1_pathes)} {len(train_latent_2_pathes)}')
        self.train_set = latent_Dataset(latent_1_pathes=train_latent_1_pathes, latent_2_pathes=train_latent_2_pathes)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


class latent_Dataset(Dataset):
    def __init__(self, latent_1_pathes, latent_2_pathes):
        self.latent_1_pathes = latent_1_pathes
        self.latent_2_pathes = latent_2_pathes

    def __getitem__(self, index):

        latent_1 = np.load(self.latent_1_pathes[index])
        latent_1 = torch.from_numpy(latent_1).float()[0,:,:,:]

        latent_2 = np.load(self.latent_2_pathes[index])
        latent_2 = torch.from_numpy(latent_2).float()[0,:,:,:]
        return {'latent_1':latent_1,'latent_2':latent_2,
                'latent_1_path': self.latent_1_pathes[index],
                'latent_2_path': self.latent_2_pathes[index]}

    def __len__(self):
        return len(self.latent_1_pathes)
