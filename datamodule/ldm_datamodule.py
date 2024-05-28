# -*- coding:utf-8 -*-
import json
import os

import cv2 as cv
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from natsort import natsorted
import numpy as np


class trainDatamodule(pl.LightningDataModule):
    def __init__(self, image_root, latent_root, train_name_json, test_name_json, batch_size=1, num_workers=8,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_root = image_root
        self.latent_root = latent_root
        with open(train_name_json, "r") as f:
            self.train_cube_names = json.load(f)
        with open(test_name_json, "r") as f:
            self.test_cube_names = json.load(f)

    def setup(self, stage=None):
        train_pathes = []
        train_latent_pathes = []
        for cube_name in self.train_cube_names:
            img_names = natsorted(os.listdir(os.path.join(self.img_root, cube_name)))
            for img_name in img_names:
                train_pathes.append(os.path.join(self.img_root, cube_name, img_name))
                train_latent_pathes.append(os.path.join(self.latent_root, cube_name))
        test_pathes = []
        test_latent_pathes = []
        for cube_name in self.test_cube_names:
            img_names = natsorted(os.listdir(os.path.join(self.img_root, cube_name)))
            for img_name in img_names:
                test_pathes.append(os.path.join(self.img_root, cube_name, img_name))
                test_latent_pathes.append(os.path.join(self.latent_root, cube_name))

        print(f'train len: {len(train_pathes)}  test len: {len(test_pathes)}')
        self.train_set = img_latent_Dataset(img_pathes=train_pathes, latent_pathes=train_latent_pathes)
        self.test_set = img_latent_Dataset(img_pathes=test_pathes, latent_pathes=test_latent_pathes)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class img_latent_Dataset(Dataset):
    def __init__(self, img_pathes, latent_pathes):
        self.img_pathes = img_pathes
        self.latent_pathes = latent_pathes

    def __getitem__(self, index):
        # print(self.img_pathes[index], self.latent_pathes[index])
        img = cv.imread(self.img_pathes[index], cv.IMREAD_COLOR)
        splits = self.img_pathes[index].split('/')
        img_id = int(splits[-1][:-4])
        if img_id == 1:
            pre_img = np.zeros_like(img)
        else:
            pre_name = str(img_id - 1) + '.png'
            pre_path = '/'.join(splits[:-1] + [pre_name])
            pre_img = cv.imread(pre_path, cv.IMREAD_COLOR)

        # latent = torch.load(self.latent_pathes[index], map_location=torch.device('cpu'))

        img = transforms.ToTensor()(img)
        img -= 0.5
        img /= 0.5
        pre_img = transforms.ToTensor()(pre_img)
        pre_img -= 0.5
        pre_img /= 0.5

        return {'image': img, 'pre_image': pre_img, 'img_path': self.img_pathes[index],
                'latent_path': self.latent_pathes[index]}

    def __len__(self):
        return len(self.img_pathes)
