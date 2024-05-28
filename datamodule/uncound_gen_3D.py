# -*- coding:utf-8 -*-
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class uncond_gen_Datamodule(pl.LightningDataModule):
    def __init__(self, total_num, batch_size=1, num_workers=8, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_num = total_num

    def setup(self, stage=None):
        self.test_set = uncond_gen_Dataset(total_num=self.total_num)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class uncond_gen_Dataset(Dataset):
    def __init__(self, total_num):
        self.pathes = [str(i) for i in range(total_num)]

    def __getitem__(self, index):
        return {'path': self.pathes[index]}

    def __len__(self):
        return len(self.pathes)
