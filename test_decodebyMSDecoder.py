# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from collections import namedtuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from datamodule.dual_latent_datamodule import test_single_latent_Datamodule
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from utils.util import save_cube_from_tensor, load_network
from einops import rearrange
import numpy as np
from networks.VQModel3D_adaptor_333 import Decoder


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--result_save_dir', type=str,
                        default='path/to/save/dir')
    parser.add_argument('--result_save_name', type=str,
                        default='save name')
    # data & tio args
    parser.add_argument('--first_stage_ckpt', type=str,
                        default='path/to/VQVAE_w_adaptor/ckpt')
    parser.add_argument('--ldm2_latent', type=str,
                        default='path/to/saved/ldm2_latent')
    # train args
    parser.add_argument("--batch_size", default=1)
    # lightning args
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--precision', default='32')
    parser.add_argument('--devices', default=[1])
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts):
    model = CascadeLDM(opts)
    datamodule = test_single_latent_Datamodule(latent_root=opts.ldm2_latent)
    trainer = pl.Trainer(accelerator=opts.accelerator, devices=opts.devices, deterministic=opts.deterministic,
                         logger=False, profiler=opts.profiler, benchmark=opts.benchmark, precision=opts.precision)
    trainer.test(model=model, datamodule=datamodule)


class CascadeLDM(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.first_stage_model = VQModelInterface()
        print('loading first_stage_model')
        load_network(self.first_stage_model, opts.first_stage_ckpt, self.device)

        self.save_dir_3 = os.path.join(opts.result_save_dir, opts.result_save_name)
        os.makedirs(self.save_dir_3, exist_ok=True)

    def test_step(self, batch, batch_idx):
        pathes = batch['latent_3D_path'][0]
        name =pathes.split('/')[-1][:-4]
        print(pathes, name)
        z_3d = batch['latent_3D']
        print(z_3d.shape)
        d_sr = z_3d.shape[-3]
        result = torch.zeros([1,1,400,640,400], device=self.device)
        for i in tqdm(range(2, d_sr - 2)):
            inputs = z_3d[:, :, i - 2:i + 3, :, :].to(self.device)
            # print(inputs.shape)
            outputs = self.first_stage_model.decode_2D(inputs)
            # outputs = outputs.to('cpu')
            # print(outputs.shape)
            result[:, :, i:i + 1, :, :] = outputs[:, :, 2, :, :]
            if i == 2:
                result[:, :, i - 2:i, :, :] = outputs[:, :, :2, :, :]
            elif i == d_sr - 3:
                result[:, :, i + 1:i + 3, :, :] = outputs[:, :, 3:, :, :]
        visuals = result.squeeze() * 0.5 + 0.5
        cube_dir = os.path.join(self.save_dir_3, name)
        save_cube_from_tensor(visuals, cube_dir)

    def decode_and_save_3D(self, z_3d, name):
        img_3d = self.first_stage_model.decode_3D(z_3d) * 0.5 + 0.5
        save_cube_from_tensor(img_3d.squeeze(), os.path.join(self.save_dir_1, name))

    def save_batch_images(self, images, save_dir, names):
        for i in range(images.shape[0]):
            save_image(images[i], os.path.join(save_dir, names[i]))


class VQModelInterface(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = 4
        n_embed = 16384
        ddconfig = {'z_channels': 4, 'resolution': 512, 'in_channels': 1, 'out_ch': 1, 'ch': 128,
                    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        self.decoder = Decoder(**ddconfig)


        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)

    def decode_2D(self, z):
        # z = self.quant_conv(h_frame)
        z = rearrange(z, "1 c b h w -> b c h w")
        quant = self.quantize(z, testing=True)
        quant = self.post_quant_conv(quant)
        outputs = self.decoder(quant)
        outputs = rearrange(outputs, "b c h w -> 1 c b h w")
        return outputs




if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    if opts.reproduce:
        pl.seed_everything(42, workers=True)
        opts.deterministic = True
        opts.benchmark = False
    else:
        opts.deterministic = False
        opts.benchmark = True
    main(opts)
