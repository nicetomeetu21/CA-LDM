# -*- coding:utf-8 -*-
import torch
import os
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn as nn
from networks.ema import LitEma
from networks.openaimodel import UNetModel
from datamodule.latent_datamodule import testDatamodule
from utils.util_for_openai_diffusion import DDPM_base
from utils.util import load_network
from networks.ldm3D_utils.vq_gan_3d.model.vqgan import DecoderSR_old as DecoderSR
from ldm.modules.diffusionmodules.model import Decoder as Decoder2D
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from torchvision.utils import save_image
import numpy as np
from einops import rearrange
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--result_save_dir', type=str, default='path/to/save/dir')
    # data & tio args
    parser.add_argument('--first_stage_ckpt', type=str,
                        default='path/to/NHVQVAE/ckpt')
    parser.add_argument('--ldm1_ckpt', type=str,
                        default='path/to/LDM3D/ckpt')
    parser.add_argument('--ldm2_ckpt', type=str,
                        default='path/to/LDM2D_refiner/ckpt')
    # train args
    parser.add_argument("--batch_size", default=1)
    # lightning args
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--precision', default=32)
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts):
    model = CascadeLDM(opts)
    datamodule = testDatamodule(latent_root='path/to/ldm1_latent')
    trainer = pl.Trainer(accelerator=opts.accelerator, devices=opts.devices, deterministic=opts.deterministic,
                        logger=False, profiler=opts.profiler, benchmark=opts.benchmark)
    trainer.test(model=model, datamodule=datamodule)


class CascadeLDM(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.first_stage_model = VQModelInterface()
        # self.ldm1 = LDM1()
        self.ldm2 = LDM2()
        print('loading first_stage_model')
        load_network(self.first_stage_model, opts.first_stage_ckpt, self.device)
        # print('loading ldm1')
        # load_network(self.ldm1, opts.ldm1_ckpt, self.device)
        print('loading ldm2')
        load_network(self.ldm2, opts.ldm2_ckpt, self.device)
        self.ldm2.switch_to_ema()

        # self.save_dir_1 = os.path.join(opts.result_save_dir, 'ldm1')
        self.save_dir_2 = os.path.join(opts.result_save_dir, 'ldm2')
        self.save_dir_2_latent = os.path.join(opts.result_save_dir, 'ldm2_latent')

    def test_step(self, batch, batch_idx):
        full_pathes = batch['latent_path'][0]

        path = full_pathes.split('/')[-1][:-4]
        print(path)
        z_3d = batch['latent']
        # print(z_3d.shape)

        h_sr = self.first_stage_model.quant_sr_3D(z_3d)
        d_sr = h_sr.shape[-3]
        k=10
        result = torch.zeros_like(h_sr)
        for i in tqdm(range(0, d_sr, k)):
            h_frame = h_sr[0, :, i:i+k, :, :].permute(1,0,2,3)
            bz = h_frame.shape[0]
            h_frame = self.first_stage_model.quant_conv(h_frame)
            h_refine = self.ldm2.sample(c=h_frame, batch_size=bz, return_intermediates=False,
                                                   clip_denoised=True)
            # frame_rec = self.first_stage_model.decode_2D(h_frame, testing=True) * 0.5 + 0.5
            refine_frame_rec = self.first_stage_model.decode_2D(h_refine, testing=True)*0.5+0.5

            names = [str(i + l+1) + '.png' for l in range(bz)]
            # self.save_batch_images(frame_rec,os.path.join(self.save_dir_1, pathes[j]), names)
            self.save_batch_images(refine_frame_rec,os.path.join(self.save_dir_2, path), names)

            h_refine = h_refine.permute(1,0,2,3).unsqueeze(0)
            print(h_refine.shape)
            result[:, :, i:i+k, :, :] = h_refine
            # names = [str(i + l+1) + '.npy' for l in range(bz)]
            # self.save_batch_npys(h_refine.cpu().numpy(),os.path.join(self.save_dir_2_latent, path), names)
        result = result.cpu().numpy()
        np.save(os.path.join(self.save_dir_2_latent, path), result)
    def save_batch_images(self, images, save_dir, names):
        os.makedirs(save_dir, exist_ok=True)
        for i in range(images.shape[0]):
            save_image(images[i],os.path.join(save_dir, names[i]))
    def save_batch_npys(self, images, save_dir, names):
        os.makedirs(save_dir, exist_ok=True)
        for i in range(images.shape[0]):
            np.save(os.path.join(save_dir, names[i]),images[i])
    # def test_step(self, batch, batch_idx):
    #     pathes = batch['path']
    #     c = None
    #     batch_size = len(pathes)
    #     z_3d = self.ldm1.sample(c=c, batch_size=batch_size, return_intermediates=False,
    #                                                clip_denoised=True)
    #     # print(z_3d.shape)
    #     h_sr = self.first_stage_model.quant_sr_3D(z_3d)
    #     d_sr = h_sr.shape[-3]
    #     x_samples = []
    #     x_samples_refine = []
    #     for i in tqdm(range(d_sr)):
    #         h_frame = h_sr[:, :, i, :, :]
    #         h_refine = self.ldm2.sample(c=h_frame, batch_size=batch_size, return_intermediates=False,
    #                                                clip_denoised=True)
    #         # print('eval',h_frame.shape)
    #         frame_rec = self.first_stage_model.decode_2D(h_frame, testing=True)
    #         x_samples.append(frame_rec.cpu())
    #         refine_frame_rec = self.first_stage_model.decode_2D(h_refine, testing=True)
    #         x_samples_refine.append(refine_frame_rec.cpu())
    #     x_samples = torch.stack(x_samples, dim=2)
    #     x_samples_refine = torch.stack(x_samples_refine, dim=2)
    #     # x_samples = self.first_stage_model.decode(z_3d) * 0.5 + 0.5
    #     # x_samples = x_samples.to('cpu')
    #     # print(img_samples.shape)
    #     for i in range(batch_size):
    #         save_cube_from_tensor(x_samples[i].squeeze()*0.5+0.5, os.path.join(self.no_refine_img_save_dir, pathes[i]))
    #
    #     for i in range(batch_size):
    #         save_cube_from_tensor(x_samples_refine[i].squeeze()*0.5+0.5, os.path.join(self.refine_img_save_dir, pathes[i]))

class VQModelInterface(nn.Module):
    def __init__(self):
        super().__init__()

        self.SR3D = DecoderSR(in_channels=4, upsample=[8, 1, 1], image_channel=1, norm_type='group', num_groups=4)
        ddconfig = {'z_channels': 4, 'resolution': 512, 'in_channels': 1, 'out_ch': 1, 'ch': 128,
                    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        self.decoder = Decoder2D(**ddconfig)
        self.embed_dim = 4
        n_embed = 16384
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)
        self.quantize3D = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        # self.quant_conv_3D = torch.nn.Conv3d(self.embed_dim, self.embed_dim, 1)
        self.post_quant_conv_3D = torch.nn.Conv3d(self.embed_dim, self.embed_dim, 1)

    def encode(self, x):
        d, h, w = x.shape[-3:]
        x = F.interpolate(x, size=(d // 2, h // 2, w // 2))
        h = self.encoder3D(x)
        return h

    def decode_2D(self, z, testing=False):
        # z = self.quant_conv(h_frame)
        if testing:
            quant = self.quantize(z, testing=True)
            quant = self.post_quant_conv(quant)
            frame = self.decoder(quant)
            return frame
        else:
            quant, emb_loss, info = self.quantize(z)
            quant = self.post_quant_conv(quant)
            frame = self.decoder(quant)
            return frame, emb_loss

    def quant_sr_3D(self, z):
        z_splits = torch.chunk(z, 10, dim=2)
        embeddings = []
        for z_split in z_splits:
            embedding = self.quantize3D.forward_3D(z_split, testing=True)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=2)
        embeddings = self.post_quant_conv_3D(embeddings)
        h_sr = self.SR3D(embeddings)
        return h_sr

    def decode(self, z):
        # z = self.quant_conv_3D(h)
        h_sr = self.quant_sr_3D(z)
        d_sr = h_sr.shape[-3]
        ret = []
        for i in tqdm(range(d_sr)):
            h_frame = h_sr[:, :, i, :, :]
            # print('eval',h_frame.shape)
            frame_rec = self.decode_2D(h_frame, testing=True)
            ret.append(frame_rec.cpu())
        ret = torch.stack(ret, dim=2)
        return ret


class LDM1(DDPM_base):
    def __init__(self):
        super().__init__()

        latent_size = (64, 64, 64)
        latent_channel = 4

        unet_config = {'image_size': latent_size, 'dims': 3, 'in_channels': latent_channel,
                       'out_channels': latent_channel, 'model_channels': 128,
                       'attention_resolutions': [4, 8], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 8],
                       'num_heads': 8, 'use_scale_shift_norm': True, 'resblock_updown': True}
        self.model = UNetModel(**unet_config)

        self.latent_size = latent_size
        self.channels = latent_channel

        self.log_every_t=100
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.register_schedule()
        self.use_ema = False
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def apply_model(self, x, t, c):
        out = self.model(x=x, timesteps=t)
        return out


class LDM2(DDPM_base):
    def __init__(self):
        super().__init__()
        self.opts = opts
        latent_size = (64, 64)
        latent_channel = 4
        unet_config = {'image_size': latent_size, 'in_channels': latent_channel*2,
                       'out_channels': latent_channel, 'model_channels': 192,
                       'attention_resolutions': [2, 4, 8], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
                       'num_heads': 8, 'use_scale_shift_norm': True, 'resblock_updown': True,
                       }
        self.model = UNetModel(**unet_config)

        self.latent_size = latent_size
        self.channels = latent_channel

        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.log_every_t = 100
        self.use_ema = True
        self.register_schedule()
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def switch_to_ema(self):
        self.model_ema.copy_to(self.model)
        print("Switched to EMA weights")

    def apply_model(self, x, t, c):
        out = self.model(x=torch.cat([x,c], dim=1), timesteps=t)
        return out

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
