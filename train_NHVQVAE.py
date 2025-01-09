# -*- coding:utf-8 -*-
import torch
# torch.set_num_threads(8)
from argparse import ArgumentParser
import os
import shutil

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from datamodule.tio_datamodule import PatchTioDatamodule
from pytorch_lightning.callbacks import ModelCheckpoint
from networks.ldm3D_utils.vq_gan_3d.model.vqgan import DecoderSR_old_v2 as DecoderSR

from networks.ldm3D_utils.vq_gan_3d.model.vqgan import Encoder as Encoder3D
from ldm.modules.diffusionmodules.model import Decoder as Decoder2D
from ldm.modules.diffusionmodules.model import Encoder as Encoder2D
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator_w_latentloss as VQLPIPSWithDiscriminator
from torchvision.utils import save_image
from utils.util import load_network
from utils.util_for_openai_diffusion import disabled_train


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='model name')
    parser.add_argument('--result_root', type=str, default='path/to/save/dir')
    parser.add_argument("--command", default="test")
    # tio args
    parser.add_argument('--image_npy_root', type=str,default='path/to/volume/npy')
    parser.add_argument('--train_name_json', type=str,default='train_volume_names.json')
    parser.add_argument('--test_name_json', type=str,default='train_volume_names.json')
    parser.add_argument('--patch_per_size', default=(256, 256, 256))
    parser.add_argument('--image_size', default=(512, 512, 512))
    parser.add_argument('--queue_length', default=100)
    parser.add_argument('--samples_per_volume', default=5)
    # train args
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=16)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    # lightning args
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--limit_train_batches", type=int, default=5000)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--precision', default=32)
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts):
    datamodule = PatchTioDatamodule(**vars(opts))

    if opts.command == "fit":
        model = VQModel(opts)
        ckpt_callback = ModelCheckpoint(save_last=False, filename="{epoch}",
                                        every_n_epochs=opts.check_val_every_n_epoch, save_top_k=-1)
        trainer = pl.Trainer(max_epochs=opts.max_epochs, limit_train_batches=opts.limit_train_batches,
                             num_sanity_val_steps=0, limit_val_batches=1,
                             accelerator=opts.accelerator, check_val_every_n_epoch=opts.check_val_every_n_epoch,
                             precision=opts.precision, devices=opts.devices, deterministic=opts.deterministic,
                             default_root_dir=opts.default_root_dir, profiler=opts.profiler,
                             benchmark=opts.benchmark, callbacks=[ckpt_callback])
        ckpt_path = 'path/to/VQVAE2D/ckpt'
        load_network(model, ckpt_path, device=model.device)

        # ckpt_path2 = 'path/to/NHVQVAE/ckpt'
        # load_network(model, ckpt_path2, device=model.device)
        # model
        model.decoder.train = disabled_train
        model.encoder.train = disabled_train
        model.post_quant_conv.train = disabled_train
        model.quant_conv.train = disabled_train
        model.quantize.train = disabled_train

        for param in model.decoder.parameters():
            param.requires_grad = False
        # for param in model.decoder.
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.post_quant_conv.parameters():
            param.requires_grad = False
        for param in model.quant_conv.parameters():
            param.requires_grad = False
        for param in model.quantize.parameters():
            param.requires_grad = False
        for param in model.decoder.conv_out.parameters():
            param.requires_grad = True
        trainer.fit(model=model, datamodule=datamodule)

    elif opts.command == "test":
        ckpt_path = ''
        opts.ckpt_name = ckpt_path.split('/')[-1].split('.')[0]

        model = VQModel(opts)
        load_network(model, ckpt_path, model.device)
        trainer = pl.Trainer(accelerator=opts.accelerator, devices=opts.devices, deterministic=opts.deterministic,
                             default_root_dir=opts.default_root_dir, profiler=opts.profiler, logger=False,
                             benchmark=opts.benchmark)
        trainer.test(model=model, datamodule=datamodule)


class VQModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # 400 200 100 50 25
        self.encoder3D = Encoder3D(z_channels=4, n_hiddens=64, downsample=[4, 4, 4], image_channel=1, norm_type='group',
                                   padding_type='replicate', num_groups=32)
        ddconfig = {'z_channels': 4, 'resolution': 512, 'in_channels': 1, 'out_ch': 1, 'ch': 128,
                    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}

        self.SR3D = DecoderSR(in_channels=4, upsample=[8, 1, 1], image_channel=1, norm_type='group', num_groups=4)
        self.decoder = Decoder2D(**ddconfig)
        self.embed_dim = 4
        n_embed = 16384
        self.quantize3D = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv_3D = torch.nn.Conv3d(self.embed_dim, self.embed_dim, 1)
        self.post_quant_conv_3D = torch.nn.Conv3d(self.embed_dim, self.embed_dim, 1)
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.embed_dim, 1)

        self.lr_g_factor = 1.0

        if opts.command == 'fit':
            self.save_hyperparameters()

            self.encoder = Encoder2D(**ddconfig)

            lossconfig = dict(disc_conditional=False, disc_in_channels=1, disc_num_layers=2, disc_start=1,
                              disc_weight=0.6,
                              codebook_weight=1.0, perceptual_weight=0.1)
            self.loss = VQLPIPSWithDiscriminator(**lossconfig)
            self.l1 = nn.L1Loss()

            os.makedirs(os.path.join(self.opts.default_root_dir, 'train_progress'), exist_ok=True)

            self.sample_batch = None

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_input(self, batch):
        x = batch['image']['data']
        return x

    def encode_3D(self, x, testing=False):
        d, h, w = x.shape[-3:]
        x = F.interpolate(x, size=(d // 2, h // 2, w // 2))
        h = self.encoder3D(x)
        # 3D VQ
        if testing:
            z = self.quant_conv_3D(h)
            z_splits = torch.chunk(z, 5, dim=2)
            embeddings = []
            for z_split in z_splits:
                embedding = self.quantize3D.forward_3D(z_split, testing=True)
                embeddings.append(embedding)
            embeddings = torch.cat(embeddings, dim=2)
            embeddings = self.post_quant_conv_3D(embeddings)
            h_sr = self.SR3D(embeddings)
            return h_sr
        else:
            h = self.quant_conv_3D(h)
            h, emb_loss, info = self.quantize3D.forward_3D(h)
            h = self.post_quant_conv_3D(h)
            h_sr = self.SR3D(h)
            return h_sr, emb_loss

    def encode_3D_nosr(self, x):
        d, h, w = x.shape[-3:]
        x = F.interpolate(x, size=(d // 2, h // 2, w // 2))
        h = self.encoder3D(x)
        return h

    def encode_2D(self, x):
        h = self.encoder(x)
        return h

    def decode_2D(self, h_frame, testing=False):
        z = self.quant_conv(h_frame)
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

    def forward(self, x):
        num = 5
        n, c, d, h, w = x.shape
        # id2 = torch.randint(0, d, (num,), device=self.device)
        id = torch.randperm(d, device=self.device)[:num]
        # print(id, id2)
        # print(id.size(),id2.size())
        id = id.view(1, 1, -1, 1, 1)
        h_3D, emb_loss1 = self.encode_3D(x)
        n, lc, d, lh, lw = h_3D.shape
        latent_id = id.expand(n, lc, num, lh, lw)
        h_3D_selected = torch.gather(h_3D, 2, latent_id)
        # print(h_3D_selected.shape)
        h_3D_selected = h_3D_selected.squeeze(0).permute(1, 0, 2, 3)
        # h_3D_selected =h_3D[:,:,id,:,:].squeeze(2)
        # h_3D_selected = h_3D_selected.permute(0, 2, 1, 3, 4).contiguous()
        # print(h_3D_selected.shape)
        # print(n*num,c,h,w)
        # h_3D_selected = h_3D_selected.view(n*num,c,h,w)
        # print(h_3D_selected.shape)

        # print('train', h_3D_selected.shape)
        frame_rec_3D, emb_loss2 = self.decode_2D(h_3D_selected)

        frame_id = id.expand(n, c, num, h, w)
        frame_target = torch.gather(x, 2, frame_id)
        frame_target = frame_target.squeeze(0).permute(1, 0, 2, 3)
        # frame_target = x[:,:,id,:,:].squeeze(2)
        # frame_target = frame_target.permute(0, 2, 1, 3, 4).contiguous()
        # frame_target = frame_target.view(n*num,c,h,w)

        h_2D = self.encode_2D(frame_target)

        latent_loss = self.l1(h_3D_selected, h_2D.detach())
        # frame_rec_2D, emb_loss2 = self.decode_2D(h_2D)
        return frame_target, frame_rec_3D, latent_loss, emb_loss1 + emb_loss2

    def check_forward(self, x):
        num = 3
        n, c, d, h, w = x.shape

        id = torch.randperm(d, device=self.device)[:num]
        id = id.view(1, 1, -1, 1, 1)

        h_3D = self.encode_3D(x, testing=True)
        n, lc, d, lh, lw = h_3D.shape
        latent_id = id.expand(n, lc, num, lh, lw)
        h_3D_selected = torch.gather(h_3D, 2, latent_id)
        h_3D_selected = h_3D_selected.squeeze(0).permute(1, 0, 2, 3)
        frame_rec_3D = self.decode_2D(h_3D_selected, testing=True)

        frame_id = id.expand(n, c, num, h, w)
        frame_target = torch.gather(x, 2, frame_id)
        frame_target = frame_target.squeeze(0).permute(1, 0, 2, 3)

        h_2D = self.encode_2D(frame_target)
        frame_rec_2D = self.decode_2D(h_2D, testing=True)
        return frame_target, frame_rec_3D, frame_rec_2D

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        if batch_idx == 0:
            self.sample_batch = batch

        x = self.get_input(batch)
        frame_target, frame_rec_3D, latent_loss, emb_loss = self(x)
        target = frame_target
        rec = frame_rec_3D

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(emb_loss, latent_loss, target, rec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer())
            self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(emb_loss, latent_loss, target, rec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer())
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.sample_batch is None: return
        batch = self.sample_batch
        x = batch['image']['data']
        name = batch['name'][0]

        # save_dir = os.path.join(self.opts.default_root_dir, 'train_visual', str(self.current_epoch) + '_' + name)
        # os.makedirs(save_dir, exist_ok=True)

        # h_sr = self.encode_3D(x)
        # d_sr = h_sr.shape[-3]
        # for i in tqdm.tqdm(range(d_sr)):
        #     h_frame = h_sr[:, :, i, :, :]
        #     # print('eval',h_frame.shape)
        #     frame_rec = self.decode_2D(h_frame, testing=True)
        #     frame_target = x[:, :, i, :, :]
        #     visuals = torch.cat([frame_target, frame_rec]) * 0.5 + 0.5
        #     save_image(visuals, os.path.join(save_dir, str(i + 1) + '.png'))

        frame_target, frame_rec_3D, frame_rec_2D = self.check_forward(x)
        visuals = torch.cat([frame_target, frame_rec_3D, frame_rec_2D], dim=0) * 0.5 + 0.5
        save_image(visuals,
                   os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch) + '.png'))

    def validation_step(self, batch, batch_idx):
        x = batch['image']['data']
        name = batch['name'][0]

        h_3D = self.encode_3D(x, testing=True)
        d_sr = h_3D.shape[-3]

        save_dir = os.path.join(self.opts.default_root_dir, 'val_visual', str(self.current_epoch) + '_' + name)
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm.tqdm(range(d_sr)):
            h_frame = h_3D[:, :, i, :, :]
            frame_rec = self.decode_2D(h_frame, testing=True)
            frame_target = x[:, :, i, :, :]
            visuals = torch.cat([frame_target, frame_rec]) * 0.5 + 0.5
            save_image(visuals, os.path.join(save_dir, str(i + 1) + '.png'))

    def test_step(self, batch, batch_idx):
        x = batch['image']['data']
        name = batch['name'][0]
        h_3D = self.encode_3D(x, testing=True)
        # h_3D = self.encode_3D_nosr(x)
        # latent_save_dir = os.path.join(self.opts.default_root_dir, 'gen_latent' + '_' + opts.ckpt_name)
        # os.makedirs(latent_save_dir, exist_ok=True)
        # h_np = h_3D.cpu().numpy()
        # np.save(os.path.join(latent_save_dir, name + '.npy'), h_np)
        d_sr = h_3D.shape[-3]
        save_dir = os.path.join(self.opts.default_root_dir, 'test_visual' + '_' + opts.ckpt_name, name)
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)

        for i in tqdm.tqdm(range(d_sr)):
            h_frame = h_3D[:, :, i, :, :]
            frame_rec = self.decode_2D(h_frame, testing=True)
            # frame_target = x[:, :, i, :, :]
            # visuals = torch.cat([frame_target, frame_rec])
            visuals = frame_rec * 0.5 + 0.5
            save_image(visuals, os.path.join(save_dir, str(i + 1) + '.png'))

    def configure_optimizers(self):
        base_lr = self.opts.base_lr
        accumulate_grad_batches = self.opts.accumulate_grad_batches
        batch_size = self.opts.batch_size
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        base_batch_size = 1
        total_steps = self.trainer.estimated_stepping_batches
        lr = base_lr * devices * nodes * batch_size * accumulate_grad_batches / base_batch_size
        print(
            "Setting learning rate to {:.2e} = {:.2e} (base_lr) * {} (batchsize) * {} (accumulate_grad_batches) * {} (num_gpus) * {} (num_nodes) / {} (base_batch_size)".format(
                lr, base_lr, batch_size, accumulate_grad_batches, devices, nodes, base_batch_size))
        print('estimated_stepping_batches:', total_steps)
        # lr = self.cfg.lr
        opt_ae = torch.optim.AdamW(list(self.encoder3D.parameters()) +
                                   list(self.SR3D.parameters())+
                                   list(self.quant_conv_3D.parameters())+
                                   list(self.post_quant_conv_3D.parameters())
                                   # list(self.encoder.parameters()) +
                                   # list(self.decoder.parameters()) +
                                   # list(self.quant_conv.parameters()) +
                                   # list(self.post_quant_conv.parameters()) +
                                   # list(self.quantize.parameters()),
                                   ,
                                   lr=lr, betas=(.9, .95), weight_decay=0.05)
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                     lr=lr, betas=(.9, .95), weight_decay=0.05)
        return [opt_ae, opt_disc], []


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
    opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name)
    if opts.command == 'fit':
        if os.getenv("LOCAL_RANK", '0') == '0':
            if not os.path.exists(opts.default_root_dir):
                os.makedirs(opts.default_root_dir)
                code_dir = os.path.abspath(os.getcwd())
                print(code_dir)
                shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
                print('save in', opts.default_root_dir)
    main(opts)
