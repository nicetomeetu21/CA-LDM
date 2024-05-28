# -*- coding:utf-8 -*-
import os
import shutil
from argparse import ArgumentParser
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torchvision.utils import save_image

from datamodule.tio_datamodule import PatchTioDatamodule
from ldm.modules.diffusionmodules.model import Encoder
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator_no_codebookloss
from networks.VQModel3D_adaptor_333 import Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from utils.util import load_network, save_cube_from_tensor


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='model name')
    parser.add_argument('--result_root', type=str, default='path/to/save/dir')
    parser.add_argument("--command", default="test")
    # tio args
    parser.add_argument('--image_npy_root', type=str, default='path/to/volume/npy')
    parser.add_argument('--train_name_json', type=str,
                        default='train_volume_names.json')
    parser.add_argument('--test_name_json', type=str,
                        default='test_volume_names.json')
    parser.add_argument('--patch_per_size', default=(5, 256, 256))
    parser.add_argument('--image_size', default=(512, 512, 512))
    parser.add_argument('--queue_length', default=40)
    parser.add_argument('--samples_per_volume', default=20)
    # train args
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=16)
    parser.add_argument("--pin_memory", default=True)
    parser.add_argument("--base_lr", type=float, default=3e-4,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', default=1.0)
    # lightning args
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--limit_train_batches", type=int, default=10000)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--precision', default='32')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    # torch.set_float32_matmul_precision('medium')
    return parser


def freeze_except_3d(model):
    for name, param in model.named_parameters():
        if '3d' not in name and 'loss' not in name:  # Check if '3d' is not in parameter name
            param.requires_grad = False
        # else:
        #     param.requires_grad = True
    for name, param in model.named_parameters():
        print(f'{name}: Requires Gradient - {param.requires_grad}')


def main(opts):
    datamodule = PatchTioDatamodule(**vars(opts))
    model = VQModel(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=False, filename="{epoch}", every_n_epochs=1, save_top_k=-1,
                                        save_on_train_epoch_end=True)
        trainer = pl.Trainer(max_epochs=opts.max_epochs, limit_train_batches=opts.limit_train_batches,limit_val_batches=1,
                             accelerator=opts.accelerator,  # strategy=opts.strategy,
                             precision=opts.precision, devices=opts.devices, deterministic=opts.deterministic,
                             default_root_dir=opts.default_root_dir, profiler=opts.profiler,
                             benchmark=opts.benchmark, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=10)])
        ckpt_path = 'path/to/VQVAE2D/ckpt'
        load_network(model, ckpt_path, model.device)
        freeze_except_3d(model)
        trainer.fit(model=model, datamodule=datamodule)
    else:
        ckpt_path = 'path/to/ckpt'
        opts.ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        opts.img_save_dir = os.path.join(opts.default_root_dir, 'test_img2_' + opts.ckpt_name)
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

        ddconfig = {'double_z': False, 'z_channels': 4, 'resolution': 512, 'in_channels': 1, 'out_ch': 1, 'ch': 128,
                    'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        # 400 200 100 50 25
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        lossconfig = dict(disc_conditional=False, disc_in_channels=1, disc_num_layers=2, disc_start=1, disc_weight=0.6,
                          codebook_weight=1.0, perceptual_weight=0.1)
        self.loss = VQLPIPSWithDiscriminator_no_codebookloss(**lossconfig)

        self.embed_dim = ddconfig["z_channels"]
        n_embed = 16384
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        self.lr_g_factor = 1.0

        self.save_hyperparameters()

    @torch.no_grad()
    def get_input(self, batch):
        x = batch['image']['data']
        names = batch['name']
        # print(x.shape)
        x = rearrange(x, "1 c b h w -> b c h w")
        h = self.encode(x)
        return x.detach(), h.detach(), names

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant = self.quantize.forward(h, testing=True)

        quant = self.post_quant_conv(quant)
        return quant

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        if batch_idx == 0:
            self.batch_sample = batch

        x, h, _ = self.get_input(batch)
        xrec = self.decode(h)
        if optimizer_idx == 0:
            # qloss = 0
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            qloss = 0
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def get_last_layer(self):
        return self.decoder.conv_out_3d.weight

    @torch.inference_mode()
    def on_train_epoch_end(self):
        x,h,_ = self.get_input(self.batch_sample)
        # h = self.encode(x)
        xrec = self.decode(h)
        os.makedirs(os.path.join(self.opts.default_root_dir, 'train_progress'), exist_ok=True)
        for i in range(x.shape[0]):
            save_image([x[i, :, :, :] * 0.5 + 0.5, xrec[i, :, :, :] * 0.5 + 0.5],
                       os.path.join(self.opts.default_root_dir, 'train_progress',
                                    str(self.current_epoch) + str(i) + '.png'))

    def val_encode(self, x):
        # x = rearrange(x, "1 c b h w -> b c h w")
        hs = []
        for i in range(x.shape[2]):
            hs.append(self.encode(x[:,:,i,:,:]).to('cpu'))
        hs = torch.stack(hs, dim=2)
        return hs
    def val_decode(self, inputs):
        inputs = rearrange(inputs, "1 c b h w -> b c h w")
        outputs = self.decoder(inputs)
        outputs = rearrange(outputs, "b c h w -> 1 c b h w")
        return outputs


    def test_step(self, batch, batch_idx):
        x = batch['image']['data']
        name = batch['name'][0]
        print(x.shape)
        hs = self.val_encode(x)
        print('hs',hs.shape)
        result = torch.zeros_like(x)

        for i in tqdm(range(2,x.shape[2]-2)):
            inputs = hs[:, :, i - 2:i + 3, :, :].to(self.device)
            print(inputs.shape)
            outputs = self.val_decode(inputs)
            outputs = outputs.to('cpu')
            print(outputs.shape)
            result[:, :, i:i+1, :, :] = outputs[:,:,2,:,:]
            if i == 2:
                result[:,:,i-2:i,:,:] = outputs[:,:,:2,:,:]
            elif i == x.shape[2]-3:
                result[:,:,i+1:i+3,:,:] = outputs[:,:,3:,:,:]
        visuals = result.squeeze() * 0.5 + 0.5
        save_cube_from_tensor(visuals,
                              os.path.join(self.opts.img_save_dir, str(name)))

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
        params = list(filter(lambda p: p.requires_grad, self.encoder.parameters())) + \
                 list(filter(lambda p: p.requires_grad, self.decoder.parameters())) + \
                 list(filter(lambda p: p.requires_grad, self.quantize.parameters())) + \
                 list(filter(lambda p: p.requires_grad, self.quant_conv.parameters())) + \
                 list(filter(lambda p: p.requires_grad, self.post_quant_conv.parameters()))
        # params =
        # print(params, list(filter(lambda p: p.requires_grad, self.decoder.parameters())))
        opt_ae = torch.optim.AdamW(params,
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
                code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
                shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
                print('save in', opts.default_root_dir)
    main(opts)
