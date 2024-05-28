# -*- coding:utf-8 -*-
import os
import shutil
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from datamodule.uncond2D_datamodule import trainDatamodule
from torchvision.utils import save_image
from utils.util import load_network
import numpy as np
def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='dirname')
    parser.add_argument('--result_root', type=str, default='path/to/dir')
    parser.add_argument("--command", default="test")
    # tio args
    parser.add_argument('--data_root', type=str,
                        default='path/to/OCT')
    parser.add_argument('--train_name_json', type=str,default='train_volume_names.json')
    parser.add_argument('--test_name_json', type=str, default='train_volume_names.json')
    parser.add_argument('--image_size', default=[512,512])
    parser.add_argument('--crop_size', default=[512,512])
    # train args
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=32)
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
    parser.add_argument('--devices', default=[1])
    parser.add_argument('--reproduce', type=int, default=False)
    # torch.set_float32_matmul_precision('medium')
    return parser


def main(opts):
    datamodule = trainDatamodule(**vars(opts))
    model = VQModel(opts)
    if opts.command == "fit":
        ckpt_callback = ModelCheckpoint(save_last=False, filename="{epoch}", every_n_epochs=1, save_top_k=-1,
                                        save_on_train_epoch_end=True)
        trainer = pl.Trainer(max_epochs=opts.max_epochs, limit_train_batches=opts.limit_train_batches,
                             accelerator=opts.accelerator,  # strategy=opts.strategy,
                             precision=opts.precision, devices=opts.devices, deterministic=opts.deterministic,
                             default_root_dir=opts.default_root_dir, profiler=opts.profiler,
                             benchmark=opts.benchmark, callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=10)])
        trainer.fit(model=model, datamodule=datamodule)
    else:
        ckpt_path = ''
        opts.ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        opts.img_save_dir = os.path.join(opts.default_root_dir, 'test_img_'+ opts.ckpt_name)
        opts.latent_save_dir = os.path.join(opts.default_root_dir, 'gen_latent_after_post_quant_conv' + '_' + opts.ckpt_name)
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


        self.embed_dim = ddconfig["z_channels"]
        n_embed = 16384
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        self.lr_g_factor = 1.0

        self.automatic_optimization = False
        if opts.command == 'fit':
            self.save_hyperparameters()

            lossconfig = dict(disc_conditional=False, disc_in_channels=1, disc_num_layers=2, disc_start=1, disc_weight=0.6,
                      codebook_weight=1.0, perceptual_weight=0.1)
            self.loss2 = VQLPIPSWithDiscriminator(**lossconfig)

    def get_input(self, batch):
        x = batch['image']
        return x

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False, testing=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if testing:
            return dec, quant
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def cal_d_loss(self, x, xrec, qloss):
        # discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx=1, global_step=self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return discloss

    def cal_g_loss(self, x, xrec, qloss):
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx=0, global_step=self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def training_step(self, batch, batch_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        g_opt, d_opt = self.optimizers()

        x = self.get_input(batch)
        if batch_idx == 0:
            self.x_sample = x
        # print(x.shape)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        # print(x.shape, xrec.shape)
        d_loss = self.cal_d_loss(x, xrec, qloss)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        self.clip_gradients(d_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        d_opt.step()

        g_loss = self.cal_g_loss(x, xrec, qloss)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.clip_gradients(g_opt, gradient_clip_val=1, gradient_clip_algorithm="norm")
        g_opt.step()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.inference_mode()
    def on_train_epoch_end(self):
        x = self.x_sample.to(self.device)
        xrec, _ = self(x, return_pred_indices=False)
        os.makedirs(os.path.join(self.opts.default_root_dir, 'train_progress'), exist_ok=True)
        for i in range(x.shape[0]):
            save_image([x[i] * 0.5 + 0.5, xrec[i] * 0.5 + 0.5],
                       os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch)+str(i) + '.png'))
    def test_step(self, batch, batch_idx):
        x = batch['image']
        pathes = batch['path']
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        quant = self.post_quant_conv(quant)
        xrec = self.decoder(quant)
        #
        quant = quant.cpu().numpy()
        xrec = xrec*0.5+0.5
        for i, path in enumerate(pathes):
            names = pathes[i].split('/')
            save_dir = '/'.join([self.opts.img_save_dir] + [names[-2]])
            os.makedirs(save_dir, exist_ok=True)
            save_image(xrec, os.path.join(save_dir, names[-1]))
            save_dir = '/'.join([self.opts.latent_save_dir] + [names[-2]])
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, names[-1][:-4] + '.npy'), quant)

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
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quantize.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()),
                                   lr=lr, betas=(.9, .95), weight_decay=0.05)
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                     lr=lr, betas=(.9, .95), weight_decay=0.05)
        return [opt_ae, opt_disc], []

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=self.device)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")


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
