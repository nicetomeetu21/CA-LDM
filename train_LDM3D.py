# -*- coding:utf-8 -*-
import torch
# torch.set_num_threads(2)
import os
import shutil
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn as nn
from networks.ema import LitEma
from networks.openaimodel import UNetModel
from datamodule.latent_datamodule import trainDatamodule
from utils.util_for_openai_diffusion import DDPM_base, disabled_train, default, LambdaLinearScheduler, \
    extract_into_tensor, noise_like
from utils.util import save_cube_from_tensor, load_network
# from networks.ldm3D_utils.vq_gan_3d.model.vqgan import Encoder as Encoder3D
from networks.ldm3D_utils.vq_gan_3d.model.vqgan import DecoderSR_old as DecoderSR
from ldm.modules.diffusionmodules.model import Decoder as Decoder2D
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--command", default="fit")
    parser.add_argument("--exp_name", default='model name')
    parser.add_argument('--result_root', type=str, default='path/to/save/dir')
    # data & tio args
    parser.add_argument('--first_stage_ckpt', type=str,default='path/to/NHVQVAE/ckpt')
    parser.add_argument('--latent_root', type=str,default='path/to/NHVQVAE/latent')
    parser.add_argument('--train_name_json', type=str,
                        default='train_volume_names.json')
    parser.add_argument('--test_name_json', type=str,
                        default='test_volume_names.json')
    # train args
    parser.add_argument("--latent_size", default=(64, 64, 64))
    parser.add_argument("--latent_channel", default=4)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--pin_memory", default=True)
    parser.add_argument("--base_lr", type=float, default=4.5e-6)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    # lightning args
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--eval_save_every_n_epoch", type=int, default=10)
    parser.add_argument("--limit_train_batches", type=int, default=4000)
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--precision', default=32)
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


def main(opts):
    datamodule = trainDatamodule(**vars(opts))
    model = LDM(opts)
    ckpt_callback = ModelCheckpoint(save_last=True, filename="model-{epoch}",
                                    every_n_epochs=opts.eval_save_every_n_epoch)
    trainer = pl.Trainer(max_epochs=opts.max_epochs, limit_train_batches=opts.limit_train_batches,
                         accelerator=opts.accelerator,  # strategy=opts.strategy,
                         precision=opts.precision, devices=opts.devices, deterministic=opts.deterministic,
                         default_root_dir=opts.default_root_dir, profiler=opts.profiler, benchmark=opts.benchmark,
                         callbacks=[ckpt_callback, TQDMProgressBar(refresh_rate=10)])
    model.instantiate_first_stage(opts)
    # del model.sample_batch
    trainer.fit(model=model, datamodule=datamodule)


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


class LDM(DDPM_base):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.save_hyperparameters()
        #
        # self.instantiate_first_stage(opts)

        unet_config = {'image_size': opts.paded_size, 'dims': 3, 'in_channels': opts.latent_channel,
                       'out_channels': opts.latent_channel, 'model_channels': 128,
                       'attention_resolutions': [4, 8], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 8],
                       'num_heads': 8, 'use_scale_shift_norm': True, 'resblock_updown': True}
        self.model = UNetModel(**unet_config)

        self.latent_size = opts.latent_size
        latent_size_rev = opts.latent_size[::-1]
        paded_size_rev = opts.paded_size[::-1]
        self.p3d = []
        for current_size, target_size in zip(latent_size_rev, paded_size_rev):
            l = (target_size - current_size) // 2
            r = target_size - current_size - l
            self.p3d.append(l)
            self.p3d.append(r)
        print(f'current size: {opts.latent_size}, target size: {opts.paded_size}, calculated pad: {self.p3d}')
        # exit()

        self.channels = opts.latent_channel

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l1"
        self.use_ema = False
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.log_every_t = 100

        self.scale_by_std = False
        scale_factor = 1.0
        if not self.scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.register_schedule()
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def instantiate_first_stage(self, opts):
        model = VQModelInterface()
        print('load vq from', opts.first_stage_ckpt)
        load_network(model, opts.first_stage_ckpt, device=self.device)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    # @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            print("### USING STD-RESCALING ###")
            if self.scale_factor == 1.:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                z = self.get_input(batch)
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())
                print(f"setting self.scale_factor to {self.scale_factor}")
            else:
                print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def apply_model(self, x, t, c):
        out = self.model(x=x, timesteps=t)
        return out

    def get_input(self, batch):
        z = batch['latent']
        return z

    def training_step(self, batch, batch_idx):
        z = self.get_input(batch)
        # print(torch.max(z), torch.min(z))
        # exit()
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        loss = self.p_losses(z, t)

        # if batch_idx == 0 and self.current_epoch==0:
        #     self.sample_batch = batch
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.current_epoch % opts.eval_save_every_n_epoch: return
        img_save_dir = os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch))
        os.makedirs(img_save_dir, exist_ok=True)

        with self.ema_scope("Plotting"):
            x_samples = self.sample(batch_size=1, return_intermediates=False, clip_denoised=True)
            x_samples = self.decode_first_stage(x_samples) * 0.5 + 0.5
            x_samples = x_samples.squeeze().to('cpu')
            # print(img_samples.shape)
            save_cube_from_tensor(x_samples, os.path.join(img_save_dir, 'sample'))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.apply_model(x_noisy, t, None)
        target = noise

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss = loss.mean()
        self.log('diffusion loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, c, t, clip_denoised, ret_x0=False):
        model_out = self.apply_model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if ret_x0:
            return model_mean, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, temperature=1., noise_dropout=0., repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, c, shape, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        # intermediates = [x]
        if return_intermediates:
            intermediates_x0 = [x]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps,
                  mininterval=2) as pbar:
            for i in pbar:
                x = self.p_sample(x, c, torch.full((b,), i, device=device, dtype=torch.long),
                                  clip_denoised=clip_denoised)
                if return_intermediates and (i % log_every_t == 0 or i == self.num_timesteps - 1):
                    intermediates_x0.append(x)
        if return_intermediates:
            return x, intermediates_x0
        return x

    @torch.no_grad()
    def sample(self, c=None, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop(c, [batch_size, self.channels] + list(self.latent_size),
                                  return_intermediates=return_intermediates, clip_denoised=clip_denoised)

    def configure_optimizers(self):
        base_lr = self.opts.base_lr
        accumulate_grad_batches = self.opts.accumulate_grad_batches
        batch_size = self.opts.batch_size
        devices, nodes = self.trainer.num_devices, self.trainer.num_nodes
        base_batch_size = 1
        # total_steps = self.trainer.estimated_stepping_batches
        lr = base_lr * devices * nodes * batch_size * accumulate_grad_batches / base_batch_size
        print(
            "Setting learning rate to {:.2e} = {:.2e} (base_lr) * {} (batchsize) * {} (accumulate_grad_batches) * {} (num_gpus) * {} (num_nodes) / {} (base_batch_size)".format(
                lr, base_lr, batch_size, accumulate_grad_batches, devices, nodes, base_batch_size))
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        scheduler_config = {'warm_up_steps': [10000], 'cycle_lengths': [10000000000000], 'f_start': [1e-06],
                            'f_max': [1.0],
                            'f_min': [1.0]}
        scheduler = LambdaLinearScheduler(**scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler


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
    if opts.command == 'fit':
        opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name)
        if os.getenv("LOCAL_RANK", '0') == '0':
            if not os.path.exists(opts.default_root_dir):
                os.makedirs(opts.default_root_dir)
                code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
                shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
                print('save in', opts.default_root_dir)
    main(opts)
