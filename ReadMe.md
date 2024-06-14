Official implementation of the paper "Memory-efficient High-resolution OCT Volume Synthesis with Cascaded Amortized Latent Diffusion Models" [Arxiv](https://arxiv.org/html/2405.16516v1).

Our codebase builds on [LatentDiffusion](https://github.com/CompVis/latent-diffusion), [MedicalDiffusion](https://github.com/FirasGit/medicaldiffusion), and [MONAI](https://docs.monai.io/en/stable/networks.html#vitautoenc).

Our proposed method is the first to synthesize 3D volumetric images with a resolution of 512^3 using only 24GB of GPU memory.

# Data

Public data 'OCTA-500' can be downloaded at: https://ieee-dataport.org/open-access/octa-500.

The partitions of data of our experiments are provided at `train_volume_names.json` and `test_volume_names.json`.


# Train

Acutally, there are many routes to train our model. Here we provide a more stable version recently discovered, which differs slightly from the version described in the paper.

1. Train a 2D VQVAE. Run `train_VQVAE2D.py` and fill following args:

```python
parser.add_argument("--exp_name", type=str, default='dirname')
parser.add_argument('--result_root', type=str, default='path/to/dir')
parser.add_argument('--data_root', type=str, default='path/to/OCT')
```

2. Train NHVQVAE. Run `train_NHVQVAE.py` and fill following args:
```python
parser.add_argument("--exp_name", type=str, default='model name')
parser.add_argument('--result_root', type=str, default='path/to/save/dir')
parser.add_argument('--data_root', type=str, default='path/to/OCT')
parser.add_argument('--image_npy_root', type=str,default='path/to/volume/npy')
```

3. Train LDM3D. Run `train_LDM3D.py` and fill following args:
```python
parser.add_argument("--exp_name", default='model name')
parser.add_argument('--result_root', type=str, default='path/to/save/dir')
parser.add_argument('--first_stage_ckpt', type=str,default='path/to/NHVQVAE/ckpt')
parser.add_argument('--latent_root', type=str,default='path/to/NHVQVAE/latent')
```
4. Train LDM2D_refiner. Run `train_LDM2D_refiner.py` and fill following args:
```python
parser.add_argument("--exp_name", type=str, default='dirname')
parser.add_argument('--result_root', type=str, default='path/to/dir')
parser.add_argument('--first_stage_ckpt', type=str, default='path/to/vqgan2d/ckpt')
parser.add_argument('--latent_1_root', type=str, default='path/to/3D/latent')
parser.add_argument('--latent_2_root', type=str, default='path/to/2D/latent')
```

5. Train multi-slice decoder. Run `train_VQVAE_w_adaptor.py` and fill following args:
```python
parser.add_argument("--exp_name", type=str, default='model name')
parser.add_argument('--result_root', type=str, default='path/to/save/dir')
parser.add_argument('--image_npy_root', type=str, default='path/to/volume/npy')
```


# Test

We split the generation procedure into three stages.

1.  Generate 3D latents. Run `test_LDM3D.py` and fill following args:
```python
parser.add_argument('--result_save_dir', type=str, default='path/to/save/dir')
parser.add_argument('--first_stage_ckpt', type=str,
default='path/to/NHVQVAE/ckpt')
parser.add_argument('--ldm1_ckpt', type=str,
default='path/to/LDM3D/ckpt')
parser.add_argument('--ldm2_ckpt', type=s
```

2.  Refine latents. Run `test_LDM2D_refiner.py` and fill following args:
```python
parser.add_argument('--result_save_dir', type=str, default='path/to/save/dir')
parser.add_argument('--first_stage_ckpt', type=str, default='path/to/NHVQVAE/ckpt')
parser.add_argument('--ldm1_ckpt', type=str, default='path/to/LDM3D/ckpt')
parser.add_argument('--ldm2_ckpt', type=str, default='path/to/LDM2D_refiner/ckpt')
datamodule = testDatamodule(latent_root='path/to/ldm1_latent')
```

3.  Decode latents to images. Run `test_decodebyMSDecoder.py` and fill following args:
```python
parser.add_argument('--result_save_dir', type=str,default='path/to/save/dir')
parser.add_argument('--result_save_name', type=str, default='save name')
parser.add_argument('--first_stage_ckpt', type=str, default='path/to/VQVAE_w_adaptor/ckpt')
parser.add_argument('--ldm2_latent', type=str, default='path/to/saved/ldm2_latent')
```


