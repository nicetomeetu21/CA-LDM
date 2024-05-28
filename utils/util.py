import importlib
import os
import sys
import cv2 as cv
import numpy as np
import torch
from natsort import natsorted
from torchvision import transforms
from torchvision.utils import save_image

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def read_cube_to_np(img_dir, stack_axis=2, cvflag=cv.IMREAD_GRAYSCALE):
    assert os.path.exists(img_dir), f"got {img_dir}"
    print(img_dir)
    imgs = []
    names = natsorted(os.listdir(img_dir))
    for name in names:
        img = cv.imread(os.path.join(img_dir, name), cvflag)
        imgs.append(img)
    imgs = np.stack(imgs, axis=stack_axis)
    return imgs


def read_cube_to_tensor(path, stack_axis=1, cvflag=cv.IMREAD_GRAYSCALE):
    imgs = []
    names = natsorted(os.listdir(path))
    for name in names:
        img = cv.imread(os.path.join(path, name), cvflag)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=stack_axis)
    return imgs

def check_interpolation(imgs, target_cube_size):
    if target_cube_size[-1] != imgs.shape[-1] or target_cube_size[-2] != imgs.shape[-2] or target_cube_size[-3] != \
            imgs.shape[-3]:
        imgs = imgs.unsqueeze(0).unsqueeze(0)
        imgs = torch.nn.functional.interpolate(imgs, size=target_cube_size, mode='trilinear', align_corners=True)
        imgs = imgs.squeeze()
    return imgs

def save_cube_from_tensor(img, result_dir, size=None):
    os.makedirs(result_dir, exist_ok=True)
    if size is not None:
        img = check_interpolation(img, size)
    for j in range(img.shape[0]):
        img_path = os.path.join(result_dir, str(j + 1) + '.png')
        save_image(img[j, :, :], img_path)


def save_cube_from_numpy(data, result_name, tonpy=False):
    if tonpy:
        np.save(result_name + '.npy', data)
    else:
        result_dir = result_name
        if not os.path.exists(result_dir): os.makedirs(result_dir)
        for i in range(data.shape[0]):
            cv.imwrite(os.path.join(result_dir, str(i + 1) + '.png'), data[i, ...])


def get_file_path_from_dir(src_dir, file_name):
    for root, dirs, files in os.walk(src_dir):
        if file_name in files:
            print(root, dirs, files)
            return os.path.join(root, file_name)

def load_network(model, save_path, device, key = 'state_dict'):
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
        raise ('path must exist!')
    else:
        # network.load_state_dict(torch.load(save_path))
        try:
            model.load_state_dict(torch.load(save_path, map_location=device)[key])
        except:
            pretrained_dict = torch.load(save_path, map_location=device)[key]
            model_dict = model.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model.load_state_dict(pretrained_dict)
                print(
                    'Pretrained network has excessive layers; Only loading layers that are used')
            except:
                print(
                    'Pretrained network has fewer layers; The following are not initialized:')
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                    # else:
                    #     print(k, v.size(), model_dict[k].size())

                if sys.version_info >= (3, 0):
                    not_initialized = set()
                else:
                    from sets import Set
                    not_initialized = Set()

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k)

                print(sorted(not_initialized))
                model.load_state_dict(model_dict)