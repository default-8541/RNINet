import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from torch.utils.data import DataLoader

def create_dataset(data_path,n_channels,noise_opt):
    test_set = DatasetMultiNoise(data_path,n_channels,noise_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    return test_loader


class DatasetMultiNoise(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, data_path,n_channels,noise_opt):
        super(DatasetMultiNoise, self).__init__()
        print('Dataset: MultiNoise. Only dataroot_H is needed.')
        self.paths_H = util.get_image_paths(data_path)
        self.n_channels=n_channels
        self.noise_opt=noise_opt
        self.noise_type=noise_opt["noise_type"]


    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        """
        # --------------------------------
        # get L/H image pairs
        # --------------------------------
        """
        img_H = util.uint2single(img_H)
        img_L = np.copy(img_H)

        # --------------------------------
        # add noise
        # --------------------------------


        img_L = Poisson_Noise(img_L, self.noise_opt)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_L = util.single2tensor3(img_L)
        img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)


def Poisson_Noise(image,noise_opt):
    out = add_poisson_noise(image, scale=noise_opt["alpha"])
    return out

def generate_poisson_noise(img, scale=1.0, gray_noise=False):
    """Generate poisson noise.
    Ref: https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L37-L219
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    if gray_noise:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # round and clip image for counting vals correctly
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(img))
    vals = 2**np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(img * vals) / float(vals))
    noise = out - img
    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return noise * scale


def add_poisson_noise(img, scale=1.0, clip=True, rounds=False, gray_noise=False):
    """Add poisson noise.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        scale (float): Noise scale. Default: 1.0.
        gray_noise (bool): Whether generate gray noise. Default: False.
    Returns:
        (Numpy array): Returned noisy image, shape (h, w, c), range[0, 1],
            float32.
    """
    noise = generate_poisson_noise(img, scale, gray_noise)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out
