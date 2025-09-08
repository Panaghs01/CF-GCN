import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):
    # imgs: list of np.ndarray (H, W, C)
    imgs = [torch.from_numpy(img.transpose(2, 0, 1)).float() for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    # Assume 12 bands for multispectral, adjust if needed
    mean = [0.5] * imgs[0].shape[0]
    std = [0.5] * imgs[0].shape[0]
    imgs = [TF.normalize(img, mean=mean, std=std) for img in imgs]
    return imgs, labels


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,] (H, W, C)
        :param labels: [ndarray,] (H, W)
        :return: [tensor,],[tensor,]
        """
        # Convert to torch tensors (C, H, W)
        imgs = [torch.from_numpy(img.transpose(2, 0, 1)).float() for img in imgs]
        labels = [torch.from_numpy(np.array(label, np.uint8)).long() for label in labels]

        # Resize
        if not self.img_size_dynamic:
            if imgs[0].shape[1:] != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], antialias=True) for img in imgs]
                labels = [TF.resize(label.unsqueeze(0), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST).squeeze(0) for label in labels]
        else:
            self.img_size = imgs[0].shape[1]

        # Random horizontal flip
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(label) for label in labels]

        # Random vertical flip
        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(label) for label in labels]

        # Random rotation (by 90, 180, 270 degrees)
        if self.with_random_rot and random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(label, angle) for label in labels]

        # Random crop (centered, or random if you want)
        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size).get_params(
                imgs[0], scale=(0.8, 1.0), ratio=(1, 1))
            imgs = [TF.resized_crop(img, i, j, h, w, size=(self.img_size, self.img_size)) for img in imgs]
            labels = [TF.resized_crop(label.unsqueeze(0), i, j, h, w, size=(self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST).squeeze(0) for label in labels]

        # NOTE: with_scale_random_crop and with_random_blur are not implemented here, as they require more custom logic for tensors.
        # You can implement them using torch.nn.functional.interpolate and torch.nn.functional.pad if needed.

        if to_tensor:
            mean = [0.5] * imgs[0].shape[0]
            std = [0.5] * imgs[0].shape[0]
            imgs = [TF.normalize(img, mean=mean, std=std) for img in imgs]

        return imgs, labels
