import random
import numpy as np

from PIL import Image
from PIL import ImageFilter
import rasterio
import data_config
from rasterio.plot import show
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
import cv2


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)

def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


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
            mean=None,
            std=None
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
        self.mean = mean
        self.std = std


    def transform(self, imgs, labels, to_tensor=True, rasterio_read=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        
        if rasterio_read:

            # Use numpy and torchvision transforms
            random_base = 0.5
            # Random horizontal flip
            if self.with_random_hflip and random.random() > 0.5:
                imgs = [np.flip(img, axis=1) for img in imgs]
                labels = [np.flip(img, axis=1) for img in labels]
            # Random vertical flip
            if self.with_random_vflip and random.random() > 0.5:
                imgs = [np.flip(img, axis=0) for img in imgs]
                labels = [np.flip(img, axis=0) for img in labels]
            # Random rotation (90, 180, 270 degrees)
            if self.with_random_rot and random.random() > random_base:
                angles = [1, 2, 3]  # 90,180,270
                index = random.randint(0, 2)
                angle = angles[index]
                imgs = [np.rot90(img, angle) for img in imgs]
                labels = [np.rot90(img, angle) for img in labels]
            # Random crop
            if self.with_random_crop and random.random() > 0:
                h, w = imgs[0].shape[:2]
                ch, cw = self.img_size, self.img_size
                if h > ch and w > cw:
                    top = np.random.randint(0, h - ch)
                    left = np.random.randint(0, w - cw)
                    imgs = [img[top:top+ch, left:left+cw] for img in imgs]
                    labels = [label[top:top+ch, left:left+cw] for label in labels]
                else:
                    pad_h = max(0, ch - h)
                    pad_w = max(0, cw - w)
                    imgs = [np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant') for img in imgs]
                    labels = [np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant') for label in labels]
            """ if self.with_random_blur and random.random() > 0.5:
                # Random Gaussian blur
                sigma = random.random()
                imgs = [cv2.GaussianBlur(img, (5, 5), sigma) for img in imgs]
            # Resize to img_size
            if imgs[0].shape[0] != self.img_size or imgs[0].shape[1] != self.img_size:
                imgs = [TF.resize(torch.from_numpy(img.transpose(2, 0, 1).copy()), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BICUBIC).numpy().transpose(1, 2, 0) for img in imgs]
                labels = [TF.resize(torch.from_numpy(label[np.newaxis, ...].copy()), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST).numpy()[0] for label in labels]
        """ 

            if to_tensor:
                
                imgs = [torch.from_numpy(img.copy()).float() for img in imgs]
                labels = [torch.from_numpy(label.copy()).long() for label in labels]
                if not self.img_size_dynamic:
                    if imgs[0].size != (self.img_size, self.img_size):
                        imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                                for img in imgs]
                        labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                                for img in labels]
                        
                #imgs = [(x - x.min()) / (x.max() - x.min()) for x in imgs]  # Normalize to [0, 1]
                #print(imgs[0].shape)
                
                imgs = [TF.normalize(img, mean=[0.5]*6,std=[0.5]*6) for img in imgs]


                #imgs = [TF.normalize(img, mean=self.mean, std=self.std) for img in imgs]
                #imgs = [torch.nan_to_num(img) for img in imgs]
                #for c in range(len(imgs[0])):
                #    print(f"channel {c} unique values: {torch.unique(imgs[0][c])}")
            return imgs, labels

        else:
            imgs = [TF.to_pil_image(img.transpose(1,2,0)) for img in imgs]
            if self.img_size is None:
                self.img_size = None

            if not self.img_size_dynamic:
                if imgs[0].size != (self.img_size, self.img_size):
                    imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                            for img in imgs]
            else:
                self.img_size = imgs[0].size[0]

            labels = [TF.to_pil_image(img) for img in labels]
            if len(labels) != 0:
                if labels[0].size != (self.img_size, self.img_size):
                    labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                            for img in labels]

            random_base = 0.5
            if self.with_random_hflip and random.random() > 0.5:
                imgs = [TF.hflip(img) for img in imgs]
                labels = [TF.hflip(img) for img in labels]

            if self.with_random_vflip and random.random() > 0.5:
                imgs = [TF.vflip(img) for img in imgs]
                labels = [TF.vflip(img) for img in labels]

            if self.with_random_rot and random.random() > random_base:
                angles = [90, 180, 270]
                index = random.randint(0, 2)
                angle = angles[index]
                imgs = [TF.rotate(img, angle) for img in imgs]
                labels = [TF.rotate(img, angle) for img in labels]

            if self.with_random_crop and random.random() > 0:
                i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                    get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))

                imgs = [TF.resized_crop(img, i, j, h, w,
                                        size=(self.img_size, self.img_size),
                                        interpolation=Image.CUBIC)
                        for img in imgs]

                labels = [TF.resized_crop(img, i, j, h, w,
                                        size=(self.img_size, self.img_size),
                                        interpolation=Image.NEAREST)
                        for img in labels]

            if self.with_scale_random_crop:
                # rescale
                scale_range = [1, 1.2]
                target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

                imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
                labels = [pil_rescale(img, target_scale, order=0) for img in labels]
                # crop
                imgsize = imgs[0].size  # h, w
                box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
                imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                        for img in imgs]
                labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                        for img in labels]

            if self.with_random_blur and random.random() > 0:
                radius = random.random()
                imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                        for img in imgs]

            # After all augmentations, before to_tensor
            if imgs[0].shape[0] != self.img_size or imgs[0].shape[1] != self.img_size:
                imgs = [TF.resize(torch.from_numpy(img.transpose(2, 0, 1)), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BICUBIC).numpy().transpose(1, 2, 0) for img in imgs]
                labels = [TF.resize(torch.from_numpy(label[np.newaxis, ...]), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST).numpy()[0] for label in labels]

            if to_tensor:
                # to tensor
                imgs = [TF.to_tensor(img) for img in imgs]
                labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                        for img in labels]
                #imgs = [(img * 255).to(torch.uint8) for img in imgs]
                #imgs = [TF.normalize(img, mean=[0.5]*8,std=[0.5]*8)
                #        for img in imgs]

            return imgs, labels


