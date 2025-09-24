"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np
import rasterio

import torch
from torch.utils import data
from datasets import FDA_source_to_target_np
from datasets.data_utils import CDDataAugmentation
import matplotlib.pyplot as plt

"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = "B"
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "MASKS"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


def load_multispectral_image(path,channels=[0,1,2,3,4,5,6,7]):
    with rasterio.open(path) as src:
        img = src.read()[channels]  # shape: (bands, H, W)
        img = img.astype(np.float32)
        #flip = img[:,:,::-1]
        """plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img[0], cmap='gray')
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(flip[0], cmap='gray')
        plt.title('Flipped')
        plt.show() """
        #img = np.transpose(img, (1, 2, 0))  # shape: (H, W, bands)
    return img

def scale(img):
    #print(img.min(), img.max())
    img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-8)
    #print(img)
    return img.astype(np.uint8)

class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
        )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        
        #img = np.asarray(Image.open(A_path).convert('RGB'))
        #img_B = np.asarray(Image.open(B_path).convert('RGB'))

        img = load_multispectral_image(A_path)  # shape: (H, W, 8)
        img_B = load_multispectral_image(B_path)

        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)
        #print(img.shape)
        #rasterio.show(img.permute(1,2,0))
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, data_name, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        self.data_name = data_name
        self.raster = True if self.data_name == 'SenForFlood' or 'OMBRIA' else False

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        if self.data_name=='SenForFlood':

            img = load_multispectral_image(A_path)  # shape: (H, W, 8)
            img_B = load_multispectral_image(B_path)
        else:
            img = np.asarray(Image.open(A_path).convert('RGB'))
            # print(img_B.type())
            img_B = np.asarray(Image.open(B_path).convert('RGB'))


        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        if self.data_name=='WHU':
            # 风格统一，使用傅立叶变换
           im_src = np.asarray(img, np.float32)
           im_trg = np.asarray(img_B, np.float32)

           im_src = im_src.transpose((2, 0, 1))
           im_trg = im_trg.transpose((2, 0, 1))

           src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.01)
            #  A偏向于B风格
           img = src_in_trg.transpose((1, 2, 0))
        with rasterio.open(L_path) as src:
            label = src.read(1)  
            label = (label > 0).astype(np.uint8)  # binarize 0/1
            #plt.imshow(label, cmap='gray')
            #plt.show()
        #print(f"loaded label {L_path} with unique values {np.unique(label)}")

        
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255
        #print(f"A:{img.shape}, B:{img_B.shape}, L:{label.shape}, uniq={torch.unique(label)}"))
        img = scale(img)
        img_B = scale(img_B)
        #rasterio.plot.show(img[4:7])
        [img, img_B], [label] = self.augm.transform([np.asarray(img, np.uint8),\
                                                      img_B], [label], to_tensor=self.to_tensor,rasterio_read=self.raster)
        label = label.long() 

        #plt.imshow(np.transpose(img.numpy(),(1,2,0)))
        #plt.show()

        #print(torch.unique(img))
        # print(label.max())
        #label = label.unsqueeze(0) 
        return {'name': name, 'A': img, 'B': img_B, 'L': label}

