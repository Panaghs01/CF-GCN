import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from datasets.CD_dataset import CDDataset

import data_config
from datasets import CD_dataset as cd



def get_dataset_mean_std(num_channels=6):
    root_dir = data_config.DataConfig().get_data_config('SenForFlood').root_dir
    img_list = cd.load_img_name_list(f"{root_dir}/list/train.txt")
    size = len(img_list)
    pixel_count = 0    
    channel_sum = torch.zeros(num_channels, dtype=torch.float64)
    channel_squared_sum = torch.zeros(num_channels, dtype=torch.float64)
    for img_name in img_list:
        for i in range(2):  # Loop over both images (pre and post)
            if i == 0:
                img = cd.get_img_path(root_dir, img_name)
            else:
                img = cd.get_img_post_path(root_dir, img_name)
            img = cd.load_multispectral_image(img)
            img = torch.from_numpy(img).float()

            c, h, w = img.shape
            pixel_count += h * w
            channel_sum += img.view(c, -1).sum(dim=1)
            channel_squared_sum += (img ** 2).view(c, -1).sum(dim=1)
    mean = (channel_sum / pixel_count).tolist()
    std = ((channel_squared_sum / pixel_count - torch.tensor(mean) ** 2).sqrt()).tolist()

    return mean, std


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,data_name=data_name,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,
                                 data_name=args.data_name,
                                 is_train=True,
                                 label_transform=label_transform,
                                )
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,
                                 data_name=args.data_name,
                                 is_train=False,
                                 label_transform=label_transform
                               )
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers,pin_memory=True)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

if __name__ == '__main__':
    mean, std = get_dataset_mean_std()
    print('mean:', mean)
    print('std:', std)