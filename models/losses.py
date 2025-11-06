import torch
import torch.nn.functional as F
import rasterio

def cross_entropy(input, target, weight=torch.Tensor([0.8,1]).to('cuda' if torch.cuda.is_available() else 'cpu'),ignore_index=255, reduction='mean'):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    #rasterio.plot.show(input[0,1,:,:].cpu().detach().numpy())
    #rasterio.plot.show(target[0,:,:].cpu().detach().numpy())
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def dice_loss(input, target, smooth=1e-5):
    """
    Computes the dice loss for binary segmentation
    Args:
        :param input: torch.Tensor, N*C*H*W
        :param target: torch.Tensor, N*1*H*W,/ N*H*W
        :param smooth: smoothing factor to avoid zero division
    returns:
        scalar Dice Loss
    """

    pred = torch.sigmoid(input)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()

def CE_with_Dice(input,target,weights=[0.7,1],smooth=1e-5,ignore_index=255):

    return weights[0]*cross_entropy(input,target) + weights[1]*dice_loss(input,target,smooth)