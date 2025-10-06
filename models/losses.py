import torch
import torch.nn.functional as F
import rasterio

def cross_entropy(input, target, weight=torch.Tensor([0.4,0.6]).to('cuda' if torch.cuda.is_available() else 'cpu'), reduction='mean',ignore_index=255):
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
