####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
from typing import Union, List

import torch


def optC(f, u, dim):
    '''
    min_c sum_{dim} ||f_dim - c*u_dim||_2^2
    :param f:
    :param u:
    :param dim:
    :return:
    '''
    return (u * f).sum(dim=dim, keepdim=True) / (u * u).sum(dim=dim, keepdim=True)


def calcRMSE(gt, est, dim: Union[int, List] = -1):
    '''
    :param gt: [..., n, ...]
    :param est: [..., n, ...]
    :param dim: List or int describing the dimension to calculate the RMSE over
    :return: tensor of same size as gt (except at dimension(s) dim) representing RMSE at dim(s)
    '''
    # calculate angular error per tensor element
    if gt.shape == est.shape:
        return torch.sqrt(((gt - est) * (gt - est)).mean(dim=dim, keepdim=True))
    else:
        raise ValueError(f"Wrong dimension.\n",
                         f"Expected: gt.shape == est.shape.\n",
                         f"Given: gt.shape = {gt.shape} and est.shape = {est.shape}.")


def calcAngularError(gt: torch.Tensor, est: torch.Tensor, dim: int = -1,
                     degrees: bool = True) -> torch.Tensor:
    '''
    :param gt: [..., n, ...]
    :param est: [..., n, ...]
    :param dim: gt.shape[dim] = est.shape[dim] = n
    :param degrees: True, if False in radians
    :return: [..., 1, ...]
    '''
    # calculate angular error per tensor element

    if gt.dim() == est.dim():
        if gt.shape != est.shape:
            raise ValueError(f"Wrong dimension.\n",
                             f"Expected: gt.shape == est.shape.\n",
                             f"Given: gt.shape = {gt.shape} and est.shape = {est.shape}.")
    else:
        if gt.shape != est.shape[-2:]:
            raise ValueError(f"Wrong dimension for batch processing.\n",
                             f"Expected: gt.shape == est.shape[-2:].\n",
                             f"Given: gt.shape = {gt.shape} and est.shape = {est.shape}.")
        # unsqueeze to enable broadcasting
        gt = torch.broadcast_tensors(gt, est)[0]

    # use more stable formular in R^3
    if gt.shape[dim] == est.shape[dim] == 3:
        x = torch.cross(gt, est, dim=dim).norm(p=2, dim=dim, keepdim=True)
        y = (gt * est).sum(dim=dim, keepdim=True)
        rad = torch.atan2(x, y)
    else:
        gt_u = gt / torch.norm(gt, p=2, dim=dim, keepdim=True)
        est_u = est / torch.norm(est, p=2, dim=dim, keepdim=True)
        x = torch.clip((gt_u * est_u).sum(dim=dim, keepdim=True), min=-1.0, max=1.0)
        rad = torch.acos(x)

    return rad * 180 / torch.pi if degrees else rad
