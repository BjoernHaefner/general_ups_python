####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
import torch
from scipy.sparse.linalg import cg
from sksparse.cholmod import cholesky

from utils.sparse import torchSparseToScipySparse


def sqL2loss(x: torch.Tensor) -> torch.Tensor:
    '''per element squared L2-norm'''
    if type(x) == list:
        return sum([sqL2loss(xi) for xi in x])  # this sum does a broadcast
    elif type(x) == torch.Tensor:
        return x * x
    else:
        RuntimeError(f"Unknown type: {type(x)}")


def gradientDescentStep(x, step_size, dir):
    '''x - step_size * dir'''
    return x - step_size * dir


def gradientAscentStep(x, step_size, dir):
    '''x + step_size * dir'''
    return gradientDescentStep(x, -step_size, dir)


def _cholesky(A, b, device):
    factor = cholesky(torchSparseToScipySparse(A))
    return torch.from_numpy(factor(b.cpu().numpy())).to(device)


def _cg(A, b, x0, tol, maxit, device):
    if x0 is None:
        x0 = torch.zeros_like(b)

    assert b.shape == x0.shape
    x = []
    for bi, x0i in zip(b.T, x0.T):
        x.append(torch.from_numpy(cg(torchSparseToScipySparse(A), bi[:, None].cpu().numpy(),
                                     x0=x0i[:, None].cpu().numpy(),
                                     tol=tol, maxiter=maxit)[0]).to(device))
    return torch.stack(x, dim=-1)
