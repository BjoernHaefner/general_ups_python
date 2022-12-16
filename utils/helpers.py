####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
import os
from os import listdir
from os.path import isfile, join
from typing import Optional

import torch
from torch.nn.functional import normalize


def allFilesInDir(filedir: str,
                  ext: Optional[str] = None,
                  contains: Optional[str] = None,
                  filedirpath: bool = False):
    '''
    Returns list of all files in filedir (optional with extension ext, optional as filedirpath)

    :param filedir: directory to look into
    :param ext: Optional: Only list files with this file extension
    :param contains: Optional: Only list files that contain 'contains'
    :param filedirpath: Return filenames with filedir path attached
    :return:
    '''
    filenames = [f for f in listdir(filedir) if isfile(join(filedir, f))]
    if ext is not None:
        filenames = [f for f in filenames if f.endswith(ext)]
    if contains is not None:
        filenames = [f for f in filenames if contains in f]
    filenames = sorted(filenames)
    if filedirpath:
        filenames = [os.path.join(filedir, f) for f in filenames]
    return filenames


def img2vec(img: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''
    :param img: assumed to have dimensions [h,w], [1,h,w], or [c,h,w]
    :param mask: is a [1,h,w] tensor
    :return: vector of size [h*w,c] (c=1 for grayscale image), stacked in column major order
    '''
    if img.dim() == 2:  # 2d grayscale image
        if mask is None:
            return img.transpose(-2, -1).reshape(-1, 1)
        else:
            if img.shape != mask.shape[1:]:
                raise AttributeError(
                    f"Image and mask dimensions don't fit: img.shape={img.shape} vs. mask.shape={mask.shape[1:]}")
            return torch.masked_select(img.transpose(-1, -2), mask.transpose(-1, -2)).reshape(-1, 1)
    if img.dim() == 3 and img.shape[0] == 1:  # 2d grayscale image with unnecessary 3rd dimension
        return img2vec(img.squeeze(0), mask)
    elif img.dim() == 3 and img.shape[0] > 1:  # Color image
        channels = img.shape[0]
        return torch.cat([img2vec(img[c, :, :], mask) for c in range(channels)], dim=1)
    else:
        raise ValueError(f"Image size not supported. Expected: (c,h,w). Got: {img.shape}")


def vec2img(vec: torch.Tensor,
            width: Optional[int] = None,
            height: Optional[int] = None,
            mask: Optional[torch.Tensor] = None,
            bg: float = 0.0) -> torch.Tensor:
    '''
    :param vec: assumed to be in column major order and to have dimensions [w*h,c] or [w*h]
    :param width: width of image
    :param height: height of image
    :param mask: of dimension [1,h,w]
    :param bg: background value between [0, 1]
    :return: image of dimensions [c,h,w] or [1,h,w]
    '''
    assert 0 <= bg <= 1
    if vec.dim() > 2:
        raise ValueError(
            f"Vector is no form of column vector [w*h,c] or [w*h]: {vec.shape}")

    if vec.dim() == 1:
        vec = vec[..., None]

    channels = vec.shape[-1]

    if mask is None:
        assert width is not None and height is not None
        return vec.transpose(-1, -2).reshape(channels, width, height).transpose(-2, -1)
    else:
        assert mask.dim() == 3 and mask.shape[0] == 1
        h, w = mask.shape[1:]
        vec_padded = bg * torch.ones((w * h, channels), dtype=vec.dtype)
        vec_padded[img2vec(mask).squeeze(), :] = vec
        return vec2img(vec_padded, w, h, None)


def normalsToLatitude(normals: torch.Tensor, dim: int = -1, rad: bool = False) -> torch.Tensor:
    '''
    Returns the latitude of the normal in degrees or radians.
    Northpole has 90 degrees. Equator has 0 degrees

    :param normals: of shape (...,3,...), where normals.shape[dim]==3
    :param dim: dim where the normals are
    :param rad: output in radians if rad true, else degrees
    :return: latitude of shape (...,1,...), where latitude.shape[dim]==1
    '''
    assert normals.shape[dim] == 3
    nxy = normals.index_select(dim=dim, index=torch.tensor([0, 1], dtype=torch.long))
    nz = normals.index_select(dim=dim, index=torch.tensor(2, dtype=torch.long))
    lat = torch.atan2(nz, nxy.norm(p=2, dim=dim, keepdim=True))
    return lat if rad else lat * 180 / torch.pi


def depthToNormals(z, mask, K: Optional[torch.Tensor], nabla: Optional[torch.Tensor] = None,
                   southern: bool = True):
    '''
    Computes normals from a given depth under perspective/orthographic projection on the northern/southern hemisphsere
    n(z(x,y)) = normalize((z_x(x,y), z_y(x,y), {+,-}1)) (orthographic projection)
    n(z(x,y)) = normalize((f_x * z_x(x,y), f_y * z_y(x,y), {+,-}(z(x,y) + (x - c_x) * z_x(x,y) + (y - c_y) * z_y(x,y)))) (perspective projection)


    :param z: [num_faces, 1]
    :param mask:
    :param K: Optional intrinsics matrix. If None, orthographic projection is used, else perspective
    :param nabla: Optional nabla operator for z. If None, Forward differences and NeumannHomogeneous boundary conditions are used
    :param southern: resulting normals on southern or northern hemisphere
    :return: normals under orthographic or perspective projection
    '''
    assert mask.dim() == 3 and mask.shape[0] == 1, f"mask.shape: {mask.shape}"
    assert z.dim() == 2 and z.shape == (mask.sum(), 1), f"z.shape: {z.shape}"

    if nabla is None:
        from utils.sparse import nabla2d
        nabla = nabla2d(mask)[0]

    nabla_z = (nabla @ z).reshape(2, -1).T  # [num_faces, 2]
    if K is not None and K.shape == (3, 3):  # perspective case
        def _pixelswrtPrincipalPoint():
            h, w = mask.shape[1:]
            x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
            x = img2vec(x, mask=mask) - K[0, 2]
            y = img2vec(y, mask=mask) - K[1, 2]
            return x, y

        nxy = K[(0, 1), (0, 1)][None, :] * nabla_z

        x, y = _pixelswrtPrincipalPoint()
        nz = z + x * nabla_z[:, 0:1] + y * nabla_z[:, 1:2]
    else:  # orthographic case
        nxy = nabla_z.detach().clone()
        nz = torch.ones(mask.sum(), 1)

    if southern:
        nz *= -1
    n_unnormalized = torch.cat([nxy, nz], dim=-1)

    dz = n_unnormalized.norm(dim=-1, keepdim=True)
    normals = normalize(n_unnormalized, dim=-1)
    return normals, dz, n_unnormalized


def normalsToDepth(n, mask, K: Optional[torch.Tensor], nabla: Optional[torch.Tensor] = None,
                   solve='cholesky'):
    '''
    Integrates normals under perspective of orthographic projection

    :param n: [num_faces, 3]
    :param mask:
    :param K: Optional intrinsics matrix. If None, orthographic projection is used, else perspective
    :param nabla: Optional nabla operator for z. If None, Forward differences and NeumannHomogeneous boundary conditions are used
    :param solve:
    :return:
    '''
    assert mask.dim() == 3 and mask.shape[0] == 1, f"mask.shape: {mask.shape}"
    assert n.dim() == 2 and n.shape == (mask.sum(), 3), f"n.shape: {n.shape}"
    from utils.sparse import nabla2d, transpose, speye
    if nabla is None:
        nabla = nabla2d(mask, stencil='Forward', bc='neumannHomogeneous')[0]

    def _smoothIntegration(p, q, persp: bool, lambda_=1e-6, z0=1):
        A = transpose(nabla, 0, 1) @ nabla + lambda_ * speye(mask.sum())
        b = transpose(nabla, 0, 1) @ torch.cat([p, q]) + lambda_ * z0
        if solve == 'cholesky':
            from utils.optimization import _cholesky
            z = _cholesky(A.double(), b.double(), p.device).float()
        elif solve == 'cg':
            from utils.optimization import _cg
            z = _cg(A.double(), b.double(), z0 * torch.ones_like(p), 1e-10, 1000,
                    p.device).float()

        return torch.exp(z) if persp else z

    # discard invalid angles
    phi_max = 85
    if K is not None and K.shape == (3, 3):
        h, w = mask.shape[1:]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = (img2vec(x, mask=mask) - K[0, 2]) / K[0, 0]
        y = (img2vec(y, mask=mask) - K[1, 2]) / K[1, 1]
        vecs = -normalize(torch.cat([x, y, torch.ones(mask.sum(), 1)], dim=-1), dim=-1)
        latitude = torch.acos((vecs * n).sum(-1)) * 180 / torch.pi
    else:  # orthographic projection
        latitude = 90 - normalsToLatitude(n).abs().squeeze(-1)  # make  northpole: 0, equator: 90
    mask_invalid_angle = latitude >= phi_max

    if K is not None and K.shape == (3, 3):
        denom = x * n[:, 0:1] + y * n[:, 1:2] + n[:, 2:3]
        p = -(n[:, 0:1] / K[0, 0]) / denom
        q = -(n[:, 1:2] / K[1, 1]) / denom
    else:  # orthographic projection
        p = n[:, 0:1] / n[:, 2:3]
        q = n[:, 1:2] / n[:, 2:3]

    p[mask_invalid_angle, :] = 0
    q[mask_invalid_angle, :] = 0
    mask_nan = vec2img(torch.logical_or(torch.isnan(p), torch.isnan(q)).float(), mask=mask).bool()
    mask[mask_nan] = False
    return _smoothIntegration(p, q, K is not None and K.shape == (3, 3))


def normalsToSH(normals: torch.Tensor, dim: int = -1, sh_order: int = 1):
    '''
    Follow layout [nx, ny, nz, 1]^T

    :param normals: shape (..., 3, ...), where normals.shape[dim]==3
    :param dim: int
    :return: spherical_harmonics basis functions of 1st order. shape (..., 4, ...), where sh.shape[dim]==4
    '''
    if -normals.dim() <= dim < 0:
        dim += normals.dim()
    if sh_order >= 1:
        ones = torch.ones(*normals.shape[:dim], 1, *normals.shape[dim + 1:])
        sh = torch.cat([normals, ones], dim=dim)
    if sh_order >= 2:
        nx = normals.select(dim, 0)
        ny = normals.select(dim, 1)
        nz = normals.select(dim, 2)
        sh2 = [nx * ny, nx * nz, ny * nz, nx * nx - ny * ny, 3 * nz * nz - 1]
        sh2 = torch.stack(sh2, dim=dim)
        sh = torch.cat([sh, sh2], dim=dim)
    return sh
