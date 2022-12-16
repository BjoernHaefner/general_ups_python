####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
import torch
import torch.nn.functional as F
from scipy import sparse

from utils.helpers import img2vec


def spcoo(r, c, v, size=None):
    assert r.dtype == torch.int64 and c.dtype == torch.int64, f"r.dtype: {r.dtype} and c.dtype: {c.dtype}"
    if size is None:
        return torch.sparse_coo_tensor(indices=vstack([r, c]), values=v,
                                       dtype=torch.float).coalesce()
    else:
        return torch.sparse_coo_tensor(indices=vstack([r, c]), values=v, size=size,
                                       dtype=torch.float).coalesce()


def vstack(list_of_mats):
    stacked = torch.vstack(list_of_mats)
    if stacked.is_sparse:
        return stacked.coalesce()
    else:
        return stacked


def hstack(list_of_mats):
    stacked = torch.hstack(list_of_mats)
    if stacked.is_sparse:
        return stacked.coalesce()
    else:
        return stacked


def transpose(mat, dim1, dim2):
    if torch.is_tensor(mat):
        if mat.is_sparse:
            return mat.transpose(dim1, dim2).coalesce()
        else:
            return mat.transpose(dim1, dim2)
    elif isinstance(mat, list):
        return [transpose(m, dim1, dim2) for m in mat]
    else:
        raise TypeError(f"Unknown type: {type(mat)}")


def dimsum(spmat, dim):
    return torch.sparse.sum(spmat, dim=dim).to_dense().reshape(-1, 1)


def spzeros(size):
    return torch.sparse_coo_tensor(size=size, dtype=torch.float).coalesce()


def speye(dim):
    return spcoo(torch.arange(dim), torch.arange(dim), torch.ones(dim))


def spdiag(vals_in, offsets, size):
    '''

    :param vals_in: (list) of row vectors or tensor of shape (1,n) if len(offsets) == 1
    :param offsets: (list) of ints describing the diag
    :param size: size of output matrix
    :return:
    '''
    if type(vals_in) == list:
        assert len(vals_in) == len(offsets)
    else:
        assert len(offsets) == 1 and vals_in.dim() == 2 and vals_in.shape[0] == 1

    rows = torch.empty((0,), dtype=torch.int64)
    cols = torch.empty((0,), dtype=torch.int64)
    vals = torch.empty((0,))

    for offset in offsets:
        if offset < 0:
            length2border = min(size[1], size[0] + offset)
            rows = hstack([rows, torch.arange(-offset, length2border - offset)])
            cols = hstack([cols, torch.arange(length2border)])
        elif offset == 0:
            rows = hstack([rows, torch.arange(min(size))])
            cols = hstack([cols, torch.arange(min(size))])
        elif offset > 0:
            length2border = min(size[1] - offset, size[0])
            rows = hstack([rows, torch.arange(length2border)])
            cols = hstack([cols, torch.arange(offset, length2border + offset)])
        vals = hstack([vals, vals_in[offsets.index(offset)]])

    return spcoo(rows, cols, vals, size)


def torchSparseToScipySparse(mat):
    values = mat.coalesce().values().to('cpu')
    indices = mat.coalesce().indices().to('cpu')
    shape = mat.size()
    return sparse.coo_matrix((values, indices), shape=shape)


def scipySparseToTorchSparse(mat):
    r = torch.tensor(mat.row, dtype=torch.int64)
    c = torch.tensor(mat.col, dtype=torch.int64)
    v = torch.tensor(mat.data)
    s = torch.Size(mat.shape)
    return spcoo(r, c, v, s)


def kron(mat1, mat2):
    mat1_ = torchSparseToScipySparse(mat1)
    mat2_ = torchSparseToScipySparse(mat2)

    result_ = sparse.kron(mat1_, mat2_).tocoo()

    return scipySparseToTorchSparse(result_)


def spdiagFromMask(mask, transpose=False):
    '''

    :param mask: of shape [1, height, width]
    :param transpose:
    :return: diagonal matrix of shape [mask.sum(), height*width] (transpose=False) or  [height*width, mask.sum()] (transpose=True)
    '''
    assert mask.dim() == 3 and mask.shape[0] == 1
    if transpose:
        return spcoo(torch.where(img2vec(mask))[0], torch.arange(mask.sum()),
                     torch.ones(mask.sum()),
                     size=(mask.numel(), mask.sum()))
    else:
        return spcoo(torch.arange(mask.sum()), torch.where(img2vec(mask))[0],
                     torch.ones(mask.sum()),
                     size=(mask.sum(), mask.numel()))


def nabla2d(mask, stencil='Forward', bc='NeumannHomogeneous'):
    '''
    Forward differences with Neumann boundary conditions
    Uses kronecker product for vectorization: https://en.wikipedia.org/wiki/Vectorization_(mathematics)

    :param mask:
    :param stencil: 'Forward', 'Backward', 'Central'
    :param bc: 'NeumannHomogeneous', 'DirichletHomogeneous'
    :return: gradient and divergence operator of size=(num_edges, num_faces) and size=(num_faces, num_edges), respectively
    '''
    assert stencil.lower() in ['forward', 'backward', 'central']
    assert bc.lower() in ['neumannhomogeneous', 'dirichlethomogeneous']

    height, width = mask.shape[1:]
    extract_mat = spdiagFromMask(mask)
    extract_mat_t = transpose(extract_mat, 0, 1)

    #####################
    # construct nabla_x #
    #####################
    if stencil.lower() == 'forward':
        e0_x = -torch.ones(width)
        e1_x = torch.ones(width - 1)
        dx_tilde_t = spdiag([e0_x, e1_x], offsets=[0, 1], size=(width, width))
    elif stencil.lower() == 'backward':
        e0_x = torch.ones(width)
        e1_x = -torch.ones(width - 1)
        dx_tilde_t = spdiag([e0_x, e1_x], offsets=[0, -1], size=(width, width))
    elif stencil.lower() == 'central':
        e0_x = -0.5 * torch.ones(width - 1)
        e1_x = 0.5 * torch.ones(width - 1)
        dx_tilde_t = spdiag([e0_x, e1_x], offsets=[-1, 1], size=(width, width))
    else:
        raise RuntimeError(f"Unknown stencil: {stencil}")
    dx = kron(dx_tilde_t, speye(height))

    # handle boundary condition in x-direction for masked case

    # true where vertical edges are no boundary edges (left and right neighbour face are valid)
    if stencil.lower() == 'forward' and bc.lower() == 'neumannhomogeneous':
        mask_r = F.pad(input=mask.float(), pad=(0, 1)).bool()
        mask_l = F.pad(input=mask.float(), pad=(1, 0)).bool()
        mask_ral = (mask_l & mask_r)[:, :, 1:]
    elif stencil.lower() == 'backward' and bc.lower() == 'neumannhomogeneous':
        mask_r = F.pad(input=mask.float(), pad=(0, 1)).bool()
        mask_l = F.pad(input=mask.float(), pad=(1, 0)).bool()
        mask_ral = (mask_l & mask_r)[:, :, :-1]
    elif stencil.lower() == 'central' and bc.lower() == 'neumannhomogeneous':
        mask_r = F.pad(input=mask.float(), pad=(0, 2)).bool()
        mask_l = F.pad(input=mask.float(), pad=(2, 0)).bool()
        mask_ral = (mask_l & mask_r)[:, :, 1:-1]
    elif stencil.lower() == 'forward' and bc.lower() == 'dirichlethomogeneous':
        mask_ral = mask.detach().clone()
    elif stencil.lower() == 'backward' and bc.lower() == 'dirichlethomogeneous':
        mask_ral = mask.detach().clone()
    elif stencil.lower() == 'central' and bc.lower() == 'dirichlethomogeneous':
        mask_r = F.pad(input=mask.float(), pad=(0, 2)).bool()
        mask_l = F.pad(input=mask.float(), pad=(2, 0)).bool()
        mask_ral = mask & (mask_l | mask_r)[:, :, 1:-1]
    else:
        raise RuntimeError(f"Unknown bc: {bc}")
    # true if left and right face exist, false otherwise
    bc_x = spdiag(img2vec(mask_ral).T, offsets=[0], size=(dx.shape[0], dx.shape[0]))
    # set boundary condition
    dx = extract_mat @ bc_x @ dx @ extract_mat_t

    #####################
    # construct nabla_y #
    #####################
    if stencil.lower() == 'forward':
        e0_y = -torch.ones(height)
        e1_y = torch.ones(height - 1)
        dy_tilde = spdiag([e0_y, e1_y], offsets=[0, 1], size=(height, height))
    elif stencil.lower() == 'backward':
        e0_y = torch.ones(height)
        e1_y = -torch.ones(height - 1)
        dy_tilde = spdiag([e0_y, e1_y], offsets=[0, -1], size=(height, height))
    elif stencil.lower() == 'central':
        e0_y = -0.5 * torch.ones(height - 1)
        e1_y = 0.5 * torch.ones(height - 1)
        dy_tilde = spdiag([e0_y, e1_y], offsets=[-1, 1], size=(height, height))
    else:
        raise RuntimeError(f"Unknown stencil: {stencil}")
    dy = kron(speye(width), dy_tilde)

    # handle boundary condition in y-direction for masked case

    # true where horizontal edges are no boundary edges (top and bottom neighbour face are valid)
    if stencil.lower() == 'forward' and bc.lower() == 'neumannhomogeneous':
        mask_t = F.pad(input=mask.unsqueeze(0).float(), pad=(0, 0, 1, 0)).squeeze(0).bool()
        mask_b = F.pad(input=mask.unsqueeze(0).float(), pad=(0, 0, 0, 1)).squeeze(0).bool()
        mask_tab = (mask_b & mask_t)[:, 1:, :]
    elif stencil.lower() == 'backward' and bc.lower() == 'neumannhomogeneous':
        mask_t = F.pad(input=mask.unsqueeze(0).float(), pad=(0, 0, 1, 0)).squeeze(0).bool()
        mask_b = F.pad(input=mask.unsqueeze(0).float(), pad=(0, 0, 0, 1)).squeeze(0).bool()
        mask_tab = (mask_b & mask_t)[:, :-1, :]
    elif stencil.lower() == 'central' and bc.lower() == 'neumannhomogeneous':
        mask_t = F.pad(input=mask.unsqueeze(0).float(), pad=(0, 0, 2, 0)).squeeze(0).bool()
        mask_b = F.pad(input=mask.unsqueeze(0).float(), pad=(0, 0, 0, 2)).squeeze(0).bool()
        mask_tab = (mask_b & mask_t)[:, 1:-1, :]
    elif stencil.lower() == 'forward' and bc.lower() == 'dirichlethomogeneous':
        mask_tab = mask.detach().clone()
    elif stencil.lower() == 'backward' and bc.lower() == 'dirichlethomogeneous':
        mask_tab = mask.detach().clone()
    elif stencil.lower() == 'central' and bc.lower() == 'dirichlethomogeneous':
        mask_t = F.pad(input=mask.float(), pad=(0, 0, 2, 0)).bool()
        mask_b = F.pad(input=mask.float(), pad=(0, 0, 0, 2)).bool()
        mask_tab = mask & (mask_b | mask_t)[:, 1:-1, :]
    else:
        raise RuntimeError(f"Unknown bc: {bc}")
    # true if upper and bottom face exist, false otherwise
    bc_y = spdiag(img2vec(mask_tab).T, offsets=[0], size=(dy.shape[0], dy.shape[0]))
    # set boundary edges to zero (neumann condition)
    dy = extract_mat @ bc_y @ dy @ extract_mat_t

    # construct nabla and divergence operator
    nabla_2d = vstack([dx, dy])
    div_2d = transpose(-nabla_2d, 0, 1)

    return nabla_2d, div_2d
