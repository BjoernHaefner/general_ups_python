####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
import os
from typing import Optional, Union

import torch

from utils.helpers import img2vec, vec2img


class DepthToOBJ:
    def __init__(self):
        return

    @classmethod
    def save(cls, depth: torch.Tensor,
             mask: torch.Tensor,
             filename: str,
             K: Optional[torch.Tensor] = None):

        '''
        Save obj file as filename
        :param depth: shape = [1,h,w]
        :param mask: shape = [1,h,w]
        :param filename: full path + filename (with or w/o '.obj' extension)
        :param K: Tensor describing the intrinsic camera parameters (if not provided orthographic projection is used)
        :return:
        '''
        assert depth.shape == mask.shape, f"depth.shape = {depth.shape}; mask.shape = {mask.shape}"
        assert depth.dim() == mask.dim() == 3  # 1xhxw
        assert depth.shape[0] == mask.shape[0] == 1  # 1xhxw
        xyz = cls.__backproject3d(img2vec(depth, mask), mask, K)
        cls.__exportObj(xyz, mask.squeeze(0), filename)
        return

    @classmethod
    def __backproject3d(cls, depth: torch.Tensor,
                        mask: torch.Tensor,
                        K: Union[torch.Tensor, type(None)]) -> torch.Tensor:
        '''

        :param depth: shape [num_faces, 1]
        :param mask:  shape [1, h, w]
        :param K: None [orthographic] or (3, 3) [perspective]
        :return: xyz points of shape [num_faces, 3]
        '''
        h, w = mask.shape[1:]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = img2vec(x, mask=mask)
        y = img2vec(y, mask=mask)

        if K is not None:  # perspective projection
            assert K.dim() == 2 and K.shape[0] == K.shape[1] == 3
            assert K[-1, -1] == 1 and K[-1, 0] == K[-1, 1] == 0
            return depth * (K.inverse() @ torch.cat([x, y, torch.ones(mask.sum(), 1)], dim=-1).T).T
        else:  # orthographic projection
            return torch.cat([x - (w / 2.0), y - (h / 2.0), depth], dim=-1)

    @classmethod
    def __exportObj(cls, vertices: torch.Tensor, mask: torch.Tensor, filename: str):
        '''
        :param vertices: shape [num_faces, 3]
        :param mask: shape [h, w]
        :param filename:
        :return:
        '''
        # change order of vectorization to match indices[mask] order: col-major vs row-major

        assert mask.dim() == vertices.dim() == 2
        assert vertices.shape[-1] == 3
        assert vertices.shape[0] == mask.sum()

        vertices = vec2img(vertices, mask=mask[None, ...])
        vertices = vertices[:, mask].T

        def sub2ind(w, r, c):
            return (c + r * w).reshape(-1)

        mask_quad = mask[:-1, :-1] & mask[1:, 1:] & mask[:-1, 1:] & mask[1:, :-1]
        rows, cols = torch.where(mask_quad)
        top_left = sub2ind(mask.shape[-1], rows, cols)
        bottom_left = sub2ind(mask.shape[-1], rows + 1, cols)
        bottom_right = sub2ind(mask.shape[-1], rows + 1, cols + 1)
        top_right = sub2ind(mask.shape[-1], rows, cols + 1)

        indices = torch.zeros(mask.shape, dtype=torch.long)
        indices[mask] = torch.arange(1, mask.sum() + 1)  # faces start counting at one
        indices = indices.reshape(-1)
        # If vertices are ordered counterclockwise around the face, both the face and the normal will point toward the viewer.
        faces = torch.stack(
            [indices[top_left],
             indices[bottom_left],
             indices[bottom_right],
             indices[top_right]],
            dim=-1)

        obj = {
            'v': vertices,  # nx3
            'f': faces,  # nx4
            # 'material': [
            #     {'type': 'newmtl', 'data': 'skin'},
            #     {'type': 'Ka', 'data': [0.5, 0.5, 0.5]},
            #     {'type': 'Kd', 'data': [1, 1, 1]},
            #     {'type': 'Ks', 'data': [0.3, 0.3, 0.3]},
            #     {'type': 'illum', 'data': 2},
            #     {'type': 'Ns', 'data': 10}
            # ],
            # 'objects':
            #     [
            #         {'type': 'f', 'data': {'vertices': faces, }},
            #         {'type': 'g', 'data': 'skin'},
            #         {'type': 'usemtl', 'data': 'skin'}
            #     ]
        }

        obj_str = ''
        for k, v in obj.items():
            if type(v) == torch.Tensor:
                # add comment for each data type
                if k == 'v':
                    obj_str += '# List of geometric vertices, with (x, y, z [,w]) coordinates, w is optional and defaults to 1.0.\n'
                elif k == 'f':
                    obj_str += '# Polygonal face element (see below)\n'

                for vi in v.cpu().numpy():
                    obj_str += ' '.join([k, ' '.join(map(str, vi)), '\n'])

        # append filetype if it's not there
        if not filename.endswith('.obj'):
            filename += '.obj'

        # create directory if it's not yet created
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        text_file = open(filename, "w")
        text_file.write(obj_str)
        text_file.close()

        return
