####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
import os
import pickle
import sys
import warnings
from subprocess import call
from typing import Optional, List, Tuple, Union

import math
import numpy as np
import scipy.io
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from utils.depth_to_obj import DepthToOBJ
from utils.helpers import vec2img, img2vec, normalsToSH, depthToNormals


class DataIO:
    _device: torch.device
    _verbose: bool
    _output: str
    _gray: bool = True

    def __init__(self, device: Optional[torch.device] = None,
                 gpu: Optional[bool] = None, gpu_id: Optional[int] = None,
                 output: Optional[str] = None,
                 verbose: bool = True):

        self._setDevice(device, gpu, gpu_id)
        self._verbose = verbose
        self._output = output
        return

    @classmethod
    def printInfo(cls):
        print("".center(80, "="))
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        if torch.cuda.is_available():
            print('__CUDA VERSION', torch.version.cuda)
            print('__CUDNN VERSION:', torch.backends.cudnn.version())
            print('__Number CUDA Devices:', torch.cuda.device_count())
            print('__Devices')
            call(["nvidia-smi", "--format=csv",
                  "--query-gpu=index,name,driver_version,memory.total"])
            print(
                f'Current cuda device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print("".center(80, "="))
        return

    @classmethod
    def setDevice(cls, gpu: bool = True, gpu_id: int = 0) -> torch.device:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and gpu:
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = "cpu"
            torch.set_default_tensor_type('torch.FloatTensor')
        cls.printInfo()
        return torch.device(device)

    def getDevice(self) -> torch.device:
        return self._device

    def getOutput(self) -> str:
        return self._output

    def loadMask(self, mask_file: str,
                 dict_key: Optional[str] = None,
                 thresh: Optional[float] = None) -> torch.Tensor:
        '''
        Load boolean mask

        :param mask_file: str to png file
        :return: torch tensor of shape [1, height, width]
        '''
        if self._extIs(mask_file, '.png'):
            mask = self._loadTorchImage(mask_file, gray=True)
        elif self._extIs(mask_file, '.pth'):
            mask = self._loadPthFile(mask_file)
        elif self._extIs(mask_file, '.mat'):
            mask = self._loadMatFile(mask_file, dict_key)
        else:
            raise NotImplementedError(f"Can't load {mask_file}-file type")

        if thresh is not None and mask.dtype != torch.bool:
            mask[mask < thresh] = 0
            mask[mask >= thresh] = 1

        return mask.bool()

    def loadImages(self, image_files: Union[List[str], str],
                   dict_key: Optional[str] = None,
                   gray: Optional[bool] = True) -> torch.Tensor:
        '''
        Load images

        :param image_files: List[str] to png files or str to single file, e.g. pth
        :param dict_key: dictionary key for mat-files
        :param gray: load png images as grayscale
        :return: torch tensor of shape [num_images, num_channels, height, width]
        '''

        if type(image_files) == list:
            images = []
            for image_path in image_files:
                images.append(self.loadImage(image_path, dict_key=dict_key, gray=gray))
            return torch.stack(images, dim=0).squeeze()  # [num_images, 1, height, width]
        elif type(image_files) == str:
            return self.loadImage(image_files, dict_key=dict_key, gray=gray)
        else:
            raise RuntimeError(f"Unknown file type: type(image_files) = {type(image_files)}")

    def loadImage(self, img_file: str,
                  dict_key: Optional[str] = None,
                  gray: Optional[bool] = True) -> torch.Tensor:
        '''
        Load image

        :param img_file: str to png file
        :param dict_key: dictionary key for mat-file
        :param gray: load png image as grayscale
        :return: torch tensor of shape [num_channels, height, width]
        '''
        if self._extIs(img_file, '.png') or self._extIs(img_file, '.tif'):
            img = self._loadTorchImage(img_file, gray)
        elif self._extIs(img_file, '.pth'):
            img = self._loadPthFile(img_file)
        elif self._extIs(img_file, '.mat'):
            img = self._loadMatFile(img_file, dict_key)
        else:
            raise NotImplementedError(f"Can't load {img_file}-file type")
        return img

    def loadNormals(self, normals_file: str,
                    dict_key: Optional[str] = None,
                    permute: Optional[Tuple[int, int, int]] = None,
                    flip_dim: Optional[Union[Tuple[int, ...], int]] = None,
                    flip_index: Optional[Union[Tuple[int, ...], int]] = None) -> torch.Tensor:
        '''
        Load surface normals

        :param normals_file: str to png file
        :return: torch tensor of shape [3, height, width], each normal in range [-1, 1]
        '''
        if self._extIs(normals_file, '.png') or self._extIs(normals_file, '.tif'):
            normals = self._loadNormalsTorchImg(normals_file)
        elif self._extIs(normals_file, '.mat'):
            normals = self._loadMatFile(normals_file, dict_key)
        elif self._extIs(normals_file, '.pth'):
            normals = self._loadPthFile(normals_file)
        else:
            raise NotImplementedError(f"Can't load {normals_file}-file type")

        if permute is not None:
            normals = self.permuteTensor(normals, permute)
        if flip_dim is not None and flip_index is not None:
            normals = self.flipTensor(normals, flip_dim, flip_index)
        return normals

    def loadIntrinsics(self, intrinsics_file: str,
                       dict_key: Optional[str] = None) -> torch.Tensor:
        if self._extIs(intrinsics_file, '.txt'):
            intrinsics = self._loadTxtFileToTensor(intrinsics_file)
        elif self._extIs(intrinsics_file, '.mat'):
            intrinsics = self._loadMatFile(intrinsics_file, dict_key)
        elif self._extIs(intrinsics_file, '.pth'):
            intrinsics = self._loadPthFile(intrinsics_file)
        else:
            raise NotImplementedError(f"Can't load {intrinsics_file}-file type")
        return intrinsics

    def loadLighting(self, lighting_file: str,
                     dict_key: Optional[str] = None,
                     permute: Optional[Tuple[int, int, int]] = None,
                     flip_dim: Optional[Union[Tuple[int, ...], int]] = None,
                     flip_index: Optional[Union[Tuple[int, ...], int]] = None) -> torch.Tensor:
        if self._extIs(lighting_file, '.pth'):
            lighting = self._loadPthFile(lighting_file)
        elif self._extIs(lighting_file, '.mat'):
            lighting = self._loadMatFile(lighting_file, dict_key)
        elif self._extIs(lighting_file, '.hdr'):
            lighting = self._loadHDRFile(lighting_file)
        else:
            raise NotImplementedError(f"Can't load {lighting_file}-file type")

        if permute is not None:
            lighting = self.permuteTensor(lighting, permute)
        if flip_dim is not None and flip_index is not None:
            lighting = self.flipTensor(lighting, flip_dim, flip_index)

        return lighting

    def loadEnvMaps(self, envmap_files: Union[List[str], str],
                    format: str = 'latlong',
                    gray: bool = True,
                    height: Optional[int] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        envmap_files = envmap_files if type(envmap_files) == list else [envmap_files]

        envmaps = []
        s2s = []
        masks = []
        for envmap_file in envmap_files:
            envmap, s2, mask = self.loadEnvMap(envmap_file, format=format, height=height, gray=gray)
            envmaps.append(envmap)
            s2s.append(s2)
            masks.append(mask)
        if sum([envmaps[0].shape == e.shape for e in envmaps]) != len(envmaps):
            sizes = [e.shape for e in envmaps]
            raise RuntimeError(
                f"Sizes of envmaps don't match. Use the 'height' input to fix all to one size: {sizes}")

        envmaps = torch.stack(envmaps, dim=0)
        s2s = torch.stack(s2s, dim=0)
        masks = torch.stack(masks, dim=0)
        return envmaps, s2s, masks

    def loadEnvMap(self, envmap_file: str,
                   format: str = 'latlong',
                   gray: bool = True,
                   height: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._extIs(envmap_file, '.hdr'):
            return self._loadHDRFile(envmap_file, format=format, height=height, gray=gray)
        else:
            raise NotImplementedError(f"Can't load {envmap_file}-file type")

    def loadDict(self, file_name: str) -> dict:
        if self._extIs(file_name, ".json"):
            d = self._loadJsonFile(file_name)
        elif self._extIs(file_name, ".pkl"):
            d = self._loadPickleFile(file_name)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")
        return d

    def saveMask(self,
                 mask_name: str,
                 mask_to_save: torch.Tensor,
                 mask_to_apply: Optional[torch.Tensor] = None,
                 dict_key: Optional[str] = None):
        if self._extIs(mask_name, '.png'):
            self._saveTorchImg(mask_name, mask_to_save.float(), mask_to_apply, bg=0.0)
        elif self._extIs(mask_name, '.pth'):
            self._savePthFile(mask_name, mask_to_save)
        elif self._extIs(mask_name, '.mat'):
            self._saveMatFile(mask_name, mask_to_save, dict_key)
        else:
            raise NotImplementedError(f"Can't save {mask_name}-file type")
        return

    def saveImages(self, img_names: Union[List[str], str],
                   imgs: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   dict_key: Optional[str] = None,
                   background: float = 1):
        if self._extIs(img_names[0], '.png'):
            assert type(img_names) == list and len(img_names) == imgs.shape[0]
            for img_name, img in zip(img_names, imgs):
                self.saveImage(img_name, img, mask, dict_key, background=background)
        else:
            assert type(img_names) == str
            self.saveImage(img_names, imgs, mask, dict_key, background=background)
        return

    def saveImage(self, img_name: str, img: torch.Tensor,
                  mask: Optional[torch.Tensor] = None,
                  dict_key: Optional[str] = None,
                  background: float = 1):
        if self._extIs(img_name, '.png'):
            self._saveTorchImg(img_name, img, mask, bg=background)
        elif self._extIs(img_name, '.pth'):
            self._savePthFile(img_name, img)
        elif self._extIs(img_name, '.mat'):
            self._saveMatFile(img_name, img, dict_key)
        else:
            raise NotImplementedError(f"Can't save {img_name}-file type")
        return

    def saveNormals(self, file_name: str, normals: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    dict_key: Optional[str] = None,
                    background: float = 0):
        if self._extIs(file_name, '.png'):
            self._saveNormalsTorchImg(file_name, normals, mask, bg=background)
        elif self._extIs(file_name, '.pth'):
            self._savePthFile(file_name, normals)
        elif self._extIs(file_name, '.mat'):
            self._saveMatFile(file_name, normals, dict_key)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")

        return

    def saveNormalsOnSphere(self, file_name: str, normals: torch.Tensor,
                            color: Optional[Union[torch.Tensor, np.ndarray, str]] = None,
                            elev: float = 0, azim: float = 0):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing {abspath} ...", end="", flush=True)
        assert normals.shape[-1] == 3

        import matplotlib.pyplot as plt

        if color is None:
            color = self.normalsToRGB(normals)
        if type(color) == torch.Tensor:
            color = color.cpu().numpy()

        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones_like(phi))

        s2 = normals.cpu().numpy()
        xi = s2[..., 0]
        yi = s2[..., 1]
        zi = s2[..., 2]

        warnings.filterwarnings("ignore")  # ignore 'VisibleDeprecationWarning' for plot_wireframe
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        ax.plot_wireframe(x, y, z, color='gray', rstride=1, cstride=1, alpha=0.2)
        ax.scatter(xi, yi, zi, s=0.2, c=color, zorder=10)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1, 1, 1])

        plt.savefig(abspath, bbox_inches='tight')
        plt.clf()
        warnings.filterwarnings("default")
        if self._verbose:
            print(
                f"\rWrote {os.path.splitext(abspath)[1][1:]}-file: {abspath}\t[{normals[..., 0].numel()}, {normals.dtype}]")
        return

    def saveDepth(self, file_name: str, depth: torch.Tensor,
                  normalize_png_01: bool = True,
                  mask: Optional[torch.Tensor] = None,
                  dict_key: Optional[str] = None,
                  K: Optional[torch.Tensor] = None,
                  background: float = 0):
        '''
        Save depth image. Supported types: [.png, .pth, .mat, .obj]. For obj-file depth must be [h,w] or [1,h,w]

        :param file_name: Ending in one of the supported types.
        :param depth:
        :param normalize_png_01: Normalize depth to [0, 1]. Optional for png-file.
        :param mask: Mandatory for obj-file. Optional for png-file.
        :param dict_key: Mandatory for mat-file
        :param K: Only for obj-file. None for orthographic depth, mandatory intrinsic (3,3) matrix for perspective depth.
        :return:
        '''
        if self._extIs(file_name, '.png'):
            if normalize_png_01:
                depth01 = (depth - depth.min()) / (depth - depth.min()).max()
                self._saveTorchImg(file_name, depth01, mask, bg=background)
            else:
                self._saveTorchImg(file_name, depth, mask, bg=background)
        elif self._extIs(file_name, '.pth'):
            self._savePthFile(file_name, depth)
        elif self._extIs(file_name, '.mat'):
            self._saveMatFile(file_name, depth, dict_key)
        elif self._extIs(file_name, '.obj'):
            self._saveObjFile(file_name, depth, mask, K)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")
        return

    def saveLighting(self, file_name: str, lighting: torch.Tensor,
                     dict_key: Optional[str] = None, img_size: Optional[int] = 256,
                     background: float = 0):
        '''
        If file_name ends with ".png", lighting is assumed to have shape [num_images, sh_dim, num_channels]
        and layout is [n, 1]

        :param file_name: file_name. Supported = [.pth, .mat, .txt, .png]
        :param lighting: lighting tensor
        :param dict_key: dictionar key for mat-file
        :param img_size: img_size for png-files
        :return:
        '''
        if self._extIs(file_name, '.pth'):
            self._savePthFile(file_name, lighting)
        elif self._extIs(file_name, '.mat'):
            self._saveMatFile(file_name, lighting, dict_key)
        elif self._extIs(file_name, '.txt'):
            self._saveTxtFile(file_name, lighting)
        elif self._extIs(file_name, '.png'):
            def lightToImages():
                '''

                :return: Returns light images of shape [num_images, num_faces, num_channels]
                '''
                r = img_size * 0.9 / 2
                X, Y = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='xy')
                zz = r ** 2 - (X - img_size / 2) ** 2 - (Y - img_size / 2) ** 2
                z_sphere = -torch.sqrt(zz)
                mask = ~torch.isnan(z_sphere)
                z_sphere += 1 - z_sphere[mask].min()  # make smallest depth value being 1
                mask = mask.unsqueeze(0)
                normals_sphere = depthToNormals(img2vec(z_sphere, mask), mask, K=None,
                                                southern=False)[0]
                sh_images = normalsToSH(normals_sphere, dim=-1,
                                        sh_order=int(math.sqrt(lighting.shape[1]) - 1))
                img_vecs = (sh_images[None, ..., None] * lighting[:, None, ...]).sum(dim=-2)
                img_vecs = torch.clamp(img_vecs, min=0, max=1)
                return img_vecs, mask

            img_vecs, mask = lightToImages()
            root, ext = os.path.splitext(file_name)
            num_images = img_vecs.shape[0]
            acc = f"0{int(math.log10(num_images)) + 2}d"
            self.saveImages([f"{root}{format(i, acc)}{ext}" for i in range(num_images)],
                            img_vecs, mask, background=background)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")

        return

    def saveTensor(self, file_name: str, tensor: torch.Tensor,
                   dict_key: Optional[str] = None):
        if self._extIs(file_name, ".pth"):
            self._savePthFile(file_name, tensor)
        elif self._extIs(file_name, ".mat"):
            self._saveMatFile(file_name, tensor, dict_key)
        elif self._extIs(file_name, ".txt"):
            self._saveTxtFile(file_name, tensor)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")

        return

    def saveStr(self, file_name: str, str_to_save: str):
        if self._extIs(file_name, ".txt"):
            self._saveTxtFile(file_name, str_to_save)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")
        return

    def saveDict(self, file_name: str, dict_to_save: dict):
        if self._extIs(file_name, ".json"):
            self._saveJsonFile(file_name, dict_to_save)
        elif self._extIs(file_name, ".txt"):
            self._saveTxtFile(file_name, dict_to_save)
        elif self._extIs(file_name, ".pkl"):
            self._savePickleFile(file_name, dict_to_save)
        else:
            raise NotImplementedError(f"Can't save {file_name}-file type")
        return

    def pillowToTorch(self, pil_img: Image) -> torch.Tensor:
        return transforms.ToTensor()(pil_img).to(self._device)

    def torchToPillow(self, torch_tensor: torch.Tensor) -> Image:
        return transforms.ToPILImage()(torch_tensor)

    def normalsToRGB(self, normals: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * normals

    def rgbToNormals(self, normals: torch.Tensor) -> torch.Tensor:
        return normals * 2 - 1

    def _setDevice(self, device: Optional[torch.device] = None, gpu: Optional[bool] = None,
                   gpu_id: Optional[int] = None):
        if device is not None and gpu is None and gpu_id is None:
            self._device = device
        elif device is None:
            if gpu is not None and gpu_id is not None:
                self._device = self.setDevice(gpu=gpu, gpu_id=gpu_id)
            elif gpu is None and gpu_id is not None:
                self._device = self.setDevice(gpu_id=gpu_id)
            elif gpu is not None and gpu_id is None:
                self._device = self.setDevice(gpu=gpu)
            else:
                self._device = self.setDevice()
        else:
            raise RuntimeError(
                f"'device' and '(gpu, gpu_id)' are both not None. Don't know what to do.")
        return

    def _loadMatFile(self, mat_file: str, dict_key: str) -> torch.Tensor:
        if self._verbose:
            print(f"Loading mat-file {mat_file} ...", end="", flush=True)
        loaded_dict = scipy.io.loadmat(mat_file)
        if dict_key in loaded_dict:
            mat = torch.from_numpy(loaded_dict[dict_key]).float().to(self._device)
        else:
            valid_keys = [k for k in loaded_dict.keys()]
            raise RuntimeError(f"Unknown key: {dict_key}. Valid keys: {valid_keys}")

        if self._verbose:
            print(f"\rLoaded mat-file: {mat_file}\t[{tuple(mat.shape)}, {mat.dtype}]")
        return mat

    def _saveMatFile(self, file_name: str, tensor: torch.Tensor, dict_key: str):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing mat-file {abspath} ...", end="", flush=True)
        scipy.io.savemat(abspath, {dict_key: tensor.cpu().numpy()})
        if self._verbose:
            print(f"\rWrote mat-file: {abspath}\t[{tuple(tensor.shape)}, {tensor.dtype}]")
        return

    def _loadPthFile(self, pth_file: str) -> torch.Tensor:
        if self._verbose:
            print(f"Loading pth-file {pth_file} ...", end="", flush=True)
        pth = torch.load(pth_file).to(self._device)
        if self._verbose:
            print(f"\rLoaded pth-file: {pth_file}\t[{tuple(pth.shape)}, {pth.dtype}]")
        return pth

    def _savePthFile(self, file_name: str, tensor: torch.Tensor):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing pth-file {abspath} ...", end="", flush=True)
        torch.save(tensor, abspath)
        if self._verbose:
            print(f"\rWrote pth-file: {abspath}\t[{tuple(tensor.shape)}, {tensor.dtype}]")
        return

    def _saveTxtFile(self, file_name: str, data: Union[torch.Tensor, str, dict]):
        if type(data) == torch.Tensor:
            self._saveTensorToTxtFile(file_name, data)
        elif type(data) == str:
            self._saveStrToTxt(file_name, data)
        elif type(data) == dict:
            self._saveDictToTxt(file_name, data)
        return

    def _loadPickleFile(self, pkl_file: str) -> dict:
        abspath = self._name2Abspath(pkl_file)
        if self._verbose:
            print(f"Loading pkl-file {abspath} ...", end="", flush=True)
        with open(pkl_file, 'rb') as f:
            dict_data = pickle.load(f)
        if self._verbose:
            print(f"\rLoaded pkl-file: {abspath}\t[{len(dict_data.keys())}, {type(dict_data)}]")
        return dict_data

    def _savePickleFile(self, file_name: str, data: dict):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing pkl-file {abspath} ...", end="", flush=True)
        with open(abspath, 'wb') as f:
            pickle.dump(data, f)
        if self._verbose:
            print(f"\rWrote pkl-file: {abspath}\t[{len(data.keys())}, {type(data)}]")
        return

    def _loadTxtFileToTensor(self, txt_file: str) -> torch.Tensor:
        if self._verbose:
            print(f"Loading txt-file {txt_file} ...", end="", flush=True)
        txt = torch.from_numpy(np.loadtxt(txt_file, dtype=np.float32)).to(self._device)
        if self._verbose:
            print(f"\rLoaded txt-file: {txt_file}\t[{tuple(txt.shape)}, {txt.dtype}]")
        return txt

    def _saveTensorToTxtFile(self, file_name: str, tensor: torch.Tensor):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing txt-file {abspath} ...", end="", flush=True)
        np.savetxt(abspath, tensor.cpu().numpy())
        if self._verbose:
            print(f"\rWrote txt-file: {abspath}\t[{tuple(tensor.shape)}, {tensor.dtype}]")
        return

    def _saveStrToTxt(self, file_name: str, data_str: str):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing txt-file {abspath} ...", end="", flush=True)
        with open(abspath, "w") as f:
            f.write(data_str)
        if self._verbose:
            print(f"\rWrote txt-file: {abspath}\t[{len(data_str)}, {type(data_str)}]")
        return

    def _saveDictToTxt(self, file_name: str, data_dict: dict):
        def _dict2str(d: dict, lvl=0) -> str:
            data_str = ""
            for k, v in sorted(d.items()):
                if type(v) == dict:
                    tabs = '  ' * (lvl)
                    data_str += f"{tabs}{k}:\n{_dict2str(v, lvl + 1)}\n"
                    data_str = data_str[:-1]  # remove last newline
                else:
                    tabs = '  ' * lvl
                    data_str += f"{tabs}{k}: {v}\n"
            return data_str

        self._saveStrToTxt(file_name, _dict2str(data_dict))
        return

    def _loadJsonFile(self, json_file: str) -> dict:
        if self._verbose:
            print(f"Loading json-file {json_file} ...", end="", flush=True)
        dic = ''
        with open(json_file, 'r') as f:
            for i in f.readlines():
                dic = i  # string
        json_data = eval(dic)

        if self._verbose:
            print(f"\rLoaded json-file: {json_file}\t[{len(json_data.keys())}, {type(json_data)}]")
        return json_data

    def _saveJsonFile(self, file_name: str, json_data: dict):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing json-file {abspath} ...", end="", flush=True)
        with open(abspath, 'w+') as fp:
            fp.write(str(dict(sorted(json_data.items()))))
        if self._verbose:
            print(f"\rWrote json-file: {abspath}\t[{len(json_data.keys())}, {type(json_data)}]")
        return

    def _saveObjFile(self, file_name: str,
                     depth: torch.Tensor,
                     mask: torch.Tensor,
                     K: torch.Tensor):
        abspath = self._name2Abspath(file_name)
        if self._verbose:
            print(f"Writing obj-file {abspath} ...", end="", flush=True)
        if depth.dim() <= 2:
            depth = vec2img(depth, mask=mask)
        DepthToOBJ.save(depth, mask, abspath, K)
        if self._verbose:
            print(f"\rWrote obj-file: {abspath}\t[{tuple(depth.shape)}, {depth.dtype}]")
        return

    def _loadHDRFile(self, hdr_file: str,
                     format: str,
                     height: Optional[int],
                     gray: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._verbose:
            print(f"Loading hdr-file {hdr_file} ...", end="", flush=True)
        from third_party.skylibs.envmap import EnvironmentMap
        envmap = EnvironmentMap(hdr_file, format_=format)

        if height is not None:
            envmap.resize(height)

        envmap = envmap.convertTo('angular')

        if gray is None:
            gray = self._gray

        if gray:
            envmap.toIntensity()

        x, y, z, mask = envmap.worldCoordinates()
        envmap = torch.from_numpy(envmap.data).float().to(self._device).permute(2, 0, 1)
        s2 = torch.from_numpy(np.concatenate([x, y, z], axis=-1)).float().to(self._device)
        s2 = s2.reshape(s2.shape[0], 3, s2.shape[1] // 3).permute(1, 0, 2)
        mask = torch.from_numpy(mask).bool().to(self._device)[None, ...]
        if self._verbose:
            print(f"\rLoaded hdr-file: {hdr_file}"
                  f"\t[{tuple(envmap.shape)}, {envmap.dtype}]"
                  f"\t[{tuple(s2.shape)}, {s2.dtype}]"
                  f"\t[{tuple(mask.shape)}, {mask.dtype}]")
        return envmap, s2, mask

    def _loadNormalsTorchImg(self, normals_file: str) -> torch.Tensor:
        return self.rgbToNormals(self._loadTorchImage(normals_file, gray=False))

    def _saveNormalsTorchImg(self, img_name: str, normals: torch.Tensor,
                             mask: Optional[torch.Tensor], bg: float):
        self._saveTorchImg(img_name, self.normalsToRGB(normals), mask, bg=bg)
        return

    def _loadTorchImage(self, img_file: str, gray: Optional[bool]) -> torch.Tensor:
        '''
        Load image

        :param img_file: str to png file
        :return: torch tensor of shape [1, height, width]
        '''
        if self._verbose:
            print(f"Loading torch image {img_file} ...", end="", flush=True)
        pil_img = self._loadPillowImg(img_file, gray)
        img = self.pillowToTorch(pil_img)

        if self._verbose:
            print(f"\rLoaded torch image: {img_file}\t[{tuple(img.shape)}, {img.dtype}]")
        return img

    def _saveTorchImg(self, img_name: str, img: torch.Tensor, mask: Optional[torch.Tensor],
                      bg: float):
        if self._verbose:
            print(f"Writing torch image image {self._name2Abspath(img_name)} ...", end="",
                  flush=True)
        if mask is not None:
            img = vec2img(img, mask=mask, bg=bg)

        self._savePillowImg(img_name, self.torchToPillow(img))
        if self._verbose:
            print(
                f"\rWrote torch image: {self._name2Abspath(img_name)}\t[{tuple(img.shape)}, {img.dtype}]")
        return

    def _loadPillowImg(self, img_file: str, gray: Optional[bool]) -> Image:
        pil_img = Image.open(img_file)

        if gray is None:
            gray = self._gray

        if gray and len(pil_img.getbands()) > 1:  # more than on channel
            pil_img = ImageOps.grayscale(pil_img)

        return pil_img

    def _savePillowImg(self, img_name: str, img: Image):
        abspath = self._name2Abspath(img_name)
        img.save(abspath)
        return

    @classmethod
    def _extIs(cls, filename: str, ext: str) -> bool:
        return filename.lower().endswith(ext)

    def _name2Abspath(self, img_name: str) -> str:
        abspath = os.path.abspath(os.path.join(self._output, img_name))
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        return abspath

    def permuteTensor(self, tensor: torch.Tensor,
                      permute: Tuple[int, int, int]) -> torch.Tensor:
        tensor = tensor.permute(*permute)
        if self._verbose:
            print(f"\tPermute tensor: {permute}")
        return tensor

    def flipTensor(self, tensor: torch.Tensor,
                   flip_dim: Union[Tuple[int, ...], int],
                   flip_index: Union[Tuple[int, ...], int]) -> torch.Tensor:
        if type(flip_dim) == tuple and type(flip_index) == tuple:
            for d, i in zip(flip_dim, flip_index):
                tensor = self.flipTensor(tensor, d, i)
        elif type(flip_dim) == int and type(flip_index) == int:
            ti = tensor.index_select(dim=flip_dim,
                                     index=torch.tensor(flip_index, dtype=torch.long))
            tensor = tensor.index_add(dim=flip_dim,
                                      index=torch.tensor(flip_index, dtype=torch.long),
                                      source=ti, alpha=-2)
            if self._verbose:
                print(f"\tFlip tensor (after permute): dim={flip_dim}, index={flip_index}")
        else:
            raise RuntimeError(
                f"Unknown combination of types: type(flip_dim)={type(flip_dim)} and type(flip_index)={type(flip_index)}. Types have to be equal.")
        return tensor
