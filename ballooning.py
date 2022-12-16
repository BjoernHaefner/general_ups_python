####################################################################################################
# The code here is used to implement the paper:                                                    #
# "Variational Uncalibrated Photometric Stereo under General Lighting"                             #
# Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers                 #
# In International Conference on Computer Vision (ICCV), 2019                                      #
#                                                                                                  #
# CopyRight (C) 2017 Bjoern Haefner (bjoern.haefner@tum.de), Computer Vision Group, TUM            #
####################################################################################################
import argparse
import os
import pathlib
import time

import math
import torch
from scipy import ndimage

from utils.data_io import DataIO
from utils.helpers import img2vec, vec2img
from utils.optimization import gradientDescentStep
from utils.sparse import nabla2d


class Balloon:
    '''
    Implementation of the paper
    Fast and Globally Optimal Single View Reconstruction of Curved Objects
    M. R. Oswald, E. Toeppe and D. Cremers,
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2012.
    https://vision.in.tum.de/_media/spezial/bib/oswald_toeppe_cremers_cvpr12.pdf

    This framework solves the following problem:

        min_{u\in C} int_S sqrt(1 + |nabla @ u|**2) dx,
        where C = {u | int_S u dx = V}
    '''
    _mask: torch.Tensor
    _volume: float
    _tau: float
    _maxit: int
    _tol: float
    _nabla: torch.Tensor
    _boundary: torch.Tensor
    _div: torch.Tensor

    def __init__(self,
                 maxit: int = 10000,
                 tol: float = 1e-15,
                 tau: float = 0.8 / math.sqrt(8.)):
        assert maxit > 0 and tol > 0 and tau > 0
        self._maxit = maxit
        self._tol = tol
        self._tau = tau
        return

    @classmethod
    def _compute_boundary(cls, mask: torch.Tensor):
        assert mask.dim() == 3 and mask.shape[0] == 1
        m = mask.squeeze(0).float().cpu().numpy()
        m2 = ndimage.binary_erosion(m).astype(m.dtype)
        mask_interior = torch.from_numpy(m2).bool().to(mask.device)[None, ...]
        return img2vec(mask & ~mask_interior, mask).squeeze(-1)

    def init(self, mask: torch.Tensor, volume: float):
        assert mask.dim() == 3 and mask.shape[0] == 1, \
            f'Expected mask shape [1, h, w] got {mask.shape}'
        self._mask = mask
        self._volume = self._mask.sum().item() * volume
        self._boundary = self._compute_boundary(self._mask)
        self._nabla, self._div = nabla2d(self._mask)
        return

    def energy(self, u):
        '''Computes surface integral int_Omega dA'''
        return self._dA(u).sum(dim=0)

    def run(self):
        def dEdu(u: torch.Tensor) -> torch.Tensor:
            '''Computes Euler-Lagrange derivative pixel-wise: dE/du(x) = -div(1/dA * nabla*u)'''
            return -self._div @ (1 / self._dA(u).repeat(2, 1) * (self._nabla @ u))

        def projectVolume(u: torch.Tensor) -> torch.Tensor:
            '''
            Projects onto depth with volume, i.e. solves:
            min_u 1/2 * int_Omega ||u - u0||^2_2 dx, s.t. int_Omega u dx = V,
            has closed-form solution: u = u0 + (V - int_Omega u0 dx) / int_Omega dx
            '''
            return u + ((self._volume - u.sum(dim=0)) / self._mask.sum())[None, :]

        def projectBoundary(u: torch.Tensor) -> torch.Tensor:
            '''
            Projects boundary of depth
            '''
            u[self._boundary, :] = 0
            return u

        u = torch.ones(self._mask.sum(), 1)
        energy_last = self.energy(u)
        for i in range(self._maxit):
            if i % 100 == 0:
                print(f"\r{i}/{self._maxit - 1}: mean energy: {self.energy(u).mean()}", end="",
                      flush=True)
            u = gradientDescentStep(u, self._tau, dEdu(u))
            u = projectBoundary(u)
            u = projectVolume(u)

            # check for convergence
            energy_new = self.energy(u)
            convergence = ((energy_last - energy_new) / energy_new).abs().max()
            if i % 100 == 0:
                print(f" | {convergence} < {self._tol}", end="", flush=True)
            if convergence < self._tol:
                print(f"\r{i}/{self._maxit - 1}: mean energy: {self.energy(u).mean()}", end="",
                      flush=True)
                print(f" | {convergence} < {self._tol} | CONVERGED", end="", flush=True)
                break
            energy_last = energy_new.detach().clone()
        print()
        u = projectBoundary(u)
        return vec2img(u, mask=self._mask)

    def _dA(self, u: torch.Tensor) -> torch.Tensor:
        '''
        Computes pixel-wise dA(x) = sqrt(1 + |nabla*u(x)|^2)

        :param u: [num_faces, 1]
        '''
        nablau = self._nabla @ u
        nablauu = nablau * nablau
        return torch.sqrt(nablauu.reshape(2, u.shape[0], u.shape[1]).sum(dim=0) + 1)


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='Ballooning',
                                     description='A variational solver for ballooning.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    subparsers = parser.add_subparsers(help='Run manually or example', dest="subcommand")
    subparsers.add_parser("example", help="Run example")
    sub_man = subparsers.add_parser("manual", help="Run manually. Run \"manual -h\" for more info",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group_required = sub_man.add_argument_group('required named arguments')
    group_required.add_argument('-m', '--mask', required=True, type=str,
                                help='Mask as .png or .pth-file [%(type)s]')
    group_required.add_argument('-v', '--volume', required=True, type=float,
                                help="Depth's volume [%(type)s]")

    group_algo = sub_man.add_argument_group('algorithmic options')
    group_algo.add_argument('--maxit', default=10000, type=int,
                            help="Max number of iterations [%(type)s]")
    group_algo.add_argument('--tol', default=1e-15, type=float,
                            help="Tolerance for convergence [%(type)s]")
    group_algo.add_argument('--tau', default=0.8 / math.sqrt(8.), type=float,
                            help="Gradient descent step size [%(type)s]")

    parser_system = parser.add_argument_group('system options')
    parser_system.add_argument('-g', '--gpu', default=False, action="store_true",
                               help="Use GPU (if not available, fall back to CPU)")
    parser_system.add_argument('--gpu_id', default=0, type=int, help="GPU ID [%(type)s]")
    output_default = os.path.join(pathlib.Path(__file__).parent.resolve(), 'output', 'ballooning')
    parser_system.add_argument('-o', '--output', type=str, default=output_default,
                               help=f"Output directory [%(type)s]")

    cli = parser.parse_args()
    if cli.subcommand == 'example':
        args = ['-g',
                f'-o={os.path.join(output_default, "synthetic_joyfulyell_hippie")}',
                'manual',
                f'--mask={os.path.join("data", "synthetic_joyfulyell_hippie", "mask.png")}',
                f'--volume=24.77']
        print(f"Execute 'ballooning.py {' '.join(args)}'")
        return parser.parse_args(args)
    else:
        return cli


def main(cli: argparse.Namespace) -> None:
    io = DataIO(gpu=cli.gpu, gpu_id=cli.gpu_id, output=cli.output)
    mask = io.loadMask(cli.mask)

    bl = Balloon(maxit=cli.maxit, tol=cli.tol, tau=cli.tau)
    t_start = time.time()
    bl.init(mask, volume=cli.volume)
    balloon = bl.run()
    t_stop = time.time()
    print(f"Elapsed time for optimization: {t_stop - t_start}")

    if cli.output != "":
        io.saveDepth(f"balloon.png", balloon)
        io.saveDepth(f"balloon.pth", balloon)
        io.saveDepth(f"balloon.obj", balloon, mask=mask)
    return


if __name__ == "__main__":
    main(parser())
