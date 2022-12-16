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
from typing import Optional, Dict

import torch

import general_ups
from ballooning import Balloon
from general_ups import GeneralUPS
from utils.data_io import DataIO
from utils.helpers import normalsToDepth, img2vec, vec2img, depthToNormals


def balloonToZInit(balloon: torch.Tensor, mask: torch.Tensor,
                   K: Optional[torch.Tensor]) -> torch.Tensor:
    '''
    Computes balloon-depth (orthographic) to normals, then integrates normals to perspective depth map

    :param balloon: [h, w] or [1, h, w]
    :param mask: [1, h, w]
    :param K: [3, 3] (perspective) or None (orthographic)
    '''
    # we need to flip the balloon as its differently oriented
    z_ortho = -balloon
    normals = depthToNormals(img2vec(z_ortho, mask), mask, K=None)[0]
    return vec2img(normalsToDepth(normals, mask, K=K), mask=mask)


def save(io: DataIO, mask: torch.Tensor, K: torch.Tensor,
         balloon: torch.Tensor, z_init: torch.Tensor,
         rho_out: torch.Tensor, l_out: torch.Tensor, z_out: torch.Tensor,
         error_metrics: Dict) -> None:
    io.saveDepth(f"balloon.png", balloon)
    io.saveDepth(f"balloon.pth", balloon)
    io.saveDepth(f"balloon.obj", balloon, mask=mask)
    io.saveDepth(f"z_init.png", z_init)
    io.saveDepth(f"z_init.pth", z_init)
    io.saveDepth(f"z_init.obj", z_init, mask=mask, K=K)
    general_ups.save(io, mask, K, rho_out, l_out, z_out, error_metrics)
    return


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='BalloonGeneralUPS',
                                     description='A variational solver for uncalibrated photometric stereo under general lighting with balloon initialization.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    subparsers = parser.add_subparsers(help='Run manually or example', dest="subcommand")
    subparsers.add_parser("example1", help="Run synthetic example")
    subparsers.add_parser("example2", help="Run real-world example")
    sub_man = subparsers.add_parser("manual", help="Run manually. Run \"manual -h\" for more info",
                                    add_help=False,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group_required = sub_man.add_argument_group('required named arguments')
    group_required.add_argument('-m', '--mask', required=True, type=str,
                                help='Mask for image files as .png or .pth-file [%(type)s]')
    group_required.add_argument('-i', '--images', required=True,
                                type=str, nargs="+",
                                help='Image file(s) as .png or .pth-file [%(type)s]')
    group_required.add_argument('-v', '--volume', required=True, type=float,
                                help="Depth's volume [%(type)s]")

    group_optional = sub_man.add_argument_group('optional arguments')
    group_optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                                help='show this help message and exit')
    group_optional.add_argument('-k', '--intrinsics', default=None, type=str,
                                help='Intrinsics matrix (default: orthographic) as .pth-file [%(type)s]')
    group_optional.add_argument('--init_light', default=None, type=str,
                                help="Initilization lighting as .pth-file [%(type)s]")
    group_optional.add_argument('--init_albedo', default=None, type=str,
                                help="Albedo for initialization as .png or .pth-file [%(type)s]")
    group_optional.add_argument('-z', '--gt_depth', default=None, type=str,
                                help="Ground truth depth as .pth-file [%(type)s]")
    group_optional.add_argument('-l', '--gt_light', default=None, type=str,
                                help="Ground truth lighting as .pth-file [%(type)s]")
    group_optional.add_argument('-a', '--gt_albedo', default=None, type=str,
                                help="Albedo for initialization as .png or .pth-file [%(type)s]")

    group_algo = sub_man.add_argument_group('high-level algorithmic options')
    group_algo.add_argument('--maxit_ballooning', default=10000, type=int,
                            help="Max number of iterations or ballooning [%(type)s]")
    group_algo.add_argument('--maxit_upssolver', default=20, type=int,
                            help="Max number of iterations for generalUPS solver [%(type)s]")

    parser_system = parser.add_argument_group('system options')
    parser_system.add_argument('-g', '--gpu', default=False, action="store_true", help="Use GPU.")
    parser_system.add_argument('--gpu_id', default=0, type=int, help="GPU ID.")
    output_default = os.path.join(pathlib.Path(__file__).parent.resolve(), 'output')
    parser_system.add_argument('-o', '--output', default=output_default,
                               help=f"Output directory")

    cli = parser.parse_args()
    if cli.subcommand == 'example1':
        args = ['-g',
                f'-o={os.path.join(output_default, "synthetic_joyfulyell_hippie")}',
                'manual', '-v=24.77',
                f'-m={os.path.join("data", "synthetic_joyfulyell_hippie", "mask.png")}',
                f'-i={os.path.join("data", "synthetic_joyfulyell_hippie", "images.pth")}',
                f'-k={os.path.join("data", "synthetic_joyfulyell_hippie", "K.pth")}',
                f'-z={os.path.join("data", "synthetic_joyfulyell_hippie", "z_gt.pth")}',
                f'-l={os.path.join("data", "synthetic_joyfulyell_hippie", "l_gt_25x9x3.pth")}',
                f'-a={os.path.join("data", "synthetic_joyfulyell_hippie", "rho_gt.pth")}']
        print(f"Execute 'main.py {' '.join(args)}'")
        return parser.parse_args(args)
    elif cli.subcommand == 'example2':
        args = ['-g',
                f'-o={os.path.join(output_default, "xtion_backpack_sf4_ups")}',
                'manual', '-v=40',
                f'-m={os.path.join("data", "xtion_backpack_sf4_ups", "mask.png")}',
                f'-i={os.path.join("data", "xtion_backpack_sf4_ups", "images.pth")}',
                f'-k={os.path.join("data", "xtion_backpack_sf4_ups", "K.pth")}']
        print(f"Execute 'main.py {' '.join(args)}'")
        return parser.parse_args(args)
    else:
        return cli


def main(cli: argparse.Namespace) -> None:
    io = DataIO(gpu=cli.gpu, gpu_id=cli.gpu_id, output=cli.output)

    mask = io.loadMask(cli.mask)
    I = io.loadImages(cli.images)
    K = io.loadIntrinsics(cli.intrinsics) if cli.intrinsics else None

    l_init = io.loadLighting(cli.init_light) if cli.init_light else None
    rho_init = io.loadImage(cli.init_albedo) if cli.init_albedo else None

    z_gt = io.loadImage(cli.gt_depth) if cli.gt_depth else None
    l_gt = io.loadLighting(cli.gt_light) if cli.gt_light else None
    rho_gt = io.loadImage(cli.gt_albedo) if cli.gt_albedo else None

    bl = Balloon(maxit=cli.maxit_ballooning)
    t_start = time.time()
    bl.init(mask, volume=cli.volume)
    balloon = bl.run()
    t_stop = time.time()
    print(f"Elapsed time for balloon optimization: {t_stop - t_start}")

    z_init = balloonToZInit(balloon, mask, K)

    gups = GeneralUPS(maxit=cli.maxit_upssolver)
    t_start = time.time()
    gups.init(I, mask, K, z_init, rho_init, l_init)
    z_out, rho_out, l_out = gups.run()
    t_stop = time.time()
    print(f"Elapsed time for general_ups optimization: {t_stop - t_start}")

    error_metrics = None
    if z_gt is not None or l_gt is not None or rho_gt is not None:
        error_metrics = general_ups.evaluate(mask, K,
                                             z_out, rho_out, l_out,
                                             z_gt, rho_gt, l_gt)
    if cli.output != "":
        save(io, mask, K, balloon, z_init, rho_out, l_out, z_out, error_metrics)
    return


if __name__ == "__main__":
    main(parser())
