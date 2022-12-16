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

import math
import numpy as np
import torch
from scipy.linalg import pinv

from utils.data_io import DataIO
from utils.error_metrics import calcRMSE, calcAngularError, optC
from utils.helpers import img2vec, normalsToSH, vec2img, depthToNormals
from utils.optimization import gradientAscentStep, _cholesky, _cg
from utils.sparse import nabla2d, speye, hstack, spdiag, spzeros, transpose, vstack


class GeneralUPS:
    '''
    Implementation of the solver detailed in the paper
    Variational Uncalibrated Photometric Stereo under General Lighting
    Bjoern Haefner, Zhenzhang Ye, Maolin Gao, Tao Wu, Yvain Quéau and Daniel Cremers,
    In IEEE/CVF International Conference on Computer Vision (ICCV),
    2019.
    https://openaccess.thecvf.com/content_ICCV_2019/html/Haefner_Variational_Uncalibrated_Photometric_Stereo_Under_General_Lighting_ICCV_2019_paper.html
    '''
    __pinv: str = "numpy"  # numpy, scipy, torch
    __solve: str = "best"  # cholesky, cg, best
    __stencil_z: str = 'forward'
    __bc_z: str = 'dirichlethomogeneous'
    __stencil_a: str = 'central'
    __bc_a: str = 'neumannhomogeneous'
    __linesearch_strict: bool = False  # original paper: False;
    __fix_clamp: bool = True  # original paper: False; mathematically correct: True
    __fix_J_sh: bool = True  # original paper: False; mathematically correct True
    __fix_mu_normalization: bool = True  # original paper: False; mathematically consistent: True
    __fix_irls_cost: bool = True  # original paper: False; mathematically correct: True

    def __init__(self, sh_order=1, c2f_lighting=8,
                 irls="cauchy", lambda_=1, mu=0.045245, huber=0.1, delta=4.5e-4,
                 maxit=20, tol=1e-6, eps=1e-6,
                 beta_init=5e-4, kappa=1,
                 albedo_pcg_tol=1e-6, albedo_pcg_maxit=1000,
                 depth_pcg_tol=1e-10, depth_pcg_maxit=1e3,
                 depth_linesearch_maxit=3, depth_linesearch_t=1e1,
                 depth_linesearch_maxit_linesearch=1e3,
                 verbose=1):
        assert irls in ["cauchy", "l2"]

        self.sh_order = sh_order
        self.c2f_lighting = c2f_lighting

        self.irls = irls
        self.lambda_ = lambda_
        self.mu_init = mu
        self.huber = huber
        self.delta = delta

        self.maxit = maxit
        self.tol = tol
        self.eps = eps

        self.beta_init = beta_init
        self.kappa = kappa

        self.albedo_pcg_tol = albedo_pcg_tol
        self.albedo_pcg_maxit = albedo_pcg_maxit

        self.depth_pcg_tol = depth_pcg_tol
        self.depth_pcg_maxit = depth_pcg_maxit

        self.depth_linesearch_maxit = depth_linesearch_maxit
        self.depth_linesearch_t = depth_linesearch_t
        self.depth_linesearch_maxit_linesearch = depth_linesearch_maxit_linesearch

        self.verbose = verbose

        # energy array
        self.energy = {
            'tab_energy': [],
            'tab_energy_irls': [],
            'tab_objective': [],
            'tab_no_smooth': [],
            'tab_rho': [],
            'tab_rho_res': [],
            'tab_s_res': [],
            'tab_s': [],
            'tab_z': [],
            'tab_z_res': [],
            'tab_theta': [],
            'tab_theta_primal': [],
            'tab_theta_dual': [],
        }
        return

    def init(self, I: torch.Tensor, mask: torch.Tensor, K: Optional[torch.Tensor],
             z_init: Optional[torch.Tensor],
             albedo_init: Optional[torch.Tensor] = None,
             light_init: Optional[torch.Tensor] = None):
        '''

        :param I: (num_images, num_channels, h, w) [torch.tensor [float]]
        :param mask: (1, h, w) [torch.tensor [bool]]
        :param K: None (orthographic projection or [3, 3] (perspective projection) [torch.tensor [float]]
        :param z_init: (1, h, w) [torch.tensor [float]]
        :param albedo_init: (num_channels, h, w) [torch.tensor [float]]
        :param light_init: (num_images, (sh_order + 1)^2, num_channels) [torch.tensor [float]]
        :return:
        '''
        # data initialization
        assert I.dim() == 4, \
            f'Expected I shape: [n, c, h, w], but got {I.shape}'
        n, c, h, w = I.shape

        assert mask.shape == (1, h, w), \
            f'Expected mask shape: [1, {h}, {w}], but got {mask.shape}'
        assert K is None or K.shape == (3, 3), \
            f'Expected K shape: [3, 3], but got {K.shape}'
        assert z_init.shape == (1, h, w) or z_init.shape == (h, w), \
            f'Expected z_init shape: [1, {h}, {w}] or  [{h}, {w}], but got {z_init.shape}'
        assert albedo_init is None or albedo_init.shape == (c, h, w), \
            f'Expected albedo_init shape: [{c}, {h}, {w}], but got {albedo_init.shape}'
        assert light_init is None or light_init.shape == (n, 4, c) or light_init.shape == (n, 3, c), \
            f'Expected light_init shape: [{n}, (4, 9), {c}], but got {light_init.shape}'

        self.mask = mask.detach().clone()
        self.K = K.detach().clone() if K is not None else None
        self.I = I.detach().clone()

        self.num_images, self.num_channels = self.I.shape[:2]
        self.num_faces = self.mask.sum()

        self.z_init = z_init.detach().clone()
        self.rho_init = self.I.median(dim=0).values if albedo_init is None else albedo_init

        if light_init is None:
            light_init = torch.tensor([0, 0, -1, 0.2, 0, 0, 0, 0, 0])[None, :, None]
            light_init = light_init.repeat(self.num_images, 1, self.num_channels)

        if self.sh_order == 1:
            if light_init.shape[1] == 4:
                self.s_init = light_init
            elif light_init.shape[1] == 9:
                self.s_init = light_init[:, :4, :]
        elif self.sh_order == 2:
            if light_init.shape[1] == 4:
                self.s_init = torch.cat(
                    [light_init, torch.zeros(self.num_images, 5, self.num_channels)], dim=1)
            elif light_init.shape[1] == 9:
                self.s_init = light_init

        self.nabla_z, self.div_z = nabla2d(self.mask, stencil=self.__stencil_z, bc=self.__bc_z)[:2]
        self.nabla_a, self.div_a = nabla2d(self.mask, stencil=self.__stencil_a, bc=self.__bc_a)[:2]
        return

    def _vec_and_init(self):
        # (num_images, num_faces, num_channels)
        self.I = torch.stack([img2vec(I, mask=self.mask) for I in self.I], dim=0)
        # (num_faces, num_channels)
        rho = img2vec(self.rho_init, mask=self.mask)
        # (num_images, (sh_order + 1)^2, num_channels)
        s = self.s_init
        # (num_faces, 1)
        z = img2vec(self.z_init, mask=self.mask)

        if self.lambda_ is None:
            self.lambda_ = self.delta * (self.I - self.I.median()).abs().median().item()

        # parameter normalization
        if self.__fix_mu_normalization:
            div_lambda = self.lambda_
        else:
            div_lambda = self.delta * (self.I - self.I.median()).abs().median()
        div_lambda /= (self.num_channels * self.num_images)
        self.mu = self.mu_init / (div_lambda * self.num_channels)

        # step size initialization
        self.beta = self.beta_init
        return z, s, rho

    @classmethod
    def _render(cls, rho, s, sh):
        # Canonical shape: [num_images, num_faces, ((sh_order+1)^1), num_channels]
        s_ = s[:, None, :, :]
        sh_ = sh[None, :, :, None]
        rho_ = rho[None, :, :]
        return (sh_ * s_).sum(dim=2) * rho_

    def __residual(self, rho, s, sh):
        return self._render(rho, s, sh) - self.I

    def calcEnergyCauchy(self, rho, s, sh, theta, drho, dz, irls_weights):
        # Photometric term
        residual = self.__residual(rho, s, sh)
        residual2 = residual * residual
        lambda2 = self.lambda_ * self.lambda_
        if self.__fix_irls_cost:
            energy_irls = 0.5 * (irls_weights * residual2).sum()
        else:
            energy_irls = 0.5 * self.lambda_ * residual2.sum()

        if self.irls == "cauchy":
            energy = lambda2 * torch.log(1 + residual2 / lambda2).sum()
        elif self.irls == "l2":
            energy = residual2.sum()
        else:
            raise ValueError(f"Unknown value for irls: {self.irls}")

        energy_irls_no_smooth = energy_irls.detach().clone()

        # Smoothness term on albedo
        if self.mu > 0:
            rho_huber = torch.zeros_like(drho)
            rho_huber[drho >= self.huber] = drho[drho >= self.huber].abs() - 0.5 * self.huber
            rho_huber[drho < self.huber] = 0.5 / self.huber \
                                           * drho[drho < self.huber] * drho[drho < self.huber]
            energy_reg = self.mu * rho_huber.sum()
            energy_irls += energy_reg
            energy += energy_reg

        # objective augmented Lagrangian
        soft_constraint = 0.5 * self.beta * ((theta - dz) * (theta - dz)).sum()
        objective = energy_irls + soft_constraint

        self.energy['tab_energy_irls'].append(energy_irls)
        self.energy['tab_objective'].append(objective)
        self.energy['tab_no_smooth'].append(energy_irls_no_smooth)
        self.energy['tab_energy'].append(energy)
        self.energy['tab_theta_primal'].append(soft_constraint)
        return energy_irls, objective, energy_irls_no_smooth, energy

    def run(self):
        '''
        Apply the Lagged Block coordinate descent to optimize the energy function

        :return:  z_out              estimated depth.          h*w
                  rho_out            estimated albedo.         h*w*c
                  s_out              estimated lighting.       9*c*N
                  plot_energy        used to plot figures.     struct
        '''

        z, s, rho = self._vec_and_init()
        normals, dz, n_unnormalized, J_n_un, J_dz = self.depthToNormals(z)
        # initialize auxiliary and dual variable
        theta = dz
        nabla_rho = self.nabla_a @ rho

        # Initial augmented normals. See the function for more details
        sh = self.normalsToSphericalHarmonics(normals)[0]
        self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, self.calcReweighting(rho, sh, s))
        if self.verbose > 0:
            print(f"Initial Energy")
            print(f"\t{self.energy['tab_energy'][-1]:.3f} "
                  f"{self.energy['tab_energy_irls'][-1]:.3f} "
                  f"{self.energy['tab_objective'][-1]:.3f} "
                  f"{self.energy['tab_no_smooth'][-1]:.3f}")

        for it in range(self.maxit):
            if self.verbose > 0:
                print(f"It {it}")
            # coarse2fine for lighting (start with sh_order1 and increase to sh_order2 after certain number of iterations)
            # sh_order2 will start if options.c2f_lighting before maximum number of iterations
            if it == self.c2f_lighting and self.sh_order == 1:
                self.sh_order += 1
                print(f"\tUse sh_order = {self.sh_order} now")
                s = torch.cat([s, torch.zeros(self.num_images, 5, self.num_channels)], dim=1)
                sh = self.normalsToSphericalHarmonics(n_unnormalized / theta)[0]

            last_rho, last_s, last_theta, last_z = self.saveOldIterates(rho, s, theta, z)
            nabla_rho, rho = self.updateAlbedo(dz, last_rho, rho, s, sh, theta)
            s = self.updateLighting(dz, last_s, nabla_rho, rho, s, sh, theta)
            J_dz, dz, irls_weights, n_unnormalized, z = self.updateDepth(last_z, nabla_rho, rho, s,
                                                                         sh, theta, z)
            sh, theta = self.auxiliaryUpdate(dz, irls_weights, last_theta, n_unnormalized,
                                             nabla_rho, rho, s)
            self.energy['energy_res'] = (self.energy['tab_energy_irls'][-5] -
                                         self.energy['tab_energy_irls'][-1]).abs() / \
                                        self.energy['tab_energy_irls'][-1]
            if self.verbose > 0:
                print(f"\tEnergy residual: {self.energy['energy_res']}")
            if self.energy['energy_res'] < self.tol:
                break

        if self.verbose > 0:
            print(f"Final Energy:")
            print(f"\t{self.energy['tab_energy'][-1]:.3f} "
                  f"{self.energy['tab_energy_irls'][-1]:.3f} "
                  f"{self.energy['tab_objective'][-1]:.3f} "
                  f"{self.energy['tab_no_smooth'][-1]:.3f}")

        self.z_out = vec2img(z, mask=self.mask)
        self.rho_out = vec2img(rho, mask=self.mask)
        self.s_out = s
        return self.z_out, self.rho_out, self.s_out

    def saveOldIterates(self, rho, s, theta, z):
        last_rho = rho.detach().clone()
        last_s = s.detach().clone()
        last_z = z.detach().clone()
        last_theta = theta.detach().clone()
        return last_rho, last_s, last_theta, last_z

    def updateAlbedo(self, dz, last_rho, rho, s, sh, theta):
        irls_weights = self.calcReweighting(rho, sh, s)
        rho, res_rho = self._updateAlbedo(rho, sh, s, irls_weights)
        nabla_rho = self.nabla_a @ rho
        self.energy['tab_rho_res'].append(res_rho)
        self.energy['tab_rho'].append((last_rho - rho).norm())
        self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)
        if self.verbose > 0:
            print(
                f"\tAfter albedo update: {self.energy['tab_energy'][-1]:.3f} "
                f"{self.energy['tab_energy_irls'][-1]:.3f} "
                f"{self.energy['tab_objective'][-1]:.3f} "
                f"{self.energy['tab_no_smooth'][-1]:.3f} "
                f"| res_rho: {res_rho}")
        return nabla_rho, rho

    def _updateAlbedo(self, rho, sh, s, irls_weights):
        shading = self._render(torch.ones(self.num_faces, self.num_channels), s, sh)
        if not self.__fix_clamp:
            shading = shading.clamp(min=0)
        a_full = torch.sqrt(2 * irls_weights) * shading
        a_data = (a_full * a_full).sum(0)
        b_ = (a_full * torch.sqrt(2 * irls_weights) * self.I).sum(0)

        A_reg = spzeros((self.num_faces, self.num_faces))
        if self.mu > 0:  # if huber regularization is used
            a_reg = 1 / (self.nabla_a @ rho).abs().clamp(min=self.huber)

        res_a = 0
        for ch in range(self.num_channels):
            # construct A and b.
            if self.mu > 0:  # if huber regularization is used
                Dk = spdiag(a_reg[:, ch:ch + 1].T, [0], (2 * self.num_faces, 2 * self.num_faces))
                A_reg = self.mu * transpose(self.nabla_a, 0, 1) @ Dk @ self.nabla_a

            A_data = spdiag(a_data[:, ch:ch + 1].T, [0], (self.num_faces, self.num_faces))
            b = b_[:, ch:ch + 1]

            # solve rho = A\b
            t_start = time.time()
            A = A_data + A_reg
            rho_ch = self._solve(A.double(), b.double(), rho[:, ch:ch + 1].double(),
                                 self.albedo_pcg_tol, self.albedo_pcg_maxit)
            t_stop = time.time()
            res_a_ch = (A.double() @ rho_ch - b.double()).norm()
            if self.verbose > 1:
                print(
                    f"\t\tAcurracy solving 'A @ rho_ch = b':{res_a_ch}")
            if self.verbose > 2:
                print(f"\t\tElapsed time for solving: 'A @ rho_ch = b': {t_stop - t_start}")
            rho[:, ch:ch + 1] = rho_ch.float().detach().clone()
            res_a += res_a_ch
        return rho, res_a

    def updateLighting(self, dz, last_s, nabla_rho, rho, s, sh, theta):
        irls_weights = self.calcReweighting(rho, sh, s)
        s, res_s = self._updateLighting(rho, s, sh, irls_weights)
        self.energy['tab_s_res'].append(res_s)
        self.energy['tab_s'].append(((last_s - s) ** 2).sum())
        self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)
        if self.verbose > 0:
            print(
                f"\tAfter light  update: {self.energy['tab_energy'][-1]:.3f} "
                f"{self.energy['tab_energy_irls'][-1]:.3f} "
                f"{self.energy['tab_objective'][-1]:.3f} "
                f"{self.energy['tab_no_smooth'][-1]:.3f} "
                f"| res_s: {res_s}")
        return s

    def _updateLighting(self, rho, s, sh, irls_weights):
        reweighted_rho = torch.sqrt(irls_weights) * rho
        reweighted_I = torch.sqrt(irls_weights) * self.I

        if not self.__fix_clamp:
            idx = self._render(torch.ones(self.num_faces, self.num_channels), s, sh) < 0
            reweighted_rho[idx] = 0
            reweighted_I[idx] = 0

        # (num_images, num_channels, num_faces, sh_dim)
        rhon_full = (reweighted_rho[..., None, :] * sh[None, ..., None]).permute(0, 3, 1, 2)
        # cast to double as it's much more accurate
        A = (transpose(rhon_full, -1, -2) @ rhon_full)
        b = (transpose(rhon_full, -1, -2) @ (reweighted_I.permute(0, 2, 1)[..., None]))
        t_start = time.time()
        s = self._pinv(A.double(), b.double())
        t_stop = time.time()
        if self.verbose > 2:
            print(f"\t\tElapsed time for solving: 'A @ s = b': {t_stop - t_start}")
        res_s = (A.double() @ s - b.double()).norm(dim=[-1, -2]).sum()
        return s.float().squeeze(-1).permute(0, 2, 1), res_s

    def updateDepth(self, last_z, nabla_rho, rho, s, sh, theta, z):
        irls_weights = self.calcReweighting(rho, sh, s)
        z, dz, n_unnormalized, sh, J_dz, res_z = self._updateDepth(rho, s, theta, z, irls_weights,
                                                                   nabla_rho)
        self.energy['tab_z_res'].append(res_z)
        self.energy['tab_z'].append((z - last_z).norm().clamp(min=self.eps))
        self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)
        if self.verbose > 0:
            print(
                f"\tAfter depth  update: {self.energy['tab_energy'][-1]:.3f} "
                f"{self.energy['tab_energy_irls'][-1]:.3f} "
                f"{self.energy['tab_objective'][-1]:.3f} "
                f"{self.energy['tab_no_smooth'][-1]:.3f}"
                f"| res_z: {res_z[-1]}")

        return J_dz, dz, irls_weights, n_unnormalized, z

    def _updateDepth(self, rho, s, theta, z, irls_weights, nabla_rho):
        res_z = torch.zeros(self.depth_linesearch_maxit)

        z0 = z.detach().clone()
        dz, n_unnormalized, J_n_un, J_dz = self.depthToNormals(z0)[1:]
        sh = self.normalsToSphericalHarmonics((n_unnormalized / theta))[0]
        if not self.__linesearch_strict:
            tab_objective = self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)[1]

        for i in range(self.depth_linesearch_maxit):
            'This loop solves (31) from the paper'
            if self.__linesearch_strict:
                tab_objective = \
                    self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)[1]
            rho_w = torch.sqrt(irls_weights) * rho
            I_w = torch.sqrt(irls_weights) * self.I

            if not self.__fix_clamp:
                idx = self._render(torch.ones(self.num_faces, self.num_channels), s, sh) < 0
                rho_w[idx] = 0
                I_w[idx] = 0

            F, b = self._buildWeightedLSMatrices(rho_w, I_w, s, n_unnormalized, dz, theta, J_n_un,
                                                 J_dz)

            t_start = time.time()
            z_step = self._solve(F.double(), b.double(), z0.double(),
                                 self.depth_pcg_tol, self.depth_pcg_maxit)
            t_stop = time.time()
            if self.verbose > 1:
                print(
                    f"\t\tAcurracy solving 'F @ z = b': {(F.double() @ z_step - b.double()).norm()}")
            if self.verbose > 2:
                print(f"\t\tElapsed time for solving: F @ z = b: {t_stop - t_start}")
            z_step = z_step.float()

            t = self.depth_linesearch_t
            t_step = 2 / (2 + 1 / t)
            energy_rec = []
            while True:
                'This loop solves (32) from the paper and ensures decrease of the irls energy in (31)'
                if self.verbose > 1:
                    print(
                        f"\r\t\tUpdateDepth: {i}/{self.depth_linesearch_maxit - 1} | {len(energy_rec)}/{self.depth_linesearch_maxit_linesearch - 1}",
                        end="", flush=True)
                z = gradientAscentStep(z0, t_step, z_step)
                dz, n_unnormalized = self.depthToNormals(z)[1:3]
                sh = self.normalsToSphericalHarmonics(n_unnormalized / theta)[0]
                objective = \
                    self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)[1]
                energy_rec.append(objective)

                if objective > tab_objective \
                        and len(energy_rec) < self.depth_linesearch_maxit_linesearch:
                    t *= 0.5
                    t_step = 2 / (2 + 1 / t)
                else:
                    if objective > tab_objective:
                        raise RuntimeWarning(f"Did NOT find descent step in z update")
                    z_last = z0.detach().clone()
                    z0 = z.detach().clone()
                    dz, n_unnormalized, J_n_un, J_dz = self.depthToNormals(z0)[1:]
                    sh = self.normalsToSphericalHarmonics(n_unnormalized / theta)[0]
                    if self.verbose > 1: print()
                    break
            res_z[i] = (1 / t_step * F @ (z0 - z_last) - b).norm().clamp(min=self.eps)
        return z, dz, n_unnormalized, sh, J_dz, res_z

    def _buildWeightedLSMatrices(self, rho_w, I_w, s, n_unnormalized, dz, theta, J_n_un, J_dz):
        'This function builds the matrices F and b to solve the weighted least squares problem resulting from (32) wrt negative gradient z, i.e. F @ x = b for x = -nabla_z'
        normals = n_unnormalized / theta
        J_n = [spdiag(1 / theta.T, [0], (self.num_faces, self.num_faces)) @ J for J in J_n_un]
        sh, J_sh = self.normalsToSphericalHarmonics(normals, J_n)

        factor_aug = torch.tensor([math.sqrt(0.5 * self.beta)])
        cost_aug = factor_aug * (theta - dz)
        J_aug = -spdiag(factor_aug.repeat(1, self.num_faces), [0],
                        (self.num_faces, self.num_faces)) @ J_dz

        J_cauchy = []
        for im in range(self.num_images):
            for ch in range(self.num_channels):
                J_sh_ic = spzeros(J_sh[0].shape)
                for si, J_shi in zip(s[im, :, ch], J_sh):
                    J_sh_ic += J_shi * si

                J_cauchy.append(spdiag(rho_w[im:im + 1, :, ch], [0],
                                       (self.num_faces, self.num_faces)) @ J_sh_ic)
        J_cauchy = vstack(J_cauchy)
        cost_cauchy = (self._render(rho_w, s, sh).squeeze(0) - I_w).permute(0, 2, 1).reshape(-1, 1)

        F = transpose(J_aug, 0, 1) @ J_aug + transpose(J_cauchy, 0, 1) @ J_cauchy
        b = -transpose(J_aug, 0, 1) @ cost_aug - transpose(J_cauchy, 0, 1) @ cost_cauchy

        return F, b

    def auxiliaryUpdate(self, dz, irls_weights, last_theta, n_unnormalized, nabla_rho, rho, s):
        theta = dz.detach().clone()
        sh = self.normalsToSphericalHarmonics((n_unnormalized / theta))[0]
        self.energy['tab_theta'].append((theta - last_theta).norm().clamp(min=self.eps))

        self.calcEnergyCauchy(rho, s, sh, theta, nabla_rho, dz, irls_weights)
        if self.verbose > 0:
            print(f"\tAfter theta  update: "
                  f"{self.energy['tab_energy'][-1]:.3f} "
                  f"{self.energy['tab_energy_irls'][-1]:.3f} "
                  f"{self.energy['tab_objective'][-1]:.3f} "
                  f"{self.energy['tab_no_smooth'][-1]:.3f}"
                  f"| res_theta: {self.energy['tab_theta_primal'][-1]}")

        self.beta = self.kappa * self.beta
        return sh, theta

    def calcReweighting(self, rho, sh, s):
        if self.irls.lower() == 'cauchy':
            rk = self.__residual(rho, s, sh)
            return 1 / (1 + (rk * rk) / (self.lambda_ * self.lambda_))
        elif self.irls.lower() == 'l2':
            return torch.ones(self.num_images, self.num_faces, self.num_channels)
        else:
            raise ValueError(f"Unknown value for irls: {self.irls}")

    def _pixelswrtPrincipalPoint(self):
        h, w = self.mask.shape[1:]
        x, y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        x = img2vec(x, mask=self.mask) - self.K[0, 2]
        y = img2vec(y, mask=self.mask) - self.K[1, 2]
        return x, y

    def depthToNormals(self, z):
        '''
        n(z(x,y)) = normalize((z_x(x,y), z_y(x,y), -1)) (orthographic projection)
        n(z(x,y)) = normalize((f_x * z_x(x,y), f_y * z_y(x,y), -(z(x,y) + (x - c_x) * z_x(x,y) + (y - c_y) * z_y(x,y)))) (perspective projection)

        :param z:
        :return: n(z), \tilde n(z) (unnormalized normal), dz(z) = |\tilde n(z)|, J_{\tilde n}(z), J_{dz}(z)
        '''
        normals, dz, n_unnormalized = depthToNormals(z, self.mask, self.K, self.nabla_z,
                                                     southern=True)

        def _calcJacobian():
            'Jacobians of n_unnormalized wrt. depth and dz wrt. depth'
            np = self.num_faces
            # Jacobian matrix of unnormalized normal regarding depth
            J_n_un = [None] * 3
            if self.K is not None:
                x, y = self._pixelswrtPrincipalPoint()
                J_n_un[0] = self.K[0, 0] * hstack([speye(np), spzeros((np, np))]) @ self.nabla_z
                J_n_un[1] = self.K[1, 1] * hstack([spzeros((np, np)), speye(np)]) @ self.nabla_z
                J_n_un[2] = - speye(np) \
                            - hstack(
                    [spdiag(x.T, [0], (np, np)), spdiag(y.T, [0], (np, np))]) @ self.nabla_z
            else:
                J_n_un[0] = hstack([speye(np), spzeros((np, np))]) @ self.nabla_z
                J_n_un[1] = hstack([spzeros((np, np)), speye(np)]) @ self.nabla_z
                J_n_un[2] = spzeros((np, np))

            # Jacobian matrix of the norm of unnormalized normal regarding depth
            J_dz = spdiag(normals[:, 0:1].T, [0], (np, np)) @ J_n_un[0] \
                   + spdiag(normals[:, 1:2].T, [0], (np, np)) @ J_n_un[1] \
                   + spdiag(normals[:, 2:3].T, [0], (np, np)) @ J_n_un[2]

            return J_n_un, J_dz

        return normals, dz, n_unnormalized, *_calcJacobian()

    def normalsToSphericalHarmonics(self, normals, J_n: Optional[torch.Tensor] = None):
        w = [None] * ((self.sh_order + 1) * (self.sh_order + 1))
        if self.sh_order >= 1:
            w[0] = math.sqrt(3 / (4 * math.pi))  # x
            w[1] = math.sqrt(3 / (4 * math.pi))  # y
            w[2] = math.sqrt(3 / (4 * math.pi))  # z
            w[3] = math.sqrt(1 / (4 * math.pi))  # constant

        if self.sh_order >= 2:
            w[4] = 3 * math.sqrt(5 / (12 * math.pi))  # 5
            w[5] = 3 * math.sqrt(5 / (12 * math.pi))  # 6
            w[6] = 3 * math.sqrt(5 / (12 * math.pi))  # 7
            w[7] = 3 / 2 * math.sqrt(5 / (12 * math.pi))  # 8
            w[8] = 0.5 * math.sqrt(5 / (4 * math.pi))  # 9

        w = torch.tensor(w)
        sh = normalsToSH(normals, dim=-1, sh_order=self.sh_order)

        def _calcJacobianwrtNormals():
            J_sh = [None] * ((self.sh_order + 1) * (self.sh_order + 1))
            if self.sh_order >= 1:
                J_sh[0] = w[0] * J_n[0]
                J_sh[1] = w[1] * J_n[1]
                J_sh[2] = w[2] * J_n[2]
                J_sh[3] = spzeros(J_n[0].shape)
            if self.sh_order >= 2:
                w5 = w[5] if self.__fix_J_sh else w[4]
                nx = spdiag(normals[:, 0:1].T, [0], (self.num_faces, self.num_faces))
                ny = spdiag(normals[:, 1:2].T, [0], (self.num_faces, self.num_faces))
                nz = spdiag(normals[:, 2:3].T, [0], (self.num_faces, self.num_faces))
                J_sh[4] = w[4] * (ny @ J_n[0] + nx @ J_n[1])
                J_sh[5] = w5 * (nz @ J_n[0] + nx @ J_n[2])
                J_sh[6] = w[6] * (nz @ J_n[1] + ny @ J_n[2])
                J_sh[7] = w[7] * 2 * (nx @ J_n[0] - ny @ J_n[1])
                J_sh[8] = w[8] * 6 * nz @ J_n[2]
            return J_sh

        return w[None, :] * sh, _calcJacobianwrtNormals() if J_n is not None else None

    def _pinv(self, A, b):
        assert A.dtype == torch.double and b.dtype == torch.double
        if self.__pinv == 'torch':
            return torch.linalg.pinv(A) @ b
        elif self.__pinv == 'numpy':
            return torch.from_numpy(np.linalg.pinv(A.cpu().numpy())).to(b.device) @ b
        elif self.__pinv == 'scipy':
            imgs = []
            for Ai in A:
                chs = []
                for Aij in Ai:
                    chs.append(torch.from_numpy(pinv(Aij.cpu().numpy())).to(b.device))
                imgs.append(torch.stack(chs))
            A_pinv = torch.stack(imgs)
            return A_pinv @ b
        else:
            raise RuntimeError(f"Unkown solver for 'A @ s = b': {self.__pinv}")

    @classmethod
    def _cholesky(cls, A, b, device):
        assert A.dtype == torch.double and b.dtype == torch.double
        return _cholesky(A, b, device)

    @classmethod
    def _cg(cls, A, b, x0, tol, maxit, device):
        assert A.dtype == torch.double and b.dtype == torch.double
        assert x0.dtype == torch.double
        return _cg(A, b, x0, tol, maxit, device)

    def _solve(self, A, b, x0, tol, maxit):
        assert A.dtype == torch.double and b.dtype == torch.double
        assert x0 is None or x0.dtype == torch.double

        if self.__solve == 'cholesky':
            try:
                return self._cholesky(A, b, b.device)
            except:
                return self._cg(A, b, x0, tol, maxit, b.device)
        elif self.__solve == 'cg':
            try:
                return self._cg(A, b, x0, tol, maxit, b.device)
            except:
                return self._cholesky(A, b, b.device)
        elif self.__solve == 'best':
            try:
                x1 = self._cholesky(A, b, b.device)
            except:
                return self._cg(A, b, x0, tol, maxit, b.device)
            try:
                x2 = self._cg(A, b, x0, tol, maxit, b.device)
            except:
                return self._cholesky(A, b, b.device)
            return x1 if (A @ x1 - b).norm() < (A @ x2 - b).norm() else x2
        else:
            raise RuntimeError(f"Unkown solver for 'F @ z = b': {self.__solve}")


def evaluate(mask, K, z_out, rho_out, l_out, z_gt, rho_gt, l_gt):
    #####################
    # Albedo evaluation #
    #####################
    rmse_a = -torch.ones(1)
    if rho_gt is not None:
        rho_gt_vec, rho_out_vec = img2vec(rho_gt, mask), img2vec(rho_out, mask)
        c_albedo = optC(rho_gt_vec, rho_out_vec, dim=0)
        rmse_a = calcRMSE(rho_gt_vec, c_albedo * rho_out_vec, dim=[-1, -2])

    #######################
    # Lighting evaluation #
    #######################
    rmse_s = -torch.ones(1)
    ae_s = -torch.ones(1)
    if l_gt is not None:
        l_out_convex = l_out.detach().clone()
        l_out_concave = l_out.detach().clone()
        l_out_concave[:, 0:2, :] *= -1
        ae_s_convex = calcAngularError(l_gt, l_out_convex, dim=-2)
        ae_s_concave = calcAngularError(l_gt, l_out_concave, dim=-2)
        l_out = l_out_convex if ae_s_convex.mean() < ae_s_concave.mean() else l_out_concave
        ae_s = ae_s_convex if ae_s_convex.mean() < ae_s_concave.mean() else ae_s_concave
        c_light = optC(l_gt, l_out, dim=[0, 1])
        rmse_s = calcRMSE(l_gt, c_light * l_out, dim=-2)

    ######################
    # normals evaluation #
    ######################
    ae_n = -torch.ones(1)
    if z_gt is not None:
        n_out_convex = depthToNormals(img2vec(z_out, mask), mask, K)[0]
        n_gt_vec = depthToNormals(img2vec(z_gt, mask), mask, K)[0]
        n_out_concave = n_out_convex.detach().clone()
        n_out_concave[:, :-1] *= -1
        ae_n_convex = calcAngularError(n_gt_vec, n_out_convex, dim=-1)
        ae_n_concave = calcAngularError(n_gt_vec, n_out_concave, dim=-1)
        ae_n = ae_n_convex if ae_n_convex.mean() < ae_n_concave.mean() else ae_n_concave

    ####################
    # depth evaluation #
    ####################
    rmse_z = -torch.ones(1)
    if z_gt is not None:
        z_out_vec = img2vec(z_out, mask)
        z_gt_vec = img2vec(z_gt, mask)
        c_z = optC(z_gt_vec, z_out_vec, dim=0)
        rmse_z = calcRMSE(z_gt_vec, c_z * z_out_vec, dim=0)

    print(f"Error metrics: ")
    print(
        f"\trmse albedo: {rmse_a.mean()}\n\trmse_s: {rmse_s.mean()}\n\tae_s: {ae_s.mean()}\n\tae_n: {ae_n.mean()}\n\trmse_z: {rmse_z.mean()}")
    results = {"rmse_a": rmse_a.mean().item(),
               "rmse_s": rmse_s.mean().item(),
               "ae_s": ae_s.mean().item(),
               "ae_n": ae_n.mean().item(),
               "rmse_z": rmse_z.mean().item()}

    return results


def save(io: DataIO, mask, K, rho_out, l_out, z_out, results: Optional[Dict] = None):
    ###############
    # Albedo save #
    ###############
    io.saveImage(f'albedo_out.png', img2vec(rho_out, mask), mask=mask)
    io.saveImage(f'albedo_out.pth', rho_out)

    #################
    # Lighting save #
    #################
    io.saveLighting(f'lighting_out/lighting_out.png', l_out)
    io.saveLighting(f'lighting_out.pth', l_out)

    ################
    # normals save #
    ################
    n_out_convex = depthToNormals(img2vec(z_out, mask), mask, K, southern=False)[0]
    n_out_concave = n_out_convex.detach().clone()
    n_out_concave[:, :-1] *= -1
    io.saveNormals(f'normals_convex.png', n_out_convex, mask=mask)
    io.saveNormals(f'normals_convex.pth', vec2img(n_out_convex, mask=mask))
    io.saveNormals(f'normals_concave.png', n_out_concave, mask=mask)
    io.saveNormals(f'normals_concave.pth', vec2img(n_out_concave, mask=mask))

    ##############
    # depth save #
    ##############
    io.saveDepth(f'z_out.png', img2vec(z_out, mask), mask=mask)
    io.saveDepth(f'z_out.pth', z_out)
    io.saveDepth(f'z_out.obj', z_out, mask=mask, K=K)

    if results is not None:
        io.saveDict(f'results.txt', results)
        io.saveDict(f'results.json', results)
        io.saveDict(f'results.pkl', results)

    return results


def parser():
    parser = argparse.ArgumentParser(prog='GeneralUPS',
                                     description='A variational solver for uncalibrated photometric stereo under general lighting',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help='Run manually or example', dest="subcommand")
    subparsers.add_parser("example", help="Run example")
    sub_man = subparsers.add_parser("manual", help="Run manually. Run \"manual -h\" for more info",
                                    add_help=False,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group_required = sub_man.add_argument_group('required named arguments')
    group_required.add_argument('-m', '--mask', required=True, type=str,
                                help='Mask for image files as .png or .pth-file [%(type)s]')
    group_required.add_argument('-i', '--images', required=True, type=str, nargs="+",
                                help='Image files as .png or .pth-file [%(type)s]')
    group_required.add_argument('--init_depth', required=True, type=str,
                                help="Depth for initialization, e.g. ballooning, as .pth-file [%(type)s]")

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
                                help="Ground truth albedo as .png or .pth-file [%(type)s]")

    group_algo = sub_man.add_argument_group('algorithmic options')
    group_algo.add_argument('--sh_order', default=1, type=int, help="Initial SH order")
    group_algo.add_argument('--c2f_lighting', default=8, type=int,
                            help="After c2f_lighting iterations sh_order is increased to 2, if it's 1")
    group_algo.add_argument('--irls', default="cauchy", type=str, choices=["cauchy", "l2"],
                            help="Used M-Estimator")
    group_algo.add_argument('--lambda', default=1., dest="lambda_", type=float,
                            help="Cauchy weight")
    group_algo.add_argument('--delta', default=4.5e-4, type=float,
                            help="Only needed if lambda is None")
    group_algo.add_argument('--mu', default=0.045245, type=float, help="Trade-off parameter")
    group_algo.add_argument('--huber', default=0.1, type=float, help="Huber weight")
    group_algo.add_argument('--beta_init', default=5e-4, type=float,
                            help="Trade-off parameter for soft constraint resulting from lagging theta, Eq. (25) in paper")
    group_algo.add_argument('--kappa', default=1, type=float,
                            help="Increase soft constraint trade-off parameter: beta_new = kappa * beta_old after every iteration")
    group_algo.add_argument('--maxit', default=20, type=int, help="Max number iterations")
    group_algo.add_argument('--tol', default=1e-6, type=float, help="Convergence tolerance")
    group_algo.add_argument('--eps', default=1e-6, type=float, help="Well, eps...")
    group_algo.add_argument('--albedo_pcg_tol', default=1e-6, type=float,
                            help="Albedo PCG tolerance")
    group_algo.add_argument('--albedo_pcg_maxit', default=1000, type=int,
                            help="Albedo PCG iterations")
    group_algo.add_argument('--depth_pcg_tol', default=1e-10, type=float,
                            help="Depth PCG tolerance")
    group_algo.add_argument('--depth_pcg_maxit', default=1000, type=int,
                            help="Depth PCG iterations")
    group_algo.add_argument('--depth_linesearch_maxit', default=3, type=int,
                            help="Maximum number of iterations for weighted least squares depth update")
    group_algo.add_argument('--depth_linesearch_t', default=10, type=float,
                            help="Initial line search step size according to: 2 / (2 + 1 / t)")
    group_algo.add_argument('--depth_linesearch_maxit_linesearch', default=1000, type=int,
                            help="Maximum number of iterations of line search for linear weighted least squares depth update")

    parser_system = parser.add_argument_group('system options')
    parser_system.add_argument('-g', '--gpu', default=False, action="store_true",
                               help="Use GPU (if not available, fall back to CPU)")
    parser_system.add_argument('--gpu_id', default=0, type=int, help="GPU ID [%(type)s]")
    output_default = os.path.join(pathlib.Path(__file__).parent.resolve(), 'output', 'general_ups')
    parser_system.add_argument('-o', '--output', default=output_default, type=str,
                               help=f"Output directory [%(type)s]")

    cli = parser.parse_args()
    if cli.subcommand == 'example':
        args = ['-g',
                f'-o={os.path.join(output_default, "synthetic_joyfulyell_hippie")}',
                'manual',
                f'--maxit=36',
                f'-m={os.path.join("data", "synthetic_joyfulyell_hippie", "mask.png")}',
                f'-i={os.path.join("data", "synthetic_joyfulyell_hippie", "images.pth")}',
                f'-k={os.path.join("data", "synthetic_joyfulyell_hippie", "K.pth")}',
                f'--init_depth={os.path.join("data", "synthetic_joyfulyell_hippie", "z_init.pth")}',
                f'-z={os.path.join("data", "synthetic_joyfulyell_hippie", "z_gt.pth")}',
                f'-l={os.path.join("data", "synthetic_joyfulyell_hippie", "l_gt_25x9x3.pth")}',
                f'-a={os.path.join("data", "synthetic_joyfulyell_hippie", "rho_gt.pth")}']
        print(f"Execute 'general_ups.py {' '.join(args)}'")
        return parser.parse_args(args)
    else:
        return cli


def main(cli):
    io = DataIO(gpu=cli.gpu, gpu_id=cli.gpu_id, output=cli.output)

    mask = io.loadMask(cli.mask)
    I = io.loadImages(cli.images)
    K = io.loadIntrinsics(cli.intrinsics) if cli.intrinsics else None

    z_init = io.loadImage(cli.init_depth)
    l_init = io.loadLighting(cli.init_light) if cli.init_light else None
    rho_init = io.loadImage(cli.init_albedo) if cli.init_albedo else None

    z_gt = io.loadImage(cli.gt_depth) if cli.gt_depth else None
    l_gt = io.loadLighting(cli.gt_light) if cli.gt_light else None
    rho_gt = io.loadImage(cli.gt_albedo) if cli.gt_albedo else None

    gups = GeneralUPS(sh_order=cli.sh_order, c2f_lighting=cli.c2f_lighting,
                      irls=cli.irls, lambda_=cli.lambda_, mu=cli.mu, huber=cli.huber,
                      delta=cli.delta,
                      beta_init=cli.beta_init, kappa=cli.kappa,
                      maxit=cli.maxit, tol=cli.tol, eps=cli.eps,
                      albedo_pcg_tol=cli.albedo_pcg_tol, albedo_pcg_maxit=cli.albedo_pcg_maxit,
                      depth_pcg_tol=cli.depth_pcg_tol, depth_pcg_maxit=cli.depth_pcg_maxit,
                      depth_linesearch_maxit=cli.depth_linesearch_maxit,
                      depth_linesearch_t=cli.depth_linesearch_t,
                      depth_linesearch_maxit_linesearch=cli.depth_linesearch_maxit_linesearch)
    t_start = time.time()
    gups.init(I, mask, K, z_init, rho_init, l_init)
    z_out, rho_out, l_out = gups.run()
    t_stop = time.time()
    print(f"Elapsed time for optimization: {t_stop - t_start}")

    error_metrics = None
    if z_gt is not None or l_gt is not None or rho_gt is not None:
        error_metrics = evaluate(mask, K,
                                 z_out, rho_out, l_out,
                                 z_gt, rho_gt, l_gt)
    if cli.output != "":
        save(io, mask, K, rho_out, l_out, z_out, error_metrics)
    return


if __name__ == "__main__":
    main(parser())
