import math
import numpy as np
import torch
from torch import nn
from utils.geometric import pairwise_distance, calc_angle, calc_dihedral
from . import spline
from common import config
from common.config import EPS
from common import constants


class OmegaRestraint(nn.Module):
    """Omega angle is defined as dehidral (CA_i, CB_i, CB_j, CA_j)"""

    def __init__(self, pred_omega):
        """
        pred_omega has shape (L, L, 37)
        """
        super().__init__()
        L = pred_omega.shape[0]
        _filter = torch.tensor(pred_omega[:, :, -1] < config.OMEGA_CUT)
        self.mask = nn.Parameter(
            torch.triu(torch.ones((L, L)).bool(), diagonal=1).__and__(_filter),
            requires_grad=False,
        )

        _step = 15.0 * math.pi / 180.0
        self.cutoffs = torch.linspace(-math.pi + 0.5 * _step, math.pi + 0.5 * _step, steps=25)
        _x = self.cutoffs
        _ref = -np.log(constants.bg_omega)
        _y = -np.log(pred_omega[:, :, :-1] + EPS) - _ref
        _y = np.concatenate([_y, _y[:, :, :1]], axis=-1)
        self.coeff = nn.Parameter(spline.cubic_spline(_x, _y, period=True), requires_grad=False)
        self.cutoffs = nn.Parameter(self.cutoffs, requires_grad=False)

    def __str__(self):
        return "Omega constraints: %i" % torch.sum(self.mask).item()

    def forward(self, coord):
        B = coord.CA.shape[0]
        x_idx, y_idx = torch.where(self.mask)
        x_CA = coord.CA[:, x_idx].view(-1, 3)
        x_CB = coord.CB[:, x_idx].view(-1, 3)
        y_CA = coord.CA[:, y_idx].view(-1, 3)
        y_CB = coord.CB[:, y_idx].view(-1, 3)
        x_idx, y_idx = x_idx.repeat(B), y_idx.repeat(B)
        x_idx, y_idx, omega = calc_dihedral(x_CA, x_CB, y_CB, y_CA, x_idx, y_idx)

        coeff = self.coeff[x_idx, y_idx]
        omega_potential = torch.sum(spline.evaluate(coeff, self.cutoffs, omega))

        return {
            "pairwise_omega": omega_potential,
        }


class ThetaRestraint(nn.Module):
    """Theta angle is defined as dehidral (N_i, CA_i, CB_i, CB_j)"""

    def __init__(self, pred_theta):
        """
        pred_theta has shape (L, L, 37)
        """
        super().__init__()
        L = pred_theta.shape[0]
        _filter = torch.tensor(pred_theta[:, :, -1] < config.THETA_CUT)
        self.mask = nn.Parameter(
            (torch.eye(pred_theta.shape[0]) == 0).__and__(_filter),
            requires_grad=False,
        )

        _step = 15.0 * math.pi / 180.0
        self.cutoffs = torch.linspace(-math.pi + 0.5 * _step, math.pi + 0.5 * _step, steps=25)
        _x = self.cutoffs
        _ref = -np.log(constants.bg_theta)
        _y = -np.log(pred_theta[:, :, :-1] + EPS) - _ref
        _y = np.concatenate([_y, _y[:, :, :1]], axis=-1)
        self.coeff = nn.Parameter(spline.cubic_spline(_x, _y, period=True), requires_grad=False)
        self.cutoffs = nn.Parameter(self.cutoffs, requires_grad=False)

    def __str__(self):
        return "Theta constraints: %i" % torch.sum(self.mask).item()

    def forward(self, coord):
        B = coord.CA.shape[0]
        x_idx, y_idx = torch.where(self.mask)
        x_N = coord.N[:, x_idx].view(-1, 3)
        x_CA = coord.CA[:, x_idx].view(-1, 3)
        x_CB = coord.CB[:, x_idx].view(-1, 3)
        y_CB = coord.CB[:, y_idx].view(-1, 3)
        x_idx, y_idx = x_idx.repeat(B), y_idx.repeat(B)
        x_idx, y_idx, theta = calc_dihedral(x_N, x_CA, x_CB, y_CB, x_idx, y_idx)

        coeff = self.coeff[x_idx, y_idx]
        theta_potential = torch.sum(spline.evaluate(coeff, self.cutoffs, theta))

        return {
            "pairwise_theta": theta_potential,
        }


class PhiRestraint(nn.Module):
    def __init__(self, pred_phi):
        super().__init__()
        step = 15.0 * math.pi / 180.0
        self.cutoffs = torch.linspace(-1.5 * step, np.pi + 1.5 * step, steps=12 + 4)
        _x = self.cutoffs.numpy().tolist()
        _ref = -np.log(constants.bg_phi)
        _y = -np.log(pred_phi[:, :, :-1] + EPS) - _ref
        _y = np.concatenate([np.flip(_y[:, :, :2], axis=-1), _y, np.flip(_y[:, :, -2:], axis=-1)], axis=-1)
        self.coeff = nn.Parameter(spline.cubic_spline(_x, _y), requires_grad=False)
        _filter = torch.tensor(pred_phi[:, :, -1] < config.PHI_CUT)
        self.mask = nn.Parameter(
            (torch.eye(pred_phi.shape[0]) == 0).__and__(_filter),
            requires_grad=False,
        )
        self.cutoffs = nn.Parameter(self.cutoffs, requires_grad=False)

    def __str__(self):
        return "Phi constraints: %i" % torch.sum(self.mask).item()

    def forward(self, coord):
        B = coord.CA.shape[0]
        x_idx, y_idx = torch.where(self.mask)
        x_CA = coord.CA[:, x_idx].view(-1, 3)
        x_CB = coord.CB[:, x_idx].view(-1, 3)
        y_CB = coord.CB[:, y_idx].view(-1, 3)
        x_idx, y_idx = x_idx.repeat(B), y_idx.repeat(B)
        x_idx, y_idx, phi = calc_angle(x_CA, x_CB, y_CB, x_idx, y_idx)

        coeff = self.coeff[x_idx, y_idx]
        phi_potential = torch.sum(spline.evaluate(coeff, self.cutoffs, phi))

        return {
            "pairwise_phi": phi_potential,
        }
